"""Objective 3 — 3D image segmentation.

Pipeline (preliminary, classical baseline):

1. Locate the tumor on the last frame of the dynamic PET. The proposal
   asks for a manual centroid + bounding box. The current implementation
   uses a smoothed-intensity peak inside the Z range that overlaps with
   the MR coverage, then applies a manual refinement when the
   ``MANUAL_CENTROID_PET_NATIVE`` constant below is set.
2. Map the bounding box from PET native voxel space to MR voxel space
   through the DICOM affines built in ``coregistration``.
3. Run a 3D region-growing segmentation seeded from the centroid inside
   that bounding box on the MR T1 volume. This is the placeholder we use
   while we evaluate the AI general-purpose models named in the proposal
   (MedSAM2 / nnInteractive / SAMed-2 / SAT). The interface is ready to
   plug a model in: it takes the same prompt (centroid + bbox) and
   returns a binary mask in MR voxel space.
4. Save figures of the mask overlaid on the MR and on the coregistered
   PET, plus a binary mask volume in NIfTI-friendly NumPy format.

Run:

    python src/segmentation.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage as ndi
from skimage.morphology import flood

from coregistration import (
    apply_rigid_to_volume,
    resample_to_grid,
    voxel_to_physical_affine,
)
from loading import compute_temporal_mean, load_dynamic_pet, load_mr
from utils import filepath


# ---------------------------------------------------------------------------
# Manual centroid override (set to None to fall back to automatic peak)
# ---------------------------------------------------------------------------
# Format: (z, y, x) in PET native voxel coordinates. None disables the
# override and the automatic peak finder is used instead.
MANUAL_CENTROID_PET_NATIVE: tuple | None = None
BBOX_HALF_EXTENT_PET_NATIVE = (4, 25, 25)  # half sizes (z, y, x) around centroid


# ---------------------------------------------------------------------------
# Centroid + bbox in PET native space
# ---------------------------------------------------------------------------

def find_candidate_centroid(
    pet_last_frame: np.ndarray,
    pet_affine: np.ndarray,
    mr_affine: np.ndarray,
    mr_shape: tuple,
    sigma: float = 2.0,
    threshold_pct: float = 0.7,
) -> tuple:
    """Find the brightest connected blob in the PET last frame whose
    voxels project inside the MR volume.

    The PET dynamic acquisition covers a Z range that extends below the
    MR coverage; we filter to the overlapping range so the candidate
    centroid is always usable downstream.
    """
    nz_pet = pet_last_frame.shape[0]
    # Compute the lowest PET z-slice that still falls inside the MR volume.
    z_indices = np.arange(nz_pet)
    physical_z = pet_affine[:3, 0][2] * z_indices + pet_affine[2, 3]
    mr_z_min = mr_affine[2, 3]
    mr_z_max = mr_affine[2, 3] + mr_affine[:3, 0][2] * (mr_shape[0] - 1)
    valid = np.where((physical_z >= mr_z_min) & (physical_z <= mr_z_max))[0]
    z_lo, z_hi = int(valid.min()), int(valid.max() + 1)

    slab = pet_last_frame[z_lo:z_hi].astype(np.float32)
    smoothed = ndi.gaussian_filter(slab, sigma=sigma)
    threshold = threshold_pct * smoothed.max()
    mask = smoothed > threshold
    labels, n = ndi.label(mask)
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    largest = 1 + int(np.argmax(sizes))
    mask_largest = labels == largest
    cz, cy, cx = ndi.center_of_mass(mask_largest)
    return float(cz + z_lo), float(cy), float(cx)


def bbox_from_centroid(centroid_voxel: tuple, half_extent: tuple, shape: tuple) -> tuple:
    """Build a (z, y, x) bounding box around ``centroid_voxel`` clipped to ``shape``."""
    z, y, x = centroid_voxel
    dz, dy, dx = half_extent
    z_lo = max(0, int(round(z - dz)))
    z_hi = min(shape[0], int(round(z + dz)) + 1)
    y_lo = max(0, int(round(y - dy)))
    y_hi = min(shape[1], int(round(y + dy)) + 1)
    x_lo = max(0, int(round(x - dx)))
    x_hi = min(shape[2], int(round(x + dx)) + 1)
    return ((z_lo, z_hi), (y_lo, y_hi), (x_lo, x_hi))


def pet_voxel_to_mr_voxel(voxel_pet: tuple, pet_affine: np.ndarray, mr_affine: np.ndarray) -> tuple:
    """Map a (z, y, x) PET native voxel onto MR voxel coordinates."""
    cz, cy, cx = voxel_pet
    physical = pet_affine @ np.array([cz, cy, cx, 1.0])
    mr_voxel = np.linalg.inv(mr_affine) @ physical
    return float(mr_voxel[0]), float(mr_voxel[1]), float(mr_voxel[2])


# ---------------------------------------------------------------------------
# Classical baseline segmentation (region growing)
# ---------------------------------------------------------------------------

def region_growing_in_bbox(
    volume: np.ndarray,
    seed_voxel: tuple,
    bbox: tuple,
    tolerance: float | None = None,
) -> np.ndarray:
    """3D region growing from ``seed_voxel`` constrained to ``bbox``.

    ``tolerance`` is the maximum absolute intensity difference between a
    voxel and the seed for the voxel to be included. If None, it defaults
    to one quarter of the seed value.
    """
    sz, sy, sx = (int(round(s)) for s in seed_voxel)
    sz = max(0, min(volume.shape[0] - 1, sz))
    sy = max(0, min(volume.shape[1] - 1, sy))
    sx = max(0, min(volume.shape[2] - 1, sx))

    seed_value = float(volume[sz, sy, sx])
    if tolerance is None:
        tolerance = max(50.0, abs(seed_value) * 0.25)

    # Confine the flood to the bbox by zeroing out everything else
    box_volume = np.zeros_like(volume, dtype=np.float32)
    (z0, z1), (y0, y1), (x0, x1) = bbox
    box_volume[z0:z1, y0:y1, x0:x1] = volume[z0:z1, y0:y1, x0:x1].astype(np.float32)

    mask = flood(box_volume, (sz, sy, sx), tolerance=float(tolerance))
    return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _planes_with_mask(volume, mask, cmap_volume, vmax_volume, suptitle, out_path):
    """Save a 3-panel figure (axial / coronal / sagittal) of volume overlaid with mask contour."""
    z, y, x = volume.shape
    if mask.any():
        zs, ys, xs = np.where(mask)
        cz, cy, cx = int(zs.mean()), int(ys.mean()), int(xs.mean())
    else:
        cz, cy, cx = z // 2, y // 2, x // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(volume[cz], cmap=cmap_volume, vmin=0, vmax=vmax_volume)
    axes[0].contour(mask[cz], levels=[0.5], colors="cyan", linewidths=1.5)
    axes[0].set_title(f"Axial (z={cz})")
    axes[0].axis("off")

    axes[1].imshow(volume[:, cy, :], cmap=cmap_volume, vmin=0, vmax=vmax_volume, origin="lower")
    axes[1].contour(mask[:, cy, :], levels=[0.5], colors="cyan", linewidths=1.5)
    axes[1].set_title(f"Coronal (y={cy})")
    axes[1].axis("off")

    axes[2].imshow(volume[:, :, cx], cmap=cmap_volume, vmin=0, vmax=vmax_volume, origin="lower")
    axes[2].contour(mask[:, :, cx], levels=[0.5], colors="cyan", linewidths=1.5)
    axes[2].set_title(f"Sagittal (x={cx})")
    axes[2].axis("off")

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    pet_path = filepath("e_1_BRAIN_DINAMIC_COLINA.dcm")
    mr_path = filepath("AX_3D_T1.dcm")

    print("Loading volumes ...")
    pet = load_dynamic_pet(pet_path)
    mr = load_mr(mr_path)
    last_frame = pet.array[-1]
    mr_vol = mr.array.astype(np.float32)

    pet_affine = voxel_to_physical_affine(pydicom.dcmread(pet_path, stop_before_pixels=True))
    mr_affine = voxel_to_physical_affine(pydicom.dcmread(mr_path, stop_before_pixels=True))

    print("Locating tumor candidate centroid in PET last frame ...")
    if MANUAL_CENTROID_PET_NATIVE is not None:
        centroid_pet = MANUAL_CENTROID_PET_NATIVE
        print(f"  Using MANUAL_CENTROID_PET_NATIVE override: {centroid_pet}")
    else:
        centroid_pet = find_candidate_centroid(
            last_frame, pet_affine, mr_affine, mr_vol.shape,
        )
        print(f"  Auto-located centroid (PET native voxels): {tuple(round(c, 1) for c in centroid_pet)}")

    bbox_pet = bbox_from_centroid(centroid_pet, BBOX_HALF_EXTENT_PET_NATIVE, last_frame.shape)
    print(f"  Bbox PET native: {bbox_pet}")

    print("Mapping centroid + bbox onto MR voxel grid ...")
    centroid_mr = pet_voxel_to_mr_voxel(centroid_pet, pet_affine, mr_affine)
    print(f"  Centroid MR voxels: {tuple(round(c, 1) for c in centroid_mr)}")

    # Build the MR-space bbox by mapping the 8 corners of the PET bbox.
    (zlo, zhi), (ylo, yhi), (xlo, xhi) = bbox_pet
    corners = np.array([(z, y, x) for z in (zlo, zhi - 1) for y in (ylo, yhi - 1) for x in (xlo, xhi - 1)])
    mr_corners = np.array([pet_voxel_to_mr_voxel(c, pet_affine, mr_affine) for c in corners])
    mr_bbox = (
        (max(0, int(mr_corners[:, 0].min())), min(mr_vol.shape[0], int(mr_corners[:, 0].max()) + 1)),
        (max(0, int(mr_corners[:, 1].min())), min(mr_vol.shape[1], int(mr_corners[:, 1].max()) + 1)),
        (max(0, int(mr_corners[:, 2].min())), min(mr_vol.shape[2], int(mr_corners[:, 2].max()) + 1)),
    )
    print(f"  Bbox MR voxel: {mr_bbox}")

    print("Running region-growing baseline segmentation on the MR (placeholder for AI model) ...")
    mask_mr = region_growing_in_bbox(mr_vol, centroid_mr, mr_bbox, tolerance=None)
    print(f"  Mask volume (voxels): {int(mask_mr.sum())}")

    print("Saving overlay figures ...")
    vmax_mr = float(np.percentile(mr_vol, 99.5))
    _planes_with_mask(
        mr_vol, mask_mr, "gray", vmax_mr,
        "Region-growing baseline mask on MR T1 (cyan contour)",
        os.path.join(out_dir, "07_segmentation_mask_on_mr.png"),
    )

    # Also overlay on the registered PET if available (so the mask sits in
    # the same voxel space as both volumes).
    coreg_path = os.path.join(out_dir, "coreg_transform.npz")
    if os.path.exists(coreg_path):
        npz = np.load(coreg_path)
        pet_mean = compute_temporal_mean(pet.array).astype(np.float32)
        pet_on_mr = resample_to_grid(pet_mean, pet_affine, mr_affine, dst_shape=mr_vol.shape, order=1)
        forward = npz["forward_params"]
        pet_registered = apply_rigid_to_volume(pet_on_mr, forward)
        vmax_pet = float(np.percentile(pet_registered, 99.5))
        _planes_with_mask(
            pet_registered, mask_mr, "hot", vmax_pet,
            "Region-growing mask on coregistered PET temporal mean (cyan contour)",
            os.path.join(out_dir, "08_segmentation_mask_on_pet.png"),
        )

    np.save(os.path.join(out_dir, "tumor_mask_mr.npy"), mask_mr)
    print(f"  Saved mask volume -> {os.path.join(out_dir, 'tumor_mask_mr.npy')}")

    # Closing the loop: bring the mask back into PET native space so the
    # tumor segmentation can be visualized on the input image of the
    # coregistration (Objective 2 mask + assessment).
    if os.path.exists(coreg_path):
        print("Propagating mask MR -> PET native via inverse rigid + inverse resample ...")
        inverse_params = npz["inverse_params"]
        mask_pre_rigid = apply_rigid_to_volume(mask_mr.astype(np.float32), inverse_params, order=0)
        mask_pet_native = resample_to_grid(
            mask_pre_rigid, mr_affine, pet_affine, dst_shape=last_frame.shape, order=0,
        )
        mask_pet_native = (mask_pet_native > 0.5).astype(np.uint8)
        np.save(os.path.join(out_dir, "tumor_mask_pet_native.npy"), mask_pet_native)
        print(f"  Mask voxels in PET native: {int(mask_pet_native.sum())}")

        vmax_last = float(np.percentile(last_frame, 99.5))
        _planes_with_mask(
            last_frame.astype(np.float32), mask_pet_native, "hot", vmax_last,
            "Tumor mask propagated to PET native space — last frame (cyan contour)",
            os.path.join(out_dir, "09_segmentation_mask_on_pet_native.png"),
        )

    print("Done.")
