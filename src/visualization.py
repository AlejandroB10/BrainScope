"""Rotating MIP animations for the project.

Implements Objective 2.b of the proposal: a GIF that displays a rotating
Maximum Intensity Projection on the coronal-sagittal plane of

    i)  the reference image (MR T1),
    ii) the coregistered input image (PET temporal mean on MR grid),
    iii) an alpha fusion of both.

The volume is rotated around its axial (z) axis and at each angle a
coronal MIP is computed by collapsing the y axis after rotation. Stacking
the frames produces a smooth turntable view.

Run via the coregistration script (this module is imported there) or
directly:

    python src/visualization.py
"""

from __future__ import annotations

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi  # intentionally kept: ndi.rotate drives the rotating-MIP display loop
import SimpleITK as sitk

from coregistration import load_rigid_transform, resample_to_reference
from loading import compute_temporal_mean, load_dynamic_pet, load_mr
from utils import filepath


def _rotated_coronal_mip(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around the axial axis (z) and collapse the y axis with max.

    ndi.rotate is kept here because this is a display effect, not a
    geometric resample. The rotation is applied to the numpy array directly.
    """
    rotated = ndi.rotate(volume, angle_deg, axes=(1, 2), reshape=False, order=1, cval=0.0)
    return rotated.max(axis=1)


def save_rotating_mip(
    reference: np.ndarray,
    registered: np.ndarray,
    out_path: str,
    n_angles: int = 24,
    fps: int = 6,
) -> dict:
    """Generate the rotating MIP GIF with reference / registered / alpha fusion."""
    angles = np.linspace(0, 360, n_angles, endpoint=False)

    ref_vmax = float(np.percentile(reference, 99.5))
    reg_vmax = float(np.percentile(registered, 99.5))

    frames = []
    for angle in angles:
        ref_mip = _rotated_coronal_mip(reference, angle)
        reg_mip = _rotated_coronal_mip(registered, angle)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

        axes[0].imshow(ref_mip, cmap="gray", vmin=0, vmax=ref_vmax, origin="lower")
        axes[0].set_title("Reference (MR T1)")
        axes[0].axis("off")

        axes[1].imshow(reg_mip, cmap="hot", vmin=0, vmax=reg_vmax, origin="lower")
        axes[1].set_title("Coregistered PET")
        axes[1].axis("off")

        axes[2].imshow(ref_mip, cmap="gray", vmin=0, vmax=ref_vmax, origin="lower")
        axes[2].imshow(reg_mip, cmap="hot", vmin=0, vmax=reg_vmax, alpha=0.45, origin="lower")
        axes[2].set_title("Alpha fusion")
        axes[2].axis("off")

        fig.suptitle(f"Rotating MIP — yaw {int(angle):3d}°", fontsize=12)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(out_path, frames, format="GIF", fps=fps)
    return {"path": out_path, "frames": len(frames)}


def save_segmentation_comparison(
    mr_image: sitk.Image,
    masks: dict,
    out_path: str,
) -> None:
    """Save a multi-method segmentation comparison figure.

    Shows three axial slices through the tumor centroid (or the overall
    centroid of all non-empty masks combined) with each method's mask
    rendered as a colour overlay on the MR T1.

    Parameters
    ----------
    mr_image:
        T1 MR volume as sitk.Image.
    masks:
        Dict mapping method name (str) to binary sitk.Image mask.
        Typical keys: ``"region_growing"``, ``"medsam2"``.
    out_path:
        Path to the output PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")

    mr_arr = sitk.GetArrayFromImage(mr_image).astype(np.float32)
    vmax_mr = float(np.percentile(mr_arr, 99.5))

    # Find the combined centroid of all non-empty masks.
    all_mask_arr = np.zeros_like(mr_arr, dtype=np.uint8)
    method_arrays = {}
    for name, mask_img in masks.items():
        arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
        method_arrays[name] = arr
        all_mask_arr |= arr

    if all_mask_arr.any():
        zs, ys, xs = np.where(all_mask_arr)
        cz = int(zs.mean())
        cy = int(ys.mean())
        cx = int(xs.mean())
    else:
        cz, cy, cx = [s // 2 for s in mr_arr.shape]

    # Choose three axial slices spaced around the centroid.
    n_z = mr_arr.shape[0]
    dz = max(5, min(10, (n_z - cz) // 3))
    z_slices = [
        max(0, cz - dz),
        cz,
        min(n_z - 1, cz + dz),
    ]

    method_names = list(masks.keys())
    n_methods = len(method_names)
    colours = ["cyan", "yellow", "magenta", "lime"]

    n_rows = max(1, n_methods)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(12, 4 * n_rows), squeeze=False
    )

    for row_idx, name in enumerate(method_names):
        arr = method_arrays[name]
        colour = colours[row_idx % len(colours)]
        for col_idx, z in enumerate(z_slices):
            ax = axes[row_idx][col_idx]
            ax.imshow(mr_arr[z], cmap="gray", vmin=0, vmax=vmax_mr)
            if arr[z].any():
                ax.contour(arr[z], levels=[0.5], colors=[colour], linewidths=1.5)
            ax.set_title(f"{name} — z={z}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Segmentation method comparison — axial slices", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _compute_slice_indices(
    mask_arr: np.ndarray,
    fallback_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return (cz, cy, cx) slice indices centred on the mask, or volume median if empty.

    For localised masks (e.g. a small tumor in a large volume) the centroid of
    the mask is far more informative than the volume median, because the
    median planes may not intersect the mask at all. For distributed masks
    (e.g. a brain parenchyma mask filling most of the volume) the centroid
    coincides with the median to within a few voxels, so the legacy figures
    remain unchanged.
    """
    nz, ny, nx = fallback_shape
    if mask_arr.sum() == 0:
        return nz // 2, ny // 2, nx // 2
    z_idx, y_idx, x_idx = np.where(mask_arr > 0)
    return (
        int(np.round(z_idx.mean())),
        int(np.round(y_idx.mean())),
        int(np.round(x_idx.mean())),
    )


def save_pet_brain_mask_overlay(
    pet_image: sitk.Image,
    brain_mask_in_pet: sitk.Image,
    out_path: str,
    title: str | None = None,
) -> None:
    """Save three median-plane views of a binary mask overlaid on the PET temporal mean.

    Renders the mask boundary as a lime contour on axial, coronal, and
    sagittal median planes of the PET volume. The PET background is rendered
    with the ``hot`` colormap; ``vmax`` is clamped at the 99th percentile of
    positive voxels to avoid hot-spot saturation.

    Parameters
    ----------
    pet_image:
        PET temporal-mean 3D volume as sitk.Image (already on PET native grid).
    brain_mask_in_pet:
        Binary mask as sitk.Image, co-aligned with pet_image (may be a brain
        parenchyma mask, a tumor mask, or any binary overlay).
    out_path:
        Path to the output PNG file.
    title:
        Optional figure title. When ``None`` the legacy title
        ``"PET (temporal mean) with brain parenchyma mask"`` is used so that
        all existing callers are backward compatible (ADR-5).
    """
    import matplotlib
    matplotlib.use("Agg")

    pet_arr = sitk.GetArrayFromImage(pet_image).astype(np.float32)
    bm_arr = sitk.GetArrayFromImage(brain_mask_in_pet).astype(np.uint8)

    positive = pet_arr[pet_arr > 0]
    if len(positive) > 0:
        vmax = float(np.percentile(positive, 99))
    else:
        vmax = 1.0  # degenerate input: flat color, PNG still created

    cz, cy, cx = _compute_slice_indices(bm_arr, pet_arr.shape)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(pet_arr[cz], cmap="hot", vmin=0, vmax=vmax)
    axes[0].contour(bm_arr[cz], levels=[0.5], colors="lime", linewidths=1.2)
    axes[0].set_title(f"Axial (z={cz})")
    axes[0].axis("off")

    axes[1].imshow(pet_arr[:, cy, :], cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[1].contour(bm_arr[:, cy, :], levels=[0.5], colors="lime", linewidths=1.2)
    axes[1].set_title(f"Coronal (y={cy})")
    axes[1].axis("off")

    axes[2].imshow(pet_arr[:, :, cx], cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[2].contour(bm_arr[:, :, cx], levels=[0.5], colors="lime", linewidths=1.2)
    axes[2].set_title(f"Sagittal (x={cx})")
    axes[2].axis("off")

    _title = title if title is not None else (
        "PET (temporal mean) with brain parenchyma mask (resampled to PET space)"
    )
    fig.suptitle(_title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_brain_mask_overlay(
    mr_image: sitk.Image,
    brain_mask: sitk.Image,
    out_path: str,
) -> None:
    """Save three median-plane views of the brain mask overlaid on the MR T1.

    Renders the brain boundary as a green contour on axial, coronal, and
    sagittal median planes, ITK-Snap style.

    Parameters
    ----------
    mr_image:
        T1 MR volume as sitk.Image.
    brain_mask:
        Binary brain mask as sitk.Image (co-aligned with mr_image).
    out_path:
        Path to the output PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")

    mr_arr = sitk.GetArrayFromImage(mr_image).astype(np.float32)
    bm_arr = sitk.GetArrayFromImage(brain_mask).astype(np.uint8)
    vmax_mr = float(np.percentile(mr_arr, 99.5))

    nz, ny, nx = mr_arr.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(mr_arr[cz], cmap="gray", vmin=0, vmax=vmax_mr)
    axes[0].contour(bm_arr[cz], levels=[0.5], colors="lime", linewidths=1.5)
    axes[0].set_title(f"Axial (z={cz})")
    axes[0].axis("off")

    axes[1].imshow(mr_arr[:, cy, :], cmap="gray", vmin=0, vmax=vmax_mr, origin="lower")
    axes[1].contour(bm_arr[:, cy, :], levels=[0.5], colors="lime", linewidths=1.5)
    axes[1].set_title(f"Coronal (y={cy})")
    axes[1].axis("off")

    axes[2].imshow(mr_arr[:, :, cx], cmap="gray", vmin=0, vmax=vmax_mr, origin="lower")
    axes[2].contour(bm_arr[:, :, cx], levels=[0.5], colors="lime", linewidths=1.5)
    axes[2].set_title(f"Sagittal (x={cx})")
    axes[2].axis("off")

    fig.suptitle("Brain mask overlay on MR T1 (TotalSegmentator)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    pet_path = filepath("e_1_BRAIN_DINAMIC_COLINA.dcm")
    mr_path = filepath("AX_3D_T1.dcm")

    print("Loading volumes ...")
    pet = load_dynamic_pet(pet_path)
    mr = load_mr(mr_path)

    print("Resampling PET temporal mean onto MR grid via sitk.Resample ...")
    # Use sitk.Image objects for physical-coordinate-correct resampling (task 5.3).
    pet_on_mr_image = resample_to_reference(
        pet.image, mr.image, transform=None, interpolator=sitk.sitkLinear
    )
    # Convert to numpy at the matplotlib boundary (design §1.6).
    pet_on_mr = sitk.GetArrayFromImage(pet_on_mr_image).astype(np.float32)
    mr_vol = sitk.GetArrayFromImage(mr.image).astype(np.float32)

    print("Loading the saved rigid transform ...")
    # Try .tfm first, fall back to .npz (task 5.5, design §6).
    coreg_base = os.path.join(out_dir, "coreg_transform")
    forward_transform = load_rigid_transform(coreg_base)
    print(f"  Loaded transform params: {np.array(forward_transform.GetParameters())}")

    print("Applying transform to full-resolution PET on MR grid via sitk.Resample ...")
    # Use the Euler3DTransform directly in sitk.Resample (task 5.4).
    pet_registered_image = resample_to_reference(
        pet_on_mr_image, mr.image,
        transform=forward_transform,
        interpolator=sitk.sitkLinear,
    )
    pet_registered = sitk.GetArrayFromImage(pet_registered_image).astype(np.float32)

    print("Building rotating MIP animation (24 angles, this can take ~1 minute) ...")
    info = save_rotating_mip(
        mr_vol, pet_registered,
        os.path.join(out_dir, "06_rotating_mip.gif"),
        n_angles=24,
        fps=6,
    )
    print(f"  -> {info['path']}  ({info['frames']} frames)")

    print("Done.")
