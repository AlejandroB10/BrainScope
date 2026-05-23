"""Objective 3 — 3D image segmentation.

Pipeline (preliminary, classical baseline):

1. Locate the tumor on the last frame of the dynamic PET. The proposal
   asks for a manual centroid + bounding box. The current implementation
   uses a smoothed-intensity peak inside the Z range that overlaps with
   the MR coverage, then applies a manual refinement when the
   MANUAL_CENTROID_PET_NATIVE constant below is set.
2. Map the bounding box from PET native voxel space to MR voxel space
   through the sitk.Image geometry.
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
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage as ndi
import SimpleITK as sitk
import yaml
from skimage.morphology import flood

from coregistration import (
    load_rigid_transform,
    resample_to_reference,
    voxel_to_physical_affine,
)
from loading import compute_temporal_mean, load_dynamic_pet, load_mr
from utils import filepath


# ---------------------------------------------------------------------------
# MedSAM2 (sam2) availability check
# ---------------------------------------------------------------------------

try:
    import sam2  # noqa: F401 — presence check only
    from sam2.build_sam import build_sam2_video_predictor_npz as _build_sam2_video_predictor_npz
    _MEDSAM2_AVAILABLE = True
except ImportError:
    _MEDSAM2_AVAILABLE = False
    _build_sam2_video_predictor_npz = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# BBox dataclass
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Tumor bounding box in PET native voxel space.

    Fields use (z, y, x) ordering throughout.

    Attributes
    ----------
    centroid_voxel:
        Center of the bounding box in PET native voxel coordinates [z, y, x].
    half_extent_voxel:
        Half-widths of the box along each axis [dz, dy, dx] in PET voxels.
    centroid_mr_voxel:
        Optional pre-computed MR-space centroid [z, y, x]. Populated by
        load_bbox_from_yaml when the YAML contains the optional field.
    half_extent_mr_voxel:
        Optional pre-computed MR-space half-extent [dz, dy, dx].
    key_slices_mr:
        Optional explicit MR z-indices for MedSAM2 key frames. When absent
        the MedSAM2 wrapper picks z_centre and z_centre ± 5.
    """

    centroid_voxel: list[int]
    half_extent_voxel: list[int]
    centroid_mr_voxel: Optional[list[int]] = None
    half_extent_mr_voxel: Optional[list[int]] = None
    key_slices_mr: Optional[list[int]] = None


# ---------------------------------------------------------------------------
# YAML bbox loader
# ---------------------------------------------------------------------------

def load_bbox_from_yaml(path: str | None = None) -> BBox:
    """Load a tumor bounding box from a YAML file.

    The YAML must contain at minimum:
        centroid_pet_native_voxel: [z, y, x]
        half_extent_voxel: [dz, dy, dx]

    Optional fields read when present:
        centroid_mr_voxel, half_extent_mr_voxel, key_slices_mr

    Parameters
    ----------
    path:
        Path to the YAML file. If None, the path defaults to
        ``data/tumor_bbox.yaml`` relative to the repository root.

    Returns
    -------
    BBox
        Loaded bounding box. Falls back to MANUAL_CENTROID_PET_NATIVE and
        BBOX_HALF_EXTENT_PET_NATIVE (module constants) when the file is
        absent. Raises a clear error when the file exists but is malformed.
    """
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if path is None:
        path = os.path.join(_ROOT, "data", "tumor_bbox.yaml")

    if not os.path.exists(path):
        warnings.warn(
            f"tumor_bbox.yaml not found at '{path}'. "
            "Falling back to MANUAL_CENTROID_PET_NATIVE / BBOX_HALF_EXTENT_PET_NATIVE constants.",
            UserWarning,
            stacklevel=2,
        )
        centroid = list(MANUAL_CENTROID_PET_NATIVE) if MANUAL_CENTROID_PET_NATIVE is not None else [0, 0, 0]
        return BBox(
            centroid_voxel=centroid,
            half_extent_voxel=list(BBOX_HALF_EXTENT_PET_NATIVE),
        )

    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    # Accept either PET-native or MR-direct schemas. MR-direct is preferred
    # when the lesion lies outside the PET spatial coverage.
    has_pet = "centroid_pet_native_voxel" in data and "half_extent_voxel" in data
    has_mr = "centroid_mr_voxel" in data and "half_extent_mr_voxel" in data

    if not (has_pet or has_mr):
        raise ValueError(
            "tumor_bbox.yaml must contain either "
            "(centroid_pet_native_voxel + half_extent_voxel) or "
            "(centroid_mr_voxel + half_extent_mr_voxel)."
        )

    def _vec3(name, value):
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            raise ValueError(f"{name} must be a list of 3 integers, got: {value!r}")
        return [int(v) for v in value]

    if has_pet:
        centroid_pet = _vec3("centroid_pet_native_voxel", data["centroid_pet_native_voxel"])
        half_pet = _vec3("half_extent_voxel", data["half_extent_voxel"])
    else:
        # MR-direct entry — use MR coordinates as the primary centroid_voxel too,
        # so any legacy caller that reads centroid_voxel still gets a valid value.
        centroid_pet = _vec3("centroid_mr_voxel", data["centroid_mr_voxel"])
        half_pet = _vec3("half_extent_mr_voxel", data["half_extent_mr_voxel"])

    return BBox(
        centroid_voxel=centroid_pet,
        half_extent_voxel=half_pet,
        centroid_mr_voxel=_vec3("centroid_mr_voxel", data["centroid_mr_voxel"]) if has_mr else None,
        half_extent_mr_voxel=_vec3("half_extent_mr_voxel", data["half_extent_mr_voxel"]) if has_mr else None,
        key_slices_mr=_vec3("key_slices_mr", data["key_slices_mr"]) if "key_slices_mr" in data else None,
    )


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
    """Build a (z, y, x) bounding box around centroid_voxel clipped to shape."""
    z, y, x = centroid_voxel
    dz, dy, dx = half_extent
    z_lo = max(0, int(round(z - dz)))
    z_hi = min(shape[0], int(round(z + dz)) + 1)
    y_lo = max(0, int(round(y - dy)))
    y_hi = min(shape[1], int(round(y + dy)) + 1)
    x_lo = max(0, int(round(x - dx)))
    x_hi = min(shape[2], int(round(x + dx)) + 1)
    return ((z_lo, z_hi), (y_lo, y_hi), (x_lo, x_hi))


def pet_voxel_to_mr_voxel(
    voxel_pet: tuple,
    pet_affine_or_image,
    mr_affine_or_image,
) -> tuple:
    """Map a (z, y, x) PET native voxel onto MR voxel coordinates.

    Accepts either:
    - Two numpy 4x4 affines (legacy path, preserved for backward compat), or
    - Two sitk.Image objects (new path using physical coordinate API).

    The sitk path is preferred: it uses TransformContinuousIndexToPhysicalPoint
    on the PET image and TransformPhysicalPointToContinuousIndex on the MR image,
    which avoids manual affine construction.
    """
    cz, cy, cx = voxel_pet

    if isinstance(pet_affine_or_image, sitk.Image) and isinstance(mr_affine_or_image, sitk.Image):
        pet_img = pet_affine_or_image
        mr_img = mr_affine_or_image
        # sitk uses (x, y, z) index order; numpy array is (z, y, x).
        # Cast to Python float explicitly — sitk rejects numpy scalar types.
        physical = pet_img.TransformContinuousIndexToPhysicalPoint(
            (float(cx), float(cy), float(cz))
        )
        mr_idx = mr_img.TransformPhysicalPointToContinuousIndex(physical)
        # Return as (z, y, x) to match the callers' convention.
        return float(mr_idx[2]), float(mr_idx[1]), float(mr_idx[0])
    else:
        # Legacy numpy path.
        pet_affine = pet_affine_or_image
        mr_affine = mr_affine_or_image
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
    """3D region growing from seed_voxel constrained to bbox.

    tolerance is the maximum absolute intensity difference between a voxel
    and the seed for the voxel to be included. If None, it defaults to one
    quarter of the seed value.
    """
    sz, sy, sx = (int(round(s)) for s in seed_voxel)
    sz = max(0, min(volume.shape[0] - 1, sz))
    sy = max(0, min(volume.shape[1] - 1, sy))
    sx = max(0, min(volume.shape[2] - 1, sx))

    seed_value = float(volume[sz, sy, sx])
    if tolerance is None:
        tolerance = max(50.0, abs(seed_value) * 0.25)

    box_volume = np.zeros_like(volume, dtype=np.float32)
    (z0, z1), (y0, y1), (x0, x1) = bbox
    box_volume[z0:z1, y0:y1, x0:x1] = volume[z0:z1, y0:y1, x0:x1].astype(np.float32)

    mask = flood(box_volume, (sz, sy, sx), tolerance=float(tolerance))
    return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask propagation via SimpleITK
# ---------------------------------------------------------------------------

def propagate_mask_to_pet(
    mask_mr: np.ndarray,
    mr_image: sitk.Image,
    pet_image: sitk.Image,
    forward_transform: sitk.Transform,
) -> np.ndarray:
    """Propagate a binary MR-space mask into PET native space.

    Uses the inverse of the forward rigid transform and nearest-neighbour
    resampling to keep the binary values intact.

    Parameters
    ----------
    mask_mr:
        Binary mask in MR voxel space, shape (z, y, x), uint8.
    mr_image:
        sitk.Image of the MR volume (provides the source grid geometry).
    pet_image:
        sitk.Image of the PET temporal mean (defines the output grid).
    forward_transform:
        The sitk.Euler3DTransform that maps PET -> MR (forward direction).
    """
    mask_sitk = sitk.GetImageFromArray(mask_mr.astype(np.float32))
    mask_sitk.CopyInformation(mr_image)

    inverse_tx = forward_transform.GetInverse()

    mask_pet_sitk = sitk.Resample(
        mask_sitk,
        pet_image,
        inverse_tx,
        sitk.sitkNearestNeighbor,
        0.0,
        mask_sitk.GetPixelID(),
    )
    return (sitk.GetArrayFromImage(mask_pet_sitk) > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Z-range clipping helper (WARNING-B fix)
# ---------------------------------------------------------------------------

def _clip_mask_to_z_range(mask: sitk.Image, z_lo: int, z_hi: int) -> sitk.Image:
    """Zero out all voxels outside the axial slice range [z_lo, z_hi).

    This function is used to constrain the MedSAM2 video propagation result to
    a biologically plausible z extent around the tumor bounding box. Without
    it, SAM2 propagates through all 156 MR slices (even empty ones far from
    the tumor), inflating the mask volume well beyond the tumor's anatomical
    extent.

    The z convention matches SimpleITK / numpy array indexing: axis 0 is z.

    Parameters
    ----------
    mask:
        Binary sitk.Image (uint8). Modified voxels are set to 0 in a copy.
    z_lo:
        First z slice to preserve (inclusive). Clipped to [0, mask.GetSize()[2]).
    z_hi:
        One-past-last z slice to preserve (exclusive). Clipped to
        [0, mask.GetSize()[2]].

    Returns
    -------
    sitk.Image
        New sitk.Image with the same geometry as ``mask`` but with all voxels
        outside [z_lo, z_hi) set to zero.
    """
    arr = sitk.GetArrayFromImage(mask).copy()  # (z, y, x)
    n_z = arr.shape[0]
    z_lo = int(max(0, z_lo))
    z_hi = int(min(n_z, z_hi))

    arr[:z_lo] = 0
    arr[z_hi:] = 0

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(mask)
    return out


# ---------------------------------------------------------------------------
# Private adapter: region growing
# ---------------------------------------------------------------------------

def _run_region_growing(mr_image: sitk.Image, bbox: BBox) -> sitk.Image:
    """Thin adapter from BBox to region_growing_in_bbox.

    Computes the MR-space centroid from ``bbox.centroid_mr_voxel`` when present,
    otherwise uses ``bbox.centroid_voxel`` directly (assumes it is already in MR
    space). Then calls ``region_growing_in_bbox`` and wraps the numpy result
    back into a ``sitk.Image``.

    The public alias ``region_growing_in_bbox`` remains unchanged and still
    callable directly. New code should prefer ``segment_tumor(..., method='region_growing')``.

    Parameters
    ----------
    mr_image:
        T1 MR volume as ``sitk.Image``.
    bbox:
        Bounding box. ``centroid_mr_voxel`` takes priority; falls back to
        ``centroid_voxel`` if absent.

    Returns
    -------
    sitk.Image
        Binary mask (uint8, values 0/1) co-aligned with ``mr_image``.
    """
    mr_vol = sitk.GetArrayFromImage(mr_image).astype(np.float32)
    mr_shape = mr_vol.shape  # (z, y, x)

    # Determine centroid in MR voxel space.
    if bbox.centroid_mr_voxel is not None:
        centroid_mr = tuple(bbox.centroid_mr_voxel)
    else:
        centroid_mr = tuple(bbox.centroid_voxel)

    # Determine half-extent in MR voxel space.
    if bbox.half_extent_mr_voxel is not None:
        half_ext_mr = tuple(bbox.half_extent_mr_voxel)
    else:
        half_ext_mr = tuple(bbox.half_extent_voxel)

    mr_bbox = bbox_from_centroid(centroid_mr, half_ext_mr, mr_shape)
    mask_np = region_growing_in_bbox(mr_vol, centroid_mr, mr_bbox)

    result = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    result.CopyInformation(mr_image)
    return result


# ---------------------------------------------------------------------------
# Private adapter: MedSAM2 inference
# ---------------------------------------------------------------------------

def _run_medsam2(mr_image: sitk.Image, bbox: BBox) -> sitk.Image:
    """Run MedSAM2 (sam2) video-propagation inference on the MR T1 volume.

    Uses ``SAM2VideoPredictorNPZ`` with pre-loaded tensor slices (no JPEG
    temp files). Key slices are ``z_centre``, ``z_centre - 5``, and
    ``z_centre + 5``, clipped to the MR depth. Propagation runs forward
    and backward from each key slice; results are union-merged.

    VRAM management:
        After inference, ``del predictor`` and ``torch.cuda.empty_cache()``
        are called unconditionally. Do not hold a reference to any model
        across this function boundary.

    Parameters
    ----------
    mr_image:
        T1 MR volume as ``sitk.Image``.
    bbox:
        Bounding box providing the centroid in MR voxel space (z, y, x order).

    Returns
    -------
    sitk.Image
        Binary mask (uint8, values 0/1) co-aligned with ``mr_image``.
        Returns a zero-filled image on inference failure (caller can detect
        and fall back to region growing).

    Raises
    ------
    RuntimeError
        If ``sam2`` is not installed. Install with:
        ``pip install -e external/MedSAM2/``
    """
    if not _MEDSAM2_AVAILABLE:
        raise RuntimeError(
            "MedSAM2 (sam2) not available. "
            "Install with: pip install -e external/MedSAM2/"
        )

    import torch
    from PIL import Image as PILImage

    # Locate weights.
    _ckpt = os.path.expanduser("~/.cache/medsam2/MedSAM2_latest.pt")
    # Hydra searches pkg://sam2, so the config path is relative to the sam2 package root.
    _cfg = "configs/sam2.1_hiera_t512.yaml"

    # Reproducibility.
    torch.manual_seed(42)

    try:
        mr_arr = sitk.GetArrayFromImage(mr_image).astype(np.float32)
        n_z, n_y, n_x = mr_arr.shape

        # Determine z-centre from bbox.
        if bbox.centroid_mr_voxel is not None:
            z_centre = int(bbox.centroid_mr_voxel[0])
        else:
            z_centre = int(bbox.centroid_voxel[0])

        # Clip to valid range.
        z_centre = max(0, min(n_z - 1, z_centre))
        z_keys = sorted({
            max(0, min(n_z - 1, z_centre - 5)),
            z_centre,
            max(0, min(n_z - 1, z_centre + 5)),
        })

        # Normalise the 3D volume to [0, 255] uint8 for SAM2 image encoder.
        v_min = float(mr_arr.min())
        v_max = float(mr_arr.max())
        if v_max > v_min:
            mr_norm = ((mr_arr - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
        else:
            mr_norm = np.zeros_like(mr_arr, dtype=np.uint8)

        image_size = 512
        img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

        # Build tensor: (n_z, 3, image_size, image_size).
        frames = np.zeros((n_z, 3, image_size, image_size), dtype=np.float32)
        for i in range(n_z):
            pil_slice = PILImage.fromarray(mr_norm[i]).convert("RGB")
            pil_resized = pil_slice.resize((image_size, image_size))
            frames[i] = np.array(pil_resized).transpose(2, 0, 1)
        frames /= 255.0

        frames_tensor = torch.from_numpy(frames)
        for i in range(n_z):
            frames_tensor[i] = (frames_tensor[i] - img_mean) / img_std

        device = "cuda" if torch.cuda.is_available() else "cpu"
        frames_tensor = frames_tensor.to(device)

        # Build 2D bbox from 3D bbox extents (x0, y0, x1, y1 in pixel coords).
        if bbox.centroid_mr_voxel is not None:
            _, cy_mr, cx_mr = bbox.centroid_mr_voxel
        else:
            _, cy_mr, cx_mr = bbox.centroid_voxel

        if bbox.half_extent_mr_voxel is not None:
            _, dy_mr, dx_mr = bbox.half_extent_mr_voxel
        else:
            _, dy_mr, dx_mr = bbox.half_extent_voxel

        # Box coordinates are in original image pixel space (n_x × n_y).
        # The predictor's normalize_coords=True divides by video_W/video_H,
        # which are n_x/n_y, so we pass coordinates directly in the original space.
        x0 = float(max(0, cx_mr - dx_mr))
        x1 = float(min(n_x - 1, cx_mr + dx_mr))
        y0 = float(max(0, cy_mr - dy_mr))
        y1 = float(min(n_y - 1, cy_mr + dy_mr))
        box_2d = np.array([x0, y0, x1, y1], dtype=np.float32)

        # Initialise predictor (uses NPZ variant that accepts pre-loaded tensors).
        predictor = _build_sam2_video_predictor_npz(_cfg, _ckpt, device=device)

        # Output accumulator.
        segs_3d = np.zeros((n_z, n_y, n_x), dtype=np.uint8)

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            for z_key in z_keys:
                inference_state = predictor.init_state(
                    frames_tensor,
                    video_height=n_y,
                    video_width=n_x,
                )
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z_key,
                    obj_id=1,
                    box=box_2d,
                )

                # Forward propagation.
                for out_frame_idx, _obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state
                ):
                    mask_2d = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    # Rescale from image_size back to original MR slice size.
                    if mask_2d.shape != (n_y, n_x):
                        from skimage.transform import resize as sk_resize
                        mask_2d = sk_resize(
                            mask_2d.astype(float),
                            (n_y, n_x),
                            order=0,
                            anti_aliasing=False,
                        ) > 0.5
                    segs_3d[out_frame_idx] |= mask_2d.astype(np.uint8)

                predictor.reset_state(inference_state)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z_key,
                    obj_id=1,
                    box=box_2d,
                )

                # Backward propagation.
                for out_frame_idx, _obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state, reverse=True
                ):
                    mask_2d = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    if mask_2d.shape != (n_y, n_x):
                        from skimage.transform import resize as sk_resize
                        mask_2d = sk_resize(
                            mask_2d.astype(float),
                            (n_y, n_x),
                            order=0,
                            anti_aliasing=False,
                        ) > 0.5
                    segs_3d[out_frame_idx] |= mask_2d.astype(np.uint8)

                predictor.reset_state(inference_state)

        raw_mask = sitk.GetImageFromArray(segs_3d.astype(np.uint8))
        raw_mask.CopyInformation(mr_image)

        # Clip the propagation result to a 2× half-extent window around the
        # bbox centroid in z. SAM2 video propagation spans all slices by
        # default; without this clip the mask inflates to anatomically
        # implausible volumes.
        if bbox.half_extent_mr_voxel is not None:
            dz_mr = bbox.half_extent_mr_voxel[0]
        else:
            dz_mr = bbox.half_extent_voxel[0]
        margin = int(dz_mr * 2)
        z_lo = max(0, z_centre - margin)
        z_hi = min(n_z, z_centre + margin + 1)
        result = _clip_mask_to_z_range(raw_mask, z_lo=z_lo, z_hi=z_hi)

        return result

    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(
            f"_run_medsam2: inference failed with: {exc!r}. "
            "Returning zero mask. Caller can fall back to region growing.",
            RuntimeWarning,
            stacklevel=2,
        )
        zero_arr = np.zeros(
            (mr_image.GetSize()[2], mr_image.GetSize()[1], mr_image.GetSize()[0]),
            dtype=np.uint8,
        )
        zero_mask = sitk.GetImageFromArray(zero_arr)
        zero_mask.CopyInformation(mr_image)
        return zero_mask

    finally:
        # Release VRAM unconditionally.
        try:
            del predictor  # noqa: F821
        except NameError:
            pass
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def segment_tumor(
    mr_image: sitk.Image,
    bbox: BBox,
    method: str = "medsam2",
) -> sitk.Image:
    """Segment the tumor from the MR T1 volume using the specified method.

    This is the single public entry point for tumor segmentation. It accepts
    a manual bounding box and dispatches to the appropriate algorithm.

    Deprecated usage note:
        The ``find_candidate_centroid`` / ``region_growing_in_bbox`` pair is
        preserved for backwards compatibility, but new code should use
        ``segment_tumor(mr_image, bbox, method='region_growing')`` instead.

    Parameters
    ----------
    mr_image:
        T1 MR volume as ``sitk.Image``.
    bbox:
        Bounding box in MR voxel space (from ``load_bbox_from_yaml``).
    method:
        Algorithm to use. One of:
        - ``"medsam2"``: MedSAM2 video propagation (requires sam2 install).
        - ``"region_growing"``: classical 3D flood-fill (no GPU required).

    Returns
    -------
    sitk.Image
        Binary mask (uint8, values 0/1) co-aligned with ``mr_image``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.
    RuntimeError
        If ``method="medsam2"`` and sam2 is not installed.
    """
    if method == "medsam2":
        return _run_medsam2(mr_image, bbox)
    elif method == "region_growing":
        return _run_region_growing(mr_image, bbox)
    else:
        raise ValueError(
            f"Unknown segmentation method: {method!r}. "
            f"Valid options: 'medsam2', 'region_growing'."
        )


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

    print("Loading manual bbox from data/tumor_bbox.yaml ...")
    bbox = load_bbox_from_yaml()
    print(f"  centroid_voxel (fallback): {bbox.centroid_voxel}")
    print(f"  half_extent_voxel:         {bbox.half_extent_voxel}")
    if bbox.centroid_mr_voxel is not None:
        print(f"  centroid_mr_voxel:         {bbox.centroid_mr_voxel}")
        print(f"  half_extent_mr_voxel:      {bbox.half_extent_mr_voxel}")

    # --- Primary AI path: MedSAM2 driven by the manual bbox ---
    print("Running MedSAM2 segmentation on the MR T1 ...")
    try:
        mask_mr_medsam2_img = segment_tumor(mr.image, bbox, method="medsam2")
        mask_mr_medsam2 = sitk.GetArrayFromImage(mask_mr_medsam2_img).astype(np.uint8)
        print(f"  MedSAM2 mask voxels: {int(mask_mr_medsam2.sum())}")
        np.save(os.path.join(out_dir, "tumor_mask_mr.npy"), mask_mr_medsam2)
    except Exception as exc:
        print(f"  MedSAM2 failed: {exc!r}. Falling back to region-growing.")
        mask_mr_medsam2 = None
        mask_mr_medsam2_img = None

    # --- Baseline path: region-growing, run regardless for comparison ---
    print("Running region-growing baseline on the MR T1 ...")
    if bbox.centroid_mr_voxel is not None:
        seed_mr = tuple(int(c) for c in bbox.centroid_mr_voxel)
        half_mr = bbox.half_extent_mr_voxel or [10, 25, 25]
    else:
        seed_mr = tuple(int(c) for c in bbox.centroid_voxel)
        half_mr = bbox.half_extent_voxel
    rg_bbox = bbox_from_centroid(seed_mr, tuple(half_mr), mr_vol.shape)
    mask_mr_rg = region_growing_in_bbox(mr_vol, seed_mr, rg_bbox, tolerance=None)
    print(f"  Region-growing mask voxels: {int(mask_mr_rg.sum())}")
    np.save(os.path.join(out_dir, "tumor_mask_mr_region_growing.npy"), mask_mr_rg)

    # Pick the primary mask used for the rest of the pipeline.
    if mask_mr_medsam2 is not None:
        mask_mr = mask_mr_medsam2
        mask_label = "MedSAM2"
    else:
        mask_mr = mask_mr_rg
        mask_label = "region-growing"

    # --- Visualization: MR with the primary mask ---
    print("Saving overlay figures ...")
    vmax_mr = float(np.percentile(mr_vol, 99.5))
    _planes_with_mask(
        mr_vol, mask_mr, "gray", vmax_mr,
        f"{mask_label} mask on MR T1 (cyan contour)",
        os.path.join(out_dir, "07_segmentation_mask_on_mr.png"),
    )

    # --- Visualization: coregistered PET with the primary mask ---
    coreg_path = os.path.join(out_dir, "coreg_transform")
    tfm_exists = os.path.exists(coreg_path + ".tfm")
    npz_exists = os.path.exists(coreg_path + ".npz")
    if tfm_exists or npz_exists:
        forward_transform = load_rigid_transform(coreg_path)

        pet_on_mr_image = resample_to_reference(pet.image, mr.image, transform=None)
        pet_registered_image = resample_to_reference(
            pet_on_mr_image, mr.image, transform=forward_transform
        )
        pet_registered = sitk.GetArrayFromImage(pet_registered_image).astype(np.float32)
        vmax_pet = float(np.percentile(pet_registered, 99.5))
        _planes_with_mask(
            pet_registered, mask_mr, "hot", vmax_pet,
            f"{mask_label} mask on coregistered PET temporal mean (cyan contour)",
            os.path.join(out_dir, "08_segmentation_mask_on_pet.png"),
        )

        # --- Propagate mask MR -> PET native ---
        print("Propagating mask MR -> PET native via inverse rigid + sitk.Resample ...")
        mask_pet_native = propagate_mask_to_pet(
            mask_mr, mr.image, pet.image, forward_transform
        )
        np.save(os.path.join(out_dir, "tumor_mask_pet_native.npy"), mask_pet_native)
        print(f"  Mask voxels in PET native: {int(mask_pet_native.sum())}")
        vmax_last = float(np.percentile(last_frame, 99.5))
        _planes_with_mask(
            last_frame.astype(np.float32), mask_pet_native, "hot", vmax_last,
            f"{mask_label} mask propagated to PET native space (last frame, cyan contour)",
            os.path.join(out_dir, "09_segmentation_mask_on_pet_native.png"),
        )

    # --- Dice between MedSAM2 and region-growing (proxy for 'provided vs auto') ---
    if mask_mr_medsam2 is not None:
        from metrics import dice
        m1 = sitk.GetImageFromArray(mask_mr_medsam2.astype(np.uint8))
        m1.CopyInformation(mr.image)
        m2 = sitk.GetImageFromArray(mask_mr_rg.astype(np.uint8))
        m2.CopyInformation(mr.image)
        d = dice(m1, m2)
        print(f"  Dice(MedSAM2, region_growing) = {d:.4f}")

    print("Done.")
