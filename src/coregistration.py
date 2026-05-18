"""Objective 2 — 3D rigid coregistration in physical coordinates.

Pipeline:

1. Load both DICOMs and build sitk.Image objects from the DICOM headers
   (origin, spacing, direction) extracted by _read_position_orientation.
2. Resample the PET temporal mean onto the MR voxel grid via sitk.Resample.
3. Find the rigid transform (sitk.Euler3DTransform) that maximises Mutual
   Information between the resampled PET and the MR. The optimisation uses
   a downsampled version of both volumes for speed; the final transform is
   applied at full resolution.
4. Save a before/after MIP overlay, dump the optimal parameters as both a
   legacy .npz file (for backward compatibility) and a new .tfm sidecar in
   ITK plain-text format.

Run:

    python src/coregistration.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.optimize import minimize

from loading import (
    _read_position_orientation,
    compute_temporal_mean,
    load_dynamic_pet,
    load_mr,
)
from utils import filepath

# Re-export _read_position_orientation for one-cycle back-compat so that
# any existing import of it from coregistration still works.
__all__ = [
    "_read_position_orientation",
    "resample_to_reference",
    "build_initial_euler3d",
    "load_rigid_transform",
    "coregister",
    "save_before_after",
    "mutual_information",
]


# ---------------------------------------------------------------------------
# Legacy compatibility shim (voxel_to_physical_affine kept for old callers)
# ---------------------------------------------------------------------------

def voxel_to_physical_affine(ds) -> np.ndarray:
    """Build a 4x4 affine from DICOM tags (kept for backward compatibility).

    New code should use sitk.Image.GetOrigin/GetSpacing/GetDirection instead.
    This function is preserved so that existing callers in segmentation.py
    and visualization.py keep working until they are migrated.
    """
    ipp, iop = _read_position_orientation(ds)
    ipp = np.asarray([float(v) for v in ipp])
    iop = np.asarray([float(v) for v in iop])

    pixel_spacing = [float(v) for v in ds.PixelSpacing]
    spacing_z = float(ds.SpacingBetweenSlices)

    row_dir = iop[0:3]
    col_dir = iop[3:6]
    slice_dir = np.cross(row_dir, col_dir)

    A = np.eye(4)
    A[:3, 0] = slice_dir * spacing_z
    A[:3, 1] = row_dir * pixel_spacing[0]
    A[:3, 2] = col_dir * pixel_spacing[1]
    A[:3, 3] = ipp
    return A


# ---------------------------------------------------------------------------
# SimpleITK resampling
# ---------------------------------------------------------------------------

def resample_to_reference(
    src: sitk.Image,
    reference: sitk.Image,
    transform=None,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Resample src onto the reference image grid.

    Parameters
    ----------
    src:
        The image to resample (moving).
    reference:
        Defines the output grid (size, spacing, origin, direction).
    transform:
        Any sitk.Transform; pass None or sitk.Transform() for identity.
    interpolator:
        sitk interpolation constant. Use sitk.sitkNearestNeighbor for masks.
    """
    if transform is None:
        transform = sitk.Transform()
    return sitk.Resample(
        src,
        reference,
        transform,
        interpolator,
        0.0,
        src.GetPixelID(),
    )


def build_initial_euler3d(moving: sitk.Image) -> sitk.Euler3DTransform:
    """Build a zero Euler3DTransform centred on the moving image's physical centre.

    The rotation centre is the physical coordinate of the voxel-array centre,
    computed via TransformContinuousIndexToPhysicalPoint. This matches the
    semantics of the previous hand-rolled implementation (rotate around the
    centre of the volume) so the MI optimisation baseline is preserved.
    """
    size = list(moving.GetSize())
    centre_index = [(s - 1) / 2.0 for s in size]
    centre_physical = moving.TransformContinuousIndexToPhysicalPoint(centre_index)
    tx = sitk.Euler3DTransform()
    tx.SetCenter(centre_physical)
    return tx


# ---------------------------------------------------------------------------
# Mutual Information loss (unchanged from the original implementation)
# ---------------------------------------------------------------------------

def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Shannon mutual information between two equal-shape arrays.

    Uses a bins x bins joint histogram to estimate the marginal and joint
    probabilities, then returns H(a) + H(b) - H(a, b).
    """
    a = a.ravel()
    b = b.ravel()
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist / (hist.sum() + 1e-12)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    def H(p):
        nz = p[p > 0]
        return -np.sum(nz * np.log2(nz))

    return float(H(px) + H(py) - H(pxy))


# ---------------------------------------------------------------------------
# Coregistration driver
# ---------------------------------------------------------------------------

def coregister(
    moving: sitk.Image,
    fixed: sitk.Image,
    bins: int = 32,
    max_iter: int = 60,
    initial_transform: sitk.Euler3DTransform | None = None,
    verbose: bool = True,
) -> dict:
    """Maximise Mutual Information by optimising a 6-DOF Euler3DTransform.

    The inner closure resamples the moving image with sitk.Resample, pulls
    both volumes as zero-copy numpy views, and passes them to the existing
    32-bin histogram MI objective. scipy.optimize.minimize with the Powell
    method drives the search.

    Returns a dict containing:
        transform         – the optimal sitk.Euler3DTransform
        inverse_transform – transform.GetInverse() as sitk.Euler3DTransform
        forward_params    – 7-element numpy array for .npz compat
        inverse_params    – 7-element numpy array for .npz compat
        mi_initial        – MI value at the first evaluation
        mi_final          – MI value at the optimum
        n_iter            – total objective evaluations
        elapsed_s         – wall-clock seconds
    """
    if initial_transform is None:
        initial_transform = build_initial_euler3d(moving)

    pixel_id = moving.GetPixelID()
    history: list[float] = []

    def negative_mi(params):
        tx = sitk.Euler3DTransform(initial_transform)
        tx.SetParameters(tuple(float(p) for p in params))
        warped = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, pixel_id)
        a = sitk.GetArrayViewFromImage(warped)
        b = sitk.GetArrayViewFromImage(fixed)
        mi = mutual_information(a, b, bins=bins)
        history.append(mi)
        if verbose and len(history) % 20 == 0:
            print(f"    iter {len(history):4d} | MI = {mi:.5f}")
        return -mi

    x0 = np.array(initial_transform.GetParameters())
    t0 = time.time()
    result = minimize(
        negative_mi,
        x0=x0,
        method="Powell",
        options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4},
    )
    elapsed = time.time() - t0

    # Build the optimal transform from the result.
    best_tx = sitk.Euler3DTransform(initial_transform)
    best_tx.SetParameters(tuple(float(p) for p in result.x))

    # GetInverse returns a generic Transform; cast it back to Euler3D.
    try:
        inv_tx = sitk.Euler3DTransform(best_tx.GetInverse())
    except Exception:
        inv_tx = best_tx.GetInverse()

    # Legacy 7-float params: [tz, ty, tx, rx, ry, rz] -> [tz,ty,tx,angle,ax,ay,az]
    # Euler3D params are (rx, ry, rz, tx, ty, tz) in radians/mm.
    p = result.x
    rx, ry, rz, tx, ty, tz = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
    angle = float(np.sqrt(rx**2 + ry**2 + rz**2))
    if angle > 1e-12:
        ax, ay, az = rx / angle, ry / angle, rz / angle
    else:
        ax, ay, az = 0.0, 0.0, 1.0
    forward_params = np.array([tz, ty, tx, angle, ax, ay, az])

    # Inverse params from the inverse transform's parameters.
    try:
        ip = np.array(inv_tx.GetParameters())
        irx, iry, irz, itx, ity, itz = ip[0], ip[1], ip[2], ip[3], ip[4], ip[5]
        iangle = float(np.sqrt(irx**2 + iry**2 + irz**2))
        if iangle > 1e-12:
            iax, iay, iaz = irx / iangle, iry / iangle, irz / iangle
        else:
            iax, iay, iaz = 0.0, 0.0, 1.0
        inverse_params = np.array([itz, ity, itx, iangle, iax, iay, iaz])
    except Exception:
        inverse_params = forward_params.copy()

    return {
        "transform": best_tx,
        "inverse_transform": inv_tx,
        "forward_params": forward_params,
        "inverse_params": inverse_params,
        "mi_initial": history[0] if history else None,
        "mi_final": -result.fun,
        "n_iter": len(history),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Transform I/O
# ---------------------------------------------------------------------------

def load_rigid_transform(path: str | Path) -> sitk.Euler3DTransform:
    """Load a rigid transform, preferring .tfm and falling back to .npz.

    The .tfm path is tried first because it is the authoritative format.
    If only a .npz exists (legacy), the 7-float forward_params vector is
    reconstructed into an Euler3DTransform with identity centre.
    """
    path = Path(path)

    # Try .tfm sidecar first.
    tfm_path = path.with_suffix(".tfm")
    if tfm_path.exists():
        tx = sitk.ReadTransform(str(tfm_path))
        if isinstance(tx, sitk.Euler3DTransform):
            return tx
        # Cast generic composite transform to Euler3D when possible.
        try:
            return sitk.Euler3DTransform(tx)
        except Exception:
            return tx

    # Fall back to .npz legacy reconstruction.
    npz_path = path.with_suffix(".npz")
    if npz_path.exists():
        data = np.load(str(npz_path))
        fp = data["forward_params"]
        # fp = [tz, ty, tx, angle, ax, ay, az]
        tz, ty, tx_, angle, ax, ay, az = (float(v) for v in fp)
        rx = ax * angle
        ry = ay * angle
        rz = az * angle
        euler = sitk.Euler3DTransform()
        euler.SetParameters((rx, ry, rz, tx_, ty, tz))
        return euler

    raise FileNotFoundError(f"No .tfm or .npz transform found near: {path}")


def save_transform(
    transform: sitk.Euler3DTransform,
    tfm_path: str | Path,
    npz_path: str | Path,
    extra: dict | None = None,
) -> None:
    """Write the transform as both a .tfm sidecar and a .npz legacy file.

    The .npz keeps all 9 original keys so downstream readers do not break.
    ``extra`` must supply: mi_initial, mi_final, downsample_factor,
    mr_shape, pet_shape, mr_affine, pet_affine.
    """
    sitk.WriteTransform(transform, str(tfm_path))

    inv_tx = transform.GetInverse()
    p = np.array(transform.GetParameters())   # (rx, ry, rz, tx, ty, tz)
    rx, ry, rz, tx_, ty, tz = p
    angle = float(np.sqrt(rx**2 + ry**2 + rz**2))
    ax, ay, az = (rx / angle, ry / angle, rz / angle) if angle > 1e-12 else (0.0, 0.0, 1.0)
    forward_params = np.array([tz, ty, tx_, angle, ax, ay, az])

    try:
        ip = np.array(inv_tx.GetParameters())
        irx, iry, irz, itx, ity, itz = ip
        iangle = float(np.sqrt(irx**2 + iry**2 + irz**2))
        iax, iay, iaz = (irx / iangle, iry / iangle, irz / iangle) if iangle > 1e-12 else (0.0, 0.0, 1.0)
        inverse_params = np.array([itz, ity, itx, iangle, iax, iay, iaz])
    except Exception:
        inverse_params = forward_params.copy()

    extra = extra or {}
    np.savez(
        str(npz_path),
        forward_params=forward_params,
        inverse_params=inverse_params,
        mi_initial=extra.get("mi_initial", np.nan),
        mi_final=extra.get("mi_final", np.nan),
        downsample_factor=extra.get("downsample_factor", 1),
        mr_shape=extra.get("mr_shape", np.array([0, 0, 0])),
        pet_shape=extra.get("pet_shape", np.array([0, 0, 0])),
        mr_affine=extra.get("mr_affine", np.eye(4)),
        pet_affine=extra.get("pet_affine", np.eye(4)),
    )


# ---------------------------------------------------------------------------
# Legacy shim: resample_to_grid kept so old call-sites continue to work
# ---------------------------------------------------------------------------

def resample_to_grid(
    src_volume: np.ndarray,
    src_affine: np.ndarray,
    dst_affine: np.ndarray,
    dst_shape: tuple,
    order: int = 1,
) -> np.ndarray:
    """Backward-compat shim — uses scipy.ndimage.affine_transform, NOT sitk.

    This function accepts numpy arrays and 4x4 affines (the legacy API) and
    performs the resample via ``scipy.ndimage.affine_transform`` internally.
    It is intentionally kept here so that existing callers continue to work
    without modification during the incremental migration.

    For new code, prefer ``resample_to_reference``, which accepts
    ``sitk.Image`` objects and uses ``sitk.Resample`` with proper
    physical-coordinate semantics.

    This function will be removed once all callers have been migrated to
    ``resample_to_reference``.
    """
    import scipy.ndimage as ndi
    M = np.linalg.inv(src_affine) @ dst_affine
    return ndi.affine_transform(
        src_volume.astype(np.float32),
        matrix=M[:3, :3],
        offset=M[:3, 3],
        output_shape=dst_shape,
        order=order,
        mode="constant",
        cval=0.0,
    )


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def save_before_after(fixed, moving_before, moving_after, out_path):
    """Save a 2x3 figure with axial / coronal / sagittal MIPs of the fixed
    volume overlaid with the moving volume before and after registration.

    Accepts both numpy arrays and sitk.Image objects (the latter are
    converted to numpy via GetArrayFromImage).
    """
    def _to_np(v):
        if isinstance(v, sitk.Image):
            return sitk.GetArrayFromImage(v)
        return np.asarray(v)

    fixed = _to_np(fixed)
    moving_before = _to_np(moving_before)
    moving_after = _to_np(moving_after)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, plane in enumerate(("axial", "coronal", "sagittal")):
        if plane == "axial":
            f = fixed.max(axis=0); mb = moving_before.max(axis=0); ma = moving_after.max(axis=0)
        elif plane == "coronal":
            f = fixed.max(axis=1); mb = moving_before.max(axis=1); ma = moving_after.max(axis=1)
        else:
            f = fixed.max(axis=2); mb = moving_before.max(axis=2); ma = moving_after.max(axis=2)

        axes[0, col].imshow(f, cmap="gray")
        axes[0, col].imshow(mb, cmap="hot", alpha=0.45)
        axes[0, col].set_title(f"Before — {plane}")
        axes[0, col].axis("off")
        axes[1, col].imshow(f, cmap="gray")
        axes[1, col].imshow(ma, cmap="hot", alpha=0.45)
        axes[1, col].set_title(f"After — {plane}")
        axes[1, col].axis("off")

    fig.suptitle("Rigid coregistration — MR (gray) overlaid with PET (hot)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path}


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

    pet_mean_image = pet.image  # sitk.Image of temporal mean
    mr_image = mr.image

    print(f"  PET sitk size (x,y,z): {pet_mean_image.GetSize()}")
    print(f"  MR  sitk size (x,y,z): {mr_image.GetSize()}")

    print("Resampling PET onto MR voxel grid (physical-coordinate aware) ...")
    pet_on_mr_image = resample_to_reference(
        pet_mean_image, mr_image, transform=None, interpolator=sitk.sitkLinear
    )
    pet_on_mr = sitk.GetArrayFromImage(pet_on_mr_image).astype(np.float32)
    mr_vol = sitk.GetArrayFromImage(mr_image).astype(np.float32)
    print(f"  pet_on_mr shape : {pet_on_mr.shape}  range: [{pet_on_mr.min():.1f}, {pet_on_mr.max():.1f}]")

    print("Downsampling 2x for the rigid optimisation ...")
    factor = 2
    pet_ds_image = sitk.Shrink(pet_on_mr_image, [factor, factor, factor])
    mr_ds_image = sitk.Shrink(mr_image, [factor, factor, factor])

    # Intensity normalise for the MI objective.
    def _normalise(img: sitk.Image) -> sitk.Image:
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        return out

    pet_ds_norm = _normalise(pet_ds_image)
    mr_ds_norm = _normalise(mr_ds_image)

    print("Running rigid coregistration (Powell, max 60 iter) ...")
    res = coregister(pet_ds_norm, mr_ds_norm, bins=32, max_iter=60, verbose=True)

    print(f"  Initial MI : {res['mi_initial']:.5f}")
    print(f"  Final   MI : {res['mi_final']:.5f}")
    print(f"  Improvement: {(res['mi_final'] - res['mi_initial']) / res['mi_initial'] * 100:+.1f} %")
    print(f"  Iterations : {res['n_iter']}")
    print(f"  Elapsed    : {res['elapsed_s']:.1f} s")
    p6 = np.array(res["transform"].GetParameters())
    print(f"  Transform params (rx,ry,rz,tx,ty,tz): {p6}")

    print("Applying the optimal transform to the full-resolution PET ...")
    pet_registered_image = resample_to_reference(
        pet_on_mr_image, mr_image,
        transform=res["transform"],
        interpolator=sitk.sitkLinear,
    )
    pet_registered = sitk.GetArrayFromImage(pet_registered_image).astype(np.float32)

    print("Saving before/after MIP overlay (full resolution) ...")
    info = save_before_after(
        mr_vol, pet_on_mr, pet_registered,
        os.path.join(out_dir, "05_coreg_before_after.png"),
    )
    print(f"  -> {info['path']}")

    # Build affines for the legacy .npz keys.
    ds_pet = pydicom.dcmread(pet_path, stop_before_pixels=True)
    ds_mr = pydicom.dcmread(mr_path, stop_before_pixels=True)
    pet_affine = voxel_to_physical_affine(ds_pet)
    mr_affine = voxel_to_physical_affine(ds_mr)

    pet_mean_arr = compute_temporal_mean(pet.array).astype(np.float32)

    tfm_out = os.path.join(out_dir, "coreg_transform.tfm")
    npz_out = os.path.join(out_dir, "coreg_transform.npz")
    save_transform(
        res["transform"],
        tfm_path=tfm_out,
        npz_path=npz_out,
        extra={
            "mi_initial": res["mi_initial"],
            "mi_final": res["mi_final"],
            "downsample_factor": factor,
            "mr_shape": np.array(mr_vol.shape),
            "pet_shape": np.array(pet_mean_arr.shape),
            "mr_affine": mr_affine,
            "pet_affine": pet_affine,
        },
    )
    print(f"  -> {npz_out}")
    print(f"  -> {tfm_out}")

    # Verify round-trip read.
    tx_back = load_rigid_transform(tfm_out)
    print(f"  .tfm re-loaded OK: params = {np.array(tx_back.GetParameters())}")

    print("Done.")
