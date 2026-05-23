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

import json
import os
import time
import warnings
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
    "normalised_mutual_information",
    "save_mi_convergence",
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


def _physical_centroid(image: sitk.Image) -> tuple[float, float, float]:
    """Return the physical coordinate of the bounding-box centre of image (mm).

    Uses TransformContinuousIndexToPhysicalPoint on the midpoint of the voxel
    index grid so that arbitrary origin/spacing/direction matrices are handled
    uniformly by SimpleITK (see ADR-2).
    """
    size = image.GetSize()
    centre_index = [(s - 1) / 2.0 for s in size]
    return image.TransformContinuousIndexToPhysicalPoint(centre_index)


def build_initial_euler3d(
    moving: sitk.Image,
    fixed: sitk.Image | None = None,
) -> sitk.Euler3DTransform:
    """Build a zero-rotation Euler3DTransform centred on the moving image.

    The rotation centre is the physical coordinate of the voxel-array centre.

    When ``fixed`` is provided the translation is initialised to
    ``centroid(fixed) - centroid(moving)`` in physical coordinates (mm), which
    centres the moving image's bounding box on the fixed image's bounding box
    before the optimiser starts (see ADR-2).

    When ``fixed`` is ``None`` the translation is ``(0, 0, 0)`` — backward
    compatible with all existing callers.
    """
    centre_moving = _physical_centroid(moving)
    tx = sitk.Euler3DTransform()
    tx.SetCenter(list(centre_moving))
    if fixed is not None:
        centre_fixed = _physical_centroid(fixed)
        translation = tuple(cf - cm for cf, cm in zip(centre_fixed, centre_moving))
        tx.SetTranslation(list(translation))
    return tx


# ---------------------------------------------------------------------------
# Normalised Mutual Information (NMI) — replaces plain MI (ADR-1, ADR-3)
# ---------------------------------------------------------------------------


def normalised_mutual_information(
    a: np.ndarray,
    b: np.ndarray,
    bins: int = 16,
) -> float:
    """Normalised Mutual Information: NMI(A, B) = (H(A) + H(B)) / H(A, B).

    Shannon entropy is estimated from a ``bins``-bin histogram of each
    marginal and the joint distribution.

    Returns ``0.0`` when ``H(A, B) == 0`` (degenerate / constant input) to
    prevent division-by-zero and avoid confusing the Powell optimiser (ADR-3).

    For identical volumes NMI approaches ``2.0``; for independent volumes NMI
    approaches ``1.0``; for multi-modal PET/MR pairs NMI typically lies in
    ``[1.0, 1.5]``.
    """
    a = a.ravel()
    b = b.ravel()
    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist_2d / hist_2d.sum() if hist_2d.sum() > 0 else hist_2d
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    nz_xy = pxy > 0
    nz_x = px > 0
    nz_y = py > 0

    h_xy = -np.sum(pxy[nz_xy] * np.log(pxy[nz_xy]))
    h_x = -np.sum(px[nz_x] * np.log(px[nz_x]))
    h_y = -np.sum(py[nz_y] * np.log(py[nz_y]))

    if h_xy == 0:
        return 0.0
    return float((h_x + h_y) / h_xy)


# Backward-compatible alias — any code importing ``mutual_information`` by
# name continues to work; new code should use the canonical name (ADR-1).
mutual_information = normalised_mutual_information


# ---------------------------------------------------------------------------
# Coregistration driver
# ---------------------------------------------------------------------------

def coregister(
    moving: sitk.Image,
    fixed: sitk.Image,
    bins: int = 16,
    max_iter: int = 60,
    initial_transform: sitk.Euler3DTransform | None = None,
    verbose: bool = True,
) -> dict:
    """Maximise Normalised Mutual Information by optimising a 6-DOF Euler3DTransform.

    The inner closure resamples the moving image with sitk.Resample, pulls
    both volumes as zero-copy numpy views, and passes them to the 16-bin
    histogram NMI objective. scipy.optimize.minimize with the Powell method
    drives the search.

    When ``initial_transform`` is ``None`` the transform is initialised with
    ``build_initial_euler3d(moving, fixed=fixed)`` so the bounding boxes are
    aligned before the first objective evaluation.

    Returns a dict containing:
        transform         – the optimal sitk.Euler3DTransform
        inverse_transform – transform.GetInverse() as sitk.Euler3DTransform
        forward_params    – 7-element numpy array for .npz compat
        inverse_params    – 7-element numpy array for .npz compat
        nmi_initial       – NMI value at iteration 0 (before any step)
        nmi_final         – NMI value at the optimum
        nmi_history       – list[tuple[int, float]] — (iteration_index, nmi)
        mi_initial        – alias of nmi_initial (backward compat)
        mi_final          – alias of nmi_final (backward compat)
        n_iter            – total objective evaluations
        elapsed_s         – wall-clock seconds
    """
    if initial_transform is None:
        initial_transform = build_initial_euler3d(moving, fixed=fixed)

    pixel_id = moving.GetPixelID()
    # nmi_history records (iteration_index, nmi_value) tuples (ADR-4).
    nmi_history: list[tuple[int, float]] = []
    iter_count = [0]

    def negative_nmi(params):
        tx = sitk.Euler3DTransform(initial_transform)
        tx.SetParameters(tuple(float(p) for p in params))
        warped = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, pixel_id)
        a = sitk.GetArrayViewFromImage(warped)
        b = sitk.GetArrayViewFromImage(fixed)
        nmi = normalised_mutual_information(a, b, bins=bins)
        nmi_history.append((iter_count[0], float(nmi)))
        iter_count[0] += 1
        if verbose and iter_count[0] % 20 == 0:
            print(f"    iter {iter_count[0]:4d} | NMI = {nmi:.5f}")
        return -nmi

    x0 = np.array(initial_transform.GetParameters())
    t0 = time.time()
    result = minimize(
        negative_nmi,
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

    nmi_initial = nmi_history[0][1] if nmi_history else None
    nmi_final = -float(result.fun)

    return {
        "transform": best_tx,
        "inverse_transform": inv_tx,
        "forward_params": forward_params,
        "inverse_params": inverse_params,
        "nmi_initial": nmi_initial,
        "nmi_final": nmi_final,
        "nmi_history": nmi_history,
        # Backward-compat aliases so existing callers using mi_initial/mi_final still work.
        "mi_initial": nmi_initial,
        "mi_final": nmi_final,
        "n_iter": len(nmi_history),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Convergence persistence helper (ADR-6)
# ---------------------------------------------------------------------------


def save_mi_convergence(
    history: list[tuple[int, float]],
    json_path,
    png_path,
) -> None:
    """Persist the NMI convergence trace to JSON and PNG.

    Parameters
    ----------
    history:
        List of ``(iteration_index, nmi_value)`` tuples as returned by
        ``coregister()["nmi_history"]``.
    json_path:
        Path for the JSON output file. JSON keys: ``iterations``,
        ``nmi_values``, ``initial``, ``final``, ``metric``.
    png_path:
        Path for the PNG convergence plot.

    Side effects only — no return value.

    When ``history`` is empty: emits a ``UserWarning`` and returns without
    writing any files (guard against degenerate optimisation runs).
    """
    if not history:
        warnings.warn(
            "Empty NMI history passed to save_mi_convergence; "
            "skipping convergence save",
            UserWarning,
            stacklevel=2,
        )
        return

    iterations = [entry[0] for entry in history]
    nmi_values = [entry[1] for entry in history]
    initial_nmi = nmi_values[0]
    final_nmi = nmi_values[-1]

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    convergence_data = {
        "iterations": iterations,
        "nmi_values": nmi_values,
        "initial": initial_nmi,
        "final": final_nmi,
        "metric": "NMI",
    }
    with open(json_path, "w") as f:
        json.dump(convergence_data, f, indent=2)

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, nmi_values, marker="o", markersize=3, linewidth=1.5,
            label="NMI per evaluation")
    ax.axhline(initial_nmi, color="orange", linestyle="--", linewidth=1.0,
               label=f"Initial NMI = {initial_nmi:.4f}")
    ax.axhline(final_nmi, color="green", linestyle="--", linewidth=1.0,
               label=f"Final NMI = {final_nmi:.4f}")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("NMI")
    ax.set_title("Coregistration NMI evolution (Powell optimiser)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


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

    print("Running rigid coregistration (Powell, max 60 iter, NMI bins=16) ...")
    res = coregister(pet_ds_norm, mr_ds_norm, bins=16, max_iter=60, verbose=True)

    print(f"  Initial NMI : {res['nmi_initial']:.5f}")
    print(f"  Final   NMI : {res['nmi_final']:.5f}")
    print(f"  Improvement: {(res['nmi_final'] - res['nmi_initial']) / res['nmi_initial'] * 100:+.1f} %")
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
            "mi_initial": res["nmi_initial"],
            "mi_final": res["nmi_final"],
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

    # Save NMI convergence trace.
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    mi_json_path = os.path.join(results_dir, "mi_convergence.json")
    mi_png_path = os.path.join(out_dir, "15_mi_convergence.png")
    save_mi_convergence(res["nmi_history"], mi_json_path, mi_png_path)
    print(f"  NMI convergence JSON -> {mi_json_path}")
    print(f"  NMI convergence PNG  -> {mi_png_path}")

    print("Done.")
