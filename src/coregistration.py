"""Objective 2 — 3D rigid coregistration in physical coordinates.

Pipeline:

1. Load both DICOMs and build their voxel-to-physical 4x4 affines from
   ``ImagePositionPatient``, ``ImageOrientationPatient``, ``PixelSpacing``
   and ``SpacingBetweenSlices``.
2. Resample the PET temporal mean onto the MR voxel grid using those
   affines, so both volumes live in the same physical / voxel space.
3. Find the rigid transform (translation + axial rotation) that maximises
   Mutual Information between the resampled PET and the MR. The
   optimisation runs on a 2x downsampled version of both volumes for
   speed; the final transform is applied at full resolution.
4. Save a before/after MIP overlay and dump the optimal parameters plus
   the inverse transform parameters (used in Objective 3 to push masks
   from MR space back into PET space).

Run:

    python src/coregistration.py
"""

from __future__ import annotations

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage as ndi
from scipy.optimize import minimize

from loading import compute_temporal_mean, load_dynamic_pet, load_mr
from utils import filepath


# ---------------------------------------------------------------------------
# DICOM affine
# ---------------------------------------------------------------------------

def _read_position_orientation(ds):
    """Locate ImagePositionPatient / ImageOrientationPatient in this dataset.

    Both DICOMs in the project store these tags inside the
    ``DetectorInformationSequence`` instead of at the top level, which is
    why we walk the structure here.
    """
    if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds:
        return ds.ImagePositionPatient, ds.ImageOrientationPatient
    seq = ds.get("DetectorInformationSequence")
    if seq:
        det = seq[0]
        return det.ImagePositionPatient, det.ImageOrientationPatient
    raise ValueError("Could not locate ImagePositionPatient / ImageOrientationPatient.")


def voxel_to_physical_affine(ds) -> np.ndarray:
    """Build a 4x4 affine that maps voxel indices (slice, row, col) to mm.

    The pixel array is indexed as ``(z, y, x)`` where z is the slice axis,
    y the row axis and x the column axis. The affine columns therefore
    correspond, in order, to the slice, row and column physical
    directions multiplied by their respective spacing.
    """
    ipp, iop = _read_position_orientation(ds)
    ipp = np.asarray([float(v) for v in ipp])
    iop = np.asarray([float(v) for v in iop])

    pixel_spacing = [float(v) for v in ds.PixelSpacing]  # [row, col]
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


def resample_to_grid(
    src_volume: np.ndarray,
    src_affine: np.ndarray,
    dst_affine: np.ndarray,
    dst_shape: tuple,
    order: int = 1,
) -> np.ndarray:
    """Resample ``src_volume`` (in src space) onto ``dst_affine``'s voxel grid.

    For each voxel in the destination grid we compute the corresponding
    voxel in the source volume through ``src_affine^{-1} dst_affine`` and
    sample with the requested interpolation order.
    """
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
# Rigid model in voxel space (volume centred)
# ---------------------------------------------------------------------------

def _axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix from an axis-angle pair via Rodrigues' formula."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def apply_rigid_to_volume(volume: np.ndarray, params: np.ndarray, order: int = 1) -> np.ndarray:
    """Apply a rigid transform (translation + axial rotation) around the volume centre."""
    tz, ty, tx, angle, ax, ay, az = params
    R = _axis_angle_matrix(np.array([ax, ay, az]), angle)
    centre = (np.array(volume.shape) - 1) / 2.0
    offset = centre - R @ centre + np.array([tz, ty, tx])
    return ndi.affine_transform(
        volume, matrix=R, offset=offset,
        order=order, mode="constant", cval=0.0,
    )


def inverse_rigid_params(params: np.ndarray) -> np.ndarray:
    """Invert a rigid transform expressed as ``[tz, ty, tx, angle, ax, ay, az]``.

    For a rigid transform ``T(x) = R x + t`` the inverse is
    ``T^{-1}(y) = R^T y - R^T t``. In axis-angle form, ``R`` is inverted by
    flipping the sign of the angle (same axis), and the translation
    becomes ``-R^T t``.
    """
    tz, ty, tx, angle, ax, ay, az = params
    R = _axis_angle_matrix(np.array([ax, ay, az]), angle)
    t = np.array([tz, ty, tx])
    t_inv = -R.T @ t
    return np.array([t_inv[0], t_inv[1], t_inv[2], -angle, ax, ay, az])


# ---------------------------------------------------------------------------
# Mutual Information loss
# ---------------------------------------------------------------------------

def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Shannon mutual information between two equal-shape arrays.

    Uses a ``bins`` x ``bins`` joint histogram to estimate the marginal
    and joint probabilities, then returns ``H(a) + H(b) - H(a, b)``.
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
    moving: np.ndarray,
    fixed: np.ndarray,
    bins: int = 32,
    max_iter: int = 60,
    initial_params: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Maximise Mutual Information by optimising a rigid transform.

    Returns a dictionary with the optimal forward and inverse parameters,
    the initial and final MI values, the iteration count and the elapsed
    time. The optimiser is ``scipy.optimize.minimize`` with the Powell
    method, which is derivative-free and robust to the ragged MI
    landscape.
    """
    if initial_params is None:
        initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    history = []

    def negative_mi(params):
        warped = apply_rigid_to_volume(moving, params)
        mi = mutual_information(warped, fixed, bins=bins)
        history.append(mi)
        if verbose and len(history) % 20 == 0:
            print(f"    iter {len(history):4d} | MI = {mi:.5f}")
        return -mi

    t0 = time.time()
    result = minimize(
        negative_mi,
        x0=initial_params,
        method="Powell",
        options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4},
    )
    elapsed = time.time() - t0

    final_params = result.x
    return {
        "params": final_params,
        "params_inverse": inverse_rigid_params(final_params),
        "mi_initial": history[0] if history else None,
        "mi_final": -result.fun,
        "n_iter": len(history),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_before_after(fixed, moving_before, moving_after, out_path):
    """Save a 2x3 figure with axial / coronal / sagittal MIPs of the
    fixed volume overlaid with the moving volume before and after the
    rigid registration."""
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
    pet_mean = compute_temporal_mean(pet.array).astype(np.float32)
    mr_vol = mr.array.astype(np.float32)

    print("Building DICOM affines ...")
    pet_affine = voxel_to_physical_affine(pydicom.dcmread(pet_path, stop_before_pixels=True))
    mr_affine = voxel_to_physical_affine(pydicom.dcmread(mr_path, stop_before_pixels=True))
    print(f"  MR  affine diag: {np.diag(mr_affine[:3, :3])}")
    print(f"  PET affine diag: {np.diag(pet_affine[:3, :3])}")
    print(f"  MR  origin     : {mr_affine[:3, 3]}")
    print(f"  PET origin     : {pet_affine[:3, 3]}")

    print("Resampling PET onto MR voxel grid (physical-coordinate aware) ...")
    pet_on_mr = resample_to_grid(pet_mean, pet_affine, mr_affine, dst_shape=mr_vol.shape, order=1)
    print(f"  pet_on_mr shape : {pet_on_mr.shape}  range: [{pet_on_mr.min():.1f}, {pet_on_mr.max():.1f}]")

    print("Downsampling 2x for the rigid optimisation ...")
    factor = 2
    pet_ds = ndi.zoom(pet_on_mr, 1 / factor, order=1)
    mr_ds = ndi.zoom(mr_vol, 1 / factor, order=1)

    print("Centring intensities ...")
    pet_ds = (pet_ds - pet_ds.mean()) / (pet_ds.std() + 1e-6)
    mr_ds = (mr_ds - mr_ds.mean()) / (mr_ds.std() + 1e-6)

    print("Running rigid coregistration (Powell, max 60 iter) ...")
    res = coregister(pet_ds, mr_ds, bins=32, max_iter=60, verbose=True)

    print(f"  Initial MI : {res['mi_initial']:.5f}")
    print(f"  Final   MI : {res['mi_final']:.5f}")
    print(f"  Improvement: {(res['mi_final'] - res['mi_initial']) / res['mi_initial'] * 100:+.1f} %")
    print(f"  Iterations : {res['n_iter']}")
    print(f"  Elapsed    : {res['elapsed_s']:.1f} s")
    p = res["params"]
    print(
        "  Forward params (downsampled vox): "
        f"t=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})  "
        f"angle={math.degrees(p[3]):+.2f}deg"
    )
    p_inv = res["params_inverse"]
    print(
        "  Inverse params (downsampled vox): "
        f"t=({p_inv[0]:+.2f},{p_inv[1]:+.2f},{p_inv[2]:+.2f})  "
        f"angle={math.degrees(p_inv[3]):+.2f}deg"
    )

    print("Applying the optimal transform to the full-resolution PET ...")
    full_params = np.array(res["params"])
    full_params[0:3] *= factor
    pet_on_mr_registered = apply_rigid_to_volume(pet_on_mr, full_params)

    print("Saving before/after MIP overlay (full resolution) ...")
    info = save_before_after(
        mr_vol, pet_on_mr, pet_on_mr_registered,
        os.path.join(out_dir, "05_coreg_before_after.png"),
    )
    print(f"  -> {info['path']}")

    # Persist the final transform for downstream use (Objective 3 mask propagation)
    npz_out = os.path.join(out_dir, "coreg_transform.npz")
    np.savez(
        npz_out,
        forward_params=full_params,
        inverse_params=inverse_rigid_params(full_params),
        mi_initial=res["mi_initial"],
        mi_final=res["mi_final"],
        downsample_factor=factor,
        mr_shape=mr_vol.shape,
        pet_shape=pet_mean.shape,
        mr_affine=mr_affine,
        pet_affine=pet_affine,
    )
    print(f"  -> {npz_out}")

    print("Done.")
