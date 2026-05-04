"""Objective 2 — 3D rigid coregistration.

The temporally averaged PET volume (input) is rigidly aligned to the MR T1
volume (reference) by maximizing Mutual Information between the two
intensity distributions.

Run:

    python src/coregistration.py

Outputs:
- `docs/figures/05_coreg_before_after.png`: MIP overlays of the PET on top
  of the MR before and after the rigid registration.
- Console log: per-iteration MI and final transform parameters.

Notes & limitations
-------------------
- The two volumes have different physical sampling: PET is
  ``(47, 256, 256)`` at ``(3.27, 1.17, 1.17)`` mm and MR is
  ``(156, 256, 256)`` at 1 mm isotropic. To run the optimization on a
  comparable grid we resample both volumes to a common downsampled shape
  (default ``(48, 64, 64)``). This is an intentional compromise for the
  intermediate submission. A full-resolution, physical-coordinate-aware
  pipeline is on the roadmap for the final submission.
- The rigid model is parameterized as in the course activities: a 3D
  translation plus an axial rotation by an angle around a unit axis.
- The optimizer is ``scipy.optimize.minimize`` with method ``Powell``,
  which is derivative-free and well-suited for the rugged MI landscape.
"""

from __future__ import annotations

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize

from loading import compute_temporal_mean, load_dynamic_pet, load_mr
from utils import filepath


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

def resample_to_shape(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Linear resample a 3D volume to ``target_shape``."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    out = ndi.zoom(volume.astype(np.float32), zoom=factors, order=1)
    # Defensive size match (zoom can be off by 1)
    crop = tuple(slice(0, t) for t in target_shape)
    out = out[crop]
    return out


# ---------------------------------------------------------------------------
# Rigid transformation
# ---------------------------------------------------------------------------

def _axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix from axis-angle (Rodrigues' formula)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)
    return R


def apply_rigid_to_volume(volume: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Apply a rigid transform (translation + axial rotation) to a 3D volume.

    ``params = [tz, ty, tx, angle_rad, ax, ay, az]``
    where translation is in voxels and the rotation is around the volume's
    centre.
    """
    tz, ty, tx, angle, ax, ay, az = params
    R = _axis_angle_matrix(np.array([ax, ay, az]), angle)
    # ndimage.affine_transform applies output = input(matrix @ output_coords + offset)
    # We want to rotate around the volume centre, so we shift by -centre, rotate, then shift back.
    centre = (np.array(volume.shape) - 1) / 2.0
    offset = centre - R @ centre + np.array([tz, ty, tx])
    return ndi.affine_transform(
        volume, matrix=R, offset=offset, order=1, mode="constant", cval=0.0,
    )


# ---------------------------------------------------------------------------
# Mutual Information loss
# ---------------------------------------------------------------------------

def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Shannon mutual information between two equal-shape volumes."""
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
    max_iter: int = 80,
    initial_params: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Maximize Mutual Information by optimizing a rigid transform.

    Returns a dict with the optimal parameters, the registered moving
    volume and the optimization trajectory.
    """
    if initial_params is None:
        initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    history = []

    def negative_mi(params):
        warped = apply_rigid_to_volume(moving, params)
        mi = mutual_information(warped, fixed, bins=bins)
        history.append(mi)
        if verbose and len(history) % 10 == 0:
            print(f"    iter {len(history):3d} | MI = {mi:.5f}")
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
    registered = apply_rigid_to_volume(moving, final_params)

    return {
        "params": final_params,
        "registered": registered,
        "mi_initial": history[0] if history else None,
        "mi_final": -result.fun,
        "n_iter": len(history),
        "elapsed_s": elapsed,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_before_after(
    fixed: np.ndarray,
    moving_before: np.ndarray,
    moving_after: np.ndarray,
    out_path: str,
) -> dict:
    """Save MIPs of MR + PET-before / MR + PET-after side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    fixed_mip_axial = fixed.max(axis=0)
    fixed_mip_coronal = fixed.max(axis=1)
    fixed_mip_sagittal = fixed.max(axis=2)

    for col, plane in enumerate(("axial", "coronal", "sagittal")):
        if plane == "axial":
            f = fixed.max(axis=0)
            mb = moving_before.max(axis=0)
            ma = moving_after.max(axis=0)
        elif plane == "coronal":
            f = fixed.max(axis=1)
            mb = moving_before.max(axis=1)
            ma = moving_after.max(axis=1)
        else:
            f = fixed.max(axis=2)
            mb = moving_before.max(axis=2)
            ma = moving_after.max(axis=2)

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

    print("Loading volumes ...")
    pet = load_dynamic_pet(filepath("e_1_BRAIN_DINAMIC_COLINA.dcm"))
    mr = load_mr(filepath("AX_3D_T1.dcm"))

    pet_mean = compute_temporal_mean(pet.array).astype(np.float32)
    mr_vol = mr.array.astype(np.float32)

    target_shape = (48, 64, 64)
    print(f"Resampling both volumes to {target_shape} for the optimization ...")
    pet_ds = resample_to_shape(pet_mean, target_shape)
    mr_ds = resample_to_shape(mr_vol, target_shape)

    print("Centering intensities ...")
    pet_ds = (pet_ds - pet_ds.mean()) / (pet_ds.std() + 1e-6)
    mr_ds = (mr_ds - mr_ds.mean()) / (mr_ds.std() + 1e-6)

    print("Running rigid coregistration (Powell, max 80 iter) ...")
    res = coregister(pet_ds, mr_ds, bins=32, max_iter=80, verbose=True)

    print(f"  Initial MI : {res['mi_initial']:.5f}")
    print(f"  Final   MI : {res['mi_final']:.5f}")
    print(f"  Iterations : {res['n_iter']}")
    print(f"  Elapsed    : {res['elapsed_s']:.1f} s")
    p = res["params"]
    print(
        "  Params (vox): "
        f"t=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})  "
        f"angle={math.degrees(p[3]):+.2f}deg  axis=({p[4]:+.2f},{p[5]:+.2f},{p[6]:+.2f})"
    )

    print("Saving before/after MIP overlay ...")
    info = save_before_after(
        mr_ds, pet_ds, res["registered"],
        os.path.join(out_dir, "05_coreg_before_after.png"),
    )
    print(f"  -> {info['path']}")

    print("Done.")
