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
import pydicom
import scipy.ndimage as ndi

from coregistration import (
    apply_rigid_to_volume,
    resample_to_grid,
    voxel_to_physical_affine,
)
from loading import compute_temporal_mean, load_dynamic_pet, load_mr
from utils import filepath


def _rotated_coronal_mip(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around the axial axis (z) and collapse the y axis with max."""
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

    pet_affine = voxel_to_physical_affine(pydicom.dcmread(pet_path, stop_before_pixels=True))
    mr_affine = voxel_to_physical_affine(pydicom.dcmread(mr_path, stop_before_pixels=True))

    print("Resampling PET onto MR grid ...")
    pet_on_mr = resample_to_grid(pet_mean, pet_affine, mr_affine, dst_shape=mr_vol.shape, order=1)

    print("Loading the saved rigid transform from coreg_transform.npz ...")
    npz = np.load(os.path.join(out_dir, "coreg_transform.npz"))
    forward = npz["forward_params"]

    print("Applying transform to full-resolution PET on MR grid ...")
    pet_registered = apply_rigid_to_volume(pet_on_mr, forward)

    print("Building rotating MIP animation (24 angles, this can take ~1 minute) ...")
    info = save_rotating_mip(
        mr_vol, pet_registered,
        os.path.join(out_dir, "06_rotating_mip.gif"),
        n_angles=24,
        fps=6,
    )
    print(f"  -> {info['path']}  ({info['frames']} frames)")

    print("Done.")
