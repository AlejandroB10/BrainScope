"""Objective 1 — DICOM loading and visualization.

Run this module directly to produce the figures and the GIF expected by
Objective 1 of the project proposal:

    python src/loading.py

Outputs land in `docs/figures/`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from utils import filepath


@dataclass
class PETStudy:
    array: np.ndarray            # shape (frames, slices, rows, cols)
    spacing_zyx_mm: tuple        # (z, y, x) in mm
    n_frames: int
    n_slices: int
    frame_durations_ms: np.ndarray  # length n_frames


@dataclass
class MRStudy:
    array: np.ndarray            # shape (slices, rows, cols)
    spacing_zyx_mm: tuple        # (z, y, x) in mm


def load_dynamic_pet(path: str) -> PETStudy:
    """Read the dynamic PET DICOM and reshape it into a 4D volume.

    The DICOM file packs every (frame, slice) pair as a row of the flat
    pixel_array. Adjacent rows share the same temporal frame and walk
    through the spatial slices from caudal to cranial; the next batch of
    rows starts the next temporal frame at the lowest slice again. The
    reshape order has been verified against `FramePositionsVector`: every
    z position repeats exactly once per temporal frame.
    """
    ds = pydicom.dcmread(path)
    flat = ds.pixel_array  # (frames * slices, rows, cols)

    n_frames = int(ds.NumberOfFrames)
    fst = np.asarray(list(ds[(0x0055, 0x1001)].value), dtype=float)
    if len(fst) != n_frames:
        n_frames = len(fst)
    n_slices = flat.shape[0] // n_frames
    rows, cols = flat.shape[1], flat.shape[2]

    array = flat.reshape(n_frames, n_slices, rows, cols)

    pixel_spacing = [float(v) for v in ds.PixelSpacing]  # [row, col] = [y, x]
    z_spacing = float(ds.SpacingBetweenSlices)
    spacing_zyx = (z_spacing, pixel_spacing[0], pixel_spacing[1])

    fdu = np.asarray(list(ds[(0x0055, 0x1004)].value), dtype=float)

    return PETStudy(
        array=array,
        spacing_zyx_mm=spacing_zyx,
        n_frames=n_frames,
        n_slices=n_slices,
        frame_durations_ms=fdu,
    )


def load_mr(path: str) -> MRStudy:
    ds = pydicom.dcmread(path)
    array = ds.pixel_array  # (slices, rows, cols)
    pixel_spacing = [float(v) for v in ds.PixelSpacing]
    z_spacing = float(getattr(ds, "SpacingBetweenSlices", 1.0))
    spacing_zyx = (z_spacing, pixel_spacing[0], pixel_spacing[1])
    return MRStudy(array=array, spacing_zyx_mm=spacing_zyx)


def compute_temporal_mean(pet_4d: np.ndarray) -> np.ndarray:
    """Average the 4D PET volume across the temporal axis."""
    return pet_4d.mean(axis=0)


def compute_last_frame(pet_4d: np.ndarray) -> np.ndarray:
    return pet_4d[-1]


def median_planes(volume: np.ndarray) -> tuple:
    """Return axial, coronal and sagittal median planes of a (z, y, x) volume."""
    z, y, x = volume.shape
    axial = volume[z // 2, :, :]
    coronal = volume[:, y // 2, :]
    sagittal = volume[:, :, x // 2]
    return axial, coronal, sagittal


def _plot_three_planes(volume: np.ndarray, spacing_zyx, cmap, vmin, vmax, axes, title):
    axial, coronal, sagittal = median_planes(volume)
    z, y, x = spacing_zyx
    axes[0].imshow(axial, cmap=cmap, vmin=vmin, vmax=vmax, aspect=y / x)
    axes[0].set_title(f"{title} — axial")
    axes[1].imshow(coronal, cmap=cmap, vmin=vmin, vmax=vmax, aspect=z / x, origin="lower")
    axes[1].set_title(f"{title} — coronal")
    axes[2].imshow(sagittal, cmap=cmap, vmin=vmin, vmax=vmax, aspect=z / y, origin="lower")
    axes[2].set_title(f"{title} — sagittal")
    for ax in axes:
        ax.axis("off")


def save_static_visualizations(pet: PETStudy, out_dir: str) -> dict:
    """Static figure with the median planes of (a) last frame and (b) temporal mean."""
    last = compute_last_frame(pet.array)
    mean = compute_temporal_mean(pet.array)

    vmax = float(np.percentile(pet.array, 99.5))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    _plot_three_planes(last, pet.spacing_zyx_mm, "hot", 0, vmax, axes[0], "Last frame")
    _plot_three_planes(mean, pet.spacing_zyx_mm, "hot", 0, vmax, axes[1], "Temporal mean")
    fig.suptitle("PET dynamic — median planes (axial / coronal / sagittal)", fontsize=13)
    fig.tight_layout()

    out = os.path.join(out_dir, "02_pet_static_views.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"path": out, "vmax_used": vmax}


def save_median_planes_gif(pet: PETStudy, out_path: str, fps: int = 4) -> dict:
    """Animated GIF: 3 median planes side by side, sweeping the 36 temporal frames."""
    z, y, x = pet.spacing_zyx_mm
    vmax = float(np.percentile(pet.array, 99.5))
    frames = []

    for t in range(pet.n_frames):
        vol = pet.array[t]
        axial, coronal, sagittal = median_planes(vol)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        axes[0].imshow(axial, cmap="hot", vmin=0, vmax=vmax, aspect=y / x)
        axes[0].set_title("axial")
        axes[1].imshow(coronal, cmap="hot", vmin=0, vmax=vmax, aspect=z / x, origin="lower")
        axes[1].set_title("coronal")
        axes[2].imshow(sagittal, cmap="hot", vmin=0, vmax=vmax, aspect=z / y, origin="lower")
        axes[2].set_title("sagittal")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"PET dynamic — frame {t+1:02d}/{pet.n_frames}", fontsize=11)
        fig.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(out_path, frames, format="GIF", fps=fps)
    return {"path": out_path, "frames": len(frames)}


def save_mr_overview(mr: MRStudy, out_dir: str) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _plot_three_planes(
        mr.array, mr.spacing_zyx_mm, "gray",
        float(np.percentile(mr.array, 1)),
        float(np.percentile(mr.array, 99.5)),
        axes, "MR T1",
    )
    fig.suptitle("MR T1 — median planes", fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, "03_mr_static_views.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"path": out}


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    pet_path = filepath("e_1_BRAIN_DINAMIC_COLINA.dcm")
    mr_path = filepath("AX_3D_T1.dcm")

    print("Loading dynamic PET ...")
    pet = load_dynamic_pet(pet_path)
    print(f"  PET 4D shape : {pet.array.shape}  spacing(z,y,x): {pet.spacing_zyx_mm}")
    print(f"  Frames: {pet.n_frames}  Slices/frame: {pet.n_slices}")
    print(f"  Frame durations (ms): min={pet.frame_durations_ms.min()}  max={pet.frame_durations_ms.max()}")

    print("Loading MR T1 ...")
    mr = load_mr(mr_path)
    print(f"  MR shape    : {mr.array.shape}  spacing(z,y,x): {mr.spacing_zyx_mm}")

    print("Saving static PET visualizations ...")
    info = save_static_visualizations(pet, out_dir)
    print(f"  -> {info['path']}")

    print("Saving MR overview ...")
    info = save_mr_overview(mr, out_dir)
    print(f"  -> {info['path']}")

    print("Building median-planes GIF (this can take a minute) ...")
    info = save_median_planes_gif(
        pet, os.path.join(out_dir, "04_pet_median_planes.gif"), fps=4,
    )
    print(f"  -> {info['path']}  ({info['frames']} frames)")

    print("Done.")
