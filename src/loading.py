"""Objective 1 — DICOM loading and visualization.

Run this module directly to produce the figures and the GIF expected by
Objective 1 of the project proposal:

    python src/loading.py

Outputs land in `docs/figures/`.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk

from utils import filepath


# ---------------------------------------------------------------------------
# DICOM header helpers
# ---------------------------------------------------------------------------

def _read_position_orientation(ds):
    """Locate ImagePositionPatient / ImageOrientationPatient in this dataset.

    Both DICOMs in the project store these tags inside the
    DetectorInformationSequence instead of at the top level, which is
    why we walk the structure here.
    """
    if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds:
        return ds.ImagePositionPatient, ds.ImageOrientationPatient
    seq = ds.get("DetectorInformationSequence")
    if seq:
        det = seq[0]
        return det.ImagePositionPatient, det.ImageOrientationPatient
    raise ValueError("Could not locate ImagePositionPatient / ImageOrientationPatient.")


def _build_sitk_image(
    array_zyx: np.ndarray,
    origin_xyz: tuple,
    spacing_xyz: tuple,
    direction_flat: tuple,
) -> sitk.Image:
    """Wrap a (z, y, x) numpy array in a sitk.Image with full geometry.

    SimpleITK uses (x, y, z) conventions for size, origin, spacing, and
    direction, while numpy stores the array as (z, y, x). GetImageFromArray
    handles the axis transposition automatically (it reverses the axis
    order when building the sitk.Image), so we pass the array as-is and
    supply the physical metadata in sitk's (x, y, z) order.

    Parameters
    ----------
    array_zyx:
        Volume in (z, y, x) numpy layout.
    origin_xyz:
        Physical origin of the first voxel in (x, y, z) mm.
    spacing_xyz:
        Voxel size in (x, y, z) mm.
    direction_flat:
        9-element row-major direction cosine matrix in (x, y, z) order.
    """
    img = sitk.GetImageFromArray(array_zyx.astype(np.float32))
    img.SetOrigin(origin_xyz)
    img.SetSpacing(spacing_xyz)
    img.SetDirection(direction_flat)
    return img


# ---------------------------------------------------------------------------
# Study dataclasses
# ---------------------------------------------------------------------------

def _assert_nifti_geometry(
    original: sitk.Image,
    path: str,
    tol: float = 1e-4,
) -> None:
    """Assert that a NIfTI file on disk has the same geometry as the original image.

    Reads the file back via sitk.ReadImage and compares origin, spacing, and
    direction element-wise within the given tolerance. Raises AssertionError
    with a descriptive message if any element exceeds the tolerance.

    Parameters
    ----------
    original:
        The sitk.Image before writing.
    path:
        Path to the NIfTI file that was written.
    tol:
        Absolute tolerance for floating-point comparisons (default 1e-4).
    """
    reloaded = sitk.ReadImage(path)

    for attr in ("GetOrigin", "GetSpacing", "GetDirection"):
        orig_vals = getattr(original, attr)()
        back_vals = getattr(reloaded, attr)()
        for i, (a, b) in enumerate(zip(orig_vals, back_vals)):
            if abs(a - b) >= tol:
                raise AssertionError(
                    f"NIfTI geometry mismatch in {attr}()[{i}]: "
                    f"original={a}, reloaded={b}, diff={abs(a-b):.2e} > tol={tol}"
                )


@dataclass
class PETStudy:
    array: np.ndarray                      # shape (frames, slices, rows, cols)
    spacing_zyx_mm: tuple                  # (z, y, x) in mm
    n_frames: int
    n_slices: int
    frame_durations_ms: np.ndarray         # length n_frames
    frame_positions_vector: np.ndarray     # length n_frames * n_slices (DICOM tag 0055,1002)
    image: sitk.Image = None              # temporal-mean 3D volume as sitk.Image

    def write_nifti(self, path: str) -> None:
        """Write the temporal-mean 3D volume to a NIfTI file.

        Parameters
        ----------
        path:
            Output file path. The extension (.nii or .nii.gz) determines
            whether the file is compressed.
        """
        sitk.WriteImage(self.image, path)


@dataclass
class MRStudy:
    array: np.ndarray            # shape (slices, rows, cols)
    spacing_zyx_mm: tuple        # (z, y, x) in mm
    image: sitk.Image = None     # 3D volume as sitk.Image

    def write_nifti(self, path: str) -> None:
        """Write the MR T1 volume to a NIfTI file.

        Parameters
        ----------
        path:
            Output file path. The extension (.nii or .nii.gz) determines
            whether the file is compressed.
        """
        sitk.WriteImage(self.image, path)


# ---------------------------------------------------------------------------
# DICOM loaders
# ---------------------------------------------------------------------------

def load_dynamic_pet(path: str) -> PETStudy:
    """Read the dynamic PET DICOM and reshape it into a 4D volume.

    The DICOM file packs every (frame, slice) pair as a row of the flat
    pixel_array. Adjacent rows share the same temporal frame and walk
    through the spatial slices from caudal to cranial; the next batch of
    rows starts the next temporal frame at the lowest slice again.

    Tag (0055, 1002) — FramePositionsVector — is read and stored as
    ``PETStudy.frame_positions_vector`` (dtype float64). The tag may store
    one or three values per (frame, slice) pair (z-only or xyz-coordinates
    depending on the scanner). A ``UserWarning`` is emitted via
    ``warnings.warn`` when the tag is absent or its length is not a clean
    integer multiple of ``n_frames * n_slices``; the reshape is never altered
    by this validation.
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

    # Read FramePositionsVector (0055, 1002) — used as a cross-check for the reshape.
    # Missing or length-mismatched tags emit a UserWarning but never raise.
    fpv_elem = ds.get((0x0055, 0x1002), None)
    if fpv_elem is None:
        warnings.warn(
            "DICOM tag (0055, 1002) FramePositionsVector not present; "
            "frame_positions_vector will be empty",
            UserWarning,
            stacklevel=2,
        )
        fpv = np.array([], dtype=np.float64)
    else:
        fpv = np.asarray(list(fpv_elem.value), dtype=np.float64)
        # FPV may store 1 or 3 values per (frame, slice) pair (scalar z-pos or xyz-pos).
        # Warn only when the length is not a clean integer multiple of n_frames*n_slices.
        base = n_frames * n_slices
        if base > 0 and len(fpv) % base != 0:
            warnings.warn(
                f"FramePositionsVector length {len(fpv)} is not an integer multiple of "
                f"n_frames*n_slices={base}; reshape preserved, field populated as-is",
                UserWarning,
                stacklevel=2,
            )

    # Build the sitk.Image from the temporal mean. We need the geometry
    # tags from the header, which may live inside DetectorInformationSequence.
    ds_hdr = pydicom.dcmread(path, stop_before_pixels=True)
    ipp, iop = _read_position_orientation(ds_hdr)
    ipp = [float(v) for v in ipp]   # physical origin [x, y, z] mm
    iop = [float(v) for v in iop]   # 6 direction cosines

    # sitk direction matrix is a 9-element flat row-major cosine matrix.
    # Columns are: x-axis direction, y-axis direction, z-axis direction.
    # IOP gives row_dir (first 3) and col_dir (last 3); slice_dir = row x col.
    row_dir = np.array(iop[0:3])   # corresponds to x in DICOM
    col_dir = np.array(iop[3:6])   # corresponds to y in DICOM
    slice_dir = np.cross(row_dir, col_dir)

    # sitk direction: columns are the physical x, y, z axes of the image.
    # For a standard axial acquisition: col 0 = x-axis, col 1 = y-axis, col 2 = z-axis.
    direction_flat = tuple(
        row_dir.tolist() + col_dir.tolist() + slice_dir.tolist()
    )
    origin_xyz = tuple(ipp)
    spacing_xyz = (pixel_spacing[1], pixel_spacing[0], z_spacing)  # (x, y, z)

    mean_3d = array.mean(axis=0)  # (slices, rows, cols) = (z, y, x)
    image = _build_sitk_image(mean_3d, origin_xyz, spacing_xyz, direction_flat)

    return PETStudy(
        array=array,
        spacing_zyx_mm=spacing_zyx,
        n_frames=n_frames,
        n_slices=n_slices,
        frame_durations_ms=fdu,
        frame_positions_vector=fpv,
        image=image,
    )


def load_mr(path: str) -> MRStudy:
    ds = pydicom.dcmread(path)
    array = ds.pixel_array  # (slices, rows, cols)
    pixel_spacing = [float(v) for v in ds.PixelSpacing]
    z_spacing = float(getattr(ds, "SpacingBetweenSlices", 1.0))
    spacing_zyx = (z_spacing, pixel_spacing[0], pixel_spacing[1])

    # Build sitk.Image geometry from DICOM headers.
    ds_hdr = pydicom.dcmread(path, stop_before_pixels=True)
    ipp, iop = _read_position_orientation(ds_hdr)
    ipp = [float(v) for v in ipp]
    iop = [float(v) for v in iop]

    row_dir = np.array(iop[0:3])
    col_dir = np.array(iop[3:6])
    slice_dir = np.cross(row_dir, col_dir)

    direction_flat = tuple(
        row_dir.tolist() + col_dir.tolist() + slice_dir.tolist()
    )
    origin_xyz = tuple(ipp)
    spacing_xyz = (pixel_spacing[1], pixel_spacing[0], z_spacing)

    image = _build_sitk_image(array, origin_xyz, spacing_xyz, direction_flat)

    return MRStudy(array=array, spacing_zyx_mm=spacing_zyx, image=image)


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

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
    print(f"  PET sitk image size (x,y,z): {pet.image.GetSize()}")

    print("Loading MR T1 ...")
    mr = load_mr(mr_path)
    print(f"  MR shape    : {mr.array.shape}  spacing(z,y,x): {mr.spacing_zyx_mm}")
    print(f"  MR  sitk image size (x,y,z): {mr.image.GetSize()}")

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
