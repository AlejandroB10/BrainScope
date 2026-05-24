"""Generate figure 16: MedSAM2 tumor mask overlaid on the PET last frame.

The tumor mask lives in MR space. This script:
1. Loads the PET DICOM and extracts the last temporal frame (peak tracer uptake).
2. Loads the coregistration transform from docs/figures/coreg_transform.tfm.
3. Loads the MR-space tumor mask from results/tumor_mask_mr_medsam2.nii.gz.
4. Resamples the tumor mask to PET space via the inverse transform.
5. Saves the overlay to docs/figures/16_pet_tumor_mask_overlay.png.

The last frame is preferred over the duration-weighted temporal mean because the
mean averages early-frame wash-in (low absolute uptake) with the late steady-state
plateau, washing out the tumor signal. The last frame shows peak tracer
concentration, which is what makes the lesion most visible in PET.

Run:
    python scripts/generate_pet_tumor_figure.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import SimpleITK as sitk

from coregistration import load_rigid_transform
from loading import load_dynamic_pet
from utils import filepath
from visualization import save_pet_brain_mask_overlay


def main() -> None:
    out_dir = os.path.join(_REPO_ROOT, "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # --- Input paths ---
    pet_dicom_path = filepath("e_1_BRAIN_DINAMIC_COLINA.dcm")
    transform_path = os.path.join(out_dir, "coreg_transform.tfm")
    tumor_mask_path = os.path.join(_REPO_ROOT, "results", "tumor_mask_mr_medsam2.nii.gz")
    out_path = os.path.join(out_dir, "16_pet_tumor_mask_overlay.png")

    # --- Guard: verify required inputs exist ---
    if not os.path.exists(pet_dicom_path):
        raise FileNotFoundError(
            f"PET DICOM not found at: {pet_dicom_path}\n"
            "Ensure the DICOM data is present in data/raw/"
        )
    if not os.path.exists(transform_path):
        raise FileNotFoundError(
            f"Coregistration transform not found at: {transform_path}\n"
            "Remediation: run `python -m src.coregistration` first to generate the transform."
        )
    if not os.path.exists(tumor_mask_path):
        raise FileNotFoundError(
            f"Tumor mask not found at: {tumor_mask_path}\n"
            "Remediation: run the MedSAM2 segmentation pipeline to generate "
            "results/tumor_mask_mr_medsam2.nii.gz."
        )

    # --- Load PET and extract the LAST FRAME (peak tracer uptake) ---
    print("Loading dynamic PET ...")
    pet_study = load_dynamic_pet(pet_dicom_path)
    last_frame_arr = pet_study.array[-1]  # shape (n_slices, rows, cols)
    pet_last_image = sitk.GetImageFromArray(last_frame_arr.astype("float32"))
    pet_last_image.CopyInformation(pet_study.image)  # preserve PET geometry
    print(f"  PET size (x,y,z): {pet_last_image.GetSize()}  (last frame, peak uptake)")

    # --- Load coregistration transform (MR-space -> PET-space via inverse) ---
    print("Loading coregistration transform ...")
    forward_transform = load_rigid_transform(transform_path)
    # The forward transform maps moving (PET) -> fixed (MR).
    # Inverse maps MR -> PET, which is what we need to bring the tumor mask
    # from MR space into PET space.
    inverse_transform = forward_transform.GetInverse()
    print(f"  Transform loaded: {type(forward_transform).__name__}")

    # --- Load MR-space tumor mask ---
    print("Loading tumor mask (MR space) ...")
    tumor_mask_mr = sitk.ReadImage(tumor_mask_path)
    print(f"  Tumor mask size (x,y,z): {tumor_mask_mr.GetSize()}")

    # --- Resample tumor mask to PET space ---
    print("Resampling tumor mask to PET space ...")
    tumor_in_pet = sitk.Resample(
        tumor_mask_mr,
        pet_last_image,
        inverse_transform,
        sitk.sitkNearestNeighbor,
        0.0,
        tumor_mask_mr.GetPixelID(),
    )
    voxels_in_pet = int(sitk.GetArrayFromImage(tumor_in_pet).sum())
    print(f"  Tumor voxels in PET space: {voxels_in_pet}")

    # --- Save overlay figure ---
    print(f"Saving overlay to {out_path} ...")
    save_pet_brain_mask_overlay(
        pet_last_image,
        tumor_in_pet,
        out_path,
        title=(
            "PET (last frame, peak uptake) with MedSAM2 tumor mask "
            "(resampled to PET space)"
        ),
    )

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  -> {out_path}  ({size_kb:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
