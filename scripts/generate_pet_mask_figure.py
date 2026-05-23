"""Generate Figure 14: brain mask resampled from MR to PET space.

Produces ``docs/figures/14_pet_brain_mask_overlay.png`` showing the PET
temporal mean (hot colormap) with the brain parenchyma mask (lime contour)
resampled into PET space via the inverse Euler3DTransform.

Prerequisites (run these first if the files are missing)::

    # Generate brain mask (TotalSegmentator):
    conda run -n medical-image-processing-11763 python src/brain_mask.py

    # Generate coregistration transform:
    conda run -n medical-image-processing-11763 python src/coregistration.py

Run from repo root::

    conda run -n medical-image-processing-11763 python scripts/generate_pet_mask_figure.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from coregistration import load_rigid_transform
from loading import load_dynamic_pet
from visualization import save_pet_brain_mask_overlay


def main() -> None:
    """Generate Figure 14: brain mask overlaid on PET temporal mean."""
    # --- Resolve all required paths ---
    pet_dicom = _REPO_ROOT / "data" / "raw" / "e_1_BRAIN_DINAMIC_COLINA.dcm"
    transform_path = _REPO_ROOT / "docs" / "figures" / "coreg_transform.tfm"
    brain_mask_mr_path = _REPO_ROOT / "results" / "brain_mask_mr.nii.gz"
    out_path = _REPO_ROOT / "docs" / "figures" / "14_pet_brain_mask_overlay.png"

    # --- Validate inputs before doing any work ---
    missing = []
    if not pet_dicom.exists():
        missing.append(str(pet_dicom))
    if not transform_path.exists():
        missing.append(
            f"{transform_path}  (run: python src/coregistration.py first)"
        )
    if not brain_mask_mr_path.exists():
        missing.append(
            f"{brain_mask_mr_path}  (run: python src/brain_mask.py first)"
        )
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n  " + "\n  ".join(missing)
        )

    # --- Load PET DICOM ---
    print(f"Loading PET DICOM from {pet_dicom} ...")
    pet = load_dynamic_pet(str(pet_dicom))
    print(f"  PET image size (x,y,z): {pet.image.GetSize()}")

    # --- Load coregistration transform (forward: PET -> MR) ---
    print(f"Loading coregistration transform from {transform_path} ...")
    forward_tx = load_rigid_transform(str(transform_path))
    print(f"  Transform params: {list(forward_tx.GetParameters())}")

    # --- Invert transform (MR -> PET direction) ---
    inverse_tx = forward_tx.GetInverse()

    # --- Load brain mask (MR space) ---
    print(f"Loading brain mask from {brain_mask_mr_path} ...")
    brain_mask_mr = sitk.ReadImage(str(brain_mask_mr_path))
    print(f"  Brain mask size (x,y,z): {brain_mask_mr.GetSize()}")

    # --- Resample brain mask from MR space to PET space ---
    print("Resampling brain mask to PET space via inverse transform ...")
    brain_mask_in_pet = sitk.Resample(
        brain_mask_mr,
        pet.image,
        inverse_tx,
        sitk.sitkNearestNeighbor,
        0,
        brain_mask_mr.GetPixelID(),
    )
    arr = sitk.GetArrayFromImage(brain_mask_in_pet)
    print(f"  Brain mask in PET space: {int((np.asarray(arr) > 0).sum()):,} voxels")

    # --- Generate and save the overlay figure ---
    print(f"Saving PET brain mask overlay to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pet_brain_mask_overlay(pet.image, brain_mask_in_pet, str(out_path))
    size_kb = out_path.stat().st_size / 1024
    print(f"  Wrote {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
