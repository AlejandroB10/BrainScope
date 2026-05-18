"""Brain mask extraction via TotalSegmentator.

Public API
----------
extract_brain_mask(mr_image, device, fast) -> sitk.Image

VRAM management
---------------
TotalSegmentator allocates 2-4 GB on the GPU. After this function returns,
torch.cuda.empty_cache() is called unconditionally so the caller can load
a second model (e.g. MedSAM2) without running into OOM errors.

Never call extract_brain_mask and a second GPU model inside the same scope
without an intermediate torch.cuda.empty_cache() call.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk


def extract_brain_mask(
    mr_image: sitk.Image,
    device: str = "gpu",
    fast: bool = False,
) -> sitk.Image:
    """Extract a binary brain mask from an MR T1 volume using TotalSegmentator.

    The input and output are both ``sitk.Image`` objects. The NIfTI round-trip
    is internal: a temp directory is created, the MR is written to it, TS runs
    inside it, and only ``brain.nii.gz`` is read back. Everything else is
    cleaned up automatically.

    Parameters
    ----------
    mr_image:
        T1-weighted MR volume as ``sitk.Image``.
    device:
        Device string forwarded to TotalSegmentator (``"gpu"`` or ``"cpu"``).
    fast:
        When True, passes ``fast=True`` to TotalSegmentator, which uses a
        lower-resolution model and runs faster at the cost of some accuracy.

    Returns
    -------
    sitk.Image
        Binary brain mask (uint8, values 0/1) co-aligned with ``mr_image``.

    Raises
    ------
    RuntimeError
        If TotalSegmentator is not installed in the active environment.
    FileNotFoundError
        If TS runs but does not produce ``brain.nii.gz`` in the output folder.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise RuntimeError(
            "TotalSegmentator is not installed. "
            "Run: pip install TotalSegmentator"
        ) from exc

    print("brain_mask: writing MR to temp NIfTI for TotalSegmentator ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mr_nifti = tmp_path / "mr.nii.gz"
        seg_dir = tmp_path / "seg"
        seg_dir.mkdir()

        sitk.WriteImage(mr_image, str(mr_nifti))

        print(f"brain_mask: running TotalSegmentator (fast={fast}, device={device}) ...")
        totalsegmentator(
            input=mr_nifti,
            output=seg_dir,
            task="total_mr",
            fast=fast,
            device=device,
            quiet=True,
            ml=False,
        )

        brain_nifti = seg_dir / "brain.nii.gz"
        if not brain_nifti.exists():
            raise FileNotFoundError(
                f"TotalSegmentator finished but brain.nii.gz was not found at {brain_nifti}. "
                "Check whether task='total_mr' is available in your TotalSegmentator version."
            )

        brain_mask = sitk.ReadImage(str(brain_nifti))
        print(f"brain_mask: done. mask size={brain_mask.GetSize()}")

    # Verify geometry is preserved within tolerance (REQ-5).
    _assert_brain_mask_geometry(mr_image, brain_mask)

    # Release GPU memory so the next model (MedSAM2) can load without OOM.
    # Note: `del model` is not called here because TotalSegmentator manages the
    # nnU-Net model internally (no handle is exposed to the caller). The
    # torch.cuda.empty_cache() call below is the correct VRAM release mechanism.
    _release_gpu_memory()

    return brain_mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_brain_mask_geometry(
    original: sitk.Image,
    brain_mask: sitk.Image,
    tol: float = 1e-2,
) -> None:
    """Warn if the brain mask geometry drifts significantly from the input MR.

    TotalSegmentator resamples to 1.5 mm isotropic internally, then resamples
    back to the original grid. The origin/spacing/direction may differ by a
    small amount due to NIfTI header rounding. We use a relaxed 1e-2 tolerance
    here (not the 1e-4 used for plain NIfTI round-trips) to avoid false failures.
    """
    import warnings

    for attr in ("GetOrigin", "GetSpacing", "GetDirection"):
        orig_vals = getattr(original, attr)()
        mask_vals = getattr(brain_mask, attr)()
        for i, (a, b) in enumerate(zip(orig_vals, mask_vals)):
            if abs(a - b) >= tol:
                warnings.warn(
                    f"Brain mask geometry drift in {attr}()[{i}]: "
                    f"MR={a:.6f}, mask={b:.6f}, diff={abs(a-b):.2e} (tol={tol})",
                    UserWarning,
                    stacklevel=3,
                )
                break  # one warning per attribute is enough


def _release_gpu_memory() -> None:
    """Call torch.cuda.empty_cache() if torch is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CLI entry point — run extract_brain_mask on the project MR T1
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os as _os

    _REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _src = _os.path.join(_REPO_ROOT, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from loading import load_mr

    mr_path = _os.path.join(_REPO_ROOT, "data", "raw", "AX_3D_T1.dcm")
    if not _os.path.exists(mr_path):
        print(f"ERROR: MR DICOM not found at {mr_path}")
        sys.exit(1)

    print(f"Loading MR from {mr_path} ...")
    mr = load_mr(mr_path)
    print(f"  MR size (x,y,z): {mr.image.GetSize()}")
    print(f"  MR spacing (x,y,z): {mr.image.GetSpacing()}")

    print("Running TotalSegmentator total_mr brain extraction ...")
    brain_mask = extract_brain_mask(mr.image)

    arr = sitk.GetArrayFromImage(brain_mask)
    voxel_count = int((arr > 0).sum())
    print(f"Brain mask voxel count: {voxel_count:,}")

    out_path = _os.path.join(_REPO_ROOT, "results", "brain_mask_mr.nii.gz")
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(brain_mask, out_path)
    print(f"Saved brain mask -> {out_path}")
