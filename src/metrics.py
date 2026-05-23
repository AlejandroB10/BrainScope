"""Quantitative metrics for segmentation evaluation.

All functions operate on ``sitk.Image`` binary masks and return scalar floats.
No numpy conversions are exposed to callers; conversions are internal.

Functions
---------
volume_mm3(mask)
    Physical volume of the positive region in cubic millimetres.
sphericity(mask)
    ITK roundness as a proxy for geometric compactness (see note below).
dice(a, b)
    Sørensen–Dice overlap coefficient between two binary masks.
mask_brain_overlap(tumor_mask, brain_mask)
    Fraction of tumor voxels that fall inside the brain mask (V-2 metric).
tumor_brain_ratio(tumor_mask, brain_mask)
    Ratio of tumor volume to brain volume (both in mm³).

Notes
-----
``sphericity`` returns ``sitk.LabelShapeStatisticsImageFilter.GetRoundness(1)``.
This is ITK's roundness measure, which is derived from the ratio of the
equivalent-sphere radius to the mean distance from the surface to the centroid.
It is NOT the same as the strict sphericity formula
    π^(1/3) · (6V)^(2/3) / A
but it correlates well for compact lesions and is fast and stable.

Edge cases
----------
- Empty masks return 0.0 (or NaN for sphericity) with a warning instead of
  dividing by zero or raising inside ITK.
- Both-empty dice returns 0.0 (convention: undefined overlap ≡ 0).
- Shape mismatch in dice raises ``ValueError``.
"""

from __future__ import annotations

import warnings

import numpy as np
import SimpleITK as sitk


def volume_mm3(mask: sitk.Image) -> float:
    """Compute the physical volume of the positive region in mm³.

    Parameters
    ----------
    mask:
        Binary sitk.Image (any pixel type; non-zero voxels count as positive).

    Returns
    -------
    float
        Volume in mm³. Returns 0.0 if the mask is empty.
    """
    arr = sitk.GetArrayFromImage(mask)
    n_pos = int((arr != 0).sum())
    if n_pos == 0:
        warnings.warn("volume_mm3: mask is empty, returning 0.0", UserWarning, stacklevel=2)
        return 0.0

    sx, sy, sz = mask.GetSpacing()  # sitk: (x, y, z)
    return float(n_pos) * sx * sy * sz


def sphericity(mask: sitk.Image) -> float:
    """Compute ITK roundness as a proxy for sphericity.

    Uses ``sitk.LabelShapeStatisticsImageFilter.GetRoundness(label=1)``.
    Returns ``float('nan')`` for empty masks.

    Parameters
    ----------
    mask:
        Binary sitk.Image with label value 1 (or any non-zero value; the
        filter treats all non-zero voxels as label 1 after thresholding).

    Returns
    -------
    float
        ITK roundness in [0, 1] range (1.0 = perfect sphere). Returns NaN
        for empty masks.
    """
    # Ensure the mask is uint8 with a clear label=1.
    binary = sitk.Cast(mask != 0, sitk.sitkUInt8)

    arr = sitk.GetArrayFromImage(binary)
    if arr.max() == 0:
        warnings.warn("sphericity: mask is empty, returning NaN", UserWarning, stacklevel=2)
        return float("nan")

    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(binary)

    if not filt.HasLabel(1):
        warnings.warn("sphericity: label 1 not found, returning NaN", UserWarning, stacklevel=2)
        return float("nan")

    return float(filt.GetRoundness(1))


def dice(a: sitk.Image, b: sitk.Image) -> float:
    """Compute the Sørensen–Dice coefficient between two binary masks.

    Parameters
    ----------
    a, b:
        Binary sitk.Image objects. Must have the same size.

    Returns
    -------
    float
        Dice coefficient in [0, 1]. Returns 0.0 if both masks are empty.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different sizes.
    """
    if a.GetSize() != b.GetSize():
        raise ValueError(
            f"dice: size mismatch: a={a.GetSize()}, b={b.GetSize()}"
        )

    arr_a = sitk.GetArrayFromImage(a)
    arr_b = sitk.GetArrayFromImage(b)

    # Guard: ITK LabelOverlapMeasuresImageFilter returns inf when both masks
    # are all-zero (denominator is zero). Convention: dice of two empty masks
    # is defined as 0.0 — undefined overlap is treated as no overlap.
    if arr_a.max() == 0 and arr_b.max() == 0:
        warnings.warn(
            "dice: both masks are empty; returning 0.0 (undefined overlap treated as no overlap).",
            UserWarning,
            stacklevel=2,
        )
        return 0.0

    filt = sitk.LabelOverlapMeasuresImageFilter()

    # Cast to same type first; the filter requires matching types.
    binary_a = sitk.Cast(a != 0, sitk.sitkUInt32)
    binary_b = sitk.Cast(b != 0, sitk.sitkUInt32)

    filt.Execute(binary_a, binary_b)
    return float(filt.GetDiceCoefficient())


def mask_brain_overlap(tumor_mask: sitk.Image, brain_mask: sitk.Image) -> float:
    """Compute the fraction of tumor voxels inside the brain mask.

    This is V-2 in the validation plan: what proportion of the tumor segmentation
    lies within the TotalSegmentator brain mask?

    Parameters
    ----------
    tumor_mask:
        Binary sitk.Image of the tumor segmentation.
    brain_mask:
        Binary sitk.Image of the brain mask.

    Returns
    -------
    float
        Fraction in [0, 1]. Returns 0.0 if the tumor mask is empty.
    """
    arr_t = (sitk.GetArrayFromImage(tumor_mask) != 0)
    arr_b = (sitk.GetArrayFromImage(brain_mask) != 0)

    n_tumor = int(arr_t.sum())
    if n_tumor == 0:
        warnings.warn(
            "mask_brain_overlap: tumor mask is empty, returning 0.0",
            UserWarning,
            stacklevel=2,
        )
        return 0.0

    n_intersection = int((arr_t & arr_b).sum())
    return float(n_intersection) / float(n_tumor)


def tumor_brain_ratio(tumor_mask: sitk.Image, brain_mask: sitk.Image) -> float:
    """Compute the ratio of tumor volume to brain volume.

    Parameters
    ----------
    tumor_mask:
        Binary sitk.Image of the tumor segmentation.
    brain_mask:
        Binary sitk.Image of the brain mask.

    Returns
    -------
    float
        Ratio in [0, inf). Returns 0.0 if the brain mask is empty.
    """
    v_tumor = volume_mm3(tumor_mask)
    v_brain = volume_mm3(brain_mask)

    if v_brain == 0.0:
        warnings.warn(
            "tumor_brain_ratio: brain mask is empty, returning 0.0",
            UserWarning,
            stacklevel=2,
        )
        return 0.0

    return v_tumor / v_brain
