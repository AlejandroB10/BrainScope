"""Objective 1 — DICOM loading and visualization.

Implementation pending. The module will:

1. Read the dynamic PET DICOM file with pydicom.
2. Reorganize ``pixel_array`` from a flat (frames * slices, rows, cols) layout
   into a 4D volume (frame, slice, row, col) using the headers documented in
   ``data/README.md``.
3. Read the MR T1 DICOM as a 3D reference volume.
4. Produce static visualizations (last frame, temporal average) and an
   animated GIF showing the three median planes across all temporal frames.
"""
