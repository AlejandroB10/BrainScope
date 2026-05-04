"""Objective 2 — 3D rigid coregistration (PET → MR).

Implementation pending. The module will:

1. Build a temporally averaged PET volume (input) and align it rigidly to the
   MR T1 volume (reference).
2. Define an initial rigid transform (translation + axial rotation) and
   optimize its parameters with ``scipy.optimize.least_squares`` driving a
   similarity metric — Mutual Information is the planned default.
3. Resample the input volume into the reference space.
4. Produce a rotating Maximum Intensity Projection animation on the
   coronal/sagittal planes for the reference volume, the coregistered input
   volume, and an alpha-fused overlay of both.

The rigid math reuses the helpers explored in the course activities
(``activity05`` / ``activity06``).
"""
