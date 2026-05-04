"""Objective 3 — 3D image segmentation of the tumor.

Implementation pending. The module will:

1. Inspect the last PET frame and locate the tumor's approximate centroid and
   bounding box manually.
2. Run a general-purpose AI segmentation model on the MR volume seeded with
   that bounding box / centroid / textual prompt. Candidates under evaluation:

   - MedSAM2 (medical adaptation of SAM 2, native 3D propagation).
   - nnInteractive (interactive segmentation built on top of nnU-Net).
   - SAMed-2 (medical fine-tune of Segment Anything 2).
   - SAT (Segment Anything in 3D Medical Images).

3. Produce a tumor mask, render it overlaid on the input image, and quantify
   the segmentation against the reference visualization (volume statistics,
   shape descriptors, qualitative comparison).
"""
