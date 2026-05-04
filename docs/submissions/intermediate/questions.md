---
title: "BrainScope — Open questions for the intermediate submission"
author: "Alejandro Bordon — Medical Image Processing (11763)"
date: "May 4, 2026"
geometry: margin=2.2cm
fontsize: 11pt
---

# Open questions

The questions below collect the decision points and difficulties that would
benefit from feedback before the corresponding objective is implemented. They
are grouped by stage and ordered roughly by their impact on the final grade.

## Objective 1 — DICOM loading and visualization

1. **Slice / frame ordering of the dynamic PET.** The proposal lists
   `(0055, 1002)` Frame Positions Vector as the source for the slice axis,
   but the same volume also exposes `ImagePositionPatient` per slice in
   regular tags. When both are present, which one should be considered the
   ground truth for the `(frame, slice)` reshape? Are there scanners where
   they disagree?

2. **Temporal average vs. last frame.** Objective 1 asks for both, but the
   coregistration in Objective 2 is only required on the temporal average.
   Is there any reason to prefer a different aggregation (e.g., median,
   weighted average using `Frame Durations`) over the simple mean?

3. **Visualization standards.** The proposal asks for a GIF that sweeps the
   three median planes across temporal frames. Do you expect us to lock the
   intensity scale across frames (`vmin/vmax` from the whole 4D volume), or
   to normalize each frame independently?

## Objective 2 — 3D rigid coregistration

4. **Custom implementation vs. PyElastix.** The proposal explicitly allows
   either path. Is implementing the rigid coregistration from scratch
   (translation + axial rotation through `least_squares`) preferred for
   grading purposes, or is using PyElastix considered equivalent if the
   results are equivalent?

5. **Choice of similarity metric.** The course activities use mutual
   information for cross-modality MAE/MSE comparisons. For PET vs. MR, mutual
   information seems the natural metric, but normalized cross-correlation or
   normalized mutual information might be more robust on this pair. Is
   there a recommended default for grading?

6. **Initial parameters.** Aligning centroids gives a sensible translation
   but no rotation prior. Would you expect us to coarsely align the
   principal axes too (e.g., via PCA of the foreground mask) before running
   the optimizer, or is a centroid-only initialization considered enough?

7. **Validation of the coregistration.** Beyond visual MIP overlays, what
   numerical figure of merit do you expect us to report? Final loss value,
   residual landmark distances, mutual information before / after, or a
   combination?

## Objective 3 — 3D image segmentation

8. **Model recommendation.** The proposal lists nnInteractive, SAMed-2, SAT,
   and MedSAM2 as candidates. For brain tumors visible on dynamic PET and
   MR T1, which would you recommend by default? MedSAM2 has good 3D
   propagation, nnInteractive is more medical-native, but the local
   environment (single GPU, limited VRAM) might favor one over the others.

9. **Modality of the segmentation.** Should the AI model be run on the MR
   volume, on the coregistered PET, or on a fused channel? The mask only
   has to be visualized in one space at the end, but the model's input
   choice changes the difficulty considerably.

10. **Reference mask.** Is a ground-truth tumor mask available for this
    study, even if approximate, that we can use to compute Dice / Jaccard /
    Hausdorff metrics? If not, is reporting volume, sphericity, and visual
    comparison considered enough for the assessment checkbox?

11. **Bounding box workflow.** For the manual bounding box step, would you
    accept a hard-coded numerical box derived once from 3D Slicer
    inspection, or do you expect an interactive widget within the report
    notebook?

## General / submission

12. **Notebook vs. scripts.** Course activities mix `if __name__ == '__main__'`
    scripts with figure generation. For the final submission, would you
    prefer a single Jupyter notebook that orchestrates all three objectives,
    or modular scripts referenced from the 5-page summary?

13. **Reproducibility data.** The DICOM data is large and excluded from the
    repository. Is a `data/README.md` with download instructions considered
    enough for the "publicly accessible" criterion, or should we host a
    sample subset in the repo?

14. **3D Slicer integration.** We are configuring a 3D Slicer MCP bridge to
    automate qualitative inspection. Is this considered acceptable evidence
    for the "third-party DICOM visualizer" requirement of Objective 1.b, or
    do you expect direct screenshots from the GUI?

15. **Final document length.** The 5-page final summary excludes figures and
    code. Does the index page count toward the limit? And does each animation
    contribute one figure (a representative frame) or zero (since GIFs cannot
    be inlined)?
