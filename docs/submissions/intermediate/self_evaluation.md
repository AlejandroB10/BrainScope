---
title: "Self-evaluation form — Intermediate submission"
author: "Alejandro Bordon — Medical Image Processing (11763)"
date: "May 4, 2026"
geometry: margin=2.2cm
fontsize: 11pt
---

# Self-evaluation form

Mark each completed item and the numerical self-qualification.

> **A note before the form.** This is the intermediate submission, but I
> ended up implementing more than I had originally planned for it. Most of
> Objective 1 is in place (load, reshape, static views, animated GIF) and
> Objective 2 has a working rigid-coregistration pipeline (translation +
> axial rotation, Mutual Information loss, Powell optimizer, before/after
> visual overlay). Objective 3 hasn't started yet. The scoring below tries
> to be conservative — I'd rather under-report than over-claim while still
> waiting for feedback.

\vspace{0.3cm}

## Objective 1 — DICOM loading and visualization \hfill **7 / 10**

**Load PET dynamic study and MR T1 reference**

- [x] Both images are loaded with PyDicom, and their corresponding headers have been studied.
- [x] The dynamic PET pixel array is correctly reorganized into a 4D volume from the relevant DICOM headers.
- [x] The MR T1 study is loaded as a 3D reference volume.
- [x] The last frame and the temporal average of the PET study are visualized.

The PET array is reshaped into `(36 frames, 47 slices, 256, 256)`. The
reshape order was confirmed both mathematically (every z position in
`FramePositionsVector` repeats once per frame) and visually (loading the
temporal mean back into 3D Slicer through the MCP bridge produces a
recognisable brain volume).

**Rotating MIP**

- [x] At least one Maximum Intensity Projection has been created.
- [x] The image and the regions are both clearly identifiable: colormaps have been correctly used, alpha fusion is used.
- [ ] An interactive animation (GIF) with at least 16 projections has been shown.

I do save a 36-frame median-planes animation; the *rotating* MIP itself is
left for the next iteration.

\vspace{0.3cm}

## Objective 2 — 3D rigid coregistration \hfill **4 / 10**

**Image coregistration**

- [x] A rigid motion has been implemented.
- [x] The initial parameters are adequate.
- [x] A loss function has been implemented.
- [x] An optimizer has been successfully used to find the optimal parameters of a rigid motion.
- [x] The correctness of the coregistration has been verified with visualizations.

The pipeline runs on a downsampled `(48, 64, 64)` grid for speed (a
limitation I want to lift before the final submission). Mutual Information
goes from `0.408` at identity to `0.526` after Powell optimisation, with a
clearly tighter PET / MR overlap on the saved before/after MIP overlay.

**Mask and assessment**

- [ ] The mask has been transformed into the input space.
- [ ] The inverse transformation has been explicitly found.
- [ ] Both the input image and the transformed mask have been visualized together.
- [ ] Numerical values have been implemented to measure the correctness of the coregistration process.

Mask propagation depends on Objective 3, so it's parked for now.

\vspace{0.3cm}

## Objective 3 — 3D image segmentation \hfill **0 / 10**

Not started yet. The progress document includes specific questions about
which model to pick (MedSAM2 / nnInteractive / SAMed-2 / SAT) and on which
modality to run it.

**Segmentation**

- [ ] The centroid and bounding box of the tumor have been calculated.
- [ ] A segmentation algorithm has been implemented, and it uses either the centroid or the bounding box.
- [ ] The segmentation algorithm works on volumetric 3D images, rather than on single slices.
- [ ] The segmentation algorithm extracts the tumoral region up to its borders.

**Assessment**

- [ ] Both the input image and the automatically segmented mask have been visualized together.
- [ ] The provided and automatically segmented masks have been visualized together, and can be easily compared.
- [ ] Numerical values have been implemented to measure the correctness of the automatic segmentation.

\vspace{0.3cm}

## Submission \hfill **3 / 10**

The repo carries actual content now (runnable scripts, generated figures,
the GIF, the registration before/after) on top of the README and the
environment file. The final 5-page document is still ahead of us.

**Document**

- [ ] Written expression is correct and accurate.
- [ ] Covers all the objectives.
- [ ] Shows figures of images / ROIs when necessary.
- [ ] Includes discussions on why certain approaches were preferred over others.
- [ ] Includes a relevant discussion of the findings and shortcomings of the project.

**Code**

- [x] Is publicly accessible — `https://github.com/AlejandroB10/BrainScope`.
- [x] Contains a README and is easy to follow.

\vspace{0.4cm}

## Aggregate self-assessment

| Section          | Weight | Score   |
|------------------|--------|---------|
| Objective 1      | 25 %   | 7 / 10  |
| Objective 2      | 25 %   | 4 / 10  |
| Objective 3      | 25 %   | 0 / 10  |
| Submission       | 25 %   | 3 / 10  |
| **Total**        | 100 %  | **3.5 / 10** |

The number is intentionally moderate. Objective 3 and the clean
physical-coordinate version of the registration are the parts that
actually carry the grade, and they're still ahead. The whole point of
this submission is the feedback, so it makes sense to leave the score
on the cautious side.
