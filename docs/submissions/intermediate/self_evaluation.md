---
title: "Self-evaluation form — Intermediate submission"
author: "Alejandro Bordon — Medical Image Processing (11763)"
date: "May 4, 2026"
geometry: margin=2.2cm
fontsize: 11pt
---

# Self-evaluation form

Mark each completed item and the numerical self-qualification.

> **Context.** This document corresponds to the **intermediate submission**.
> No objective has been implemented at the date of this delivery. The current
> work has focused exclusively on planning, environment setup, repository
> bootstrap, and study of the reference course activities. Every checkbox is
> therefore left unmarked, and every section is reported as **0 / 10**.
> Detailed information about the work performed so far is provided in the
> companion progress report.

\vspace{0.3cm}

## Objective 1 — DICOM loading and visualization \hfill **0 / 10**

**Load PET dynamic study and MR T1 reference**

- [ ] Both images are loaded with PyDicom, and their corresponding headers have been studied.
- [ ] The dynamic PET pixel array is correctly reorganized into a 4D volume from the relevant DICOM headers.
- [ ] The MR T1 study is loaded as a 3D reference volume.
- [ ] The last frame and the temporal average of the PET study are visualized.

**Rotating MIP**

- [ ] At least one Maximum Intensity Projection has been created.
- [ ] The image and the regions are both clearly identifiable: colormaps have been correctly used, alpha fusion is used.
- [ ] An interactive animation (GIF) with at least 16 projections has been shown.

\vspace{0.3cm}

## Objective 2 — 3D rigid coregistration \hfill **0 / 10**

**Image coregistration**

- [ ] A rigid motion has been implemented.
- [ ] The initial parameters are adequate.
- [ ] A loss function has been implemented.
- [ ] An optimizer has been successfully used to find the optimal parameters of a rigid motion.
- [ ] The correctness of the coregistration has been verified with visualizations.

**Mask and assessment**

- [ ] The mask has been transformed into the input space.
- [ ] The inverse transformation has been explicitly found.
- [ ] Both the input image and the transformed mask have been visualized together.
- [ ] Numerical values have been implemented to measure the correctness of the coregistration process.

\vspace{0.3cm}

## Objective 3 — 3D image segmentation \hfill **0 / 10**

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

## Submission \hfill **0 / 10**

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
| Objective 1      | 25 %   | 0 / 10  |
| Objective 2      | 25 %   | 0 / 10  |
| Objective 3      | 25 %   | 0 / 10  |
| Submission       | 25 %   | 0 / 10  |
| **Total**        | 100 %  | **0 / 10** |
