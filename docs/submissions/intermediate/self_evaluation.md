---
title: "Self-evaluation form — Intermediate submission"
author: "Alejandro Bordon — Medical Image Processing (11763)"
date: "May 4, 2026"
geometry: margin=2.2cm
fontsize: 11pt
---

# Self-evaluation form

Mark each completed item and the numerical self-qualification.

> **A note before the form.** I went further than I had planned for this
> intermediate. Objective 1 is done from load all the way to the rotating
> MIP. Objective 2 runs end-to-end in physical coordinates with the
> inverse transform and the mask propagation already in place. Objective
> 3 has a working scaffold around a classical region-growing baseline,
> with the AI general-purpose model (MedSAM2, nnInteractive, SAMed-2 or
> SAT) as the next step. Project completion sits around 60-65%.

\vspace{0.3cm}

## Objective 1 — DICOM loading and visualization \hfill **9 / 10**

**Load PET dynamic study and MR T1 reference**

- [x] Both images are loaded with PyDicom, and their corresponding headers have been studied.
- [x] The dynamic PET pixel array is correctly reorganized into a 4D volume from the relevant DICOM headers.
- [x] The MR T1 study is loaded as a 3D reference volume.
- [x] The last frame and the temporal average of the PET study are visualized.

The PET array is reshaped into `(36 frames, 47 slices, 256, 256)`. The
reshape order was confirmed both mathematically (every z position in
`FramePositionsVector` repeats once per frame) and visually by reloading
the temporal mean back into 3D Slicer through the MCP bridge.

**Rotating MIP**

- [x] At least one Maximum Intensity Projection has been created.
- [x] The image and the regions are both clearly identifiable: colormaps have been correctly used, alpha fusion is used.
- [x] An interactive animation (GIF) with at least 16 projections has been shown.

The 24-frame rotating MIP animation lives at
`docs/figures/06_rotating_mip.gif`. It shows the MR alone, the
coregistered PET alone, and the alpha-fused overlay turning around the
axial axis.

\vspace{0.3cm}

## Objective 2 — 3D rigid coregistration \hfill **8 / 10**

**Image coregistration**

- [x] A rigid motion has been implemented.
- [x] The initial parameters are adequate.
- [x] A loss function has been implemented.
- [x] An optimizer has been successfully used to find the optimal parameters of a rigid motion.
- [x] The correctness of the coregistration has been verified with visualizations.

The pipeline now works in physical coordinates. PET is resampled onto
the MR voxel grid via DICOM affines, and Mutual Information goes from
`0.068` at identity to `0.131` after Powell optimisation (+93%). The
optimisation runs on a 2× downsampled grid for speed; the final transform
is applied at full resolution.

**Mask and assessment**

- [x] The mask has been transformed into the input space.
- [x] The inverse transformation has been explicitly found.
- [x] Both the input image and the transformed mask have been visualized together.
- [x] Numerical values have been implemented to measure the correctness of the coregistration process.

The inverse rigid parameters are computed analytically from the forward
ones (`apply_rigid_in_volume` plus axis-angle inversion) and stored in
`coreg_transform.npz`. The Objective 3 mask is propagated MR → PET
native through the chain *inverse rigid · inverse DICOM resample* and
shown overlaid on the PET last frame in
`docs/figures/09_segmentation_mask_on_pet_native.png`.

\vspace{0.3cm}

## Objective 3 — 3D image segmentation \hfill **4 / 10**

The segmentation pipeline runs end-to-end with a classical baseline,
but the AI general-purpose model named in the proposal is still pending.

**Segmentation**

- [x] The centroid and bounding box of the tumor have been calculated.
- [x] A segmentation algorithm has been implemented, and it uses either the centroid or the bounding box.
- [x] The segmentation algorithm works on volumetric 3D images, rather than on single slices.
- [ ] The segmentation algorithm extracts the tumoral region up to its borders.

The candidate centroid is auto-located inside the PET-MR Z overlap and
mapped to MR voxel coordinates. A 3D `skimage.morphology.flood`
region-growing then runs on the MR, constrained to the bounding box.
The current mask volume (around 36 k voxels in MR, 8 k after propagation back
to PET native) is too generous compared to a real tumor — that's why
the AI model is the priority next step.

**Assessment**

- [x] Both the input image and the automatically segmented mask have been visualized together.
- [ ] The provided and automatically segmented masks have been visualized together, and can be easily compared.
- [ ] Numerical values have been implemented to measure the correctness of the automatic segmentation.

A reference mask isn't available yet, so Dice / Jaccard / Hausdorff are
in the open questions. The visual overlay on the PET last frame stands
in for the qualitative comparison for now.

\vspace{0.3cm}

## Submission \hfill **5 / 10**

The repository now carries five Python modules (`loading`,
`coregistration`, `visualization`, `segmentation`, `utils`), nine figures
in `docs/figures/`, the saved transform, and the binary masks. The README
walks through structure, environment, and usage. The 5-page final
document is still ahead.

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
| Objective 1      | 25 %   | 9 / 10  |
| Objective 2      | 25 %   | 8 / 10  |
| Objective 3      | 25 %   | 4 / 10  |
| Submission       | 25 %   | 5 / 10  |
| **Total**        | 100 %  | **6.5 / 10** |

The number is meant to reflect actual progress (~65% of the project) and
flag the remaining work: integrating the AI segmentation model on the
MR side, computing proper similarity / overlap metrics, and writing the
final 5-page document.
