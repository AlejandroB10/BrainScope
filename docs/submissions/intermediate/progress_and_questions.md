---
title: "BrainScope — Intermediate progress report and open questions"
author: "Alejandro Bordon — Medical Image Processing (11763)"
date: "May 4, 2026"
geometry: margin=1.8cm
fontsize: 10pt
---

**Repository.** `https://github.com/AlejandroB10/BrainScope`
(public; you can read the README and browse the layout from there).

# 1. What the project is about

The plan is to implement the three objectives of the proposal on a single
brain study made of a dynamic PET (`e_1_BRAIN_DINAMIC_COLINA`) and a MR
T1 (`AX_3D_T1`): load and visualize the DICOMs, rigidly coregister the
temporally averaged PET onto the MR, and segment the tumor visible on the
last PET frame with one of the AI models suggested in the proposal. The
architecture stays small on purpose: one Python module per objective, the
course conda environment `medical-image-processing-11763`, and a bridge
to 3D Slicer through the `mcp-slicer` MCP server for live qualitative
inspection.

# 2. Where I am right now

I ended up pushing further than the intermediate strictly required.
Objective 1 is fully covered, Objective 2 runs end-to-end in physical
coordinates with the inverse transform and the mask propagation already
wired, and Objective 3 has a classical baseline scaffold with the AI
swap-in as the next step. Total project completion sits around 60-65%.

**Objective 1.** `src/loading.py` reads both DICOMs, reshapes the flat
`(1692, 256, 256)` `int16` PET array into `(36, 47, 256, 256)`, computes
the temporal mean and the last frame, and saves median-plane composites
plus a 36-frame GIF that sweeps the three median planes across time.
The reshape order was confirmed against `FramePositionsVector` and
validated visually by reloading the result into 3D Slicer through the
MCP bridge (Figure 1).

![PET temporal mean reloaded into 3D Slicer through the MCP bridge. The volume is anatomically coherent.](../../figures/01_pet_temporal_mean_in_slicer.png){width=55%}

The Objective 2.b rotating MIP has also landed:
`src/visualization.py` builds a 24-frame turntable showing the MR alone,
the coregistered PET alone and the alpha fusion of both
(`docs/figures/06_rotating_mip.gif`).

**Objective 2.** `src/coregistration.py` is now fully physical-coordinate
aware. It builds 4×4 affines from `ImagePositionPatient`,
`ImageOrientationPatient`, `PixelSpacing` and `SpacingBetweenSlices`,
resamples PET onto the MR voxel grid through those affines, and only
then runs the rigid optimisation. Mutual Information climbs from
`0.068` at identity to `0.131` after Powell (+93%). The inverse rigid
parameters are derived analytically from the forward ones and saved next
to the figures, ready to push masks back to PET space (Figure 2).

![Rigid coregistration before (top) vs after (bottom) on the MR voxel grid.](../../figures/05_coreg_before_after.png){width=55%}

**Objective 3.** `src/segmentation.py` locates a candidate centroid
inside the PET-MR Z overlap, builds a bounding box around it, maps both
to MR voxel coordinates through the DICOM affines, and runs a 3D
`skimage.morphology.flood` region growing constrained to that box. The
mask is then propagated back to the PET native grid through the chain
*inverse rigid · inverse DICOM resample*, closing the loop the proposal
asks for. Region growing is a placeholder until the AI general-purpose
model is integrated; the interface already takes the same prompt
(centroid + bbox), so the swap is mechanical.

**Still missing.** The AI general-purpose segmentation model
(MedSAM2 / nnInteractive / SAMed-2 / SAT) is the open item — see
question 9. There's no ground-truth mask yet, so Dice / Jaccard /
Hausdorff aren't computed. The 5-page final document hasn't been
written.

# 3. How I'm planning to attack the rest

The biggest open item is the AI general-purpose segmentation. I'd like
to land MedSAM2 on the MR volume prompted by the bounding box already in
place, then evaluate the mask against whatever ground truth is
available. Once that's running I'll close Objective 2's last detail (a
target registration error from a few Slicer-placed fiducials, on top of
the Mutual Information already reported). Then the 5-page final document
ties everything together.

# 4. Risks I'm watching

| Risk                                                          | Mitigation                                                              |
|---------------------------------------------------------------|-------------------------------------------------------------------------|
| AI segmentation model too heavy for the local hardware        | Pre-screen MedSAM2 / nnInteractive sizes; keep region growing as fallback |
| Mutual Information stuck at a local maximum                   | Multi-start initialisation; PyElastix as a sanity check                   |
| No ground-truth mask, segmentation hard to assess numerically | Use volume / sphericity as proxies; ask for a reference mask if available |
| Final document writing time                                   | Outline now, write while the model integration is running                 |

The list of open questions on the next pages is the input I need from
you to lock those decisions before going further.

\newpage

# Open questions

These are the points where I'd like input before committing to an approach.
Grouped by stage, ordered roughly by impact on the final grade.

## Objective 1 — DICOM loading and visualization

1. **Slice ordering of the dynamic PET.** I'm using
   `(0055, 1002)` Frame Positions Vector, but the file also exposes
   `ImagePositionPatient` per slice. I confirmed against
   `FramePositionsVector` that every z position appears once per frame.
   In your experience, are there scanners where Frame Positions Vector and
   `ImagePositionPatient` disagree, and would you trust either one as the
   ground truth?

2. **Temporal aggregate for the coregistration input.** Right now I'm
   using a plain unweighted mean across the 36 frames. Frame durations
   range from 5 s to 300 s, so a duration-weighted mean would emphasise
   the late frames. Is the unweighted mean fine, or do you prefer the
   weighted one?

3. **Intensity scale for the GIF.** I lock `vmax` at the 99.5 percentile
   of the whole 4D volume and reuse it across frames so the early-frame
   wash-in vs. late-frame plateau remains visible. Does that match what
   you usually expect?

## Objective 2 — 3D rigid coregistration

4. **Custom vs. PyElastix.** My implementation is from scratch (rigid +
   MI + Powell). I plan to keep it that way and only use PyElastix as a
   sanity check. Does that match the spirit of the assignment?

5. **Loss function.** I'm on Mutual Information with a 32-bin joint
   histogram. Would you push for normalised MI or normalised cross
   correlation on PET vs. MR specifically, or is plain MI good enough?

6. **Initial rotation prior.** I currently start from the identity
   (zero translation, zero rotation). For a same-session study this seems
   safe, but a PCA-based principal-axes alignment might bring the
   optimiser closer to the global maximum. Worth doing, or overkill?

7. **What to report numerically.** Right now I report MI before / after.
   What's the figure of merit you'd like to see in the final report?
   Final MI, residual landmark distances, target registration error from
   Slicer-placed fiducials, or a combination?

8. **Optimisation grid resolution.** The pipeline now resamples PET
   onto the MR grid in physical coordinates, but the rigid optimisation
   itself still runs on a 2× downsampled version of both volumes for
   speed. Is that downsampling acceptable in the final report, or do you
   want the optimisation at full resolution even if it costs minutes
   per run?

## Objective 3 — 3D image segmentation

9. **Model recommendation.** Among nnInteractive, SAMed-2, SAT, and
   MedSAM2, which one would you pick as the default for brain tumors on
   PET / MR T1? Hardware-wise I have one consumer-grade GPU, which I
   think rules out the heaviest configurations.

10. **Modality the model sees.** Do you expect the model to run on the
    MR volume, on the coregistered PET, or on a fused channel? The
    decision really changes how I structure Objective 3.

11. **Reference mask.** Is there a ground-truth tumor mask available for
    this study? If yes, I'd compute Dice / Jaccard / Hausdorff. If not,
    I'll fall back to volume / sphericity / qualitative comparison and
    document the limitation.

12. **Bounding box workflow.** For the manual bounding box step, is a
    hard-coded tuple (taken once from a 3D Slicer inspection) acceptable,
    or do you want an interactive widget inside the report notebook?

## General / submission

13. **Notebook vs. scripts.** Activities use `__main__` scripts. For the
    final 5-page summary, do you prefer a single Jupyter notebook that
    orchestrates everything, or modular scripts referenced from the doc?

14. **Reproducibility data.** The DICOMs are out of git because of size.
    Is a `data/README.md` with download instructions enough for the
    "publicly accessible" criterion, or should I host a sample subset?

15. **3D Slicer evidence.** I'm wiring up an MCP bridge to drive Slicer
    programmatically (already used to validate the PET reshape, see
    Figure 1). Does that count as the "third-party DICOM visualizer"
    Objective 1.b asks for, or do you expect manual GUI screenshots in
    the final report too?
