# BrainScope

DICOM processing toolkit for dynamic PET/MR brain studies with semi-automatic tumor segmentation.

Course: **Medical Image Processing (11763)** — Master in Intelligent Systems, UIB.

## Overview

BrainScope implements a three-stage pipeline that operates on a co-acquired
brain study composed of a dynamic PET acquisition (`e_1_BRAIN_DINAMIC_COLINA`) and
a high-resolution MR T1 reference (`AX_3D_T1`):

1. **DICOM loading and visualization** — Parse the dynamic PET pixel array,
   reorder it according to the relevant DICOM headers, and produce both static
   and animated visualizations of its three median planes.
2. **3D rigid coregistration** — Align the temporally averaged PET volume to the
   MR reference through a rigid transform optimized with Mutual Information.
3. **3D image segmentation** — Delineate the tumor visible in the last PET frame
   by combining a manual bounding box with a general-purpose AI segmentation
   model (MedSAM2 / nnInteractive / SAMed-2 / SAT under evaluation).

## Project structure

```
BrainScope/
├── src/                      # Python source code
│   ├── loading.py            # Objective 1: DICOM loading & visualization
│   ├── coregistration.py     # Objective 2: rigid PET↔MR coregistration
│   ├── segmentation.py       # Objective 3: tumor segmentation
│   └── utils.py              # Shared helpers
├── data/                     # DICOM inputs (gitignored, see data/README.md)
├── docs/
│   ├── figures/              # Figures produced for the report
│   └── submissions/          # Intermediate and final deliverables
├── results/                  # Generated images, GIFs, masks (gitignored)
├── environment.yml           # Conda environment definition
└── README.md
```

## Environment

This project reuses the conda environment used throughout the course
(`medical-image-processing-11763`). Recreate it from scratch with:

```bash
conda env create -f environment.yml
conda activate medical-image-processing-11763
```

Python 3.11. Key dependencies: `pydicom`, `numpy`, `scipy`, `matplotlib`,
`scikit-image`, `SimpleITK`.

## Data

The dataset is not tracked in this repository. See `data/README.md` for
download instructions and the DICOM headers that drive the pixel-array
reordering.

## Status

This repository corresponds to the **intermediate submission** of the project.
The current state, planned approach, and open questions are documented at
`docs/submissions/`.
