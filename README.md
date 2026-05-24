# BrainScope

BrainScope is a Python pipeline for dynamic PET / MR T1 brain studies. It loads and reshapes the DICOM stack, coregisters PET to MR via Normalised Mutual Information, and segments the tumor with TotalSegmentator and MedSAM2.

**Repository:** <https://github.com/AlejandroB10/BrainScope>

Course: **Medical Image Processing (11763)** — Master in Intelligent Systems, UIB.

---

## Quick start

Requirements: conda, CUDA-capable GPU (tested on RTX 4060 Mobile, 8 GB VRAM).
Python **3.11** (3.12+ is not supported — MedSAM2 and TotalSegmentator pin 3.11).

```bash
git clone https://github.com/AlejandroB10/BrainScope.git
cd BrainScope
conda env create -f environment.yml
conda activate medical-image-processing-11763

# MedSAM2 must be installed as an editable local clone — it is NOT in environment.yml
git clone https://github.com/bowang-lab/MedSAM2.git external/MedSAM2
pip install -e external/MedSAM2/

# Download model weights
totalseg_download_weights -t total_mr
huggingface-cli download wanglab/MedSAM2 --local-dir ~/.cache/medsam2/ MedSAM2_latest.pt

# Confirm everything is wired up
pytest -q tests/
```

> **CPU fallback.** Both TotalSegmentator and MedSAM2 support `--device cpu` when no
> GPU is available, at the cost of longer runtimes (10-20× slower).

---

## Run the pipeline

Each objective has an entry point in `src/`. Run them in order from the repository root:

```bash
# Objective 1 — DICOM loading and visualization
conda run -n medical-image-processing-11763 \
    python -m src.loading

# Objective 2 — 3D rigid coregistration
conda run -n medical-image-processing-11763 \
    python -m src.coregistration

# Objective 3 — Brain mask (TotalSegmentator)
conda run -n medical-image-processing-11763 \
    python -m src.brain_mask

# Objective 3 — Tumor segmentation (MedSAM2 or region-growing fallback)
conda run -n medical-image-processing-11763 \
    python -m src.segmentation
```

DICOM inputs live under `data/` (gitignored — see `data/README.md` for the dataset).
Figures are written to `docs/figures/`; masks and GIFs go to `results/`.

---

## Test

```bash
conda run -n medical-image-processing-11763 pytest -q tests/
# Expected: 58 passed
```

The suite covers DICOM loading geometry, coregistration smoke, brain mask API,
segmentation unit tests, and metrics.

---

## Project structure

```
BrainScope/
├── src/
│   ├── loading.py          # Objective 1: DICOM loading & 4D reshape
│   ├── coregistration.py   # Objective 2: rigid PET ↔ MR coregistration
│   ├── segmentation.py     # Objective 3: tumor segmentation (MedSAM2 + fallback)
│   ├── brain_mask.py       # TotalSegmentator brain parenchyma mask
│   ├── metrics.py          # V-1 / V-2 validation metrics
│   ├── visualization.py    # plotting and GIF helpers
│   └── utils.py            # shared helpers
├── scripts/
│   ├── grab_bbox_from_slicer.py   # capture tumor bbox from 3D Slicer via MCP
│   ├── extract_gif_frames.py      # extract static PNG frames from animated GIFs
│   └── run_validation.py          # run V-1 / V-2 validation and write JSON
├── tests/                  # pytest suite (58 tests)
├── data/                   # DICOM inputs — gitignored
├── docs/
│   ├── figures/            # PNG and GIF figures referenced in the document
│   │   └── static/         # static PNG frames extracted from GIFs (PDF-safe)
│   └── submissions/
│       ├── intermediate/   # intermediate submission (May 4, 2026 snapshot)
│       └── final/          # final technical document and slide deck
├── results/                # masks, transforms, validation JSON — gitignored
├── external/               # third-party clones (MedSAM2) — not committed
├── environment.yml         # conda environment definition
└── README.md
```

---

## Reproduce the deliverables

The final technical document and slide deck are authored in Markdown and rendered via
`pandoc + xelatex`. Run from the repository root:

```bash
cd docs/submissions/final
mkdir -p build

# 5-page technical document
pandoc technical_document.md \
    -o build/technical_document.pdf \
    --pdf-engine=xelatex \
    --toc --number-sections \
    -V colorlinks=true

# 14-slide Beamer deck
pandoc slides.md \
    -t beamer \
    --pdf-engine=xelatex \
    -o build/slides.pdf
```

Both PDFs are committed under `docs/submissions/final/build/`.

---

## Hardware

Tested on a laptop with an NVIDIA RTX 4060 Mobile (8 GB VRAM) running CUDA 13.0
(driver backward-compatible with `cu124` wheels).

Peak VRAM usage (sequential, never simultaneous):

| Step | VRAM |
|------|------|
| TotalSegmentator `total_mr` | ~2–4 GB |
| MedSAM2 video propagation | ~2–3 GB |

Both models are unloaded after inference (`del model; torch.cuda.empty_cache()`), so
they can run sequentially on an 8 GB card without OOM.

---

## License

Code: Apache 2.0.
Documentation and figures: CC-BY-SA 4.0.

MedSAM2 is developed by the Wang Lab (Bowang Lab); TotalSegmentator is developed by
the University Hospital Basel. Both are used here under their respective open-source
licenses.
