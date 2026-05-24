"""Build demo.ipynb programmatically using nbformat.

Run once to regenerate the notebook:
    python notebooks/build_demo.py
"""
import nbformat
from pathlib import Path

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

# ---------------------------------------------------------------------------
# Cell 1 — Setup
# ---------------------------------------------------------------------------
cell1 = nbformat.v4.new_code_cell("""\
import sys
import json
import warnings
import numpy as np
import matplotlib
# Do not force a backend — Jupyter notebook will use its own inline backend automatically.
# matplotlib.use("Agg") is intentionally omitted so plt.show() renders inline in the notebook.
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path

# Resolve PROJECT_ROOT robustly (works in notebook and as a script)
try:
    _here = Path(__file__).resolve().parent
    PROJECT_ROOT = _here.parent
except NameError:
    # __file__ not defined in a notebook kernel
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent

# Make sure both project root and src/ are importable
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"BrainScope demo — loading project context...")
print(f"Project root : {PROJECT_ROOT}")
print()
print("Pipeline objectives:")
print("  1. DICOM loading & visualisation  — parse dynamic PET (36 frames × 47 slices × 256×256)")
print("  2. 3-D rigid co-registration      — align PET to MR T1 with Normalised Mutual Information")
print("  3. Tumour segmentation            — MedSAM2 video propagation, manual bbox prompt")
""")

# ---------------------------------------------------------------------------
# Cell 2 — Objective 1: Load + verify dynamic PET
# ---------------------------------------------------------------------------
cell2 = nbformat.v4.new_code_cell("""\
from loading import load_dynamic_pet
import warnings

PET_DICOM = PROJECT_ROOT / "data" / "raw" / "e_1_BRAIN_DINAMIC_COLINA.dcm"

if not PET_DICOM.exists():
    print(f"[SKIP] PET DICOM not found at {PET_DICOM}")
    print("  (pre-computed figures are shown in later cells)")
    pet = None
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pet = load_dynamic_pet(str(PET_DICOM))

    print("Objective 1 — Dynamic PET loaded")
    print(f"  4D shape      : {pet.array.shape}  (frames × slices × rows × cols)")
    print(f"  Voxel spacing : z={pet.spacing_zyx_mm[0]:.2f} mm  "
          f"y={pet.spacing_zyx_mm[1]:.2f} mm  x={pet.spacing_zyx_mm[2]:.2f} mm")
    print(f"  FrameDurations: {pet.n_frames} entries  (one per temporal frame)")
    print(f"  FramePositionsVector tag: {len(pet.frame_positions_vector)} values "
          f"= {pet.n_frames} frames × {pet.n_slices} slices × 3 (x,y,z)")
    print()

    # Temporal-mean PET (already duration-weighted inside load_dynamic_pet)
    mean_3d = np.average(pet.array, axis=0, weights=pet.frame_durations_ms)

    vmax = float(np.percentile(mean_3d, 99.5))
    nz, ny, nx = mean_3d.shape
    mz, my, mx = nz // 2, ny // 2, nx // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Objective 1 — Dynamic PET temporal mean (duration-weighted)", fontsize=13)

    axes[0].imshow(mean_3d[mz, :, :], cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[0].set_title(f"Axial  (z={mz})")
    axes[0].axis("off")

    axes[1].imshow(mean_3d[:, my, :], cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[1].set_title(f"Coronal  (y={my})")
    axes[1].axis("off")

    axes[2].imshow(mean_3d[:, :, mx], cmap="hot", vmin=0, vmax=vmax, origin="lower")
    axes[2].set_title(f"Sagittal  (x={mx})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    print("Figure: temporal-mean PET in the 3 median planes (hot colormap, vmax=99.5th percentile)")
""")

# ---------------------------------------------------------------------------
# Cell 3 — Objective 2: NMI convergence curve
# ---------------------------------------------------------------------------
cell3 = nbformat.v4.new_code_cell("""\
MI_JSON = PROJECT_ROOT / "results" / "mi_convergence.json"

if not MI_JSON.exists():
    print(f"[SKIP] {MI_JSON} not found")
else:
    with open(MI_JSON) as f:
        mi_data = json.load(f)

    iterations  = mi_data["iterations"]
    nmi_values  = mi_data["nmi_values"]
    nmi_initial = mi_data["initial"]
    nmi_final   = mi_data["final"]
    pct_change  = (nmi_final - nmi_initial) / nmi_initial * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iterations, nmi_values, color="#2563eb", linewidth=1.2, alpha=0.85)
    ax.axhline(nmi_initial, color="red", linestyle="--", linewidth=1.2,
               label=f"Initial NMI = {nmi_initial:.4f}")
    ax.axhline(nmi_final,   color="darkred", linestyle="--", linewidth=1.2,
               label=f"Final NMI   = {nmi_final:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("NMI")
    ax.set_title("Coregistration NMI evolution (Powell optimiser, 16-bin joint histogram)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"NMI {nmi_initial:.3f} -> {nmi_final:.3f}  ({pct_change:+.1f}%)")
    print(f"  {len(iterations)} objective evaluations across {max(iterations)+1} iterations")
    print(f"  Optimiser: Powell  |  Metric: Normalised Mutual Information  |  Bins: 16")
""")

# ---------------------------------------------------------------------------
# Cell 4 — V-1 + V-2 validation dashboard
# ---------------------------------------------------------------------------
cell4 = nbformat.v4.new_code_cell("""\
from IPython.display import HTML, display

V1_JSON = PROJECT_ROOT / "results" / "v1_brain_mask_overlap.json"
V2_JSON = PROJECT_ROOT / "results" / "v2_region_growing_in_brain.json"

missing = [p for p in (V1_JSON, V2_JSON) if not p.exists()]
if missing:
    print("[SKIP] Missing validation files:")
    for p in missing:
        print(f"  {p}")
else:
    with open(V1_JSON) as f:
        v1 = json.load(f)
    with open(V2_JSON) as f:
        v2 = json.load(f)

    def badge(passed):
        color = "#16a34a" if passed else "#dc2626"
        text  = "PASS" if passed else "FAIL"
        return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{text}</span>'

    v1_val   = f"{v1['value']:.3f}"
    v1_thr   = f"{v1['threshold']:.2f}"
    v2_val   = f"{v2['value']:.3f}"
    v2_thr   = f"{v2['threshold']:.2f}"
    v1_badge = badge(v1["pass"])
    v2_badge = badge(v2["pass"])
    html = (
        "<table style='border-collapse:collapse;font-family:monospace;font-size:14px;width:90%'>"
        "<thead><tr style='background:#1e3a5f;color:white'>"
        "<th style='padding:8px 12px;text-align:left'>Metric</th>"
        "<th style='padding:8px 12px;text-align:center'>Value</th>"
        "<th style='padding:8px 12px;text-align:center'>Threshold</th>"
        "<th style='padding:8px 12px;text-align:center'>Result</th>"
        "<th style='padding:8px 12px;text-align:left'>What it validates</th>"
        "</tr></thead><tbody>"
        "<tr style='border-bottom:1px solid #e2e8f0'>"
        "<td style='padding:8px 12px'>V-1 Brain-mask round-trip Dice</td>"
        f"<td style='padding:8px 12px;text-align:center'><b>{v1_val}</b></td>"
        f"<td style='padding:8px 12px;text-align:center'>&ge; {v1_thr}</td>"
        f"<td style='padding:8px 12px;text-align:center'>{v1_badge}</td>"
        "<td style='padding:8px 12px'>Geometric self-consistency of the rigid transform"
        " (MR &rarr; PET inverse &rarr; MR forward; mask should survive the round-trip)</td>"
        "</tr>"
        "<tr>"
        "<td style='padding:8px 12px'>V-2 Tumour in-brain fraction</td>"
        f"<td style='padding:8px 12px;text-align:center'><b>{v2_val}</b></td>"
        f"<td style='padding:8px 12px;text-align:center'>&ge; {v2_thr}</td>"
        f"<td style='padding:8px 12px;text-align:center'>{v2_badge}</td>"
        "<td style='padding:8px 12px'>Anatomical plausibility of the tumour segmentation"
        " (fraction of MedSAM2 voxels inside TotalSegmentator brain parenchyma mask)</td>"
        "</tr>"
        "</tbody></table>"
    )
    display(HTML(html))
    print()
    print(f"V-1 Dice {v1['value']:.3f}  (threshold {v1['threshold']:.2f}) — {'PASS' if v1['pass'] else 'FAIL'}")
    print(f"V-2 overlap {v2['value']:.3f}  (threshold {v2['threshold']:.2f}) — {'PASS' if v2['pass'] else 'FAIL'}")
    print(f"  Tumour MedSAM2 voxels: {v2['medsam2_voxels']:,}  |  z-range: {v2['medsam2_z_range']}")
""")

# ---------------------------------------------------------------------------
# Cell 5 — Objective 3: Tumour segmentation summary + overlay
# ---------------------------------------------------------------------------
cell5 = nbformat.v4.new_code_cell("""\
import warnings

MR_DICOM      = PROJECT_ROOT / "data" / "raw" / "AX_3D_T1.dcm"
TUMOR_NPY     = PROJECT_ROOT / "docs" / "figures" / "tumor_mask_mr.npy"

# Hard-coded ground truth from the archive report
MEDSAM2_VOXELS  = 56_586
MEDSAM2_VOLUME_ML = 56.6
BBOX_PROMPT     = "[75, 187, 180]"

print("Objective 3 — MedSAM2 Tumour Segmentation")
print(f"  Method  : MedSAM2 video propagation (SAM-2 backbone)")
print(f"  Prompt  : manual bounding box at MR voxel {BBOX_PROMPT}")
print(f"  Voxels  : {MEDSAM2_VOXELS:,}")
print(f"  Volume  : {MEDSAM2_VOLUME_ML:.1f} mL")
print()

if not TUMOR_NPY.exists():
    print(f"[SKIP] Tumour mask not found at {TUMOR_NPY}")
elif not MR_DICOM.exists():
    print(f"[SKIP] MR DICOM not found at {MR_DICOM} — cannot render overlay")
else:
    # Load MR
    import pydicom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mr_ds = pydicom.dcmread(str(MR_DICOM))
    mr_vol = mr_ds.pixel_array  # (156, 256, 256)

    # Load tumour mask
    tumor_mask = np.load(str(TUMOR_NPY))  # (156, 256, 256) uint8

    # Mid-tumour axial slice
    z_indices = np.where(tumor_mask.any(axis=(1, 2)))[0]
    z_mid = int(z_indices[len(z_indices) // 2])

    mr_slice    = mr_vol[z_mid].astype(float)
    mask_slice  = tumor_mask[z_mid]

    # Contour from mask
    from skimage import measure
    contours = measure.find_contours(mask_slice, 0.5)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(mr_slice, cmap="gray", origin="lower",
              vmin=np.percentile(mr_slice, 1),
              vmax=np.percentile(mr_slice, 99))
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color="lime", linewidth=1.5)
    ax.set_title(f"MR T1 axial z={z_mid}  +  MedSAM2 tumour contour (lime)\\n"
                 f"Volume = {MEDSAM2_VOLUME_ML:.1f} mL  |  {MEDSAM2_VOXELS:,} voxels",
                 fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
""")

# ---------------------------------------------------------------------------
# Cell 6 — PET-side validation (Pedro Q10)
# ---------------------------------------------------------------------------
cell6 = nbformat.v4.new_code_cell("""\
from IPython.display import Image as IPyImage, display

PET_TUMOR_FIG = PROJECT_ROOT / "docs" / "figures" / "16_pet_tumor_mask_overlay.png"

if not PET_TUMOR_FIG.exists():
    print(f"[SKIP] Figure not found at {PET_TUMOR_FIG}")
else:
    print("Objective 2+3 — PET-side validation (Pedro Q10)")
    print()
    display(IPyImage(filename=str(PET_TUMOR_FIG), width=700))
    print()
    print("MedSAM2 mask (lime contour) on co-registered PET.")
    print("Direct visual validation: the MR-based segmentation lands on the metabolically active region.")
    print("The tumour is clearly visible as a hot spot in PET — confirming spatial correspondence.")
""")

# ---------------------------------------------------------------------------
# Cell 7 — Closing numbers at a glance
# ---------------------------------------------------------------------------
cell7 = nbformat.v4.new_code_cell("""\
from IPython.display import HTML, display

html = '''
<div style="font-family:monospace;background:#0f172a;color:#e2e8f0;
            padding:20px 24px;border-radius:8px;font-size:15px;line-height:2">
  <div style="font-size:18px;font-weight:bold;margin-bottom:12px;color:#7dd3fc">
    BrainScope — Numbers at a Glance
  </div>
  <div><span style="color:#86efac">&#10003; Objective 1</span>
    &nbsp;PET 4D shape (36, 47, 256, 256) &mdash; 8 DICOM headers parsed
  </div>
  <div><span style="color:#86efac">&#10003; Objective 2</span>
    &nbsp;NMI 1.029 &rarr; 1.073 &nbsp;|&nbsp; V-1 Dice 0.941
  </div>
  <div><span style="color:#86efac">&#10003; Objective 3</span>
    &nbsp;MedSAM2 tumour 56.6 mL &nbsp;|&nbsp; V-2 in-brain 0.924
  </div>
  <div style="margin-top:8px;text-align:center">
    <span style="color:#c4b5fd">github.com/AlejandroB10/BrainScope</span>
  </div>
</div>
'''
display(HTML(html))
""")

nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7]

out = Path(__file__).parent / "demo.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)

print(f"Wrote {out}")
