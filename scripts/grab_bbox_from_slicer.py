"""Grab a tumor bounding box from a live 3D Slicer session.

This script connects to the 3D Slicer Web Server on port 2016, reads the
first active MarkupsROI or MarkupsFiducial node from the MRML scene, converts
the centre coordinates from RAS physical space to PET native voxel space, and
writes the result to ``data/tumor_bbox.yaml``.

Prerequisites
-------------
1. Launch 3D Slicer:
   /home/alejandro/Applications/Slicer-5.10.0-linux-amd64/Slicer

2. Load the PET and MR DICOM series via the DICOM module.

3. Place a Markups ROI (or a single fiducial) at the tumor center:
   - Open the Markups module (or press Ctrl+Shift+M).
   - Click "Add ROI" and position the box around the tumor.
   - Alternatively, click "Add point" and place a single fiducial at the tumor center.

4. Enable the Slicer Web Server on port 2016:
   - Modules → Developer Tools → Web Server
   - Set port to 2016, enable "Slicer API", click Start.

5. Run this script:
   conda run -n medical-image-processing-11763 python scripts/grab_bbox_from_slicer.py

6. Verify the output:
   cat data/tumor_bbox.yaml

Notes
-----
- The script reads ONE markup node (the first ROI, or first fiducial if no ROI
  exists). Place exactly one markup at the tumor center before running.
- Coordinates are automatically converted: Slicer uses RAS, DICOM uses LPS.
  The LPS→PET-voxel transform uses the PET sitk.Image geometry, which is
  identical to the coregistration module's physical-coordinate API.
- The ``half_extent_voxel`` is taken from the ROI node's half-extents when
  available; for a point fiducial the module constant
  ``BBOX_HALF_EXTENT_PET_NATIVE`` is used.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SLICER_PORT = 2016
SLICER_BASE = f"http://localhost:{SLICER_PORT}"

# The path to write the bbox YAML, relative to the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBOX_YAML_PATH = os.path.join(_REPO_ROOT, "data", "tumor_bbox.yaml")

# Fall-back half-extent if the markup node has no explicit extent.
DEFAULT_HALF_EXTENT = [4, 25, 25]  # [dz, dy, dx] in PET voxels


# ---------------------------------------------------------------------------
# Slicer communication helpers
# ---------------------------------------------------------------------------

def _slicer_exec(python_code: str) -> str:
    """Execute Python code inside the running Slicer instance via the exec API.

    Parameters
    ----------
    python_code:
        Python source code to evaluate in the Slicer Python kernel.

    Returns
    -------
    str
        The text printed by the code to stdout inside Slicer.

    Raises
    ------
    RuntimeError
        When the connection fails or Slicer returns an error status.
    """
    url = f"{SLICER_BASE}/slicer/exec"
    payload = python_code.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8").strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Slicer on port {SLICER_PORT}. "
            "Make sure Slicer is running with the Web Server module enabled on "
            f"port {SLICER_PORT}. Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Markup reading
# ---------------------------------------------------------------------------

_QUERY_CODE = """
import slicer, json

result = {}

# Try a MarkupsROI node first.
roi_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsROINode")
roi_nodes.UnRegister(None)

if roi_nodes.GetNumberOfItems() > 0:
    node = roi_nodes.GetItemAsObject(0)
    center_ras = [0.0, 0.0, 0.0]
    node.GetXYZ(center_ras)
    size = [0.0, 0.0, 0.0]
    node.GetSize(size)  # full extents (x, y, z) in mm
    result = {
        "type": "ROI",
        "name": node.GetName(),
        "center_ras": center_ras,
        "half_extent_mm_xyz": [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0],
    }
else:
    # Fall back to a fiducial point.
    fid_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")
    fid_nodes.UnRegister(None)
    if fid_nodes.GetNumberOfItems() > 0:
        node = fid_nodes.GetItemAsObject(0)
        if node.GetNumberOfControlPoints() > 0:
            pt = [0.0, 0.0, 0.0]
            node.GetNthControlPointPosition(0, pt)
            result = {
                "type": "Fiducial",
                "name": node.GetName(),
                "center_ras": pt,
                "half_extent_mm_xyz": None,
            }

print(json.dumps(result))
"""


def _query_slicer_markup() -> dict:
    """Read the first markup node from the active Slicer scene.

    Returns
    -------
    dict
        Keys: ``type``, ``name``, ``center_ras`` ([x, y, z] in mm RAS),
        ``half_extent_mm_xyz`` ([hx, hy, hz] in mm, or None for fiducials).

    Raises
    ------
    RuntimeError
        When no markup node is found or the Slicer connection fails.
    """
    raw = _slicer_exec(_QUERY_CODE)
    data = json.loads(raw)

    if not data:
        raise RuntimeError(
            "No Markups ROI or Fiducial node found in the Slicer scene. "
            "Place a Markups ROI at the tumor centre before running this script."
        )
    return data


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def _ras_to_lps(ras_xyz: list[float]) -> list[float]:
    """Convert RAS physical coordinates to LPS by negating x and y."""
    return [-ras_xyz[0], -ras_xyz[1], ras_xyz[2]]


def _lps_to_pet_voxel(lps_xyz: list[float], pet_image_path: str) -> tuple[int, int, int]:
    """Map LPS physical coordinates to PET native voxel indices (z, y, x).

    Loads the PET DICOM via the project loading module to obtain the sitk.Image
    with correct geometry, then calls TransformPhysicalPointToContinuousIndex.

    Parameters
    ----------
    lps_xyz:
        Physical coordinates [x, y, z] in mm (LPS).
    pet_image_path:
        Path to the PET DICOM file.

    Returns
    -------
    tuple[int, int, int]
        Rounded voxel indices [z, y, x] in PET native voxel space.
    """
    # Local import to avoid circular dependency if this is run standalone.
    _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from loading import load_dynamic_pet

    pet = load_dynamic_pet(pet_image_path)
    # TransformPhysicalPointToContinuousIndex expects (x, y, z) and returns (x, y, z).
    idx_xyz = pet.image.TransformPhysicalPointToContinuousIndex(
        (float(lps_xyz[0]), float(lps_xyz[1]), float(lps_xyz[2]))
    )
    # Return as (z, y, x) to match the BBox convention.
    return (int(round(idx_xyz[2])), int(round(idx_xyz[1])), int(round(idx_xyz[0])))


def _half_extent_mm_to_pet_voxels(
    half_extent_mm_xyz: list[float] | None,
    pet_spacing_zyx: tuple[float, float, float],
) -> list[int]:
    """Convert half-extents from mm to PET voxels.

    Parameters
    ----------
    half_extent_mm_xyz:
        Half-extents [hx, hy, hz] in mm, or None to use the default.
    pet_spacing_zyx:
        PET voxel spacing (sz, sy, sx) in mm.

    Returns
    -------
    list[int]
        Half-extents [dz, dy, dx] in PET voxels.
    """
    if half_extent_mm_xyz is None:
        return DEFAULT_HALF_EXTENT

    hx_mm, hy_mm, hz_mm = half_extent_mm_xyz
    sz, sy, sx = pet_spacing_zyx
    return [
        max(1, int(round(hz_mm / sz))),
        max(1, int(round(hy_mm / sy))),
        max(1, int(round(hx_mm / sx))),
    ]


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """Read the Slicer markup and write data/tumor_bbox.yaml."""
    from utils import filepath

    pet_path = filepath("e_1_BRAIN_DINAMIC_COLINA.dcm")
    if not os.path.exists(pet_path):
        print(
            "ERROR: PET DICOM not found. Make sure the data/ directory is "
            "present and the file 'e_1_BRAIN_DINAMIC_COLINA.dcm' is in data/raw/."
        )
        sys.exit(1)

    print(f"Connecting to Slicer on port {SLICER_PORT} ...")
    markup = _query_slicer_markup()
    print(f"Found {markup['type']} node: '{markup['name']}'")

    center_ras = markup["center_ras"]
    print(f"  Center (RAS): {center_ras}")

    center_lps = _ras_to_lps(center_ras)
    print(f"  Center (LPS): {center_lps}")

    # We need the PET image to convert LPS → voxel index.
    _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from loading import load_dynamic_pet

    print("Loading PET DICOM to obtain voxel geometry ...")
    pet = load_dynamic_pet(pet_path)

    idx_xyz = pet.image.TransformPhysicalPointToContinuousIndex(
        (float(center_lps[0]), float(center_lps[1]), float(center_lps[2]))
    )
    centroid_zyx = [int(round(idx_xyz[2])), int(round(idx_xyz[1])), int(round(idx_xyz[0]))]
    print(f"  Centroid (PET voxel, z/y/x): {centroid_zyx}")

    half_extent = _half_extent_mm_to_pet_voxels(
        markup["half_extent_mm_xyz"], pet.spacing_zyx_mm
    )
    print(f"  Half-extent (PET voxel, dz/dy/dx): {half_extent}")

    bbox_data = {
        "schema_version": 1,
        "centroid_pet_native_voxel": centroid_zyx,
        "half_extent_voxel": half_extent,
    }

    os.makedirs(os.path.dirname(BBOX_YAML_PATH), exist_ok=True)
    with open(BBOX_YAML_PATH, "w") as fh:
        fh.write(
            "# Generated by scripts/grab_bbox_from_slicer.py\n"
            "# Coordinates are in PET native voxel space (z, y, x).\n"
        )
        yaml.dump(bbox_data, fh, default_flow_style=False)

    print(f"\nWrote: {BBOX_YAML_PATH}")
    print("BBox YAML contents:")
    with open(BBOX_YAML_PATH) as fh:
        print(fh.read())


if __name__ == "__main__":
    # Smoke check: if no args are given, just print a readiness message.
    # Pass --run to actually connect to Slicer.
    if "--run" not in sys.argv:
        print("scripts/grab_bbox_from_slicer.py is ready; run me with Slicer up on port 2016")
        print("Usage: python scripts/grab_bbox_from_slicer.py --run")
    else:
        main()
