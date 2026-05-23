"""Extract representative static PNG frames from an animated GIF.

PDF cannot embed animations, so this helper extracts key frames into
``docs/figures/static/`` for use in pandoc-compiled documents.

Usage (command line)::

    python scripts/extract_gif_frames.py \\
        --gif docs/figures/04_pet_median_planes.gif \\
        --out docs/figures/static/

    python scripts/extract_gif_frames.py \\
        --gif docs/figures/06_rotating_mip.gif \\
        --out docs/figures/static/ \\
        --indices 0 6 12 18

"""
from __future__ import annotations

import argparse
import os

import imageio.v3 as iio


def extract_frames(
    gif_path: str,
    out_dir: str,
    indices: list[int] | None = None,
) -> list[str]:
    """Extract frames from a GIF and save them as PNG files.

    Parameters
    ----------
    gif_path:
        Path to the source animated GIF.
    out_dir:
        Directory where PNG files will be written.  Created if absent.
    indices:
        Frame indices to extract.  When ``None``, defaults to four
        evenly-spaced frames: ``[0, n//4, n//2, 3*n//4]``.

    Returns
    -------
    list[str]
        Absolute paths to the written PNG files, in the order they were
        written.
    """
    frames = iio.imread(gif_path, plugin="pillow", index=None)
    n_frames = len(frames)

    if indices is None:
        indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4]

    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(gif_path))[0]
    paths: list[str] = []

    for idx in indices:
        out = os.path.join(out_dir, f"{stem}_frame_{idx:03d}.png")
        iio.imwrite(out, frames[idx])
        paths.append(out)

    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract static PNG frames from an animated GIF."
    )
    parser.add_argument(
        "--gif",
        required=True,
        help="Path to the source GIF file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for PNG frames.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Frame indices to extract (0-based).  "
            "Defaults to four evenly-spaced frames."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    written = extract_frames(args.gif, args.out, args.indices)
    for path in written:
        print(f"Wrote: {path}")
