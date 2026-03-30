#!/usr/bin/env python3
"""Process sinograms with streak mode BM3D, reconstruct via FBP, and save images.

Produces side-by-side comparison of:
  - Raw (no ring removal) FBP reconstruction
  - SVD ring removal FBP reconstruction (reference)
  - Streak mode ring removal FBP reconstruction (our fix)

Output images are saved to output_recon/ for visual inspection.

Usage:
    pixi run python scripts/reconstruct_and_compare.py
    pixi run python scripts/reconstruct_and_compare.py --slice 0140
    pixi run python scripts/reconstruct_and_compare.py --center 2100 --angles 180
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import correlate
from skimage.transform import iradon

from bm3dornl.bm3d import bm3d_ring_artifact_removal

DATA_DIR = Path(__file__).resolve().parent.parent / "sinogram_extract"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_recon"


def load_tiff(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.float32)


def find_rotation_center(sinogram: np.ndarray, angle_range: float = 180.0) -> float:
    """Find center of rotation by cross-correlating proj_0 with flipped proj_180.

    Uses the central 50% of the detector to avoid edge effects.
    For a 360-deg scan, the projection at 180 degrees is at index n_proj//2.
    For a 180-deg scan, we use the first and last projections.
    """
    n_proj, n_cols = sinogram.shape

    if angle_range >= 360.0:
        idx_180 = n_proj // 2
        proj_0 = sinogram[0, :]
        proj_180 = sinogram[idx_180, ::-1]
    else:
        proj_0 = sinogram[0, :]
        proj_180 = sinogram[-1, ::-1]

    # Use central strip to avoid edge artifacts
    margin = n_cols // 4
    strip = slice(margin, n_cols - margin)
    corr = correlate(proj_0[strip], proj_180[strip], mode="full")
    shift = np.argmax(corr) - (len(proj_180[strip]) - 1)
    center = (n_cols + shift) / 2.0
    return center


def shift_sinogram_to_center(sinogram: np.ndarray, center: float) -> np.ndarray:
    """Shift sinogram columns so rotation center aligns with detector midpoint.

    Uses Fourier-based sub-pixel shift for accuracy.
    """
    n_cols = sinogram.shape[1]
    midpoint = n_cols / 2.0
    pixel_shift = midpoint - center

    if abs(pixel_shift) < 0.1:
        return sinogram

    from scipy.ndimage import shift as ndi_shift

    shifted = ndi_shift(sinogram, [0, pixel_shift], mode="nearest")
    return shifted


def reconstruct_fbp(sinogram: np.ndarray, angle_range: float = 360.0) -> np.ndarray:
    """FBP reconstruction from sinogram.

    sinogram shape: (n_projections, n_detector_columns)
    iradon expects: (n_detector_columns, n_projections)
    """
    n_proj = sinogram.shape[0]
    theta = np.linspace(0, angle_range, n_proj, endpoint=False)
    recon = iradon(sinogram.T, theta=theta, filter_name="ramp", circle=True)
    return recon


def to_uint8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct sinograms and compare ring removal methods")
    parser.add_argument("--slice", type=str, default=None, help="Process only this slice ID (e.g. 0140)")
    parser.add_argument("--center", type=float, default=None, help="Rotation center (pixel). Auto-detected if omitted.")
    parser.add_argument("--angles", type=float, default=180.0, help="Angular range in degrees (default: 180)")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.slice:
        pre_files = sorted(DATA_DIR.glob(f"sino_pre_ring_slice{args.slice}.tiff"))
    else:
        pre_files = sorted(DATA_DIR.glob("sino_pre_ring_slice*.tiff"))

    if not pre_files:
        print("No matching sinograms found.")
        sys.exit(1)

    for pre_path in pre_files:
        slice_id = pre_path.stem.replace("sino_pre_ring_slice", "")
        svd_path = DATA_DIR / f"sino_post_ring_svd_slice{slice_id}.tiff"

        print(f"\n=== Slice {slice_id} ===")
        sinogram = load_tiff(pre_path)
        print(f"  Sinogram shape: {sinogram.shape}")

        # Find or use provided rotation center
        if args.center is not None:
            center = args.center
        else:
            center = find_rotation_center(sinogram, args.angles)
        midpoint = sinogram.shape[1] / 2.0
        print(f"  Rotation center: {center:.1f} (midpoint: {midpoint:.1f}, offset: {center - midpoint:.1f})")

        # Shift sinogram so center of rotation is at detector midpoint
        sinogram_centered = shift_sinogram_to_center(sinogram, center)

        # 1. Raw FBP (no ring removal)
        print("  Reconstructing raw FBP...")
        recon_raw = reconstruct_fbp(sinogram_centered, args.angles)

        # 2. SVD reference
        recon_svd = None
        if svd_path.exists():
            print("  Reconstructing SVD reference FBP...")
            sino_svd = load_tiff(svd_path)
            sino_svd_centered = shift_sinogram_to_center(sino_svd, center)
            recon_svd = reconstruct_fbp(sino_svd_centered, args.angles)

        # 3. Streak mode (our fix) — denoise original, then shift for recon
        print("  Running streak mode BM3D...")
        sino_streak = bm3d_ring_artifact_removal(sinogram, mode="streak")
        sino_streak_centered = shift_sinogram_to_center(sino_streak, center)
        print("  Reconstructing streak FBP...")
        recon_streak = reconstruct_fbp(sino_streak_centered, args.angles)

        # Shared percentile bounds for fair comparison
        all_recons = [recon_raw, recon_streak]
        if recon_svd is not None:
            all_recons.append(recon_svd)
        combined = np.concatenate([r.ravel() for r in all_recons])
        vmin, vmax = float(np.percentile(combined, 1)), float(np.percentile(combined, 99))

        # Save individual images
        raw_img = to_uint8(recon_raw, vmin, vmax)
        streak_img = to_uint8(recon_streak, vmin, vmax)
        Image.fromarray(raw_img).save(OUTPUT_DIR / f"recon_raw_slice{slice_id}.png")
        Image.fromarray(streak_img).save(OUTPUT_DIR / f"recon_streak_slice{slice_id}.png")

        panels = [raw_img, streak_img]
        labels = ["Raw", "Streak"]

        if recon_svd is not None:
            svd_img = to_uint8(recon_svd, vmin, vmax)
            Image.fromarray(svd_img).save(OUTPUT_DIR / f"recon_svd_slice{slice_id}.png")
            panels.append(svd_img)
            labels.append("SVD")

        # Side-by-side comparison
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        gap = 4

        total_w = len(panels) * max_w + (len(panels) - 1) * gap
        canvas = np.zeros((max_h, total_w), dtype=np.uint8)

        for i, panel in enumerate(panels):
            x_off = i * (max_w + gap)
            y_off = (max_h - panel.shape[0]) // 2
            canvas[y_off : y_off + panel.shape[0], x_off : x_off + panel.shape[1]] = panel

        comp_path = OUTPUT_DIR / f"comparison_slice{slice_id}.png"
        Image.fromarray(canvas).save(comp_path)
        print(f"  Saved: {comp_path}")
        print(f"  Individual: recon_raw_slice{slice_id}.png, recon_streak_slice{slice_id}.png", end="")
        if recon_svd is not None:
            print(f", recon_svd_slice{slice_id}.png")
        else:
            print()

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Compare Raw vs Streak vs SVD — look for center ring artifact in streak results.")


if __name__ == "__main__":
    main()
