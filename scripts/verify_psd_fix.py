#!/usr/bin/env python
"""Diagnostic script to verify the PSD symmetry fix for streak mode.

Loads sinograms from sinogram_extract/, processes with streak mode,
and compares column variance / center bias against SVD reference.
"""

import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    import tifffile  # type: ignore[import-untyped]

    class _TiffReader:
        @staticmethod
        def open(path):
            class _Img:
                def __init__(self, arr):
                    self._arr = arr

                def __array__(self):
                    return self._arr

            return _Img(tifffile.imread(str(path)))

    Image = _TiffReader  # type: ignore[misc,assignment]

from bm3dornl.bm3d import bm3d_ring_artifact_removal

DATA_DIR = Path(__file__).resolve().parent.parent / "sinogram_extract"


def load_tiff(path: Path) -> np.ndarray:
    img = Image.open(path)
    return np.array(img).astype(np.float32)


def column_variance(sinogram: np.ndarray) -> np.ndarray:
    """Variance along projection axis (axis=0) for each column."""
    return np.var(sinogram, axis=0)


def center_bias_metric(col_var: np.ndarray, margin_frac: float = 0.25) -> float:
    """Ratio of center column variance to edge column variance.

    Values close to 1.0 indicate no center artifact.
    Values >> 1.0 indicate center artifact (excess variance at center).
    """
    n = len(col_var)
    margin = int(n * margin_frac)
    if margin < 1:
        margin = 1
    edge_var = np.mean(np.concatenate([col_var[:margin], col_var[-margin:]]))
    center_var = np.mean(col_var[margin:-margin])
    if edge_var == 0:
        return float("inf")
    return float(center_var / edge_var)


def main():
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Place sinogram TIFF files in sinogram_extract/ to run this verification.")
        sys.exit(1)

    pre_files = sorted(DATA_DIR.glob("sino_pre_ring_slice*.tiff"))
    svd_files = sorted(DATA_DIR.glob("sino_post_ring_svd_slice*.tiff"))

    if not pre_files:
        print("No pre-ring sinograms found in sinogram_extract/")
        sys.exit(1)

    print(f"Found {len(pre_files)} pre-ring sinograms, {len(svd_files)} SVD references")
    print("=" * 72)

    all_pass = True

    for pre_path in pre_files:
        slice_id = pre_path.stem.replace("sino_pre_ring_slice", "")
        svd_path = DATA_DIR / f"sino_post_ring_svd_slice{slice_id}.tiff"

        print(f"\nSlice {slice_id}:")

        sinogram = load_tiff(pre_path)
        print(f"  Input shape: {sinogram.shape}, range: [{sinogram.min():.4f}, {sinogram.max():.4f}]")

        # Process with streak mode
        streak_result = bm3d_ring_artifact_removal(sinogram, mode="streak")

        streak_col_var = column_variance(streak_result)
        streak_bias = center_bias_metric(streak_col_var)
        print(f"  Streak mode center bias: {streak_bias:.4f}")

        if svd_path.exists():
            svd_ref = load_tiff(svd_path)
            svd_col_var = column_variance(svd_ref)
            svd_bias = center_bias_metric(svd_col_var)
            print(f"  SVD reference center bias: {svd_bias:.4f}")
            ratio = streak_bias / svd_bias if svd_bias > 0 else float("inf")
            print(f"  Streak/SVD bias ratio: {ratio:.4f}")

            # Pass if streak bias is within 2x of SVD bias
            if ratio > 2.0:
                print(f"  ** WARN: streak bias ratio {ratio:.2f}x exceeds 2.0x threshold **")
                all_pass = False
            else:
                print("  OK")
        else:
            print("  (no SVD reference available)")
            # Without reference, just check bias isn't extreme
            if streak_bias > 3.0:
                print(f"  ** WARN: streak center bias {streak_bias:.2f} seems high **")
                all_pass = False
            else:
                print("  OK")

    print("\n" + "=" * 72)
    if all_pass:
        print("PASS: No center artifact detected in streak mode results.")
    else:
        print("WARN: Some slices show elevated center bias. Review results above.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
