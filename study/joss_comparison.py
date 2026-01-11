#!/usr/bin/env python
"""
JOSS Paper Benchmark Comparison Study

Compares bm3dornl ring artifact removal against TomoPy methods.
On Linux x86_64, also includes bm3d-streak-removal (Mäkinen et al., 2021).

Usage:
    pixi run python joss_comparison.py
"""

import platform
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# bm3dornl imports
from bm3dornl import bm3d_ring_artifact_removal
from bm3dornl.phantom import (
    generate_sinogram,
    shepp_logan_phantom,
    simulate_detector_gain_error,
)

# TomoPy imports
import tomopy


def get_platform_info():
    """Detect platform and return appropriate output directory."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        return "apple_silicon", False
    elif system == "Linux" and machine == "x86_64":
        return "linux_x86_64", True  # Can use bm3d-streak-removal
    elif system == "Linux" and machine == "aarch64":
        return "linux_arm64", False
    else:
        return f"{system.lower()}_{machine}", False


PLATFORM_NAME, CAN_USE_BM3D_STREAK = get_platform_info()

# Configuration - output to platform-specific directory
STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / "results" / PLATFORM_NAME
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
np.random.seed(42)

# Test parameters
PHANTOM_SIZE = 256
SCAN_STEP = 0.5  # degrees
NUM_TIMING_RUNS_FAST = 3  # For bm3dornl (fast methods)
NUM_TIMING_RUNS_SLOW = 1  # For TomoPy and bm3d-streak-removal (slower methods)
DETECTOR_GAIN_RANGE = (0.95, 1.05)
DETECTOR_GAIN_ERROR = 0.02

# Try to import bm3d-streak-removal on Linux x86_64
BM3D_STREAK_AVAILABLE = False
if CAN_USE_BM3D_STREAK:
    try:
        # Fix scipy.signal.gaussian import for bm3d-streak-removal compatibility
        import scipy.signal
        from scipy.signal import windows
        if not hasattr(scipy.signal, 'gaussian'):
            scipy.signal.gaussian = windows.gaussian

        from bm3d_streak_removal import multiscale_streak_removal, extreme_streak_attenuation
        BM3D_STREAK_AVAILABLE = True
        print("bm3d-streak-removal: Available")
    except ImportError as e:
        print(f"bm3d-streak-removal: Not available ({e})")


def generate_test_data():
    """Generate synthetic sinogram with ring artifacts."""
    print("Generating synthetic test data...")

    # Create Shepp-Logan phantom
    phantom = shepp_logan_phantom(size=PHANTOM_SIZE, contrast_factor=2.0)

    # Generate sinogram (Radon transform)
    sinogram, theta = generate_sinogram(phantom, scan_step=SCAN_STEP)

    # Add ring artifacts via detector gain error
    sinogram_with_rings, detector_gain = simulate_detector_gain_error(
        sinogram,
        detector_gain_range=DETECTOR_GAIN_RANGE,
        detector_gain_error=DETECTOR_GAIN_ERROR,
    )

    print(f"  Phantom shape: {phantom.shape}")
    print(f"  Sinogram shape: {sinogram.shape}")
    print(f"  Theta range: {theta[0]:.1f} to {theta[-1]:.1f} degrees")

    return {
        "phantom": phantom,
        "sinogram_clean": sinogram,
        "sinogram_rings": sinogram_with_rings,
        "theta": theta,
        "detector_gain": detector_gain,
    }


def time_function(func, *args, num_runs=NUM_TIMING_RUNS_FAST, **kwargs):
    """Time a function over multiple runs, return mean time."""
    times = []
    result = None
    for i in range(num_runs):
        if num_runs > 1:
            print(f"    Run {i+1}/{num_runs}...", end=" ", flush=True)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if num_runs > 1:
            print(f"{elapsed:.3f}s")
    return result, np.mean(times), np.std(times)


def compute_metrics(result, ground_truth):
    """Compute PSNR and SSIM against ground truth."""
    # Ensure same dtype and range
    result = np.asarray(result, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    # Normalize to [0, 1] for consistent metrics
    result_norm = (result - result.min()) / (result.max() - result.min() + 1e-10)
    gt_norm = (ground_truth - ground_truth.min()) / (
        ground_truth.max() - ground_truth.min() + 1e-10
    )

    psnr = peak_signal_noise_ratio(gt_norm, result_norm, data_range=1.0)
    ssim = structural_similarity(gt_norm, result_norm, data_range=1.0)

    return psnr, ssim


def run_bm3dornl_streak(sinogram):
    """Apply bm3dornl in streak mode."""
    return bm3d_ring_artifact_removal(
        sinogram.astype(np.float32),
        mode="streak",
        sigma_random=0.05,
        patch_size=8,
        step_size=4,
        search_window=24,
        max_matches=16,
    )


def run_bm3dornl_generic(sinogram):
    """Apply bm3dornl in generic mode."""
    return bm3d_ring_artifact_removal(
        sinogram.astype(np.float32),
        mode="generic",
        sigma_random=0.05,
        patch_size=8,
        step_size=4,
        search_window=24,
        max_matches=16,
    )


def run_bm3d_streak_removal(sinogram):
    """Apply bm3d-streak-removal (Mäkinen et al., 2021)."""
    sino_3d = sinogram.astype(np.float64)[np.newaxis, :, :]
    sino_log = np.log1p(sino_3d)
    sino_attenuated = extreme_streak_attenuation(sino_log)
    sino_denoised = multiscale_streak_removal(sino_attenuated)
    result = np.expm1(sino_denoised)
    return result[0]


def run_tomopy_fw(sinogram):
    """Apply TomoPy wavelet-Fourier stripe removal (Münch et al.)."""
    # TomoPy expects 3D array (nslices, nrows, ncols)
    sino_3d = sinogram[np.newaxis, :, :]
    result = tomopy.remove_stripe_fw(sino_3d, level=7, wname="db5", sigma=2, pad=True)
    return result[0]


def run_tomopy_sf(sinogram):
    """Apply TomoPy sorting-fitting stripe removal (Vo et al.)."""
    sino_3d = sinogram[np.newaxis, :, :]
    result = tomopy.remove_stripe_sf(sino_3d, size=5)
    return result[0]


def run_tomopy_based(sinogram):
    """Apply TomoPy ring removal based on Fourier-wavelet approach."""
    sino_3d = sinogram[np.newaxis, :, :]
    result = tomopy.remove_stripe_based_sorting(sino_3d, size=5, dim=1)
    return result[0]


def run_benchmarks(data):
    """Run all benchmark methods and collect results."""
    sinogram = data["sinogram_rings"]
    ground_truth = data["sinogram_clean"]

    # Methods: (name, function, do_timing, num_runs)
    methods = [
        ("Input (with rings)", lambda x: x, False, 1),
        ("bm3dornl (streak)", run_bm3dornl_streak, True, NUM_TIMING_RUNS_FAST),
        ("bm3dornl (generic)", run_bm3dornl_generic, True, NUM_TIMING_RUNS_FAST),
        ("TomoPy FW (Münch)", run_tomopy_fw, True, NUM_TIMING_RUNS_SLOW),
        ("TomoPy SF (Vo)", run_tomopy_sf, True, NUM_TIMING_RUNS_SLOW),
        ("TomoPy BSD (sort)", run_tomopy_based, True, NUM_TIMING_RUNS_SLOW),
    ]

    # Add bm3d-streak-removal if available (Linux x86_64 only)
    if BM3D_STREAK_AVAILABLE:
        methods.append(
            ("bm3d-streak-removal", run_bm3d_streak_removal, True, NUM_TIMING_RUNS_SLOW)
        )

    results = []

    for name, method, do_timing, num_runs in methods:
        print(f"\nRunning: {name}...")

        if do_timing:
            output, mean_time, std_time = time_function(method, sinogram, num_runs=num_runs)
            psnr, ssim = compute_metrics(output, ground_truth)
            print(f"  Time: {mean_time:.3f} ± {std_time:.3f} s")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  SSIM: {ssim:.4f}")
        else:
            output = method(sinogram)
            mean_time, std_time = 0.0, 0.0
            psnr, ssim = compute_metrics(output, ground_truth)

        results.append(
            {
                "method": name,
                "output": output,
                "time_mean": mean_time,
                "time_std": std_time,
                "psnr": psnr,
                "ssim": ssim,
            }
        )

    return results


def create_comparison_grid(data, results):
    """Create visual comparison grid."""
    print("\nGenerating comparison grid figure...")

    n_methods = len(results)
    # Determine grid size based on number of methods
    if n_methods <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 2, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Determine common color scale from clean sinogram
    vmin = data["sinogram_clean"].min()
    vmax = data["sinogram_clean"].max()

    for i, ax in enumerate(axes):
        if i < n_methods:
            res = results[i]
            ax.imshow(res["output"], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
            title = res["method"]
            if res["time_mean"] > 0:
                title += f"\n({res['time_mean']:.2f}s, PSNR={res['psnr']:.1f})"
            ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "comparison_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_DIR / 'comparison_grid.png'}")


def create_timing_chart(results):
    """Create bar chart of processing times."""
    print("\nGenerating timing comparison figure...")

    # Filter out input (no timing)
    timed_results = [r for r in results if r["time_mean"] > 0]

    methods = [r["method"] for r in timed_results]
    times = [r["time_mean"] for r in timed_results]
    errors = [r["time_std"] for r in timed_results]

    # Color coding: blue for bm3dornl, green for bm3d-streak-removal, orange for TomoPy
    def get_color(m):
        if "bm3dornl" in m:
            return "#1f77b4"  # blue
        elif "bm3d-streak" in m:
            return "#2ca02c"  # green
        else:
            return "#ff7f0e"  # orange

    colors = [get_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.5), 5))
    bars = ax.bar(methods, times, yerr=errors, color=colors, capsize=5)

    ax.set_ylabel("Processing Time (seconds)")
    ax.set_title(f"Ring Artifact Removal - Processing Time ({PHANTOM_SIZE}×{PHANTOM_SIZE} sinogram)\nPlatform: {PLATFORM_NAME}")
    ax.tick_params(axis="x", rotation=20)

    # Add value labels on bars
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{t:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "timing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_DIR / 'timing_comparison.png'}")


def create_quality_chart(results):
    """Create PSNR/SSIM comparison chart."""
    print("\nGenerating quality metrics figure...")

    # Filter out input
    metric_results = [r for r in results if r["time_mean"] > 0]

    methods = [r["method"] for r in metric_results]
    psnr_values = [r["psnr"] for r in metric_results]
    ssim_values = [r["ssim"] for r in metric_results]

    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(methods) * 2), 5))

    # Color coding
    def get_color(m):
        if "bm3dornl" in m:
            return "#1f77b4"
        elif "bm3d-streak" in m:
            return "#2ca02c"
        else:
            return "#ff7f0e"

    colors = [get_color(m) for m in methods]

    # PSNR chart
    bars1 = ax1.bar(x, psnr_values, width, color=colors)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Peak Signal-to-Noise Ratio (higher = better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right")

    for bar, v in zip(bars1, psnr_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # SSIM chart
    bars2 = ax2.bar(x, ssim_values, width, color=colors)
    ax2.set_ylabel("SSIM")
    ax2.set_title("Structural Similarity Index (higher = better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20, ha="right")
    ax2.set_ylim(0, 1.05)

    for bar, v in zip(bars2, ssim_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "quality_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_DIR / 'quality_metrics.png'}")


def save_results_csv(results):
    """Save results to CSV file."""
    print("\nSaving results to CSV...")

    df_data = []
    for r in results:
        if r["time_mean"] > 0:
            df_data.append(
                {
                    "Method": r["method"],
                    "Time (s)": f"{r['time_mean']:.3f} ± {r['time_std']:.3f}",
                    "PSNR (dB)": f"{r['psnr']:.2f}",
                    "SSIM": f"{r['ssim']:.4f}",
                }
            )

    df = pd.DataFrame(df_data)
    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Also print table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return df


def main():
    """Main benchmark entry point."""
    print("=" * 70)
    print("bm3dornl JOSS Paper - Benchmark Comparison Study")
    print("=" * 70)
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"bm3d-streak-removal available: {BM3D_STREAK_AVAILABLE}")

    # Generate test data
    data = generate_test_data()

    # Run benchmarks
    results = run_benchmarks(data)

    # Generate figures
    create_comparison_grid(data, results)
    create_timing_chart(results)
    create_quality_chart(results)

    # Save results
    df = save_results_csv(results)

    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_DIR}/")
    print("Files generated:")
    print("  - figures/comparison_grid.png")
    print("  - figures/timing_comparison.png")
    print("  - figures/quality_metrics.png")
    print("  - results.csv")

    return results


if __name__ == "__main__":
    main()
