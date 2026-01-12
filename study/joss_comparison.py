#!/usr/bin/env python
"""
JOSS Paper Benchmark Comparison Study

Unified comparison of ring artifact removal methods:
- bm3dornl (streak and generic modes)
- TomoPy methods (FW, SF, BSD)
- bm3d-streak-removal (run separately in isolated environment)

Uses 512x512 phantom to accommodate bm3d-streak-removal's size requirements.

Usage:
    pixi run benchmark  # Run main benchmark (bm3dornl + TomoPy)
    cd bm3d_streak_test && pixi run run  # Run bm3d-streak-removal
    pixi run visualize  # Generate unified comparison figures
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
DATA_DIR = RESULTS_DIR / "data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
np.random.seed(42)

# Test parameters - 512x512 for bm3d-streak-removal compatibility
PHANTOM_SIZE = 512
SCAN_STEP = 0.5  # degrees
NUM_TIMING_RUNS_FAST = 3  # For bm3dornl (fast methods)
NUM_TIMING_RUNS_SLOW = 1  # For TomoPy (slower methods)
DETECTOR_GAIN_RANGE = (0.95, 1.05)
DETECTOR_GAIN_ERROR = 0.02


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

    mean_time = np.mean(times)
    std_time = np.std(times) if len(times) > 1 else 0.0
    return result, mean_time, std_time


def compute_metrics(result, ground_truth):
    """Compute PSNR and SSIM between result and ground truth."""
    result = np.asarray(result, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    # Normalize to [0, 1] for metric computation
    result_norm = (result - result.min()) / (result.max() - result.min() + 1e-10)
    gt_norm = (ground_truth - ground_truth.min()) / (
        ground_truth.max() - ground_truth.min() + 1e-10
    )

    psnr = peak_signal_noise_ratio(gt_norm, result_norm, data_range=1.0)
    ssim = structural_similarity(gt_norm, result_norm, data_range=1.0)
    return psnr, ssim


# Method implementations
def run_bm3dornl_streak(sinogram):
    """Apply bm3dornl in streak mode."""
    return bm3d_ring_artifact_removal(
        sinogram,
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
        sinogram,
        mode="generic",
        sigma_random=0.05,
        patch_size=8,
        step_size=4,
        search_window=24,
        max_matches=16,
    )


def run_tomopy_fw(sinogram):
    """Apply TomoPy wavelet-Fourier stripe removal (Münch et al.)."""
    sino_3d = sinogram[np.newaxis, :, :]
    result = tomopy.remove_stripe_fw(sino_3d, level=7, wname="db5", sigma=2, pad=True)
    return result[0]


def run_tomopy_sf(sinogram):
    """Apply TomoPy sorting-fitting stripe removal (Vo et al.)."""
    sino_3d = sinogram[np.newaxis, :, :]
    result = tomopy.remove_stripe_sf(sino_3d, size=5)
    return result[0]


def run_tomopy_bsd(sinogram):
    """Apply TomoPy sorting-based stripe removal."""
    sino_3d = sinogram[np.newaxis, :, :]
    result = tomopy.remove_stripe_based_sorting(sino_3d, size=5)
    return result[0]


def run_benchmarks(data):
    """Run all benchmark methods."""
    sinogram = data["sinogram_rings"]
    ground_truth = data["sinogram_clean"]

    # Define methods to test
    methods = [
        ("bm3dornl (streak)", run_bm3dornl_streak, True, NUM_TIMING_RUNS_FAST),
        ("bm3dornl (generic)", run_bm3dornl_generic, True, NUM_TIMING_RUNS_FAST),
        ("TomoPy FW (Münch)", run_tomopy_fw, True, NUM_TIMING_RUNS_SLOW),
        ("TomoPy SF (Vo)", run_tomopy_sf, True, NUM_TIMING_RUNS_SLOW),
        ("TomoPy BSD (sort)", run_tomopy_bsd, True, NUM_TIMING_RUNS_SLOW),
    ]

    results = []

    # Add input (with rings) as reference
    results.append(
        {
            "method": "Input (with rings)",
            "output": sinogram,
            "time_mean": 0.0,
            "time_std": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
        }
    )

    for name, method, do_timing, num_runs in methods:
        print(f"\nRunning: {name}...")

        if do_timing:
            output, mean_time, std_time = time_function(
                method, sinogram, num_runs=num_runs
            )
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


def save_data_for_bm3d_streak(data, results):
    """Save test data and results for bm3d-streak-removal processing."""
    print("\nSaving test data for bm3d-streak-removal...")

    # Save test data
    np.save(DATA_DIR / "sinogram_clean.npy", data["sinogram_clean"])
    np.save(DATA_DIR / "sinogram_rings.npy", data["sinogram_rings"])
    print(f"  Saved test data to: {DATA_DIR}")

    # Save method results
    for r in results:
        if r["method"] != "Input (with rings)":
            safe_name = r["method"].replace(" ", "_").replace("(", "").replace(")", "")
            np.save(DATA_DIR / f"result_{safe_name}.npy", r["output"])

    # Save metrics
    metrics = []
    for r in results:
        if r["time_mean"] > 0:
            metrics.append({
                "method": r["method"],
                "time_mean": r["time_mean"],
                "time_std": r["time_std"],
                "psnr": r["psnr"],
                "ssim": r["ssim"],
            })
    pd.DataFrame(metrics).to_csv(DATA_DIR / "metrics.csv", index=False)
    print(f"  Saved metrics to: {DATA_DIR / 'metrics.csv'}")


def create_comparison_grid_with_diff(data, results):
    """Create visual comparison grid with diff images.

    Layout:
    Row 1: Input (noisy) | Clean (ground truth)
    Row 2: Method results (6 methods)
    Row 3: Diff images (result - clean) with diverging colormap
    """
    print("\nGenerating comparison grid with diff images...")

    ground_truth = data["sinogram_clean"]
    sinogram_rings = data["sinogram_rings"]

    # Filter to only actual methods (not input)
    method_results = [r for r in results if r["time_mean"] > 0]

    n_methods = len(method_results)
    ncols = max(n_methods, 2)  # At least 2 columns for row 1

    fig = plt.figure(figsize=(3 * ncols, 10))

    # Common color scale for sinograms
    vmin = ground_truth.min()
    vmax = ground_truth.max()

    # Row 1: Input and Clean reference (centered)
    ax1 = fig.add_subplot(3, ncols, 1)
    ax1.imshow(sinogram_rings, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title("Input (with rings)", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(3, ncols, 2)
    ax2.imshow(ground_truth, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title("Clean (ground truth)", fontsize=10)
    ax2.axis("off")

    # Hide unused axes in row 1
    for i in range(2, ncols):
        ax = fig.add_subplot(3, ncols, i + 1)
        ax.axis("off")

    # Row 2: Method results
    for i, r in enumerate(method_results):
        ax = fig.add_subplot(3, ncols, ncols + i + 1)
        ax.imshow(r["output"], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        title = r["method"]
        if r["time_mean"] > 0:
            title += f"\n{r['time_mean']:.2f}s | PSNR={r['psnr']:.1f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 3: Diff images (result - clean)
    # First, compute all diffs to find common scale
    diffs = []
    for r in method_results:
        diff = r["output"] - ground_truth
        diffs.append(diff)

    # Zero-centered diverging colormap
    max_abs_diff = max(np.abs(d).max() for d in diffs)
    diff_vmin, diff_vmax = -max_abs_diff, max_abs_diff

    for i, (r, diff) in enumerate(zip(method_results, diffs)):
        ax = fig.add_subplot(3, ncols, 2 * ncols + i + 1)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=diff_vmin, vmax=diff_vmax, aspect="auto")
        ax.set_title("Diff", fontsize=9)
        ax.axis("off")

    # Add colorbar for diff images
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.25])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Result - Ground Truth", fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
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
    ax.set_title(f"Ring Artifact Removal - Processing Time ({PHANTOM_SIZE}x{PHANTOM_SIZE} sinogram)\nPlatform: {PLATFORM_NAME}")
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
    print(f"Phantom size: {PHANTOM_SIZE}x{PHANTOM_SIZE}")
    print(f"Results directory: {RESULTS_DIR}")

    if CAN_USE_BM3D_STREAK:
        print("\nNote: bm3d-streak-removal available on this platform.")
        print("Run separately: cd bm3d_streak_test && pixi run run")

    # Generate test data
    data = generate_test_data()

    # Run benchmarks
    results = run_benchmarks(data)

    # Save data for bm3d-streak-removal
    save_data_for_bm3d_streak(data, results)

    # Generate figures
    create_comparison_grid_with_diff(data, results)
    create_timing_chart(results)
    create_quality_chart(results)

    # Save CSV
    save_results_csv(results)

    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_DIR}/")
    print("Files generated:")
    print("  - data/sinogram_clean.npy")
    print("  - data/sinogram_rings.npy")
    print("  - data/metrics.csv")
    print("  - figures/comparison_grid.png")
    print("  - figures/timing_comparison.png")
    print("  - figures/quality_metrics.png")
    print("  - results.csv")

    if CAN_USE_BM3D_STREAK:
        print("\nNext step: Run bm3d-streak-removal in isolated environment:")
        print("  cd bm3d_streak_test && pixi run run")
        print("Then generate unified visualization:")
        print("  pixi run visualize")


if __name__ == "__main__":
    main()
