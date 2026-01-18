#!/usr/bin/env python
"""
Unified Visualization for JOSS Paper Benchmark

Combines results from all methods (including bm3d-streak-removal run separately)
into a single unified comparison grid with diff images.

Usage:
    pixi run visualize
"""

import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_platform_name():
    """Detect platform."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        return "apple_silicon"
    elif system == "Linux" and machine == "x86_64":
        return "linux_x86_64"
    elif system == "Linux" and machine == "aarch64":
        return "linux_arm64"
    else:
        return f"{system.lower()}_{machine}"


PLATFORM_NAME = get_platform_name()
STUDY_DIR = Path(__file__).parent
RESULTS_DIR = STUDY_DIR / "results" / PLATFORM_NAME
DATA_DIR = RESULTS_DIR / "data"
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    """Load all results including bm3d-streak-removal."""
    print("Loading results...")

    # Load test data
    sinogram_clean = np.load(DATA_DIR / "sinogram_clean.npy")
    sinogram_rings = np.load(DATA_DIR / "sinogram_rings.npy")

    # Load main benchmark metrics
    main_metrics = pd.read_csv(DATA_DIR / "metrics.csv")

    # Load method results
    results = []

    # Main benchmark methods
    method_files = {
        "bm3dornl (streak)": "result_bm3dornl_streak.npy",
        "bm3dornl (generic)": "result_bm3dornl_generic.npy",
        "bm3dornl (multiscale)": "result_bm3dornl_multiscale.npy",
        "Fourier-SVD": "result_Fourier-SVD.npy",
        "TomoPy FW (Münch)": "result_TomoPy_FW_Münch.npy",
        "TomoPy SF (Vo)": "result_TomoPy_SF_Vo.npy",
        "TomoPy BSD (sort)": "result_TomoPy_BSD_sort.npy",
    }

    for method, filename in method_files.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            output = np.load(filepath)
            metrics_row = main_metrics[main_metrics["method"] == method]
            if not metrics_row.empty:
                results.append({
                    "method": method,
                    "output": output,
                    "time_mean": metrics_row["time_mean"].values[0],
                    "time_std": metrics_row["time_std"].values[0],
                    "psnr": metrics_row["psnr"].values[0],
                    "ssim": metrics_row["ssim"].values[0],
                })
                print(f"  Loaded: {method}")

    # bm3d-streak-removal (if available)
    bm3d_streak_file = DATA_DIR / "result_bm3d-streak-removal.npy"
    bm3d_streak_metrics = DATA_DIR / "bm3d_streak_metrics.csv"

    if bm3d_streak_file.exists() and bm3d_streak_metrics.exists():
        output = np.load(bm3d_streak_file)
        metrics = pd.read_csv(bm3d_streak_metrics)
        results.append({
            "method": "bm3d-streak-removal",
            "output": output,
            "time_mean": float(metrics["time_mean"].values[0]),
            "time_std": float(metrics["time_std"].values[0]),
            "psnr": float(metrics["psnr"].values[0]),
            "ssim": float(metrics["ssim"].values[0]),
        })
        print(f"  Loaded: bm3d-streak-removal")
    else:
        print(f"  Note: bm3d-streak-removal results not found")
        print(f"        Run: cd bm3d_streak_test && pixi run run")

    return {
        "sinogram_clean": sinogram_clean,
        "sinogram_rings": sinogram_rings,
    }, results


def create_unified_comparison_grid(data, results):
    """Create unified comparison grid with all methods and diff images.

    Layout:
    Row 1: Input (noisy) | Clean (ground truth) | [empty cells]
    Row 2: Method results (all 6 methods)
    Row 3: Diff images (result - clean) with diverging colormap
    """
    print("\nGenerating unified comparison grid...")

    ground_truth = data["sinogram_clean"]
    sinogram_rings = data["sinogram_rings"]

    n_methods = len(results)
    ncols = max(n_methods, 2)

    fig = plt.figure(figsize=(3.5 * ncols, 11))

    # Common color scale for sinograms
    vmin = ground_truth.min()
    vmax = ground_truth.max()

    # Row 1: Input and Clean reference
    ax1 = fig.add_subplot(3, ncols, 1)
    ax1.imshow(sinogram_rings, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title("Input\n(with ring artifacts)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(3, ncols, 2)
    ax2.imshow(ground_truth, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title("Ground Truth\n(clean)", fontsize=10, fontweight="bold")
    ax2.axis("off")

    # Hide unused axes in row 1
    for i in range(2, ncols):
        ax = fig.add_subplot(3, ncols, i + 1)
        ax.axis("off")

    # Row 2: Method results
    for i, r in enumerate(results):
        ax = fig.add_subplot(3, ncols, ncols + i + 1)
        ax.imshow(r["output"], cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")

        # Color-code title
        method = r["method"]
        if "bm3dornl" in method:
            color = "#1f77b4"  # blue
        elif "Fourier-SVD" in method:
            color = "#17becf"  # cyan/teal (also bm3dornl)
        elif "bm3d-streak" in method:
            color = "#2ca02c"  # green
        else:
            color = "#ff7f0e"  # orange (TomoPy)

        title = f"{method}\n{r['time_mean']:.2f}s | PSNR={r['psnr']:.1f}"
        ax.set_title(title, fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

    # Row 3: Diff images (result - clean)
    diffs = []
    for r in results:
        diff = r["output"] - ground_truth
        diffs.append(diff)

    # Zero-centered diverging colormap
    max_abs_diff = max(np.abs(d).max() for d in diffs)
    diff_vmin, diff_vmax = -max_abs_diff, max_abs_diff

    for i, (r, diff) in enumerate(zip(results, diffs)):
        ax = fig.add_subplot(3, ncols, 2 * ncols + i + 1)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=diff_vmin, vmax=diff_vmax, aspect="auto")
        ax.set_title("Difference", fontsize=9)
        ax.axis("off")

    # Add colorbar for diff images
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.22])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Result - Ground Truth", fontsize=9)

    # Add row labels
    fig.text(0.02, 0.83, "Reference", fontsize=11, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.50, "Results", fontsize=11, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.17, "Difference", fontsize=11, fontweight="bold", rotation=90, va="center")

    plt.tight_layout(rect=[0.04, 0, 0.9, 0.96])
    fig.suptitle(f"Ring Artifact Removal Comparison - {PLATFORM_NAME}", fontsize=12, fontweight="bold")

    output_path = FIGURE_DIR / "unified_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_unified_timing_chart(results):
    """Create timing comparison bar chart with all methods."""
    print("\nGenerating unified timing chart...")

    methods = [r["method"] for r in results]
    times = [r["time_mean"] for r in results]
    errors = [r["time_std"] for r in results]

    # Color coding
    def get_color(m):
        if "bm3dornl" in m:
            return "#1f77b4"  # blue
        elif "Fourier-SVD" in m:
            return "#17becf"  # cyan/teal (also bm3dornl)
        elif "bm3d-streak" in m:
            return "#2ca02c"  # green
        else:
            return "#ff7f0e"  # orange (TomoPy)

    colors = [get_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(methods, times, yerr=errors, color=colors, capsize=5)

    ax.set_ylabel("Processing Time (seconds)", fontsize=11)
    ax.set_title(f"Ring Artifact Removal - Processing Time Comparison\nPlatform: {PLATFORM_NAME}", fontsize=12)
    ax.tick_params(axis="x", rotation=25, labelsize=10)

    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.02,
            f"{t:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="bm3dornl"),
        Patch(facecolor="#17becf", label="Fourier-SVD (bm3dornl)"),
        Patch(facecolor="#2ca02c", label="bm3d-streak-removal"),
        Patch(facecolor="#ff7f0e", label="TomoPy"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    output_path = FIGURE_DIR / "unified_timing.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_unified_quality_chart(results):
    """Create quality metrics comparison with all methods."""
    print("\nGenerating unified quality metrics chart...")

    methods = [r["method"] for r in results]
    psnr_values = [r["psnr"] for r in results]
    ssim_values = [r["ssim"] for r in results]

    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color coding
    def get_color(m):
        if "bm3dornl" in m:
            return "#1f77b4"  # blue
        elif "Fourier-SVD" in m:
            return "#17becf"  # cyan/teal (also bm3dornl)
        elif "bm3d-streak" in m:
            return "#2ca02c"  # green
        else:
            return "#ff7f0e"  # orange (TomoPy)

    colors = [get_color(m) for m in methods]

    # PSNR chart
    bars1 = ax1.bar(x, psnr_values, width, color=colors)
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.set_title("Peak Signal-to-Noise Ratio\n(higher = better)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)

    for bar, v in zip(bars1, psnr_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # SSIM chart
    bars2 = ax2.bar(x, ssim_values, width, color=colors)
    ax2.set_ylabel("SSIM", fontsize=11)
    ax2.set_title("Structural Similarity Index\n(higher = better)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.1)

    for bar, v in zip(bars2, ssim_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path = FIGURE_DIR / "unified_quality.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_consolidated_results(results):
    """Save consolidated results table."""
    print("\nSaving consolidated results...")

    df_data = []
    for r in results:
        df_data.append({
            "Method": r["method"],
            "Time (s)": f"{r['time_mean']:.3f} ± {r['time_std']:.3f}",
            "PSNR (dB)": f"{r['psnr']:.2f}",
            "SSIM": f"{r['ssim']:.4f}",
        })

    df = pd.DataFrame(df_data)
    csv_path = RESULTS_DIR / "consolidated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print table
    print("\n" + "=" * 70)
    print("CONSOLIDATED RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return df


def main():
    print("=" * 70)
    print("Unified Visualization - JOSS Paper Benchmark")
    print("=" * 70)
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Results directory: {RESULTS_DIR}")

    # Check if data exists
    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        print("Please run the main benchmark first:")
        print("  pixi run benchmark")
        return

    # Load all results
    data, results = load_all_results()

    if not results:
        print("\nERROR: No results found")
        return

    # Generate unified figures
    create_unified_comparison_grid(data, results)
    create_unified_timing_chart(results)
    create_unified_quality_chart(results)

    # Save consolidated results
    save_consolidated_results(results)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Figures saved to: {FIGURE_DIR}/")
    print("  - unified_comparison.png")
    print("  - unified_timing.png")
    print("  - unified_quality.png")
    print(f"Results saved to: {RESULTS_DIR}/consolidated_results.csv")


if __name__ == "__main__":
    main()
