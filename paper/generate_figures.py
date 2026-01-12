#!/usr/bin/env python3
"""
Generate publication-quality figures for JOSS paper.

Produces three separate figures:
- Figure 1: Input (with ring artifacts) + Ground Truth (clean)
- Figure 2: Method results (cropped) + Difference images
- Figure 3: Metrics (Processing time + PSNR vs SSIM)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("study/results/linux_x86_64/data")
OUTPUT_DIR = Path("paper")
DPI = 300

# Font sizes (publication quality)
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
ANNOTATION_SIZE = 10

# Crop region for zoomed views (square region to avoid distortion)
# Centered on columns 310-410 where ring artifacts are most visible
CROP_Y = slice(200, 450)  # rows (250 pixels)
CROP_X = slice(235, 485)  # columns (250 pixels, centered on artifact region)

# Method definitions with display names, colors, and data file mapping
METHODS = [
    {"key": "bm3dornl_streak", "name": "bm3dornl\n(streak)", "color": "#1f77b4", "file": "result_bm3dornl_streak.npy"},
    {"key": "bm3dornl_generic", "name": "bm3dornl\n(generic)", "color": "#1f77b4", "file": "result_bm3dornl_generic.npy"},
    {"key": "TomoPy_FW", "name": "TomoPy FW\n(Münch)", "color": "#ff7f0e", "file": "result_TomoPy_FW_Münch.npy"},
    {"key": "TomoPy_SF", "name": "TomoPy SF\n(Vo)", "color": "#ff7f0e", "file": "result_TomoPy_SF_Vo.npy"},
    {"key": "TomoPy_BSD", "name": "TomoPy BSD\n(sort)", "color": "#ff7f0e", "file": "result_TomoPy_BSD_sort.npy"},
    {"key": "bm3d_streak", "name": "bm3d-streak\n-removal", "color": "#2ca02c", "file": "result_bm3d-streak-removal.npy"},
]

# Timing and metrics data (from consolidated_results.csv)
METRICS = {
    "bm3dornl_streak": {"time": 0.256, "std": 0.009, "psnr": 32.63, "ssim": 0.6160},
    "bm3dornl_generic": {"time": 0.219, "std": 0.002, "psnr": 32.93, "ssim": 0.5760},
    "TomoPy_FW": {"time": 0.318, "std": 0.008, "psnr": 20.61, "ssim": 0.5831},
    "TomoPy_SF": {"time": 0.278, "std": 0.008, "psnr": 34.50, "ssim": 0.9591},
    "TomoPy_BSD": {"time": 0.349, "std": 0.009, "psnr": 34.69, "ssim": 0.9333},
    "bm3d_streak": {"time": 41.033, "std": 0.139, "psnr": 36.34, "ssim": 0.8697},
}

# Marker styles for scatter plot
MARKERS = {
    "bm3dornl_streak": "s",  # square
    "bm3dornl_generic": "D",  # diamond
    "TomoPy_FW": "^",  # triangle up
    "TomoPy_SF": "v",  # triangle down
    "TomoPy_BSD": ">",  # triangle right
    "bm3d_streak": "o",  # circle
}


def load_data():
    """Load all sinogram data."""
    data = {
        "input": np.load(DATA_DIR / "sinogram_rings.npy"),
        "ground_truth": np.load(DATA_DIR / "sinogram_clean.npy"),
    }
    for method in METHODS:
        data[method["key"]] = np.load(DATA_DIR / method["file"])
    return data


def figure1_input_groundtruth(data):
    """
    Figure 1: Input (with ring artifacts) + Ground Truth (clean)
    Two large sinogram images side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)

    # Determine common color scale
    vmin = min(data["input"].min(), data["ground_truth"].min())
    vmax = max(data["input"].max(), data["ground_truth"].max())

    # (a) Input with artifacts
    ax = axes[0]
    im = ax.imshow(data["input"], cmap="gray", aspect=1, vmin=vmin, vmax=vmax)
    ax.set_title("(a) Input (with ring artifacts)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Detector channel", fontsize=LABEL_SIZE)
    ax.set_ylabel("Projection angle", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Draw crop region indicator
    rect = Rectangle(
        (CROP_X.start, CROP_Y.start),
        CROP_X.stop - CROP_X.start,
        CROP_Y.stop - CROP_Y.start,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
        linestyle="--"
    )
    ax.add_patch(rect)

    # (b) Ground truth
    ax = axes[1]
    ax.imshow(data["ground_truth"], cmap="gray", aspect=1, vmin=vmin, vmax=vmax)
    ax.set_title("(b) Ground Truth (clean)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Detector channel", fontsize=LABEL_SIZE)
    ax.set_ylabel("Projection angle", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Draw crop region indicator
    rect = Rectangle(
        (CROP_X.start, CROP_Y.start),
        CROP_X.stop - CROP_X.start,
        CROP_Y.stop - CROP_Y.start,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
        linestyle="--"
    )
    ax.add_patch(rect)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "figure1_input.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def figure2_results_diff(data):
    """
    Figure 2: Method results (cropped regions) + Difference images

    Layout: 6 columns × 2 rows + colorbar column
    Row 1: Cropped results for each method
    Row 2: Difference from ground truth
    Colorbar: spans row 2 only (applies to diff images)
    """
    # Extract cropped ground truth for diff calculation
    gt_crop = data["ground_truth"][CROP_Y, CROP_X]

    # Prepare cropped data
    crops = []
    diffs = []
    for method in METHODS:
        result = data[method["key"]]
        crop = result[CROP_Y, CROP_X]
        crops.append(crop)
        diffs.append(crop - gt_crop)

    # Determine color scales
    crop_vmin = min(c.min() for c in crops)
    crop_vmax = max(c.max() for c in crops)

    diff_absmax = max(abs(d).max() for d in diffs)
    diff_vmin, diff_vmax = -diff_absmax, diff_absmax

    # Create figure with GridSpec
    # 6 image columns + 1 narrow colorbar column
    # Crop is 250x250 pixels, so aspect is 1:1
    # Figure width: 6 images + colorbar, height: 2 rows
    fig = plt.figure(figsize=(13, 4.5), dpi=DPI)
    gs = GridSpec(
        2, 7,  # 2 rows, 7 columns (6 images + 1 colorbar)
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.05],  # colorbar is narrow
        height_ratios=[1, 1],
        wspace=0.03,
        hspace=0.08
    )

    # Row 1: Cropped results
    for i, (method, crop) in enumerate(zip(METHODS, crops)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(crop, cmap="gray", aspect=1, vmin=crop_vmin, vmax=crop_vmax)

        # Title with method name and time
        time_str = f"{METRICS[method['key']]['time']:.2f}s"
        title = f"{method['name']}\n{time_str}"
        ax.set_title(title, fontsize=ANNOTATION_SIZE, color=method["color"], fontweight="bold")

        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2: Difference images
    diff_im = None
    for i, (method, diff) in enumerate(zip(METHODS, diffs)):
        ax = fig.add_subplot(gs[1, i])
        diff_im = ax.imshow(diff, cmap="RdBu_r", aspect=1, vmin=diff_vmin, vmax=diff_vmax)

        ax.set_xticks([])
        ax.set_yticks([])

        # Set ylabel only for the first diff image
        if i == 0:
            ax.set_ylabel("Diff", fontsize=ANNOTATION_SIZE)

    # Colorbar spanning row 2 only
    cbar_ax = fig.add_subplot(gs[1, 6])
    cbar = fig.colorbar(diff_im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    output_path = OUTPUT_DIR / "figure2_results.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def figure3_metrics():
    """
    Figure 3: Metrics
    (a) Processing time bar chart (linear scale)
    (b) PSNR vs SSIM scatter plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)

    # (a) Processing time bar chart
    ax = axes[0]

    names = [m["name"].replace("\n", " ") for m in METHODS]
    times = [METRICS[m["key"]]["time"] for m in METHODS]
    stds = [METRICS[m["key"]]["std"] for m in METHODS]
    colors = [m["color"] for m in METHODS]

    bars = ax.bar(range(len(METHODS)), times, yerr=stds, color=colors,
                  capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(names, fontsize=TICK_SIZE, rotation=45, ha="right")
    ax.set_ylabel("Processing Time (s)", fontsize=LABEL_SIZE)
    ax.set_title("(a) Processing Time (n=30 runs)", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(
            f"{time:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE - 1
        )

    ax.set_ylim(0, max(times) * 1.15)  # Add headroom for labels

    # (b) PSNR vs SSIM scatter plot
    ax = axes[1]

    for method in METHODS:
        key = method["key"]
        psnr = METRICS[key]["psnr"]
        ssim = METRICS[key]["ssim"]

        ax.scatter(
            psnr, ssim,
            c=method["color"],
            marker=MARKERS[key],
            s=120,
            edgecolors="black",
            linewidths=0.5,
            label=method["name"].replace("\n", " "),
            zorder=3
        )

    ax.set_xlabel("PSNR (dB)", fontsize=LABEL_SIZE)
    ax.set_ylabel("SSIM", fontsize=LABEL_SIZE)
    ax.set_title("(b) Quality Metrics", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, zorder=0)

    # Legend outside plot
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=TICK_SIZE,
        framealpha=0.9
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "figure3_metrics.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate all figures."""
    print("Loading data...")
    data = load_data()
    print(f"  Input shape: {data['input'].shape}")
    print(f"  Ground truth shape: {data['ground_truth'].shape}")

    print("\nGenerating figures...")

    # Set matplotlib defaults
    plt.rcParams.update({
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "figure.dpi": DPI,
    })

    figure1_input_groundtruth(data)
    figure2_results_diff(data)
    figure3_metrics()

    print("\nDone! Generated figures:")
    print("  - paper/figure1_input.png")
    print("  - paper/figure2_results.png")
    print("  - paper/figure3_metrics.png")


if __name__ == "__main__":
    main()
