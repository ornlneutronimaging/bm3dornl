<!-- Badges -->

[![Build Status](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/test.yml/badge.svg?branch=next)](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/test.yml?query=branch?next)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11811/badge)](https://www.bestpractices.dev/projects/11811)
[![Documentation Status](https://readthedocs.org/projects/bm3dornl/badge/?version=latest)](https://bm3dornl.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/bm3dornl)](https://pypi.org/project/bm3dornl/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18276016.svg)](https://doi.org/10.5281/zenodo.18276016)

<!-- End Badges -->
BM3D ORNL
=========

A high-performance BM3D denoising library for neutron imaging, optimized for streak/ring artifact removal from sinograms.

The BM3D algorithm was originally proposed by K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian in the paper "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering" (2007).

**BM3D ORNL** provides a Python API with a **Rust backend** for efficient, parallel processing of tomography data. Key features:

- **Streak/Ring Artifact Removal**: Specialized mode for removing vertical streak artifacts common in neutron and X-ray imaging
- **Multi-Scale Processing**: True multi-scale BM3D for handling wide streaks that single-scale cannot capture (based on Mäkinen et al. 2021)
- **Fourier-SVD Method**: Alternative fast algorithm (~2.6x faster than BM3D) combining FFT-based energy detection with rank-1 SVD
- **Stack Processing**: Efficient batched processing of 3D sinogram stacks
- **High Performance**: Rust backend with optimized block matching (integral images, early termination) and transforms (Hadamard, FFT)

How to install
--------------

**Using Pip**

```bash
# Core library only
pip install bm3dornl

# With GUI application
pip install bm3dornl[gui]
```

**Supported Platforms**

| Platform | Architecture | Library | GUI |
|----------|--------------|---------|-----|
| Linux | x86_64 | ✅ | ✅ |
| macOS | ARM64 (Apple Silicon) | ✅ | ✅ |

**Using Pixi (Development)**

```bash
pixi install
pixi run build
```

Usage
-----

```python
from bm3dornl import bm3d_ring_artifact_removal
import numpy as np

# Load sinogram data - 2D (H, W) or 3D stack (N, H, W)
sinogram = np.load("sinogram.npy")

# Standard BM3D denoising (generic white noise)
denoised = bm3d_ring_artifact_removal(sinogram, mode="generic", sigma_random=0.1)

# Streak artifact removal (recommended for ring artifacts)
denoised = bm3d_ring_artifact_removal(sinogram, mode="streak", sigma_random=0.1)

# With custom parameters (all parameters are flat, no dict wrapping)
denoised = bm3d_ring_artifact_removal(
    sinogram,
    mode="streak",
    sigma_random=0.1,
    patch_size=8,           # Patch size (7 or 8 recommended)
    step_size=4,            # Step size for patch extraction
    search_window=40,       # Max search distance
    max_matches=64,         # Similar patches per 3D group
    batch_size=32,          # Batch size for stack processing
)

# Multi-scale BM3D for wide streaks (v0.7.0+)
denoised = bm3d_ring_artifact_removal(
    sinogram,
    mode="streak",
    multiscale=True,        # Enable multi-scale pyramid processing
    num_scales=None,        # Auto-detect (or set explicitly)
    filter_strength=1.0,    # Filtering intensity multiplier
)
```

### Fourier-SVD Method (v0.7.0+)

For faster processing with excellent results on many datasets:

```python
from bm3dornl.fourier_svd import fourier_svd_removal

# Fast streak removal (~2.6x faster than BM3D)
denoised = fourier_svd_removal(
    sinogram,
    fft_alpha=1.0,          # FFT-guided trust factor (0.0 disables FFT guidance)
    notch_width=2.0,        # Gaussian notch filter width
)
```

Performance
-----------

The **Rust backend** provides high performance for tomography stacks:

| Metric | Value |
|--------|-------|
| **Speed** | ~0.63s per frame (512×512) on Apple Silicon |
| **Memory** | >50% reduction via chunked processing |
| **Parallelism** | Zero-overhead parallel processing via Rayon |

Key optimizations:
- Integral image pre-screening for fast block matching
- Early termination in distance calculations
- Pre-computed FFT plans
- Fast Walsh-Hadamard transform for 8×8 patches

Development
-----------

We use [pixi](https://prefix.dev) for development environment management.

1.  Clone repo.
2.  Run `pixi run build` to compile the Rust backend and install in editable mode.
3.  Run `pixi run test` to run tests.
4.  Run `pixi run bench` to run performance benchmarks.

```bash
git clone https://github.com/ornlneutronimaging/bm3dornl.git
cd bm3dornl
pixi run build
pixi run test
```

GUI Application
---------------

BM3DORNL includes a standalone GUI application for interactive ring artifact removal.

### Installation

```bash
pip install bm3dornl[gui]
```

Or install the GUI separately:

```bash
pip install bm3dornl-gui
```

### Launching

```bash
bm3dornl-gui
```

### Features

- Load HDF5 files with tree browser for dataset selection
- Interactive slice viewer with histogram
- Side-by-side comparison of original and processed images
- Real-time parameter adjustment
- ROI selection for histogram (Shift+drag to select region)
- Export processed data to TIFF or HDF5

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Shift+Drag | Select ROI for histogram |
| Scroll | Zoom in/out on image |
| Drag | Pan image |

Parameter Reference
-------------------

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"streak"` | `"generic"` for white noise, `"streak"` for ring artifacts |
| `sigma_random` | `0.1` | Noise standard deviation |
| `patch_size` | `8` | Patch size (7 or 8 recommended) |
| `step_size` | `3` | Step size for patch extraction |
| `search_window` | `39` | Maximum search distance for similar patches |
| `max_matches` | `16` | Maximum similar patches per 3D group |
| `batch_size` | `32` | Batch size for stack processing |
| `streak_sigma_smooth` | `1.0` | Smoothing for streak mode (streak mode only) |
| `multiscale` | `False` | Enable multi-scale processing for wide streaks |
| `num_scales` | `None` | Number of scales (auto-detected if None) |
| `filter_strength` | `1.0` | Filtering strength multiplier for multi-scale |
| `debin_iterations` | `30` | Debinning iterations for multi-scale |

### Fourier-SVD Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_alpha` | `1.0` | FFT-guided trust factor (0.0 disables FFT guidance) |
| `notch_width` | `2.0` | Gaussian notch filter width in frequency domain |
