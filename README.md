<!-- Badges -->

[![Build Status](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/test.yml/badge.svg?branch=next)](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/test.yml?query=branch?next)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11811/badge)](https://www.bestpractices.dev/projects/11811)
[![Documentation Status](https://readthedocs.org/projects/bm3dornl/badge/?version=latest)](https://bm3dornl.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/bm3dornl)](https://pypi.org/project/bm3dornl/)
[![Crates.io](https://img.shields.io/crates/v/bm3d_core)](https://crates.io/crates/bm3d_core)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18276016.svg)](https://doi.org/10.5281/zenodo.18276016)

<!-- End Badges -->
BM3D ORNL
=========

BM3D ORNL removes streak and ring artifacts from neutron-imaging sinograms.
BM3D means block-matching and three-dimensional filtering.

The BM3D algorithm was originally proposed by K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian in the paper "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering" (2007).

The library provides a Python API backed by Rust. Key features include:

- **Streak removal** targets vertical detector artifacts.
- **Multiscale processing** handles streaks wider than one scale can capture.
- **Fourier-SVD** combines frequency detection with rank-one matrix decomposition.
- **Stack processing** batches three-dimensional sinogram data.
- **Rust processing** accelerates block matching and transforms.

Documentation
-------------

Full documentation: <https://bm3dornl.readthedocs.io>

It covers the installation guide, a tutorial, the parameter reference, the GUI
guide, and the API reference.

How to install
--------------

Use pip for published releases and [Pixi](https://prefix.dev) for a clone.

Do not run `pip install -e .` in a clone. `pip install -e .` is unsupported
because clone builds compile the Rust extension with Maturin inside the pinned
Pixi environment. The `pixi run build` task performs that step.

The `[gui]` extra installs the separate `bm3dornl-gui` binary wheel from PyPI.
It does not build the GUI in a clone. Run the clone's GUI with `pixi run gui`.

**Using Pip (released packages)**

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

For an alternative streak-removal method:

```python
from bm3dornl.fourier_svd import fourier_svd_removal

# Fourier-guided streak removal
denoised = fourier_svd_removal(
    sinogram,
    fft_alpha=1.0,          # Weight for detected vertical-frequency energy
    notch_width=2.0,        # Gaussian notch width in frequency bins
)
```

Performance
-----------

Processing time for one 720×725 sinogram, mean of 100 runs (measured
2026-08-31 and 2026-09-02):

| Method | Linux x86_64, 24-core Threadripper | Apple M2 Max laptop |
|--------|------------------------------------|---------------------|
| bm3dornl, streak mode | 0.094 s | 0.20 s |
| bm3dornl, Fourier-SVD | 0.017 s | 0.022 s |
| bm3d-streak-removal (reference, x86_64 only) | 31.8 s | not available |

The Rust backend runs in parallel across cores with Rayon. The full method
comparison is in `notebooks/evaluation_performance.ipynb`.

Key optimizations:
- Integral image pre-screening for fast block matching
- Early termination in distance calculations
- Pre-computed FFT plans
- Fast Walsh-Hadamard transform for 8×8 patches

Development
-----------

Use Pixi for every command in a clone. See [How to install](#how-to-install)
for the install policy. Pixi supplies Python, Rust, HDF5, and the remaining
build dependencies.

| Task | Command |
|------|---------|
| Create the environment without running a task (optional) | `pixi install` |
| Build the Rust extension | `pixi run build` |
| Run all tests (Rust and Python) | `pixi run test` |
| Lint (rustfmt and clippy) | `pixi run lint` |
| Run the clone's GUI | `pixi run gui` (release build) or `pixi run gui-debug` |
| Run the benchmarks | `pixi run bench` |

Clone the repository, enter its directory, then run `pixi run build`,
`pixi run test`, and `pixi run gui`, in that order.

```bash
git clone https://github.com/ornlneutronimaging/bm3dornl.git
cd bm3dornl
pixi run build
pixi run test
pixi run gui
```

Each `pixi run` command installs the environment on demand. Run `pixi install`
first only when you want to create the environment separately.

The first `pixi run gui` compiles the GUI application, which takes a few
minutes; later runs start immediately.

### Optional: test-data submodule

`tests/bm3dornl-data` is a Git submodule with notebook reference data. It is
not required to build the package or run tests. The clone command above is
enough for `pixi run build` and `pixi run test`.

Fetch it only if you want to run the notebooks:

```bash
git submodule update --init tests/bm3dornl-data
git -C tests/bm3dornl-data lfs pull
```

The submodule is hosted on `code.ornl.gov`. It clones anonymously over HTTPS,
without an ORNL account or SSH key. The `tomostack_small.h5` file is about
1.4 GB. It uses Git Large File Storage (Git LFS). Without Git LFS, the file
remains a small text pointer. The `lfs pull` step can take time.

GUI Application
---------------

BM3DORNL includes a standalone GUI application for interactive ring artifact removal.

### Installation

Released binaries (Linux x86_64 and macOS Apple Silicon):

```bash
pip install bm3dornl[gui]
```

Or install the GUI separately:

```bash
pip install bm3dornl-gui
```

In a clone, run the GUI through Pixi. See [How to install](#how-to-install).

```bash
pixi run gui
```

### Launching

The released binary installs a `bm3dornl-gui` command:

```bash
bm3dornl-gui
```

### Features

- Load HDF5 files with tree browser for dataset selection
- Interactive slice viewer with histogram
- Side-by-side comparison of original and processed images
- Adjust parameters before processing
- View the processed result after processing finishes
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
| `step_size` | `4` | Step size for patch extraction |
| `search_window` | `24` | Maximum search distance for similar patches |
| `max_matches` | `16` | Maximum similar patches per 3D group |
| `batch_size` | `32` | Batch size for stack processing |
| `streak_sigma_smooth` | `3.0` | Smoothing for streak mode (streak mode only) |
| `multiscale` | `False` | Enable multi-scale processing for wide streaks |
| `num_scales` | `None` | Number of scales (`None` selects it automatically) |
| `filter_strength` | `1.0` | Filtering strength multiplier for multi-scale |
| `debin_iterations` | `30` | Cubic-spline expansion iterations for multiscale corrections |

### Fourier-SVD Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_alpha` | `1.0` | Weight for detected vertical-frequency energy |
| `notch_width` | `2.0` | Gaussian notch width in frequency bins |

Contributing and Support
------------------------

- **Contributing**: see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding
  standards, testing, and pull requests. Participation follows our
  [Code of Conduct](CODE_OF_CONDUCT.md).
- **Reporting issues**: open an issue at
  <https://github.com/ornlneutronimaging/bm3dornl/issues>. Please include your
  platform, the bm3dornl version, and a minimal reproduction.
- **Getting support**: for usage questions, start a
  [GitHub Discussion](https://github.com/ornlneutronimaging/bm3dornl/discussions)
  or email the maintainer at <zhangc@ornl.gov>.
