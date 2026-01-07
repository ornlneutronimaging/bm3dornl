<!-- Badges -->

[![Build Status](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/unittest.yml/badge.svg?branch=next)](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/unittest.yml?query=branch?next)
[![codecov](https://codecov.io/gh/ornlneutronimaging/bm3dornl/branch/next/graph/badge.svg)](https://codecov.io/gh/ornlneutronimaging/bm3dornl/tree/next)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6650/badge)](https://bestpractices.coreinfrastructure.org/projects/6650)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ornlneutronimaging/bm3dornl/next.svg)](https://results.pre-commit.ci/latest/github/ornlneutronimaging/bm3dornl/next)
[![Documentation Status](https://readthedocs.org/projects/bm3dornl/badge/?version=latest)](https://bm3dornl.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/neutronimaging/bm3dornl/badges/version.svg)](https://anaconda.org/neutronimaging/bm3dornl)

<!-- End Badges -->
BM3D ORNL
=========

A high-performance BM3D denoising library for neutron imaging, optimized for streak/ring artifact removal from sinograms.

The BM3D algorithm was originally proposed by K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian in the paper "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering" (2007).

**BM3D ORNL** provides a Python API with a **Rust backend** for efficient, parallel processing of tomography data. Key features:

- **Streak/Ring Artifact Removal**: Specialized mode for removing vertical streak artifacts common in neutron and X-ray imaging
- **Stack Processing**: Efficient batched processing of 3D sinogram stacks
- **High Performance**: Rust backend with optimized block matching (integral images, early termination) and transforms (Hadamard, FFT)

How to install
--------------

**Using Pixi (Recommended)**

```bash
pixi install
pixi run install
```

**Using Pip**

```bash
pip install bm3dornl
```

Usage
-----

```python
from bm3dornl import bm3d_ring_artifact_removal
import numpy as np

# Load sinogram data - 2D (H, W) or 3D stack (N, H, W)
sinogram = np.load("sinogram.npy")

# Standard BM3D denoising (generic white noise)
denoised = bm3d_ring_artifact_removal(sinogram, mode="generic", sigma=0.1)

# Streak artifact removal (recommended for ring artifacts)
denoised = bm3d_ring_artifact_removal(sinogram, mode="streak", sigma=0.1)

# With custom parameters
denoised = bm3d_ring_artifact_removal(
    sinogram,
    mode="streak",
    sigma=0.1,
    block_matching_kwargs={
        "patch_size": 8,        # Patch size (7 or 8 recommended)
        "stride": 2,            # Step size for patch extraction
        "cut_off_distance": (40, 40),  # Max search distance
        "num_patches_per_group": 64,   # Patches per 3D group
        "batch_size": 32,       # Batch size for stack processing
    }
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

