# bm3d_core

[![Crates.io](https://img.shields.io/crates/v/bm3d_core.svg)](https://crates.io/crates/bm3d_core)
[![Documentation](https://docs.rs/bm3d_core/badge.svg)](https://docs.rs/bm3d_core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This pure Rust crate implements block-matching and three-dimensional filtering
(BM3D). It includes processing for streak and ring artifacts in tomography data.

## Features

- **Generic float support:** Use `f32` or `f64` through the `Bm3dFloat` trait.
- **Streak removal:** Target vertical artifacts in neutron and X-ray imaging.
- **Multiscale processing:** Process streaks wider than one scale can capture.
- **Fourier-SVD method:** Combine frequency detection with rank-one matrix decomposition.
- **Parallel processing:** Use Rayon and integral-image block matching.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
bm3d_core = "0.10.0"
```

## Quick Start

```rust
use bm3d_core::{bm3d_ring_artifact_removal, Bm3dConfig, RingRemovalMode};
use ndarray::Array2;

// Create a noisy 2D image (H x W)
let image: Array2<f32> = /* your image data */;

// Configure BM3D for streak removal
let config = Bm3dConfig {
    sigma_random: 0.1,
    patch_size: 8,
    step_size: 4,
    search_window: 24,
    max_matches: 16,
    ..Default::default()
};

// Run denoising
let denoised = bm3d_ring_artifact_removal(image.view(), RingRemovalMode::Streak, &config);
```

## Main API

### High-Level Functions

- `bm3d_ring_artifact_removal` - Main entry point for ring and streak removal
- `multiscale_bm3d_streak_removal` - Multiscale processing for wide streaks

### Configuration

- `Bm3dConfig` - BM3D parameter configuration
- `RingRemovalMode` - `Generic` for white noise or `Streak` for directional artifacts
- `MultiscaleConfig` - Multiscale parameter configuration

### Low-Level Components

- `run_bm3d_kernel` - BM3D kernel for one image
- `run_bm3d_step` - One hard-threshold or Wiener-filtering step
- `estimate_noise_sigma` - Vertical-streak noise estimate used when `sigma_random` is at or below `1e-6`, for example `0.0`

## Performance

The implementation uses:
- Integral image pre-screening for fast block matching
- Early termination in distance calculations
- Pre-computed FFT plans (`Bm3dPlans`)
- Fast Walsh-Hadamard transform for 8×8 patches
- Parallel processing through Rayon

## References

- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). Image denoising by sparse 3D transform-domain collaborative filtering. *IEEE TIP*.
- Mäkinen, Y., et al. (2021). Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching.

## License

The crate uses the MIT License. See
[LICENSE](https://github.com/ornlneutronimaging/bm3dornl/blob/main/LICENSE).

## Related

This crate belongs to the
[bm3dornl](https://github.com/ornlneutronimaging/bm3dornl) project. The project
also provides:
- Python bindings via PyO3
- GUI application for interactive denoising
