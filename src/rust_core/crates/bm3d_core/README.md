# bm3d_core

[![Crates.io](https://img.shields.io/crates/v/bm3d_core.svg)](https://crates.io/crates/bm3d_core)
[![Documentation](https://docs.rs/bm3d_core/badge.svg)](https://docs.rs/bm3d_core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Rust implementation of the BM3D (Block-Matching and 3D filtering) denoising algorithm, optimized for streak/ring artifact removal in tomographic imaging.

## Features

- **Generic Float Support**: Works with both `f32` and `f64` precision via the `Bm3dFloat` trait
- **Streak Artifact Removal**: Specialized mode for vertical streak artifacts common in neutron and X-ray imaging
- **Multi-Scale Processing**: Pyramid-based processing for wide streaks that single-scale cannot capture
- **Fourier-SVD Method**: Fast alternative algorithm combining FFT-based detection with rank-1 SVD
- **High Performance**: Parallelized with Rayon, optimized block matching with integral images

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
bm3d_core = "0.7"
```

## Quick Start

```rust
use bm3d_core::{bm3d_ring_artifact_removal, Bm3dConfig, RingRemovalMode};
use ndarray::Array2;

// Create a noisy 2D image (H x W)
let image: Array2<f32> = /* your image data */;

// Configure BM3D for streak removal
let config = Bm3dConfig {
    mode: RingRemovalMode::Streak,
    sigma_random: 0.1,
    patch_size: 8,
    step_size: 3,
    search_window: 39,
    max_matches: 16,
    ..Default::default()
};

// Run denoising
let denoised = bm3d_ring_artifact_removal(&image, &config);
```

## Main API

### High-Level Functions

- `bm3d_ring_artifact_removal` - Main entry point for ring/streak artifact removal
- `multiscale_bm3d_streak_removal` - Multi-scale processing for wide streaks

### Configuration

- `Bm3dConfig` - Configuration struct for BM3D parameters
- `RingRemovalMode` - `Generic` (white noise) or `Streak` (directional artifacts)
- `MultiscaleConfig` - Configuration for multi-scale processing

### Low-Level Components

- `run_bm3d_kernel` - Core BM3D kernel for a single image
- `run_bm3d_step` - Single BM3D step (hard/wiener threshold)
- `estimate_noise_sigma` - Automatic noise level estimation

## Performance

Optimizations include:
- Integral image pre-screening for fast block matching
- Early termination in distance calculations
- Pre-computed FFT plans (`Bm3dPlans`)
- Fast Walsh-Hadamard transform for 8×8 patches
- Zero-overhead parallelism via Rayon

## References

- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). Image denoising by sparse 3D transform-domain collaborative filtering. *IEEE TIP*.
- Mäkinen, Y., et al. (2021). Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching.

## License

MIT License - see [LICENSE](https://github.com/ornlneutronimaging/bm3dornl/blob/main/LICENSE) for details.

## Related

This crate is part of the [bm3dornl](https://github.com/ornlneutronimaging/bm3dornl) project, which also provides:
- Python bindings via PyO3
- GUI application for interactive denoising
