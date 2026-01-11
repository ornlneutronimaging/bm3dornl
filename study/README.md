# bm3dornl JOSS Paper - Benchmark Comparison Study

This directory contains an isolated benchmark environment for comparing bm3dornl against existing ring artifact removal methods. The results support the "State of the Field" section in the JOSS paper.

## Multi-Platform Benchmark Strategy

Due to platform-specific compatibility issues with some packages, benchmarks are run on multiple platforms:

| Platform | bm3dornl | TomoPy | bm3d-streak-removal |
|----------|----------|--------|---------------------|
| macOS Apple Silicon | Yes | Yes | No (bm4d x86_64 only) |
| Linux x86_64 | Yes | Yes | Yes |

Results are stored separately and consolidated for the paper.

## Methods Compared

**bm3dornl (this package):**

- **bm3dornl (streak mode)** - Specialized for ring/streak artifacts
- **bm3dornl (generic mode)** - Standard BM3D denoising

**TomoPy methods:**

- **remove_stripe_fw** - Wavelet-Fourier method (Münch et al.)
- **remove_stripe_sf** - Sorting-fitting method (Vo et al.)
- **remove_stripe_based_sorting** - Sorting-based stripe removal

**bm3d-streak-removal (Linux only):**

- Mäkinen et al., 2021 - Multiscale BM3D for streak noise

## Setup

```bash
cd study
pixi install
```

## Running Benchmarks

```bash
pixi run benchmark
```

The script auto-detects the platform and saves results to the appropriate subfolder:

- `results/apple_silicon/` - macOS arm64 results
- `results/linux_x86_64/` - Linux x86_64 results

## Results Structure

```
study/
├── results/
│   ├── apple_silicon/
│   │   ├── results.csv
│   │   └── figures/
│   │       ├── comparison_grid.png
│   │       ├── timing_comparison.png
│   │       └── quality_metrics.png
│   └── linux_x86_64/
│       └── .gitkeep  (placeholder until Linux benchmarks run)
├── joss_comparison.py
├── pixi.toml
└── README.md
```

## Test Data

Synthetic data generated with bm3dornl's phantom module:

1. Shepp-Logan phantom (256x256)
2. Radon transform to generate sinogram (720x363)
3. Simulated detector gain errors (ring artifacts)

## Apple Silicon Results (Complete)

| Method | Time (s) | PSNR (dB) | SSIM | Visual Quality |
|--------|----------|-----------|------|----------------|
| bm3dornl (streak) | 0.135 | 30.43 | 0.5955 | Stripes removed |
| bm3dornl (generic) | 0.143 | 32.17 | 0.5724 | Stripes removed |
| TomoPy FW (Münch) | 2.049 | 16.03 | 0.5293 | Failed |
| TomoPy SF (Vo) | 1.994 | 36.58 | 0.9521 | Stripes visible |
| TomoPy BSD (sort) | 2.020 | 34.88 | 0.9349 | Stripes visible |

**Key observations:**

- bm3dornl is ~15x faster than TomoPy methods
- PSNR/SSIM metrics can be misleading - TomoPy methods show higher metrics but visual inspection reveals stripes remain
- bm3dornl effectively removes ring artifacts while TomoPy methods largely fail on this synthetic data

## Running on Linux

On a Linux x86_64 system:

```bash
git pull
cd study
pixi install
pixi run benchmark
```

The benchmark will automatically:

1. Detect Linux x86_64 platform
2. Include bm3d-streak-removal in the comparison
3. Save results to `results/linux_x86_64/`

## bm3d-streak-removal Compatibility Notes

The `bm3d-streak-removal` package has several compatibility constraints:

1. **scipy version**: Requires scipy < 1.11 (uses deprecated `scipy.signal.gaussian`)
2. **Architecture**: bm4d library only provides x86_64 binaries
3. **Maintenance**: No releases since 2022

These issues are handled by:

- Separate pixi environment in `bm3d_streak_test/` for isolated testing
- Platform detection in main benchmark script
- Graceful fallback when package unavailable

## Parameters Used

bm3dornl parameters (tuned for this synthetic data):

```python
sigma_random=0.05
patch_size=8
step_size=4
search_window=24
max_matches=16
```

Note: `sigma_random` controls denoising strength. Values tested: 0.005, 0.01, 0.05, 0.1. The value 0.05 provided best balance between artifact removal and detail preservation.
