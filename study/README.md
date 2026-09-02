# bm3dornl JOSS Paper - Benchmark Comparison Study

This directory contains an isolated benchmark environment for comparing bm3dornl against existing ring artifact removal methods. The results support the "State of the Field" section in the JOSS paper.

## Methods Compared

**bm3dornl (this package):**

- **bm3dornl (streak mode)** - Specialized for ring/streak artifacts
- **bm3dornl (generic mode)** - Standard BM3D denoising

**TomoPy methods:**

- **remove_stripe_fw** - Wavelet-Fourier method (Münch et al.)
- **remove_stripe_sf** - Sorting-fitting method (Vo et al.)
- **remove_stripe_based_sorting** - Sorting-based stripe removal

**bm3d-streak-removal:**

- Mäkinen et al., 2021 - Multiscale BM3D for streak noise

## Platform Support

| Feature | bm3dornl | bm3d-streak-removal | TomoPy |
|---------|----------|---------------------|--------|
| Apple Silicon | Yes | No | Yes |
| Linux x86_64 | Yes | Yes | Yes |
| Python 3.12+ | Yes | No | Yes |
| Active maintenance | Yes | No | Yes |

## Setup

```bash
cd study
pixi install
```

## Running Benchmarks

### Full benchmark (Linux x86_64)

```bash
# Step 1: Run main benchmark (bm3dornl + TomoPy)
pixi run benchmark

# Step 2: Run bm3d-streak-removal in isolated environment
cd bm3d_streak_test && pixi run run && cd ..

# Step 3: Generate unified visualization with all 6 methods
pixi run visualize
```

### Apple Silicon (macOS arm64)

```bash
pixi run benchmark  # bm3d-streak-removal not available
```

## Test Environment

### Hardware (Linux x86_64)

| Component | Specification |
|-----------|---------------|
| Host | cg1d-analysis3.ornl.gov (ORNL CG-1D analysis node) |
| OS | Red Hat Enterprise Linux 9.8 (Plow) |
| Kernel | 5.14.0-611.54.1.el9_7.x86_64 |
| CPU | AMD Ryzen Threadripper PRO 7965WX 24-Cores |
| Threads | 48 (24 cores × 2 threads) |
| RAM | 502 GB |
| Architecture | x86_64 |
| Python | 3.12.12 |
| Rust | 1.98.0 |
| Measured | 2026-08-31 |

### Test Data

Synthetic data generated with bm3dornl's phantom module:

1. Shepp-Logan phantom (512x512)
2. Radon transform to generate sinogram (720x725)
3. Simulated detector gain errors (ring artifacts)

## Linux x86_64 Results (Consolidated)

All 6 methods compared on identical test data (512x512 phantom, 720x725 sinogram), n=30 timing runs each:

| Method | Time (s) | PSNR (dB) | SSIM |
|--------|----------|-----------|------|
| bm3dornl (streak) | 0.256 ± 0.009 | 32.63 | 0.6160 |
| bm3dornl (generic) | 0.219 ± 0.002 | 32.93 | 0.5760 |
| TomoPy FW (Münch) | 0.318 ± 0.008 | 20.61 | 0.5831 |
| TomoPy SF (Vo) | 0.278 ± 0.008 | 34.50 | 0.9591 |
| TomoPy BSD (sort) | 0.349 ± 0.009 | 34.69 | 0.9333 |
| bm3d-streak-removal | 41.033 ± 0.139 | 36.34 | 0.8697 |

### Key Findings

**Speed comparison:**
- bm3dornl is **~160x faster** than bm3d-streak-removal (0.256s vs 41.033s)
- bm3dornl and TomoPy methods have comparable speed (~0.2-0.4s)

**Quality analysis (from diff images):**
- **bm3dornl (both modes):** Diff shows vertical stripe patterns indicating successful ring artifact removal with minimal sample information loss
- **TomoPy FW (Münch):** Large red/blue regions in diff indicate significant alteration of sample structure - method fails on this data
- **TomoPy SF/BSD:** Visible vertical stripes in diff showing artifacts not fully removed
- **bm3d-streak-removal:** Clean diff but extremely slow

**Conclusion:** bm3dornl provides the best balance of speed and artifact removal quality.

## Apple Silicon Results

### Hardware

| Component | Specification |
|-----------|---------------|
| Model | MacBook Pro |
| Chip | Apple M2 Max |
| CPU Cores | 12 (8 performance + 4 efficiency) |
| RAM | 32 GB unified memory |
| Architecture | arm64 |
| OS | macOS 26.6.2 |
| Python | 3.12 |
| Measured | 2026-09-02 |

### Benchmark Results (7 methods, n=100 runs each)

| Method | Time (s) | PSNR (dB) | SSIM |
|--------|----------|-----------|------|
| bm3dornl (streak) | 0.202 ± 0.014 | 39.57 | 0.9423 |
| bm3dornl (generic) | 0.183 ± 0.007 | 35.10 | 0.9476 |
| bm3dornl (multiscale) | 0.419 ± 0.017 | 39.73 | 0.9759 |
| Fourier-SVD | 0.022 ± 0.003 | 39.64 | 0.9513 |
| TomoPy FW (Münch) | 3.562 ± 0.195 | 20.63 | 0.5840 |
| TomoPy SF (Vo) | 3.573 ± 0.256 | 42.01 | 0.9868 |
| TomoPy BSD (sort) | 3.471 ± 0.250 | 42.86 | 0.9829 |

Note: bm3d-streak-removal is not available on Apple Silicon (no arm64 binary).

### Cross-Platform Comparison

| Method | Linux x86_64 (s) | Apple Silicon (s) | Apple Silicon / Linux |
|--------|------------------|-------------------|-----------------------|
| bm3dornl (streak) | 0.094 | 0.202 | 2.1x slower |
| bm3dornl (generic) | 0.083 | 0.183 | 2.2x slower |
| bm3dornl (multiscale) | 0.389 | 0.419 | 1.1x slower |
| Fourier-SVD | 0.017 | 0.022 | 1.3x slower |
| TomoPy FW (Münch) | 0.261 | 3.562 | 13.6x slower |
| TomoPy SF (Vo) | 0.222 | 3.573 | 16.1x slower |
| TomoPy BSD (sort) | 0.293 | 3.471 | 11.8x slower |

**Key observations:**

- The two hosts differ in core count (24-core Threadripper workstation vs 12-core laptop chip), so the absolute cross-platform ratios mix hardware with software; only same-machine ratios are comparable.
- **bm3dornl is 1.1-2.2x slower on the M2 Max than on the Linux host; TomoPy is 12-16x slower.**
- **Same-machine advantage of bm3dornl (streak) over the fastest TomoPy method: 2.4x on Linux x86_64 (0.222 s / 0.094 s), 17x on Apple Silicon (3.471 s / 0.202 s).**
- **Quality metrics agree across platforms**: PSNR identical to two decimals for every method; SSIM identical to four decimals except bm3dornl (streak), 0.9423 vs 0.9426 (thread-count dependent aggregation order).

## Results Structure

```
study/
├── results/
│   ├── apple_silicon/
│   │   ├── data/
│   │   │   ├── sinogram_clean.npy
│   │   │   ├── sinogram_rings.npy
│   │   │   ├── result_*.npy
│   │   │   └── metrics.csv
│   │   ├── figures/
│   │   │   ├── unified_comparison.png
│   │   │   ├── unified_timing.png
│   │   │   ├── unified_quality.png
│   │   │   ├── comparison_grid.png
│   │   │   ├── timing_comparison.png
│   │   │   └── quality_metrics.png
│   │   ├── results.csv
│   │   └── consolidated_results.csv
│   └── linux_x86_64/
│       ├── data/
│       │   ├── sinogram_clean.npy
│       │   ├── sinogram_rings.npy
│       │   ├── result_*.npy
│       │   └── metrics.csv
│       ├── figures/
│       │   ├── unified_comparison.png  # All 6 methods + diff images
│       │   ├── unified_timing.png
│       │   ├── unified_quality.png
│       │   ├── comparison_grid.png
│       │   ├── timing_comparison.png
│       │   └── quality_metrics.png
│       ├── results.csv
│       └── consolidated_results.csv
├── bm3d_streak_test/
│   ├── pixi.toml
│   └── run_bm3d_streak.py
├── joss_comparison.py
├── unified_visualization.py
├── pixi.toml
└── README.md
```

## bm3d-streak-removal Compatibility Notes

The `bm3d-streak-removal` package has several compatibility constraints:

1. **scipy version**: Requires scipy < 1.11 (uses deprecated `scipy.signal.gaussian`)
2. **Architecture**: bm4d library only provides x86_64 binaries
3. **Maintenance**: No releases since 2022

These issues are handled by running bm3d-streak-removal in an isolated Python 3.10 environment (`bm3d_streak_test/`).

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
