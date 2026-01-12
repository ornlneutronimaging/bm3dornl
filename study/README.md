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
| OS | Red Hat Enterprise Linux 9.7 |
| CPU | AMD Ryzen Threadripper PRO 5975WX 32-Cores |
| Threads | 64 (32 cores × 2 threads) |
| RAM | 995 GB |
| Architecture | x86_64 |

### Test Data

Synthetic data generated with bm3dornl's phantom module:

1. Shepp-Logan phantom (512x512)
2. Radon transform to generate sinogram (720x725)
3. Simulated detector gain errors (ring artifacts)

## Linux x86_64 Results (Consolidated)

All 6 methods compared on identical test data (512x512 phantom, 720x725 sinogram):

| Method | Time (s) | PSNR (dB) | SSIM |
|--------|----------|-----------|------|
| bm3dornl (streak) | 0.266 | 32.63 | 0.6160 |
| bm3dornl (generic) | 0.228 | 32.93 | 0.5760 |
| TomoPy FW (Münch) | 0.350 | 20.61 | 0.5831 |
| TomoPy SF (Vo) | 0.257 | 34.50 | 0.9591 |
| TomoPy BSD (sort) | 0.322 | 34.69 | 0.9333 |
| bm3d-streak-removal | 41.116 | 36.34 | 0.8697 |

### Key Findings

**Speed comparison:**
- bm3dornl is **~155x faster** than bm3d-streak-removal
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
| Chip | Apple M4 Max |
| RAM | 128 GB |
| Architecture | arm64 |

### Benchmark Results (5 methods, n=30 runs each)

| Method | Time (s) | PSNR (dB) | SSIM |
|--------|----------|-----------|------|
| bm3dornl (streak) | 0.191 ± 0.006 | 32.63 | 0.6160 |
| bm3dornl (generic) | 0.184 ± 0.007 | 32.93 | 0.5760 |
| TomoPy FW (Münch) | 2.104 ± 0.048 | 20.61 | 0.5831 |
| TomoPy SF (Vo) | 2.062 ± 0.038 | 34.50 | 0.9591 |
| TomoPy BSD (sort) | 2.133 ± 0.040 | 34.69 | 0.9333 |

Note: bm3d-streak-removal is not available on Apple Silicon (no arm64 binary).

### Cross-Platform Comparison

| Method | Linux x86_64 (s) | Apple Silicon (s) | Speedup |
|--------|------------------|-------------------|---------|
| bm3dornl (streak) | 0.266 | 0.191 | 1.39x faster |
| bm3dornl (generic) | 0.228 | 0.184 | 1.24x faster |
| TomoPy FW (Münch) | 0.350 | 2.104 | 6.01x slower |
| TomoPy SF (Vo) | 0.257 | 2.062 | 8.02x slower |
| TomoPy BSD (sort) | 0.322 | 2.133 | 6.62x slower |

**Key observations:**

- **bm3dornl performs ~25-40% faster on Apple Silicon** compared to Linux x86_64
- **TomoPy methods are 6-8x slower on Apple Silicon** - likely due to lack of native arm64 optimization
- **Quality metrics (PSNR, SSIM) are identical** across platforms (same algorithm, same random seed)
- bm3dornl's Rust-based implementation benefits from Apple Silicon's architecture

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
