# Benchmark Results Summary

Consolidated results from JOSS paper benchmark comparison study.

## Test Configuration

- **Test data**: Shepp-Logan phantom (512×512), sinogram (720×725)
- **Timing runs**: n=100 per method
- **Metrics**: Processing time (mean ± std), PSNR (dB), SSIM

## Cross-Platform Timing Comparison

The two hosts differ in core count (24-core Threadripper workstation vs 12-core laptop chip), so the platform ratio mixes hardware with software; see the same-machine ratios under Key Findings.

| Method | Apple Silicon (s) | Linux x86_64 (s) | Apple Silicon / Linux |
|--------|-------------------|------------------|-----------------------|
| bm3dornl (streak) | 0.202 ± 0.014 | 0.094 ± 0.004 | 2.1× slower |
| bm3dornl (generic) | 0.183 ± 0.007 | 0.083 ± 0.003 | 2.2× slower |
| bm3dornl (multiscale) | 0.419 ± 0.017 | 0.389 ± 0.004 | 1.1× slower |
| Fourier-SVD | 0.022 ± 0.003 | 0.017 ± 0.001 | 1.3× slower |
| TomoPy FW (Münch) | 3.562 ± 0.195 | 0.261 ± 0.011 | 13.6× slower |
| TomoPy SF (Vo) | 3.573 ± 0.256 | 0.222 ± 0.007 | 16.1× slower |
| TomoPy BSD (sort) | 3.471 ± 0.250 | 0.293 ± 0.007 | 11.8× slower |
| bm3d-streak-removal | N/A | 31.813 ± 0.058 | N/A (x86_64 only) |

## Quality Metrics (Platform-Independent)

Quality metrics agree across platforms: PSNR identical to two decimals, SSIM identical to four decimals except bm3dornl (streak), 0.9423 on Apple Silicon vs 0.9426 on Linux (thread-count dependent aggregation order). Values below are from the Linux x86_64 run.

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| bm3dornl (streak) | 39.57 | 0.9426 |
| bm3dornl (generic) | 35.10 | 0.9476 |
| bm3dornl (multiscale) | 39.73 | 0.9759 |
| Fourier-SVD | 39.64 | 0.9513 |
| TomoPy FW (Münch) | 20.63 | 0.5840 |
| TomoPy SF (Vo) | 42.01 | 0.9868 |
| TomoPy BSD (sort) | 42.86 | 0.9829 |
| bm3d-streak-removal | 43.79 | 0.9670 |

## Platform Support

| Feature | bm3dornl | bm3d-streak-removal | TomoPy |
|---------|----------|---------------------|--------|
| Apple Silicon (arm64) | ✓ | ✗ | ✓ |
| Linux x86_64 | ✓ | ✓ | ✓ |
| Python 3.12+ | ✓ | ✗ | ✓ |
| Active maintenance | ✓ | ✗ | ✓ |
| Open source | ✓ | ✗ | ✓ |

## Key Findings

### Speed Advantage

- **bm3dornl vs bm3d-streak-removal**: 340× faster (0.094 s vs 31.813 s on Linux x86_64); Fourier-SVD 1900× faster (0.017 s)
- **bm3dornl (streak) vs the fastest TomoPy method**: 2.4× faster on Linux x86_64 (0.094 s vs 0.222 s); 17× faster on Apple Silicon (0.202 s vs 3.471 s)

### Cross-Platform Performance

- **bm3dornl**: 1.1-2.2× slower on the 12-core M2 Max laptop than on the 24-core Linux host
- **TomoPy**: 12-16× slower on the same M2 Max than on the Linux host
- Only the same-machine ratios above compare software; the absolute ratios also reflect the hardware difference

### Quality Trade-offs

- **Highest PSNR**: bm3d-streak-removal (43.79 dB) > TomoPy BSD (42.86 dB) > TomoPy SF (42.01 dB) > bm3dornl multiscale (39.73 dB)
- **Highest SSIM**: TomoPy SF (0.9868) > TomoPy BSD (0.9829) > bm3dornl multiscale (0.9759) > bm3d-streak-removal (0.9670)
- **TomoPy FW fails on this dataset**: PSNR 20.63 dB indicates significant quality degradation

## Hardware Specifications

### Apple Silicon Test System

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

### Linux x86_64 Test System

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

## Conclusion

BM3DORNL provides the best balance of:
1. **Speed**: 340× faster than bm3d-streak-removal
2. **Portability**: Works on Apple Silicon where bm3d-streak-removal does not
3. **Openness**: Fully open-source (MIT) vs closed-source non-commercial license
4. **Maintainability**: Actively maintained with Python 3.12+ support

While TomoPy SF and BSD achieve higher SSIM scores on this synthetic dataset, BM3DORNL's speed advantage enables interactive parameter tuning and high-throughput batch processing that would be impractical with bm3d-streak-removal's 32-second processing time per sinogram.
