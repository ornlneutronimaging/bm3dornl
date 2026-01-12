# Benchmark Results Summary

Consolidated results from JOSS paper benchmark comparison study.

## Test Configuration

- **Test data**: Shepp-Logan phantom (512×512), sinogram (720×725)
- **Timing runs**: n=30 per method
- **Metrics**: Processing time (mean ± std), PSNR (dB), SSIM

## Cross-Platform Timing Comparison

| Method | Apple Silicon (s) | Linux x86_64 (s) | Platform Ratio |
|--------|-------------------|------------------|----------------|
| bm3dornl (streak) | 0.191 ± 0.006 | 0.256 ± 0.009 | 1.34× faster on Apple Silicon |
| bm3dornl (generic) | 0.184 ± 0.007 | 0.219 ± 0.002 | 1.19× faster on Apple Silicon |
| TomoPy FW (Münch) | 2.104 ± 0.048 | 0.318 ± 0.008 | 6.62× slower on Apple Silicon |
| TomoPy SF (Vo) | 2.062 ± 0.038 | 0.278 ± 0.008 | 7.42× slower on Apple Silicon |
| TomoPy BSD (sort) | 2.133 ± 0.040 | 0.349 ± 0.009 | 6.11× slower on Apple Silicon |
| bm3d-streak-removal | N/A | 41.033 ± 0.139 | N/A (x86_64 only) |

## Quality Metrics (Platform-Independent)

Quality metrics are identical across platforms (deterministic algorithms, same random seed).

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| bm3dornl (streak) | 32.63 | 0.6160 |
| bm3dornl (generic) | 32.93 | 0.5760 |
| TomoPy FW (Münch) | 20.61 | 0.5831 |
| TomoPy SF (Vo) | 34.50 | 0.9591 |
| TomoPy BSD (sort) | 34.69 | 0.9333 |
| bm3d-streak-removal | 36.34 | 0.8697 |

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

- **bm3dornl vs bm3d-streak-removal**: ~160× faster (0.256s vs 41.033s on Linux x86_64)
- **bm3dornl vs TomoPy**: Comparable speed on Linux (~0.2-0.4s); 10× faster on Apple Silicon

### Cross-Platform Performance

- **bm3dornl**: 20-35% faster on Apple Silicon vs Linux x86_64 (Rust backend benefits from ARM architecture)
- **TomoPy**: 6-7× slower on Apple Silicon (likely lacks native ARM optimization)

### Quality Trade-offs

- **Highest PSNR**: bm3d-streak-removal (36.34 dB) > TomoPy BSD (34.69 dB) > TomoPy SF (34.50 dB) > bm3dornl generic (32.93 dB)
- **Highest SSIM**: TomoPy SF (0.9591) > TomoPy BSD (0.9333) > bm3d-streak-removal (0.8697) > bm3dornl streak (0.6160)
- **TomoPy FW fails on this dataset**: PSNR 20.61 dB indicates significant quality degradation

## Hardware Specifications

### Apple Silicon Test System

| Component | Specification |
|-----------|---------------|
| Model | MacBook Pro |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| Architecture | arm64 |

### Linux x86_64 Test System

| Component | Specification |
|-----------|---------------|
| OS | Red Hat Enterprise Linux 9.7 |
| CPU | AMD Ryzen Threadripper PRO 5975WX 32-Cores |
| Threads | 64 (32 cores × 2 threads) |
| RAM | 995 GB |
| Architecture | x86_64 |

## Conclusion

BM3DORNL provides the best balance of:
1. **Speed**: ~160× faster than bm3d-streak-removal
2. **Portability**: Works on Apple Silicon where bm3d-streak-removal does not
3. **Openness**: Fully open-source (MIT) vs closed-source non-commercial license
4. **Maintainability**: Actively maintained with Python 3.12+ support

While TomoPy SF and BSD achieve higher SSIM scores on this synthetic dataset, BM3DORNL's speed advantage enables interactive parameter tuning and high-throughput batch processing that would be impractical with bm3d-streak-removal's 41-second processing time per sinogram.
