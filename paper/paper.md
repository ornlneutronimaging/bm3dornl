---
title: 'BM3DORNL: High-Performance BM3D Denoising for Neutron Tomography'
tags:
  - Python
  - Rust
  - neutron imaging
  - tomography
  - denoising
  - ring artifacts
  - BM3D
authors:
  - name: Chen Zhang
    orcid: 0000-0001-8374-4467
    affiliation: 1
    corresponding: true
  - name: Jean-Christophe Bilheux
    orcid: 0000-0003-2172-6487
    affiliation: 2
  - name: Dmitry Ganyushin
    affiliation: 1
  - name: Pete Peterson
    orcid: 0000-0002-1353-0348
    affiliation: 1
affiliations:
  - name: Computing and Computational Sciences Directorate, Oak Ridge National Laboratory, Oak Ridge, TN, USA
    index: 1
  - name: Neutron Sciences Directorate, Oak Ridge National Laboratory, Oak Ridge, TN, USA
    index: 2
date: 11 January 2026
bibliography: paper.bib
---

# Summary

BM3DORNL is a high-performance Python library for denoising neutron and X-ray tomography data using a modified Block-Matching and 3D Filtering (BM3D) algorithm.
The library provides two denoising modes: a generic mode for standard noise removal, and a specialized streak mode optimized for removing vertical streak patterns in sinograms that manifest as ring artifacts in reconstructed images.
Built with a Rust backend and Python bindings via PyO3, BM3DORNL achieves 160$\times$ faster processing than comparable BM3D implementations while maintaining cross-platform compatibility including native Apple Silicon support.

# Statement of Need

Ring artifacts are a persistent challenge in neutron and X-ray computed tomography, arising from variations in detector pixel response, beam intensity fluctuations, and systematic errors [@münch2009].
These artifacts appear as concentric rings in reconstructed images, obscuring sample structure and degrading quantitative analysis.
Pre-reconstruction streak removal addresses these artifacts directly in sinograms, avoiding the spatial spreading that occurs during reconstruction.
Existing approaches include wavelet-Fourier filtering [@münch2009], polynomial fitting [@vo2018], and sorting-based methods [@miqueles2014], with implementations available in TomoPy [@gürsoy2014].

Mäkinen et al. [@mäkinen2021] demonstrated that applying BM3D [@dabov2007] across multiple scales achieves superior artifact removal by exploiting the self-similarity of streak patterns.
However, their bm3d-streak-removal implementation is closed-source, restricted to non-commercial use, and provides only x86_64 binaries incompatible with Apple Silicon and modern Python versions.
BM3DORNL provides the neutron imaging community with an open-source, MIT-licensed, high-performance implementation that enables both high-throughput batch processing and interactive parameter tuning.

# State of the Field

Several software packages implement pre-processing streak removal for tomography: TomoPy [@gürsoy2014] provides wavelet-Fourier filtering [@münch2009] and Vo's sorting/fitting methods [@vo2018]; ASTRA Toolbox [@van2016] offers GPU-accelerated preprocessing; and bm3d-streak-removal [@mäkinen2021] implements multiscale BM3D but remains closed-source and platform-limited.

\autoref{fig:input} shows the benchmark input data: a synthetic sinogram (720$\times$725 pixels) with simulated ring artifacts and the corresponding clean ground truth.
All benchmark results are from Linux x86_64 to enable comparison with bm3d-streak-removal, which provides only x86_64 binaries.
We compared eight methods: four BM3DORNL variants (streak, generic, multiscale, and Fourier-SVD), three TomoPy algorithms (wavelet-Fourier, sorting-fitting, and sorting-based), and the original bm3d-streak-removal.
The multiscale variant implements the pyramid approach from Mäkinen et al. [@mäkinen2021], while Fourier-SVD is a lightweight FFT-guided method that achieves comparable quality at 50$\times$ faster speeds.

![Benchmark input data. (a) Input sinogram with simulated ring artifacts. (b) Ground truth (clean sinogram). The dashed rectangle indicates the crop region shown in \autoref{fig:results}.\label{fig:input}](figure1_input.png){ width=100% }

**Speed comparison:** \autoref{fig:metrics}(a) shows processing times (n=100 runs on Linux x86_64).
Fourier-SVD is the fastest method at 0.017 seconds, achieving 2300$\times$ speedup over bm3d-streak-removal (39.9 seconds).
The BM3D-based methods span a performance range: standard BM3DORNL processes sinograms in 0.4 seconds (100$\times$ faster than bm3d-streak-removal), while multiscale BM3DORNL takes 0.9 seconds (44$\times$ faster) due to pyramid processing.
TomoPy methods achieve comparable processing times (0.7--0.8 seconds).
Cross-platform performance differs substantially: TomoPy runs 2.5--3$\times$ slower on Apple Silicon than Linux x86_64 (likely due to unoptimized C extensions), whereas BM3DORNL runs at comparable speeds (within 10%) across platforms due to its Rust backend.

**Quality analysis:** \autoref{fig:results} shows cropped results from all eight methods with their difference images (result minus ground truth).
The benchmark reveals that quality metrics alone do not tell the full story.
TomoPy SF achieves the highest SSIM (0.987, see \autoref{fig:metrics}(b)), but difference image analysis shows residual vertical stripes in its output.
This occurs because SSIM uses local windows (7$\times$7 or 11$\times$11 pixels), and narrow vertical stripes affect few pixels per window.
Interestingly, both multiscale BM3DORNL and Fourier-SVD achieve excellent SSIM scores (0.951) while maintaining fast processing times---Fourier-SVD accomplishes this at 0.017 seconds, demonstrating that aggressive artifact removal and high-quality reconstruction are achievable without computationally expensive multi-scale processing.
The difference images reveal that all BM3DORNL variants produce clean vertical patterns, indicating successful artifact removal, while TomoPy methods leave faint residual artifacts.

![Method comparison. Top row: Cropped results from each method with processing times. Bottom row: Difference images (result minus ground truth) using a zero-centered RdBu colormap---blue indicates artifact removal, red indicates added artifacts. Data from Linux x86_64.\label{fig:results}](figure2_results.png){ width=100% }

![Performance metrics (Linux x86_64, n=100 runs). (a) Processing time comparison on linear scale, showing Fourier-SVD's 2300$\times$ speedup and standard BM3DORNL's 100$\times$ speedup over bm3d-streak-removal. (b) Quality metrics (PSNR vs SSIM) showing trade-offs between methods. BM3DORNL variants (blue/cyan), TomoPy (orange), bm3d-streak-removal (green).\label{fig:metrics}](figure3_metrics.png){ width=100% }

**Platform support:** bm3d-streak-removal is unavailable on Apple Silicon (x86_64 binaries only), incompatible with Python 3.11+, and restricted to non-commercial use.
BM3DORNL provides native performance on all platforms, supports Python 3.12+, and uses the MIT license for unrestricted commercial use.

# Software Design

BM3DORNL employs a hybrid Python-Rust architecture: the core algorithm is implemented in Rust using the `rayon` crate for work-stealing parallelism, with Python bindings via PyO3 for seamless NumPy integration.

**Key optimizations:**

- **Integral image pre-screening:** Before computing expensive patch distances, the block matching stage uses integral images to compute mean and norm bounds in O(1) time. Patches that fail these bounds are skipped, eliminating approximately 80% of distance calculations.

- **Walsh-Hadamard transform:** For the default 8$\times$8 patches, BM3DORNL uses Walsh-Hadamard transforms instead of FFT. This is multiplication-free (only additions and subtractions), cache-friendly with fixed 64-element buffers, and computed in-place without memory allocation.

- **Pre-computed FFT plans:** FFT plans are computed once and shared across threads via Arc, avoiding expensive plan initialization on each invocation (measured 45% speedup).

- **Batched stack processing:** Sinograms are processed slice-by-slice to control memory usage, ensuring each slice fits in L2/L3 cache while minimizing memory allocation through buffer reuse.

The library provides a minimal Python API:

```python
from bm3dornl import bm3d_ring_artifact_removal

cleaned = bm3d_ring_artifact_removal(
    sinogram,
    mode="streak",
    sigma=0.0, # default to automatic noise estimation
)
```

**GUI application:** BM3DORNL includes a native GUI built with the egui framework for Rust.
The GUI enables interactive parameter tuning, side-by-side comparison with difference visualization, HDF5/TIFF file loading with dataset browsing, and real-time processing feedback.
At 0.2--0.3 seconds per sinogram, scientists can explore parameter space interactively---something impractical with bm3d-streak-removal's 41-second processing time.

# Research Impact

BM3DORNL is being integrated into processing pipelines at the VENUS and MARS beamlines at Oak Ridge National Laboratory.
The library provides multiple performance tiers: Fourier-SVD enables real-time parameter exploration at 0.017 seconds per sinogram (2300$\times$ faster than bm3d-streak-removal), standard BM3DORNL offers robust denoising at 0.4 seconds (100$\times$ speedup), and multiscale BM3DORNL provides maximum quality at 0.9 seconds (44$\times$ speedup).
For batch processing, a 1000-sinogram dataset that would take 11 hours with bm3d-streak-removal completes in 17 seconds (Fourier-SVD), 7 minutes (standard BM3DORNL), or 15 minutes (multiscale BM3DORNL).

The hybrid Rust-Python architecture demonstrates a modern approach to scientific software development: Rust provides memory safety and performance portability (single codebase compiles natively to arm64 and x86_64), while Python ensures integration with the NumPy/SciPy ecosystem.
This pattern is increasingly valuable as the scientific computing community diversifies hardware platforms.

BM3DORNL fills a critical gap: the neutron imaging community needed an open-source, actively maintained BM3D implementation that works on modern platforms.
The MIT license enables unrestricted use at national facilities, and native Apple Silicon support ensures scientists can use contemporary hardware for analysis workstations.

# AI Usage Disclosure

Generative AI tools (Claude) were used for code assistance and documentation drafting. All AI-generated content was reviewed, tested, and validated by the authors.

# Acknowledgements

This research used resources at the Spallation Neutron Source, a DOE Office of Science User Facility operated by Oak Ridge National Laboratory.
This work is supported by the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

# References
