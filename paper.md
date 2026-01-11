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
  - image processing
authors:
  - name: Chen Zhang
    orcid: 0000-0001-8374-4467
    affiliation: 1
    corresponding: true
  - name: Dmitry Ganyushin
    affiliation: 1
  - name: Pete Peterson
    orcid: 0000-0002-1353-0348
    affiliation: 1
affiliations:
  - name: Oak Ridge National Laboratory, Oak Ridge, TN, USA
    index: 1
date: 10 January 2026
bibliography: paper.bib
---

# Summary

BM3DORNL is a high-performance Python library for denoising neutron and X-ray tomography data using a modified Block-Matching and 3D Filtering (BM3D) algorithm.
The library provides two denoising modes: a generic mode for standard white noise removal, and a specialized streak mode optimized for removing the vertical streak patterns in sinograms that manifest as ring artifacts in reconstructed tomographic images.
Built with a Rust backend for computational efficiency, BM3DORNL enables high-throughput batch processing of sinogram stacks and fast response during interactive reconstruction workflows.

# Statement of Need

Ring artifacts are a persistent challenge in neutron and X-ray computed tomography, arising from variations in detector pixel response, beam intensity fluctuations, and other systematic errors [@münch2009].
These artifacts can be addressed at different stages of the tomography pipeline: pre-reconstruction methods remove vertical streaks directly from sinograms before reconstruction, while post-reconstruction methods operate on the reconstructed images, often using polar coordinate transformations or iterative techniques [@vo2018; @prell2009].

Pre-reconstruction streak removal is particularly advantageous as it addresses the artifact source directly, avoiding the spatial spreading that occurs during reconstruction.
Existing pre-reconstruction approaches include wavelet-Fourier filtering [@münch2009], polynomial fitting and smoothing [@vo2018], and sorting-based methods [@miqueles2014].
TomoPy [@gürsoy2014] provides implementations of many of these algorithms.
However, these methods can struggle to balance artifact removal with preservation of genuine image features, particularly when streaks have varying intensities or widths.

Mäkinen et al. [@mäkinen2021] demonstrated that modeling streak noise as spatially correlated noise and applying the BM3D algorithm [@dabov2007] across multiple scales achieves superior artifact removal while preserving image structure.
Their bm3d-streak-removal implementation, however, is closed-source and licensed for non-commercial use only.

BM3DORNL provides the neutron imaging community with an open-source implementation of this advanced denoising method, promoting open science and accelerating scientific advancement.
The library prioritizes not only denoising quality but also computational performance—achieving fast processing speeds essential for high-throughput batch processing of large tomography datasets and interactive parameter tuning during reconstruction workflows.

# State of the Field

[PLACEHOLDER: Complete this section with comparison to existing tools]

Software-based ring artifact removal methods fall into three categories [@prell2009]:

- **Pre-processing methods**: Remove vertical streaks from sinograms before reconstruction
- **Post-processing methods**: Remove ring artifacts from reconstructed images
- **In-processing methods**: Address artifacts during iterative reconstruction

Several software packages implement pre-processing streak removal:

- **TomoPy** [@gürsoy2014]: Comprehensive tomography toolkit implementing wavelet-Fourier filtering [@münch2009], Titarenko's algorithm [@miqueles2014], and Vo's suite of sorting, filtering, and fitting methods [@vo2018]
- **ASTRA Toolbox** [@van2016]: GPU-accelerated reconstruction with preprocessing capabilities
- **bm3d-streak-removal** [@mäkinen2021]: Multiscale BM3D for correlated streak noise (closed-source, non-commercial license)

[PLACEHOLDER: Add specific comparison of approaches and justify why BM3DORNL was developed rather than contributing to existing projects. Consider addressing:
- What performance/quality advantages does BM3DORNL provide over existing open-source methods?
- Benchmark results comparing to TomoPy methods and/or bm3d-streak-removal]

# Software Design

BM3DORNL employs a hybrid Python-Rust architecture to balance usability with performance.
The core algorithm is implemented in Rust using the `rayon` crate for data parallelism, while Python bindings via PyO3 provide seamless integration with the scientific Python ecosystem (NumPy, SciPy).

Key design decisions include:

- **Dual denoising modes**: Generic mode for standard white noise removal; streak mode for vertically-correlated artifact patterns, inspired by the approach of Mäkinen et al. [@mäkinen2021].
- **Streak-aware block matching**: In streak mode, the algorithm prioritizes matching blocks along vertical orientations to group streak artifact patterns while avoiding false matches across genuine horizontal features.
- **Integral image pre-screening**: Fast block distance estimation using integral images enables early termination of non-matching candidates, reducing computational cost.
- **Batched stack processing**: 3D sinogram stacks are processed in configurable batches to balance memory usage against parallel efficiency.
- **Fast transforms**: Optimized Walsh-Hadamard and FFT implementations for 8×8 patch transforms.

The library exposes a minimal API centered on the `bm3d_ring_artifact_removal()` function, which accepts 2D sinograms or 3D stacks and returns denoised arrays with matching dimensions.

# Research Impact Statement

BM3DORNL is being integrated into the data processing pipelines at the VENUS beamline at the Spallation Neutron Source (SNS) and the MARS beamline (formerly CG1D) at the High Flux Isotope Reactor (HFIR), both neutron user facilities at Oak Ridge National Laboratory.
The software was developed to support the Neutron User Facility under the Neutron Data Project.

By providing an open-source implementation of multiscale BM3D-based streak removal, BM3DORNL enables the neutron imaging community to adopt this advanced denoising method without licensing restrictions, promoting reproducible research and collaborative development.

# AI Usage Disclosure

[PLACEHOLDER: Confirm or modify this statement]

Generative AI tools were used during the development of this software for code assistance and documentation drafting.
All AI-generated content was reviewed, tested, and validated by the authors.

# Acknowledgements

A portion of this research used resources at the Spallation Neutron Source (SNS), a Department of Energy (DOE) Office of Science User Facility operated by Oak Ridge National Laboratory.
This work is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

# References
