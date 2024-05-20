# FAQ

## General

1. **What is the bm3dornl library?**
   - `bm3dornl` is a Python library for removing streak artifacts in normalized sinograms to reduce ring artifacts in the final reconstruction. It uses a multiscale BM3D algorithm accelerated with CuPy and Numba.

2. **What is the purpose of the bm3dornl library?**
   - The library aims to provide open-source, high-performance ring artifact removal for neutron imaging using advanced denoising techniques.

3. **Who are the contributors to the bm3dornl project?**
   - Developed by the ORNL neutron software engineering team and maintained by the neutron imaging team, including MARS@HFIR and VENUS@SNS.

## Installation

1. **How do I install the bm3dornl library?**
   - Currently under development. Install in developer mode:
     - Clone the repository and checkout a feature branch (use `next` for latest features).
     - Create a virtual environment, activate it, and install dependencies from `environment.yml`.
     - Perform a developer install using `pip install -e .`.

2. **What are the system requirements for bm3dornl?**
   - Requires Python 3.10 or later and a CUDA-enabled GPU for CuPy acceleration.

3. **How can I set up the environment to use bm3dornl?**
   - Use the `environment.yml` file: `conda env create -f environment.yml`.

## Usage

1. **Can you provide a basic example of how to use bm3dornl for ring artifact removal?**
   - For simple usage:
     ```python
     from bm3dornl.denoiser import bm3d_streak_removal

     sino_bm3dornl = bm3d_streak_removal(
         sinogram=sinogram_noisy,
         background_threshold=0.1,
         patch_size=(8, 8),
         stride=3,
         cut_off_distance=(128, 128),
         intensity_diff_threshold=0.2,
         num_patches_per_group=300,
         shrinkage_threshold=1 - 1e-4,
         k=0,
         fast_estimate=True,
     )
     ```

2. **How do I use bm3dornl with CuPy for accelerated performance?**
   - Ensure you have a CUDA-enabled GPU and install dependencies from `environment.yml`. Use bm3dornl functions that leverage CuPy for collaborative filtering and hard thresholding (`fast_estimate=False`).

3. **What are the main functions provided by bm3dornl?**
   - Key components:
     - `PatchManager` for block matching.
     - `Numba` accelerated functions.
     - `CuPy` accelerated functions.
     - `bm3d_streak_removal` for ring artifact removal.
     - Helper functions for visualization and data manipulation.

## Code and Implementation

1. **What does the `PatchManager` class do in bm3dornl?**
   - Manages image patches, groups them based on spatial and intensity thresholds, and generates groups of similar patches as 4D arrays.

2. **How does the bm3dornl library utilize Numba for performance optimization?**
   - `Numba` accelerates functions by compiling Python code to machine code at runtime.

3. **Can you explain the process of block matching in bm3dornl?**
   - Involves finding similar patches in an image and grouping them for collaborative denoising, leveraging similar patches to enhance denoising effectiveness.

## Documentation and Support

1. **Where can I find the official documentation for bm3dornl?**
   - Available in the repository’s [README](https://github.com/ornlneutronimaging/bm3dornl/blob/main/README.md) and additional docs.

2. **How can I contribute to the bm3dornl project?**
   - Fork the repository, make your changes, and submit a pull request. Follow the contribution guidelines in the README.

3. **Are there any tutorials available for learning bm3dornl?**
   - Check the repository for example notebooks or additional tutorial links.
