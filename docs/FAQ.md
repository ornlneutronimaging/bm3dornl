FAQ
===

General
-------

1. **What is the bm3dornl library?**

   - `bm3dornl` is a Python library for removing streak artifacts in normalized sinograms to reduce ring artifacts in the final reconstruction. It uses a multiscale BM3D algorithm accelerated with CuPy and Numba.

1. **What is the purpose of the bm3dornl library?**

   - The library aims to provide open-source, high-performance ring artifact removal for neutron imaging using advanced denoising techniques.

1. **Who are the contributors to the bm3dornl project?**

   - Developed by the ORNL neutron software engineering team and maintained by the neutron imaging team, including MARS@HFIR and VENUS@SNS.

Installation
------------

1. **How do I install the bm3dornl library?**

   - See user guide for installation instructions.
   - For developers, follow these steps:

      . Clone the repository and checkout a feature branch (use `next` for latest features).
      . Create a virtual environment, activate it, and install dependencies from `environment.yml`.
      . Perform a developer install using `pip install -e .`.

1. **What are the system requirements for bm3dornl?**

   - Requires Python 3.10 or later and a CUDA-enabled GPU for CuPy acceleration.

1. **How can I set up the environment to use bm3dornl?**

   - Use the `environment.yml` file: `conda env create -f environment.yml`.

1. **Can I use bm3dornl without a GPU?**

   - At present, the library requires a CUDA-enabled GPU for accelerated performance. We are working on a version that will allow CPU-only operation when a GPU is not available.

Usage
-----

1. **Can you provide a basic example of how to use bm3dornl for ring artifact removal?**

   - See the user guide for a step-by-step example.
   - Basic usage

   ```python
   import numpy as np
   from bm3dornl.bm3d import bm3d_ring_artifact_removal
   sinogram_denoised = bm3d_ring_artifact_removal(np.array(sinogram_input), mode="simple")
   ```

1. **How do I use bm3dornl with CuPy for accelerated performance?**

   - Ensure you have a CUDA-enabled GPU and CuPy installed. The library will automatically use CuPy for acceleration.

1. **What are the main functions provided by bm3dornl?**

   - The main function is `bm3d_ring_artifact_removal`, which takes a normalized sinogram and returns a denoised sinogram with reduced ring artifacts.
   - It also provides functions to perform
     - fast estimate with FFT-based notch filter
     - block-matching, noise variance weighted hard-thresholding, and aggregation for noise-free estimate pilot
     - block-matching, collaborative filtering, and aggregation for final denoising
     - global fourier refiltering derived from a Wiener filter

Code and Implementation
-----------------------

1. **How does the bm3dornl library utilize Numba for performance optimization?**

   - `Numba` accelerates functions by compiling Python code to machine code at runtime.

1. **Can you explain the process of block matching in bm3dornl?**

   - Involves finding similar patches in an image and grouping them for collaborative denoising, leveraging similar patches to enhance denoising effectiveness.

1. **How is CuPy used in bm3dornl for GPU acceleration?**

   - CuPy is used to perform computationally intensive tasks on the GPU, such as collaborative filtering and hard thresholding, to speed up the denoising process.

Documentation and Support
-------------------------

1. **Where can I find the official documentation for bm3dornl?**

   - Available in the repository’s [README](https://github.com/ornlneutronimaging/bm3dornl/blob/main/README.md) and additional docs.

1. **How can I contribute to the bm3dornl project?**

   - Fork the repository, make your changes, and submit a pull request. Follow the contribution guidelines in the README.

1. **Are there any tutorials available for learning bm3dornl?**

   - Check the repository for example notebooks or additional tutorial links.

1. **Where can I report bugs or request features?**

   - You can report bugs or request features by opening an issue in the GitHub repository.

1. **Is there a community or forum for bm3dornl users?**

   - Currently, the primary community interaction happens through the GitHub issues and pull requests. Consider checking the repository for any updates on community forums or mailing lists.
