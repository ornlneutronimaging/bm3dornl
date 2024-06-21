FAQ
===

General
-------

1. **What is the bm3dornl library?**
   - `bm3dornl` is a Python library for removing streak artifacts in normalized sinograms to reduce ring artifacts in the final reconstruction. It uses a multiscale BM3D algorithm accelerated with CuPy and Numba.

2. **What is the purpose of the bm3dornl library?**
   - The library aims to provide open-source, high-performance ring artifact removal for neutron imaging using advanced denoising techniques.

3. **Who are the contributors to the bm3dornl project?**
   - Developed by the ORNL neutron software engineering team and maintained by the neutron imaging team, including MARS@HFIR and VENUS@SNS.

Installation
------------

1. **How do I install the bm3dornl library?**
   - Currently under development. Install in developer mode:

      . Clone the repository and checkout a feature branch (use `next` for latest features).
      . Create a virtual environment, activate it, and install dependencies from `environment.yml`.
      . Perform a developer install using `pip install -e .`.

2. **What are the system requirements for bm3dornl?**
   - Requires Python 3.10 or later and a CUDA-enabled GPU for CuPy acceleration.

3. **How can I set up the environment to use bm3dornl?**
   - Use the `environment.yml` file: `conda env create -f environment.yml`.

Usage
-----

1. **Can you provide a basic example of how to use bm3dornl for ring artifact removal?**

2. **How do I use bm3dornl with CuPy for accelerated performance?**

3. **What are the main functions provided by bm3dornl?**

Code and Implementation
-----------------------

1. **How does the bm3dornl library utilize Numba for performance optimization?**
   - `Numba` accelerates functions by compiling Python code to machine code at runtime.

2. **Can you explain the process of block matching in bm3dornl?**
   - Involves finding similar patches in an image and grouping them for collaborative denoising, leveraging similar patches to enhance denoising effectiveness.

Documentation and Support
-------------------------

1. **Where can I find the official documentation for bm3dornl?**
   - Available in the repository’s [README](https://github.com/ornlneutronimaging/bm3dornl/blob/main/README.md) and additional docs.

2. **How can I contribute to the bm3dornl project?**
   - Fork the repository, make your changes, and submit a pull request. Follow the contribution guidelines in the README.

3. **Are there any tutorials available for learning bm3dornl?**
   - Check the repository for example notebooks or additional tutorial links.
