<!-- Badges -->

[![Build Status](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/actions.yml/badge.svg?branch=next)](https://github.com/ornlneutronimaging/bm3dornl/actions/workflows/actions.yml?query=branch?next)
[![codecov](https://codecov.io/gh/ornlneutronimaging/bm3dornl/branch/next/graph/badge.svg)](https://codecov.io/gh/ornlneutronimaging/bm3dornl/tree/next)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6650/badge)](https://bestpractices.coreinfrastructure.org/projects/6650)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ornlneutronimaging/bm3dornl/next.svg)](https://results.pre-commit.ci/latest/github/ornlneutronimaging/bm3dornl/next)
[![Documentation Status](https://readthedocs.org/projects/bm3dornl/badge/?version=latest)](https://bm3dornl.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/neutronimaging/bm3dornl/badges/version.svg)](https://anaconda.org/neutronimaging/bm3dornl)

<!-- End Badges -->
BM3D ORNL
=========

This repository contains the BM3D ORNL code, which is a Python implementation of the BM3D denoising algorithm. The BM3D algorithm was originally proposed by K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian in the paper "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering" (2007).
The BM3D algorithm is a state-of-the-art denoising algorithm that is widely used in the image processing community.
The BM3D ORNL code is a Python implementation of the BM3D algorithm that has been optimized for performance using both `Numba` and `CuPy`.
The BM3D ORNL code is designed to be easy to use and easy to integrate into existing Python workflows.
The BM3D ORNL code is released under an open-source license, and is freely available for download and use.

For more information, check out our [FAQ](docs/FAQ.md).

How to install
--------------

For **users**, you can install the latest version published on our [anaconda channel](https://anaconda.org/neutronimaging/bm3dornl) with:

```bash
conda install neutronimaging::bm3dornl
```

Or use `pip` to install from [PyPI](https://pypi.org/project/bm3dornl/0.3.1/)

```bash
pip install bm3dornl
```

Since the `PyPI` version relies on pre-built `CuPy` from PyPI, it is possible that the bundled latest version of `CuPy` might not be compatible.
In such situations, please either install the correct pre-built version via pip, e.g. `pip install cupy-cuda11x` for Nvidia card with older drivers, or use local `nvcc` to build `CuPy` from source with `pip install cupy`.

How to contribute
-----------------

For **developers**, please fork this repo and create a conda development environment with

```bash
conda env create -f environment.yml
```

followed by

```bash
pip install --no-deps -e .
```

The option `--no-deps` here is critical as `pip` will try to install a second set of dependencies based on information from `pyproject.toml`.
Since conda does not check packages compatibility installed from `pip`, we need to avoid bring in in-compatible packages.

Once your feature implementation is ready, please make a pull request and ping one of our developers for review.
