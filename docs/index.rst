BM3D ORNL Documentation
========================

A high-performance BM3D denoising library for neutron imaging, optimized for streak/ring artifact removal from sinograms.

Features
--------

- **Streak/Ring Artifact Removal**: Specialized mode for removing vertical streak artifacts common in neutron and X-ray imaging
- **Stack Processing**: Efficient batched processing of 3D sinogram stacks
- **High Performance**: Rust backend with optimized block matching and transforms
- **GUI Application**: Interactive application for processing HDF5 tomography data

Installation
------------

**Using pip**

.. code-block:: bash

    # Core library only
    pip install bm3dornl

    # With GUI application
    pip install bm3dornl[gui]

**Supported Platforms**

- Linux x86_64
- macOS ARM64 (Apple Silicon)

Quick Start
-----------

.. code-block:: python

    from bm3dornl import bm3d_ring_artifact_removal
    import numpy as np

    # Load sinogram data - 2D (H, W) or 3D stack (N, H, W)
    sinogram = np.load("sinogram.npy")

    # Streak artifact removal (recommended for ring artifacts)
    denoised = bm3d_ring_artifact_removal(
        sinogram,
        mode="streak",
        sigma_random=0.1,
    )

    # With custom parameters
    denoised = bm3d_ring_artifact_removal(
        sinogram,
        mode="streak",
        sigma_random=0.1,
        patch_size=8,
        step_size=4,
        search_window=24,
        max_matches=16,
    )

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   parameters
   gui
   tutorial
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
