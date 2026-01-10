BM3D ORNL Documentation
========================

A high-performance BM3D denoising library for neutron imaging, optimized for streak/ring artifact removal from sinograms.

Features
--------

- **Streak/Ring Artifact Removal**: Specialized mode for removing vertical streak artifacts common in neutron and X-ray imaging
- **Stack Processing**: Efficient batched processing of 3D sinogram stacks
- **High Performance**: Rust backend with optimized block matching and transforms

Quick Start
-----------

.. code-block:: python

    from bm3dornl import bm3d_ring_artifact_removal
    import numpy as np

    # Load sinogram data - 2D (H, W) or 3D stack (N, H, W)
    sinogram = np.load("sinogram.npy")

    # Streak artifact removal (recommended for ring artifacts)
    denoised = bm3d_ring_artifact_removal(sinogram, mode="streak", sigma=0.1)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
