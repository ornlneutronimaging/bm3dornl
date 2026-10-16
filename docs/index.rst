BM3D ORNL Documentation
========================

BM3DORNL removes streak and ring artifacts from neutron-imaging sinograms.

Features
--------

- **Streak removal:** Streak mode removes vertical detector artifacts.
- **Stack processing:** Batch three-dimensional sinogram stacks.
- **Rust backend:** Accelerate block matching and transforms.
- **GUI application:** Process HDF5 tomography data interactively.

Installation
------------

Use pip for published packages:

.. code-block:: bash

    # Core library only
    pip install bm3dornl

    # With GUI application
    pip install bm3dornl[gui]

For a clone, use Pixi. Follow the README's
`How to install <https://github.com/ornlneutronimaging/bm3dornl#how-to-install>`_
section.

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
