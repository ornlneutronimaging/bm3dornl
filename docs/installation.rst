Installation
============

Requirements
------------

- Python 3.12 or later
- NumPy, SciPy, scikit-image (installed automatically)

Installing from PyPI
--------------------

**Core Library**

.. code-block:: bash

    pip install bm3dornl

**With GUI Application**

.. code-block:: bash

    pip install bm3dornl[gui]

Or install the GUI separately:

.. code-block:: bash

    pip install bm3dornl-gui

Supported Platforms
-------------------

+----------+-----------------------+---------+-----+
| Platform | Architecture          | Library | GUI |
+==========+=======================+=========+=====+
| Linux    | x86_64                | Yes     | Yes |
+----------+-----------------------+---------+-----+
| macOS    | ARM64 (Apple Silicon) | Yes     | Yes |
+----------+-----------------------+---------+-----+

Development Installation
------------------------

For development, we use `pixi <https://prefix.dev>`_ for environment management:

.. code-block:: bash

    git clone https://github.com/ornlneutronimaging/bm3dornl.git
    cd bm3dornl
    pixi install
    pixi run build
    pixi run test

Verifying Installation
----------------------

.. code-block:: python

    import bm3dornl
    print(bm3dornl.__version__)

    # Test basic functionality
    from bm3dornl import bm3d_ring_artifact_removal
    import numpy as np

    test_image = np.random.rand(100, 100).astype(np.float32)
    result = bm3d_ring_artifact_removal(test_image, mode="generic", sigma_random=0.1)
    print(f"Input shape: {test_image.shape}, Output shape: {result.shape}")
