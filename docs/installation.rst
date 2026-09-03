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

Working from a Source Checkout
------------------------------

Everything in a clone of the repository runs through `pixi <https://prefix.dev>`_,
which provides Python, Rust, HDF5, and the other build dependencies. It is the
only supported way to build, test, or run the code in a checkout. ``pip install -e .``
is not supported: the Rust extension is built by ``pixi run build``, and the
``[gui]`` extra downloads the released GUI binary from PyPI instead of building
the GUI in your checkout.

.. code-block:: bash

    git clone https://github.com/ornlneutronimaging/bm3dornl.git
    cd bm3dornl
    pixi install        # create the environment
    pixi run build      # build the Rust extension, install the package in editable mode
    pixi run test       # run the Rust and Python test suites
    pixi run gui        # run the GUI from your checkout (compiles it on first use)

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
