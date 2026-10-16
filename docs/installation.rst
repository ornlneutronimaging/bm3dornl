Installation
============

Requirements
------------

- Python 3.12 or later
- NumPy, SciPy, scikit-image (installed automatically)

Installing from PyPI
--------------------

Install the core library:

.. code-block:: bash

    pip install bm3dornl

Install the library and the released GUI:

.. code-block:: bash

    pip install bm3dornl[gui]

To install only the released GUI, run:

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

Working in a Clone
------------------

Use Pixi for all commands in a clone. Follow the README's
`How to install <https://github.com/ornlneutronimaging/bm3dornl#how-to-install>`_
section.

.. code-block:: bash

    git clone https://github.com/ornlneutronimaging/bm3dornl.git
    cd bm3dornl
    pixi run build      # build the Rust extension
    pixi run test       # run the Rust and Python tests
    pixi run gui        # run the clone's GUI

Each ``pixi run`` command installs the environment on demand. Run
``pixi install`` first only to create the environment separately.

Verifying the Installation
--------------------------

After a pip install, start Python with ``python``. In a clone, use
``pixi run python``. Then run:

.. code-block:: python

    import bm3dornl
    print(bm3dornl.__version__)

    # Test basic functionality
    from bm3dornl import bm3d_ring_artifact_removal
    import numpy as np

    test_image = np.random.rand(100, 100).astype(np.float32)
    result = bm3d_ring_artifact_removal(test_image, mode="generic", sigma_random=0.1)
    print(f"Input shape: {test_image.shape}, Output shape: {result.shape}")
