GUI Application
===============

BM3DORNL includes a standalone GUI for processing tomography data.

Installation
------------

Install a released binary for Linux x86_64 or macOS Apple Silicon:

.. code-block:: bash

    # Install with the main package
    pip install bm3dornl[gui]

    # Or install separately
    pip install bm3dornl-gui

For a clone, use Pixi. Follow the README's
`How to install <https://github.com/ornlneutronimaging/bm3dornl#how-to-install>`_
section.

.. code-block:: bash

    pixi run gui

Launching
---------

The released binary installs a ``bm3dornl-gui`` command:

.. code-block:: bash

    bm3dornl-gui

Features
--------

**Data Loading**

- Load HDF5 files with interactive tree browser for dataset selection
- Load TIFF files (single images or stacks)
- Support for 2D and 3D datasets
- Automatic data type detection

**Visualization**

- Interactive slice viewer with a slice slider and scroll-wheel zoom
- Real-time histogram display
- Adjustable window/level (contrast) controls
- Side-by-side comparison of original and processed images

**Processing**

- Adjust parameters before starting processing
- Support for both ``generic`` and ``streak`` modes
- View the processed result after the run finishes

**ROI Selection**

- Shift+drag to select a region of interest
- Histogram updates to show ROI statistics
- Useful for evaluating local noise characteristics

**Export**

- Export processed data to TIFF format
- Export to HDF5 format
- Batch export of full stacks

Keyboard Shortcuts
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Shortcut
     - Action
   * - Scroll wheel
     - Zoom in and out
   * - Drag
     - Pan image
   * - Shift + Drag
     - Select ROI for histogram

Workflow
--------

1. **Load data:** Click "Open" and select an HDF5 or TIFF file. For HDF5,
   select a dataset in the tree browser.

2. **Adjust the view:** Use the window and level controls for contrast. Use the
   slice slider for a three-dimensional stack. Use the scroll wheel to zoom.

3. **Select parameters:** Set the denoising parameters before processing.

   - Mode: ``streak`` for ring artifacts, ``generic`` for random noise
   - Sigma: Start at 0.02 to 0.05 and increase as needed
   - Other parameters: Start with the defaults

4. **Process:** Click "Process" to apply denoising. Results appear after the run
   finishes.

5. **Evaluate:** Compare the result with the original in split view. Select a
   region of interest with Shift+drag to inspect local noise statistics.

6. **Export:** Save the result as TIFF or HDF5.

Tips
----

- **Start with defaults:** The defaults work well for most cases.

- **Use streak mode:** Choose ``mode="streak"`` for ring artifacts.

- **Check the histogram:** A narrower histogram can indicate less noise.

- **Use ROI selection:** Select flat regions to evaluate noise reduction.

- **Inspect the difference:** In streak mode, removed content should be mainly
  vertical streaks. Horizontal structure may indicate signal loss.
