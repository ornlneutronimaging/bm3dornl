GUI Application
===============

BM3DORNL includes a standalone GUI application for interactive ring artifact removal
from tomography data.

Installation
------------

.. code-block:: bash

    # Install with the main package
    pip install bm3dornl[gui]

    # Or install separately
    pip install bm3dornl-gui

Launching
---------

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

- Interactive slice viewer with scroll-wheel navigation
- Real-time histogram display
- Adjustable window/level (contrast) controls
- Side-by-side comparison of original and processed images

**Processing**

- Real-time parameter adjustment
- Support for both ``generic`` and ``streak`` modes
- Live preview of denoising results

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
     - Navigate through slices (3D data) or zoom
   * - Drag
     - Pan image
   * - Shift + Drag
     - Select ROI for histogram

Workflow
--------

1. **Load Data**: Click "Open" and select an HDF5 or TIFF file. For HDF5, use the tree browser to select a dataset.

2. **Adjust View**: Use window/level controls or scroll to navigate through slices.

3. **Select Parameters**: Adjust denoising parameters in the control panel:

   - Mode: ``streak`` for ring artifacts, ``generic`` for random noise
   - Sigma: Start low (0.02-0.05) and increase as needed
   - Other parameters: Usually defaults work well

4. **Process**: Click "Process" to apply denoising. Compare with original using the split view.

5. **Evaluate**: Use ROI selection (Shift+drag) to check noise statistics in specific regions.

6. **Export**: Save processed data to TIFF or HDF5 format.

Tips
----

- **Start with defaults**: The default parameters work well for most cases.

- **Use streak mode**: For ring artifacts in sinograms, always use ``mode="streak"``.

- **Check the histogram**: The histogram should narrow after denoising, indicating reduced noise.

- **Use ROI selection**: Select flat regions to evaluate noise reduction without signal interference.

- **Monitor the difference**: The removed signal should show primarily vertical streaks (for streak mode), not horizontal structure.
