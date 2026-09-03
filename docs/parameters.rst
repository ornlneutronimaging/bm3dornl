Parameter Reference
===================

This page describes every ``bm3d_ring_artifact_removal`` parameter. BM3D means
block-matching and three-dimensional filtering.

Mode Selection
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``mode``
     - ``"streak"``
     - Operation mode. Use ``"generic"`` for white noise, ``"streak"`` for ring artifacts.

.. note::

   Use ``mode="streak"`` for ring artifacts in tomography sinograms. It targets
   vertical artifacts from detector response variations.

Noise Parameters
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``sigma_random``
     - ``0.1``
     - Estimated noise standard deviation. Higher values apply stronger denoising.

**Guidelines for sigma_random:**

- Light artifacts: ``0.02 - 0.05``
- Moderate artifacts: ``0.05 - 0.10``
- Heavy artifacts: ``0.10 - 0.20``

.. tip::

   Start with a low value and increase gradually. Check the difference image
   (input - output) to ensure you're removing artifacts, not signal.

Block Matching Parameters
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``patch_size``
     - ``8``
     - Size of patches for block matching. Use 7 or 8 for best results.
   * - ``step_size``
     - ``4``
     - Stride for patch extraction. Lower values can improve quality but take longer.
   * - ``search_window``
     - ``24``
     - Maximum search distance for finding similar patches.
   * - ``max_matches``
     - ``16``
     - Maximum number of similar patches per 3D group.

**Quality vs Speed Tradeoffs:**

- Faster processing: ``step_size=6-8``, ``max_matches=8``
- Higher quality: ``step_size=2-3``, ``max_matches=32``

Streak Mode Parameters
----------------------

These parameters only apply when ``mode="streak"``:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``streak_sigma_smooth``
     - ``3.0``
     - Sigma for Gaussian smoothing in streak profile estimation.
   * - ``streak_iterations``
     - ``2``
     - Number of iterations for robust streak estimation.
   * - ``sigma_map_smoothing``
     - ``20.0``
     - Smoothing width for the noise map. This map stores a local noise estimate
       for each input position.
   * - ``streak_sigma_scale``
     - ``1.1``
     - Scale factor for streak sigma estimation.
   * - ``psd_width``
     - ``0.6``
     - Gaussian width of the power spectral density (PSD) model. PSD describes
       how noise power varies across frequencies.

Advanced Parameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``threshold``
     - ``None``
     - Hard thresholding coefficient. ``None`` uses the backend default:
       ``2.7`` for single-scale BM3D and ``3.5`` for multi-scale BM3D.
   * - ``batch_size``
     - ``32``
     - Number of stack slices processed together. Smaller chunks use less memory.
   * - ``sigma_map``
     - ``None``
     - Optional map of local noise estimates for three-dimensional processing.

Multiscale Mode (Experimental)
------------------------------

.. warning::

   Multiscale mode is experimental. It works well for wide ring artifacts
   (>39 pixels) but may over-process regular sinograms.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Default
     - Description
   * - ``multiscale``
     - ``False``
     - Enable multi-scale BM3D for wide streaks. Only works with ``mode="streak"``.
   * - ``num_scales``
     - ``None``
     - Override automatic scale selection. ``None`` uses
       ``min(3, max(0, floor(log2(width / 40))))``.
   * - ``filter_strength``
     - ``1.0``
     - Multiplier for BM3D filtering intensity.
   * - ``debin_iterations``
     - ``30``
     - Iterations used to expand each binned correction to the original width.

Example Usage
-------------

**Basic streak removal:**

.. code-block:: python

    from bm3dornl import bm3d_ring_artifact_removal

    denoised = bm3d_ring_artifact_removal(
        sinogram,
        mode="streak",
        sigma_random=0.05,
    )

**High quality processing:**

.. code-block:: python

    denoised = bm3d_ring_artifact_removal(
        sinogram,
        mode="streak",
        sigma_random=0.05,
        step_size=2,
        max_matches=32,
    )

**Fast processing for large datasets:**

.. code-block:: python

    denoised = bm3d_ring_artifact_removal(
        sinogram_stack,
        mode="streak",
        sigma_random=0.05,
        step_size=6,
        max_matches=8,
        batch_size=16,  # Reduce memory usage
    )
