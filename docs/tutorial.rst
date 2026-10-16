Tutorial
========

Use this tutorial to remove ring artifacts from a synthetic sinogram.

For an interactive version of this tutorial, see the Jupyter notebook:
`notebooks/tutorial.ipynb <https://github.com/ornlneutronimaging/bm3dornl/blob/main/notebooks/tutorial.ipynb>`_

Overview
--------

Detector response variations can cause rings in reconstructed tomography images.
These artifacts appear as vertical streaks in a sinogram. BM3DORNL targets those
streaks while preserving image structure.

Generating Test Data
--------------------

BM3DORNL includes a phantom module for generating synthetic test data:

.. code-block:: python

    import numpy as np
    from bm3dornl.phantom import (
        shepp_logan_phantom,
        generate_sinogram,
        simulate_detector_gain_error,
        get_synthetic_noise,
    )

    # Generate Shepp-Logan phantom
    phantom = shepp_logan_phantom(size=256)

    # Create sinogram via Radon transform
    clean_sinogram, angles = generate_sinogram(phantom, scan_step=0.5)

    # Add detector gain errors (causes ring artifacts)
    noisy_sinogram, gain = simulate_detector_gain_error(
        clean_sinogram,
        detector_gain_range=(0.95, 1.05),
        detector_gain_error=0.02,
    )

    # Add synthetic noise
    noise = get_synthetic_noise(
        image_size=noisy_sinogram.shape,
        streak_kernel_width=1,
        streak_kernel_length=100,
        white_noise_intensity=0.02,
        streak_noise_intensity=0.03,
    )
    noisy_sinogram = noisy_sinogram + noise.astype(np.float32)

Basic Usage
-----------

The main function is ``bm3d_ring_artifact_removal``:

.. code-block:: python

    from bm3dornl import bm3d_ring_artifact_removal

    # Use streak mode for ring artifacts
    denoised = bm3d_ring_artifact_removal(
        noisy_sinogram,
        mode="streak",
        sigma_random=0.05,
    )

Generic vs Streak Mode
----------------------

BM3D means block-matching and three-dimensional filtering.
BM3DORNL provides two modes:

- **generic:** Standard BM3D for white random noise
- **streak:** Processing for vertical streak artifacts

Use ``mode="streak"`` for vertical ring-artifact streaks. This mode preserves
horizontal angular information.

.. code-block:: python

    # Compare both modes
    denoised_generic = bm3d_ring_artifact_removal(
        noisy_sinogram,
        mode="generic",
        sigma_random=0.05,
    )

    denoised_streak = bm3d_ring_artifact_removal(
        noisy_sinogram,
        mode="streak",
        sigma_random=0.05,
    )

    # Compare the modes on your data

Parameter Tuning
----------------

``sigma_random`` sets denoising strength:

.. code-block:: python

    # Light denoising (preserves more detail)
    denoised_light = bm3d_ring_artifact_removal(
        sinogram, mode="streak", sigma_random=0.02
    )

    # Moderate denoising (balanced)
    denoised_moderate = bm3d_ring_artifact_removal(
        sinogram, mode="streak", sigma_random=0.05
    )

    # Aggressive denoising (removes more artifacts, may over-smooth)
    denoised_heavy = bm3d_ring_artifact_removal(
        sinogram, mode="streak", sigma_random=0.15
    )

Adjust ``step_size`` to trade speed for quality:

.. code-block:: python

    # Higher quality (slower)
    denoised = bm3d_ring_artifact_removal(
        sinogram, mode="streak", sigma_random=0.05, step_size=2
    )

    # Faster processing (slightly lower quality)
    denoised = bm3d_ring_artifact_removal(
        sinogram, mode="streak", sigma_random=0.05, step_size=6
    )

Processing 3D Stacks
--------------------

Pass a three-dimensional array to process a sinogram stack:

.. code-block:: python

    # stack_3d has shape (N, H, W) - N slices
    denoised_stack = bm3d_ring_artifact_removal(
        stack_3d,
        mode="streak",
        sigma_random=0.05,
        batch_size=32,  # Control memory usage
    )

Evaluating Results
------------------

Inspect the difference image for removed signal:

.. code-block:: python

    import matplotlib.pyplot as plt

    difference = noisy_sinogram - denoised

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(noisy_sinogram, cmap='gray')
    axes[0].set_title('Input')
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('Denoised')
    axes[2].imshow(difference, cmap='bwr')
    axes[2].set_title('Removed (should show vertical streaks)')
    plt.show()

The difference should contain mainly vertical streaks. Reduce ``sigma_random``
if it contains horizontal structure.

Best Practices
--------------

1. **Use streak mode** for ring artifacts in sinograms.

2. **Keep the original value range.** Single-scale processing normalizes
   nonconstant data to [0, 1] internally. It restores the original range.
   Multiscale mode expects linear transmission data unless
   ``log_domain_input=True``. Python converts values to ``float32`` before Rust
   processing.

3. **Start with a low ``sigma_random``.** Increase it gradually.

4. **Check the difference image.** Confirm that it contains artifacts, not signal.

5. **Set ``batch_size``** to control memory for large stacks.

6. **Compare both modes** on your data.
