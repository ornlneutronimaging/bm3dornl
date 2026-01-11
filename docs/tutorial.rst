Tutorial
========

This tutorial demonstrates how to use BM3DORNL for removing ring artifacts from
tomography sinograms using synthetic phantom data.

For an interactive version of this tutorial, see the Jupyter notebook:
`notebooks/tutorial.ipynb <https://github.com/ornlneutronimaging/bm3dornl/blob/main/notebooks/tutorial.ipynb>`_

Overview
--------

Ring artifacts are a common problem in neutron and X-ray tomography caused by
detector pixel response variations. In sinogram space, these appear as vertical
streaks. BM3DORNL provides specialized algorithms to remove these artifacts while
preserving image structure.

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

    # For ring artifacts, always use mode="streak"
    denoised = bm3d_ring_artifact_removal(
        noisy_sinogram,
        mode="streak",
        sigma_random=0.05,
    )

Generic vs Streak Mode
----------------------

BM3DORNL provides two modes:

- **generic**: Standard BM3D for white (random) noise
- **streak**: Specialized mode for vertical streak artifacts

For ring artifact removal, always use ``mode="streak"``. It specifically targets
vertical structures while preserving horizontal (angular) information.

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

    # Streak mode will show better artifact removal with less structure loss

Parameter Tuning
----------------

The most important parameter is ``sigma_random``, which controls denoising strength:

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

For quality vs speed tradeoffs, adjust ``step_size``:

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

BM3DORNL handles 3D sinogram stacks automatically:

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

Always check the difference image to verify you're removing artifacts, not signal:

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

The difference image should show primarily vertical streaks (the artifacts being removed).
If you see horizontal structure, reduce ``sigma_random``.

Best Practices
--------------

1. **Always use streak mode** for ring artifacts in sinograms

2. **Normalize your data** to [0, 1] range for best results

3. **Start with low sigma_random** and increase gradually

4. **Check the difference image** to ensure you're removing artifacts, not signal

5. **Use batch_size** to control memory for large 3D stacks

6. **Compare with generic mode** to verify streak mode is providing benefit
