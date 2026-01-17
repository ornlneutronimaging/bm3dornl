#!/usr/bin/env python3
"""Denoising functions using Rust backend."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from . import bm3d_rust
from .bm3d_rust import estimate_streak_profile_rust


def estimate_streak_profile(sinogram, sigma_smooth=3.0, iterations=3):
    """
    Estimate the static vertical streak profile using an iterative robust approach.

    This is a wrapper that calls the Rust implementation for performance.

    Parameters
    ----------
    sinogram : np.ndarray
        Input 2D sinogram (H x W), dtype should be float32.
    sigma_smooth : float
        Sigma for vertical Gaussian smoothing (default 3.0).
    iterations : int
        Number of refinement iterations (default 3).

    Returns
    -------
    np.ndarray
        1D streak profile of length W.
    """
    # Ensure float32 for Rust compatibility
    sinogram_f32 = np.ascontiguousarray(sinogram, dtype=np.float32)
    return estimate_streak_profile_rust(sinogram_f32, sigma_smooth, iterations)


def bm3d_ring_artifact_removal(
    sinogram: np.ndarray,
    mode: str = "generic",
    sigma_random: float = 0.1,
    patch_size: int = 8,
    step_size: int = 4,
    search_window: int = 24,
    max_matches: int = 16,
    batch_size: int = 32,
    threshold: float = 2.7,
    streak_sigma_smooth: float = 3.0,
    streak_iterations: int = 2,
    sigma_map_smoothing: float = 20.0,
    streak_sigma_scale: float = 1.1,
    psd_width: float = 0.6,
    sigma_map: np.ndarray | None = None,
    multiscale: bool = False,
    num_scales: int | None = None,
    filter_strength: float = 1.0,
    debin_iterations: int = 30,
) -> np.ndarray:
    """Remove ring artifacts (streaks) from a sinogram or a stack of sinograms.

    This function leverages a customized BM3D implementation optimized for streak removal.
    It supports both 2D (single sinogram) and 3D (stack of sinograms) inputs.

    For 2D inputs, the entire pipeline runs in Rust for maximum performance.
    For 3D inputs, Python orchestrates batch processing with Rust kernels.

    Parameters
    ----------
    sinogram : np.ndarray
        The input data. Can be 2D (H, W) or 3D (N, H, W).
    mode : str, optional
        Operation mode:
        - "generic": Standard BM3D (assume white noise).
        - "streak": Additive Streak Removal (Residual Median) + Standard BM3D.
        By default "generic".
    sigma_random : float, optional
        Random noise standard deviation, by default 0.1.
    patch_size : int, optional
        Size of patches for block matching, by default 8.
    step_size : int, optional
        Step size (stride) for patch extraction, by default 4.
    search_window : int, optional
        Search window size for block matching, by default 24.
    max_matches : int, optional
        Maximum number of similar patches per group, by default 16.
    batch_size : int, optional
        Chunk size for 3D stack processing, by default 32.
    threshold : float, optional
        Hard thresholding coefficient, by default 2.7.
    streak_sigma_smooth : float, optional
        Sigma for smoothing in streak estimation, by default 3.0.
    streak_iterations : int, optional
        Number of iterations for streak estimation, by default 2.
    sigma_map_smoothing : float, optional
        Sigma for sigma map profile smoothing, by default 20.0.
    streak_sigma_scale : float, optional
        Scale factor for streak sigma estimation, by default 1.1.
    psd_width : float, optional
        PSD Gaussian width for streak mode, by default 0.6.
    sigma_map : np.ndarray | None, optional
        Optional pre-computed sigma map (3D processing only), by default None.
    multiscale : bool, optional
        Enable multi-scale processing for handling wide streaks (default False).
    num_scales : int | None, optional
        Number of scales for multi-scale pyramid. If None, automatic.
    filter_strength : float, optional
        Filtering strength multiplier for multi-scale mode (default 1.0).
    debin_iterations : int, optional
        Number of debinning iterations (default 30).

    Returns
    -------
    np.ndarray
        Denoised data with same shape as input.

    Raises
    ------
    ValueError
        If multiscale=True with mode other than "streak".
    """
    # Validate multiscale + mode combination
    if multiscale and mode != "streak":
        raise ValueError(f"multiscale=True only supports mode='streak', got mode='{mode}'")



    # Check Dimension
    is_stack = sinogram.ndim == 3

    if not is_stack:
        # =====================================================================
        # 2D Case: Use unified Rust implementation
        # =====================================================================
        sinogram_f32 = np.ascontiguousarray(sinogram, dtype=np.float32)

        if multiscale:
            # Multi-scale BM3D (streak mode only, validated above)
            return bm3d_rust.multiscale_bm3d_streak_removal_2d(
                sinogram_f32,
                num_scales=num_scales,
                filter_strength=filter_strength,
                threshold=threshold,
                debin_iterations=debin_iterations,
                patch_size=patch_size,
                step_size=step_size,
                search_window=search_window,
                max_matches=max_matches,
                sigma_random=float(sigma_random),
            )
        else:
            # Single-scale BM3D
            return bm3d_rust.bm3d_ring_artifact_removal_2d(
                sinogram_f32,
                mode,
                sigma_random=float(sigma_random),
                patch_size=patch_size,
                step_size=step_size,
                search_window=search_window,
                max_matches=max_matches,
                threshold=threshold,
                streak_sigma_smooth=streak_sigma_smooth,
                streak_iterations=streak_iterations,
                sigma_map_smoothing=sigma_map_smoothing,
                streak_sigma_scale=streak_sigma_scale,
                psd_width=psd_width,
            )

    # =========================================================================
    # 3D Case: Python-orchestrated batch processing
    # =========================================================================

    # Handle 3D + multiscale: process slice-by-slice
    if multiscale:
        n_slices = sinogram.shape[0]
        output_result = np.empty_like(sinogram, dtype=np.float32)
        for i in range(n_slices):
            slice_f32 = np.ascontiguousarray(sinogram[i], dtype=np.float32)
            output_result[i] = bm3d_rust.multiscale_bm3d_streak_removal_2d(
                slice_f32,
                num_scales=num_scales,
                filter_strength=filter_strength,
                threshold=threshold,
                debin_iterations=debin_iterations,
                patch_size=patch_size,
                step_size=step_size,
                search_window=search_window,
                max_matches=max_matches,
                sigma_random=float(sigma_random),
            )
        return output_result

    # 1. Global Stats (Min/Max)
    d_min = float(sinogram.min())
    d_max = float(sinogram.max())

    # --- Spatially Adaptive BM3D Setup Helpers ---
    def compute_slice_map(slice_img):
        streak_profile_rough = estimate_streak_profile(slice_img, sigma_smooth=5.0, iterations=1)
        profile_smooth = gaussian_filter1d(streak_profile_rough, sigma_map_smoothing)
        streak_signal = streak_profile_rough - profile_smooth
        sigma_streak_1d = np.abs(streak_signal).astype(np.float32)
        return np.tile(sigma_streak_1d * streak_sigma_scale, (slice_img.shape[0], 1))

    # Prepare PSD (Shared)
    sigma_psd = np.zeros((1, 1), dtype=np.float32)
    if mode == "streak" and patch_size > 0:
        sigma_psd = np.zeros((patch_size, patch_size), dtype=np.float32)
        y_coords = np.arange(patch_size)
        psd_profile = np.exp(-0.5 * (y_coords / psd_width) ** 2)
        for x in range(patch_size):
            sigma_psd[:, x] = psd_profile
        sigma_psd = sigma_psd.astype(np.float32)

    # Output allocation
    output_result = np.empty_like(sinogram)

    n_slices = sinogram.shape[0]
    # Validate sigma_map if provided
    sigma_map_full = None
    if sigma_map is not None:
        sigma_map_full = np.ascontiguousarray(sigma_map, dtype=np.float32)
        if sigma_map_full.ndim != 3:
            raise ValueError("sigma_map must be 3D for stack input")

    # Process in batches to control memory usage
    for i in range(0, n_slices, batch_size):
        end = min(i + batch_size, n_slices)

        # 1. Extract and Normalize Chunk
        chunk = sinogram[i:end].astype(np.float32)

        z_chunk = chunk
        # Normalize
        if d_max > d_min:
            z_chunk = (chunk - d_min) / (d_max - d_min)

        # 2. Sigma Map Chunk
        if sigma_map_full is not None:
            chunk_map = sigma_map_full[i:end]
        else:
            # Compute per slice
            c_n, c_h, c_w = z_chunk.shape
            chunk_map = np.zeros((c_n, c_h, c_w), dtype=np.float32)
            for k in range(c_n):
                chunk_map[k] = compute_slice_map(z_chunk[k])

        chunk_map = np.ascontiguousarray(chunk_map, dtype=np.float32)

        # 3. Streak Pre-Subtraction
        if mode == "streak":
            for k in range(z_chunk.shape[0]):
                prof = estimate_streak_profile(
                    z_chunk[k],
                    sigma_smooth=streak_sigma_smooth,
                    iterations=streak_iterations,
                )
                corr = np.tile(prof, (z_chunk[k].shape[0], 1))
                z_chunk[k] -= corr

        # 4. Run Rust Stack Processing
        yhat_ht = bm3d_rust.bm3d_hard_thresholding_stack(
            z_chunk,
            z_chunk,
            sigma_psd,
            chunk_map,
            sigma_random,
            threshold,
            patch_size,
            step_size,
            search_window,
            max_matches,
        )

        yhat_final_chunk = bm3d_rust.bm3d_wiener_filtering_stack(
            z_chunk,
            yhat_ht,
            sigma_psd,
            chunk_map,
            sigma_random,
            patch_size,
            step_size,
            search_window,
            max_matches,
        )

        # 5. Denormalize and Store
        if d_max > d_min:
            yhat_final_chunk = yhat_final_chunk * (d_max - d_min) + d_min

        output_result[i:end] = yhat_final_chunk

        # Clean up explicit large temps if Python GC is lazy
        del z_chunk
        del chunk_map
        del yhat_ht
        del yhat_final_chunk

    return output_result
