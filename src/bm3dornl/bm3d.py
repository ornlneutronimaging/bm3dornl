#!/usr/bin/env python3
"""Denoising functions using Rust backend."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from . import bm3d_rust
from .bm3d_rust import estimate_streak_profile_rust

# Default parameters - aligned with Rust implementation defaults
default_block_matching_kwargs = {
    "patch_size": 8,
    "stride": 4,
    "cut_off_distance": (24, 24),
    "num_patches_per_group": 16,
}
# Legacy filter kwargs preserved just for signature compatibility if needed
default_filter_kwargs = {}

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
    sigma: float = 0.1,
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
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
    sigma : float, optional
        Noise standard deviation, by default 0.1.
    block_matching_kwargs : dict, optional
        Parameters for block matching:
        - patch_size (tuple[int, int] | int): Size of patches (default 8).
        - stride (int): Step size for patch extraction (default 4).
        - cut_off_distance (tuple[int, int]): Max distance for block matching (default (24, 24)).
        - num_patches_per_group (int): Number of similar patches to group (default 16).
        - batch_size (int): Chunk size for stack processing (default 32).
        By default `default_block_matching_kwargs`.
    filter_kwargs : dict, optional
        Parameters for filtering:
        - sigma_map (np.ndarray): Optional pre-computed sigma map (3D processing only).
        - streak_sigma_scale (float): Scale factor for streak sigma estimation (default 1.1).
        - sigma_smooth (float): Sigma for smoothing in streak estimation (default 3.0).
        - streak_iterations (int): Iterations for streak estimation (default 2).
        - sigma_map_smoothing (float): Sigma for sigma map profile smoothing (default 20.0).
        - psd_width (float): PSD Gaussian width for streak mode (default 0.6).
        - threshold (float): Hard thresholding coefficient (default 2.7).
        By default `default_filter_kwargs`.
    multiscale : bool, optional
        If True, use multi-scale BM3D for handling wide streaks.
        Only supported with mode="streak". By default False.
    num_scales : int | None, optional
        Override automatic scale calculation for multi-scale mode.
        If None, uses floor(log2(width/40)). By default None.
    filter_strength : float, optional
        Multiplier for BM3D filtering intensity (multi-scale only). By default 1.0.
    debin_iterations : int, optional
        Iterations for cubic spline debinning (multi-scale only). By default 30.

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
        raise ValueError(
            f"multiscale=True only supports mode='streak', got mode='{mode}'"
        )
    # Unpack parameters
    patch_size = block_matching_kwargs.get("patch_size", 8)
    if isinstance(patch_size, tuple):
        patch_size_dim = patch_size[0]
    else:
        patch_size_dim = patch_size

    step_size = block_matching_kwargs.get("stride", 4)
    cut_off = block_matching_kwargs.get("cut_off_distance", (24, 24))
    if isinstance(cut_off, tuple):
        search_window = cut_off[0]
    else:
        search_window = cut_off
    num_patches = block_matching_kwargs.get("num_patches_per_group", 16)
    batch_size = block_matching_kwargs.get("batch_size", 32)

    # Filter parameters
    scale_factor = filter_kwargs.get("streak_sigma_scale", 1.1)
    sigma_smooth = filter_kwargs.get("sigma_smooth", 3.0)
    streak_iterations = filter_kwargs.get("streak_iterations", 2)
    sigma_map_smoothing = filter_kwargs.get("sigma_map_smoothing", 20.0)
    psd_width = filter_kwargs.get("psd_width", 0.6)
    threshold_hard = filter_kwargs.get("threshold", 2.7)
    sigma_map_arg = filter_kwargs.get("sigma_map", None)

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
                threshold=threshold_hard,
                debin_iterations=debin_iterations,
                patch_size=patch_size_dim,
                step_size=step_size,
                search_window=search_window,
                max_matches=num_patches,
            )
        else:
            # Single-scale BM3D
            return bm3d_rust.bm3d_ring_artifact_removal_2d(
                sinogram_f32,
                mode,
                sigma_random=float(sigma),
                patch_size=patch_size_dim,
                step_size=step_size,
                search_window=search_window,
                max_matches=num_patches,
                threshold=threshold_hard,
                streak_sigma_smooth=sigma_smooth,
                streak_iterations=streak_iterations,
                sigma_map_smoothing=sigma_map_smoothing,
                streak_sigma_scale=scale_factor,
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
                threshold=threshold_hard,
                debin_iterations=debin_iterations,
                patch_size=patch_size_dim,
                step_size=step_size,
                search_window=search_window,
                max_matches=num_patches,
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
        return np.tile(sigma_streak_1d * scale_factor, (slice_img.shape[0], 1))

    # Prepare PSD (Shared)
    sigma_psd = np.zeros((1, 1), dtype=np.float32)
    if mode == "streak" and patch_size_dim > 0:
        sigma_psd = np.zeros((patch_size_dim, patch_size_dim), dtype=np.float32)
        y_coords = np.arange(patch_size_dim)
        psd_profile = np.exp(-0.5 * (y_coords / psd_width)**2)
        for x in range(patch_size_dim):
            sigma_psd[:, x] = psd_profile
        sigma_psd = sigma_psd.astype(np.float32)

    sigma_random = float(sigma)

    # Output allocation
    output_result = np.empty_like(sinogram)

    n_slices = sinogram.shape[0]
    # Validate sigma_map if provided
    sigma_map_full = None
    if sigma_map_arg is not None:
        sigma_map_full = np.ascontiguousarray(sigma_map_arg, dtype=np.float32)
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
                prof = estimate_streak_profile(z_chunk[k], sigma_smooth=sigma_smooth, iterations=streak_iterations)
                corr = np.tile(prof, (z_chunk[k].shape[0], 1))
                z_chunk[k] -= corr

        # 4. Run Rust Stack Processing
        yhat_ht = bm3d_rust.bm3d_hard_thresholding_stack(
            z_chunk, z_chunk, sigma_psd,
            chunk_map, sigma_random,
            threshold_hard,
            patch_size_dim, step_size, search_window, num_patches
        )

        yhat_final_chunk = bm3d_rust.bm3d_wiener_filtering_stack(
            z_chunk, yhat_ht, sigma_psd,
            chunk_map, sigma_random,
            patch_size_dim, step_size, search_window, num_patches
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
