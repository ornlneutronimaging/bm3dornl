#!/usr/bin/env python3
"""Denoising functions using Rust backend."""

import logging
import numpy as np
from skimage.transform import pyramid_reduce, pyramid_expand, resize
from skimage.transform import pyramid_reduce, pyramid_expand, resize
from scipy.ndimage import median_filter, sobel, gaussian_filter
from scipy.signal.windows import gaussian
from . import bm3d_rust

# Default parameters (simplified for Rust backend)
default_block_matching_kwargs = {
    "patch_size": (7, 7),
    "stride": 2,
    "cut_off_distance": (40, 40),
    "num_patches_per_group": 64,
}
# Legacy filter kwargs preserved just for signature compatibility if needed
default_filter_kwargs = {}

from scipy.ndimage import zoom

from scipy.ndimage import gaussian_filter1d

from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def _estimate_static_streaks(sino):
    """
    Estimate static streaks using Gradient-Guided Column Statistics.
    
    Standard column median is biased by object structure (sinogram edges are vertical-ish).
    By excluding pixels with high vertical gradients, we select 'Air' or 'Flat' regions
    to estimate the detector artifacts, significantly reducing structure leakage.
    """
    # 1. Compute Vertical Gradient (to detect object edges)
    # Using Sobel for robustness
    grad_y = np.abs(sobel(sino, axis=0))
    
    # 2. Determine Threshold (keep only the flattest regions)
    # T=20% was found to minimize correlation with object structure while retaining enough signal.
    # We want low gradient pixels.
    threshold_percentile = 20
    threshold = np.percentile(grad_y, threshold_percentile)
    mask = grad_y < threshold
    
    H, W = sino.shape
    col_profile = np.zeros(W, dtype=np.float32)
    
    # 3. Compute Robust Mean per column using Mask
    # We use Mean instead of Median for lower variance, since we already masked out "Structure/Outliers"
    for c in range(W):
        col_data = sino[:, c]
        col_mask = mask[:, c]
        
        valid_data = col_data[col_mask]
        
        # Fallback if mask is too aggressive
        if len(valid_data) > max(10, H * 0.01):
            # Simple Mean (or Sigma Clipped Mean if needed, but Mask handled gradients)
            # Let's use a simple sigma clip to be safe against salt-and-pepper noise
            m = np.mean(valid_data)
            s = np.std(valid_data)
            valid_data = valid_data[np.abs(valid_data - m) < 3 * s]
            if len(valid_data) > 0:
                col_profile[c] = np.mean(valid_data)
            else:
                col_profile[c] = m
        else:
            col_profile[c] = np.median(col_data) # Fallback to standard median
    
    # 4. Remove Background Trend (Low-pass)
    filter_size = 21
    if len(col_profile) < filter_size:
        filter_size = max(3, len(col_profile) // 2 * 2 + 1)
        
    background_trend = median_filter(col_profile, size=filter_size)
    
    static_streaks = col_profile - background_trend
    
    return static_streaks.astype(np.float32)

def estimate_streak_sigma(sinogram: np.ndarray, patch_size: tuple[int, int]) -> float:
    """Estimate the standard deviation of the streak noise using robust statistics on vertical profile.
    
    1. Compute mean vertical profile (averaging out random noise).
    2. High-pass filter (subtract smooth trend) to isolate streaks from object structure.
    3. Use MAD (Median Absolute Deviation) to estimate noise level, ignoring outliers (edges).
    """
    profile = np.mean(sinogram, axis=0) # Shape (W,)
    
    # 2. Separate Streaks from Object
    sigma_smooth = 15.0 # Tunable parameter for object smoothness
    profile_smooth = gaussian_filter1d(profile, sigma_smooth)
    
    streak_signal = profile - profile_smooth
    
    # 3. Robust Estimation (MAD)
    median_val = np.median(streak_signal)
    mad = np.median(np.abs(streak_signal - median_val))
    
    sigma_est = mad * 1.4826
    
    return sigma_est

def construct_streak_psd(
    patch_size: tuple[int, int], 
    sigma: float, 
    sinogram: np.ndarray,
    streak_strength: float = 1.0
) -> np.ndarray:
    """Construct colored noise PSD using robust MAD estimation."""
    H, W = patch_size
    # Base Random Noise PSD (White) -> Handled by separate scalar in Rust.
    # This map should ONLY contain the STATIC noise component.
    sigma_map = np.zeros((H, W), dtype=np.float32)
    
    # 1. Estimate robust streak variance
    sigma_streak = estimate_streak_sigma(sinogram, patch_size)
    
    # 2. Scale by strength
    sigma_target = sigma_streak * streak_strength
    
    # 3. Populate DC Row with Frequency Weighting
    # Streaks (Vertical Lines) have energy at Vertical DC (r=0).
    # In Horizontal domain (columns), they are sharp transitions => High Frequencies.
    # Object variations are usually smooth => Low Horizontal Frequencies.
    # If we apply high sigma to Low Horizontal Frequencies at r=0, we destroy object structure (shading).
    # Solution: Ramp up sigma from 0 (at low freq) to sigma_target (at high freq).
    
    # Simple Ramp or Sigmoid
    # Frequencies c range from 0 to W-1 (or W/2 if real FFT, but BM3D transform is full?)
    # BM3D Rust core uses full 2D FFT on patches (7x7).
    # Col indices 0..patch_size.
    # c=0 is DC (Protected).
    # c=1 is Lowest AC freq.
    
    freq_weight = np.ones(W, dtype=np.float32)
    # Ramp 0 to 1 over first few coefficients?
    # e.g. c=0: 0.0, c=1: 0.5, c=2: 1.0 ...
    # Let's use a smoother transition to avoid ringing.
    # Sigmoid centered at freq_cutoff?
    # Or just linear ramp.
    
    for c in range(W):
        if c == 0:
            freq_weight[c] = 0.0 # Protected explicitly later, but weight 0 implies no streak sigma
        elif c < 2:
            freq_weight[c] = 0.2 * c # 0.2, 0.4... gentler start
        else:
            freq_weight[c] = 1.0

    sigma_map[0, :] = sigma_target * freq_weight
    
    # PROTECT DC Component (Patch Mean / Global Brightness)
    # Base random noise floor
    sigma_map[0, 0] = sigma
    
    return sigma_map

def fft_streak_removal(sinogram, sigma_y=1.0, sigma_protect_x=20.0):
    """
    Remove vertical streaks using a FFT-based Notch Filter.
    
    Streaks correspond to the DC component in the vertical frequency axis (v_y = 0).
    We dampen this axis using a Gaussian notch, while protecting the low-frequency 
    horizontal components (centered at v_x=0) to preserve the object's mean structure.
    """
    H, W = sinogram.shape
    
    # 1. FFT
    f = fft2(sinogram)
    fshift = fftshift(f)
    
    # 2. Construct Filter
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
    
    # Gaussian Notch along v_y
    notch_y = np.exp(-(y**2) / (2 * sigma_y**2))
    
    # Protection Mask along v_x (Center)
    protect_x = np.exp(-(x**2) / (2 * sigma_protect_x**2))
    
    # Combined Mask: 1 - Notch + Notch*Protection
    # Equivalently: 1 - Notch * (1 - Protection)
    # If Protection=1 (Center), Mask=1 (No filtering).
    # If Protection=0 (High Freq X), Mask=1-Notch (Filter out v_y=0).
    mask = 1.0 - notch_y * (1.0 - protect_x)
    
    # 3. Apply
    fshift_filtered = fshift * mask
    
    # 4. IFFT
    img_filtered = np.real(ifft2(ifftshift(fshift_filtered)))
    
    return img_filtered.astype(np.float32)

def bm3d_ring_artifact_removal(
    sinogram: np.ndarray,
    mode: str = "generic",  # "generic" or "streak"
    sigma: float = 0.1,    
    block_matching_kwargs: dict = default_block_matching_kwargs,
    filter_kwargs: dict = default_filter_kwargs,
) -> np.ndarray:
    """Remove ring artifacts (streaks) from a sinogram.
    
    Parameters
    ----------
    sinogram : np.ndarray
        The sinogram (2D float array).
    mode : str
        "generic": Standard BM3D (assume white noise).
        "streak": Multiscale BM3D with Streak PSD (remove vertical lines).
    sigma : float
        Noise standard deviation.
    """
    # Unpack parameters
    patch_size = block_matching_kwargs.get("patch_size", (7, 7))
    if isinstance(patch_size, int): patch_size_dim = patch_size
    else: patch_size_dim = patch_size[0] # Assume square for now as Rust core uses 'patch_size: usize'

    step_size = block_matching_kwargs.get("stride", 2)
    cut_off = block_matching_kwargs.get("cut_off_distance", (40, 40))[0]
    num_patches = block_matching_kwargs.get("num_patches_per_group", 64)
    threshold_hard = 2.7
    
    logger = logging.getLogger(__name__)
    
    # Ensure float32 and normalize
    z = sinogram.astype(np.float32)
    d_min = z.min()
    d_max = z.max()
    z_norm = z
    if d_max > d_min:
        z_norm = (z - d_min) / (d_max - d_min)

        if mode == "streak":
            use_fft = filter_kwargs.get("use_fft", True) # Default to FFT (New SOTA)
            
            if use_fft:
                # Spectral Filtering for Streaks
                # This handles the "Streak Removal" part.
                # We typically don't need Multiscale for this if FFT works well.
                # But we might need BM3D for the *random* noise.
                
                # 1. FFT Streak Removal
                sigma_notch = filter_kwargs.get("fft_notch_sigma", 1.0)
                sigma_protect = filter_kwargs.get("fft_protect_width", 20.0)
                
                # Apply FFT Denoising
                z_denoised_streaks = fft_streak_removal(z_norm, sigma_notch, sigma_protect)
                
                # 2. Run Generic BM3D on the streak-corrected image to remove Random Noise
                # We treat the result as a "Generic" image now.
                # Recursive call with mode="generic"
                # Avoid infinite recursion? No, mode changed.
                
                # But we need to handle the denormalization wrapper.
                # It's cleaner to just call the generic path logic here or recurse.
                # Let's recurse.
                
                # Denormalize first? No, pass normalized.
                # Wait, recursive call expects raw input.
                # Better to just set z_norm = z_denoised_streaks and fall through to Generic Logic?
                
                # IMPORTANT: If we use FFT, we skip the Multiscale BM3D logic below.
                # We switch to Standard BM3D on the filtered image.
                
                z_norm = z_denoised_streaks
                # Fall through to Generic Mode logic (by changing 'mode' variable? No, risky).
                # Explicitly run Generic logic here.
                
                # Generic Mode (Scalar Sigma)
                sigma_psd_dummy = np.zeros((1, 1), dtype=np.float32)
                sigma_norm = sigma
                
                yhat_ht = bm3d_rust.bm3d_hard_thresholding(
                    z_norm, z_norm, sigma_psd_dummy, sigma_norm, threshold_hard,
                    patch_size_dim, step_size, cut_off, num_patches
                )
                
                z_gft_ht = yhat_ht 
                yhat_ht_gft = bm3d_rust.bm3d_hard_thresholding(
                    z_norm, z_gft_ht, sigma_psd_dummy, sigma_norm, threshold_hard,
                    patch_size_dim, step_size, cut_off, num_patches
                )
                
                yhat_wie = bm3d_rust.bm3d_wiener_filtering(
                     z_norm, yhat_ht_gft, sigma_psd_dummy, sigma_norm,
                     patch_size_dim, step_size, cut_off, num_patches
                )
                
                z_gft_wie = yhat_wie
                yhat_final = bm3d_rust.bm3d_wiener_filtering(
                     z_norm, z_gft_wie, sigma_psd_dummy, sigma_norm,
                     patch_size_dim, step_size, cut_off, num_patches
                )
                
                # Denormalize and Return
                if d_max > d_min:
                    yhat_final = yhat_final * (d_max - d_min) + d_min
                return yhat_final

        # Fallback to Legacy Multiscale if use_fft=False
        # Multiscale Strategy: Laplacian Pyramid
        
        # Manual Decomposition to safely handle ranges
        from skimage.transform import pyramid_reduce, pyramid_expand
        
        # Extract variables needed for legacy multiscale (Fallback)
        scales = filter_kwargs.get("scales", 3)
        # Default strength for feedback if fallback used (though FFT is separate)
        strength = filter_kwargs.get("streak_strength", 0.6)
        
        # 1. Build Gaussian Pyramid
        gaussians = [z_norm]
        for i in range(scales - 1):
            # Downscale
            g_next = pyramid_reduce(gaussians[-1], downscale=2)
            gaussians.append(g_next)
            
        # 2. Build Laplacian Pyramid
        pyramid = []
        for i in range(scales - 1):
            g_curr = gaussians[i]
            g_next = gaussians[i+1]
            # Expand next to match current
            g_next_expanded = pyramid_expand(g_next, upscale=2, order=1)
            
            # Match shape
            if g_next_expanded.shape != g_curr.shape:
                from skimage.transform import resize
                g_next_expanded = resize(g_next_expanded, g_curr.shape, order=1, mode='reflect', preserve_range=True)
                
            lap = g_curr - g_next_expanded
            pyramid.append(lap)
            
        # Last band is the Gaussian Low Pass
        pyramid.append(gaussians[-1])
        
        denoised_pyramid = []
        
        for i, band in enumerate(pyramid):
            # 1. Construct Streak PSD for this band
            # We estimate the streak component from the band itself.
            # We use the input base sigma for the random component.
            
            # Ensure band is contiguous float32 for Rust
            band_c = np.ascontiguousarray(band, dtype=np.float32)
            
            # Robust estimate on the band
            sigma_map_band = construct_streak_psd(
                (patch_size_dim, patch_size_dim), 
                sigma, # Base random sigma passed as is
                sinogram=band_c, 
                streak_strength=strength
            )
            
            # --- Step 1: HT ---
            yhat_ht = bm3d_rust.bm3d_hard_thresholding(
                band_c, band_c, sigma_map_band, sigma, threshold_hard,
                patch_size_dim, step_size, cut_off, num_patches
            )
            
            # --- Step 2: Wiener ---
            # Reuse HT output as pilot
            yhat_wie = bm3d_rust.bm3d_wiener_filtering(
                 band_c, yhat_ht, sigma_map_band, sigma,
                 patch_size_dim, step_size, cut_off, num_patches
            )
            
            denoised_pyramid.append(yhat_wie)
            
        # Reconstruct
        # Start with the simplified lower frequency band (last element)
        y_est = denoised_pyramid[-1]
        for detail in reversed(denoised_pyramid[:-1]):
            # Expand current guess to match detail scale
            expanded = pyramid_expand(y_est, upscale=2, order=1)
            
            # Handle shape mismatch due to odd dimensions during downscaling
            if expanded.shape != detail.shape:
                er, ec = expanded.shape
                dr, dc = detail.shape
                from skimage.transform import resize
                expanded = resize(expanded, detail.shape, order=1, mode='reflect', preserve_range=True)
            
            y_est = expanded + detail
        

        
        # --- Spatial Masking (Anti-Structure-Damage) ---
        # If enabled, we blend the result with the original image (or generic denoised) 
        # based on the vertical gradient of the original image.
        # High Gradient (Edge) -> Keep Original (Preserve Structure)
        # Low Gradient (Flat) -> Keep Denoised (Remove Streak)
        use_spatial_mask = filter_kwargs.get("use_spatial_mask", True)
        if use_spatial_mask:
            # 1. Compute Vertical Gradient of Original (Smoothed to avoid noise)
            g_smooth = gaussian_filter(z_norm, (2, 2))
            grad_y = np.abs(sobel(g_smooth, axis=0))
            
            # 2. Determine Sigmoid Parameters
            # We want to protect strong edges.
            # Mask -> 1.0 (Keep Original) when Gradient is High
            # Mask -> 0.0 (Keep Denoised) when Gradient is Low
            
            g_mean = np.mean(grad_y)
            g_std = np.std(grad_y)
            
            # Center transition at mean + 1 std (Capture clear edges)
            center = g_mean + 1.0 * g_std
            width = g_std * 0.5
            
            mask = 1.0 / (1.0 + np.exp(-(grad_y - center) / width))
            
            # 3. Blend
            # y_est = mask * z_norm + (1 - mask) * y_est
            # Wait, blending with z_norm keeps NOISE in high-grad regions.
            # Ideally we blend with a "Generic" denoised version?
            # But z_norm has very low random noise (RMSE ~0.005).
            # The structure/sharpness is most preserved in the original.
            # So blending with z_norm is acceptable for High SNR cases.
            y_est = mask * z_norm + (1.0 - mask) * y_est

        # Denormalize
        if d_max > d_min:
            y_est = y_est * (d_max - d_min) + d_min
            
        return y_est

    else:
        # Generic Mode (Scalar Sigma)
        # We must now pass a 1x1 array for sigma to match new Rust API
        # And sigma_random
        
        # 1x1 Dummy PSD (ignored by Rust when use_colored_noise=false)
        sigma_psd_dummy = np.zeros((1, 1), dtype=np.float32)
        sigma_norm = sigma
        
        yhat_ht = bm3d_rust.bm3d_hard_thresholding(
            z_norm, z_norm, sigma_psd_dummy, sigma_norm, threshold_hard,
            patch_size_dim, step_size, cut_off, num_patches
        )
        
        z_gft_ht = yhat_ht 
        yhat_ht_gft = bm3d_rust.bm3d_hard_thresholding(
            z_norm, z_gft_ht, sigma_psd_dummy, sigma_norm, threshold_hard,
            patch_size_dim, step_size, cut_off, num_patches
        )
        
        yhat_wie = bm3d_rust.bm3d_wiener_filtering(
             z_norm, yhat_ht_gft, sigma_psd_dummy, sigma_norm,
             patch_size_dim, step_size, cut_off, num_patches
        )
        
        z_gft_wie = yhat_wie
        yhat_final = bm3d_rust.bm3d_wiener_filtering(
             z_norm, z_gft_wie, sigma_psd_dummy, sigma_norm,
             patch_size_dim, step_size, cut_off, num_patches
        )
    
    # Restore original range
    if d_max > d_min:
        yhat_final = yhat_final * (d_max - d_min) + d_min
    
    return yhat_final
