
import numpy as np
import time
from scipy.ndimage import gaussian_filter
import skimage.metrics
from bm3dornl.bm3d import bm3d_ring_artifact_removal

def generate_synthetic_sinogram(size=(512, 512), snr_db=10):
    """Generates a synthetic sinogram with vertical streaks."""
    x = np.linspace(-1, 1, size[1])
    y = np.linspace(-1, 1, size[0])
    X, Y = np.meshgrid(x, y)
    clean = np.zeros(size, dtype=np.float32)
    clean += 0.5 * np.exp(-((X)**2 + (Y)**2) / 0.1)
    
    np.random.seed(42)
    streak_profile = (np.random.rand(size[1]) - 0.5) * 0.2
    streak_profile = gaussian_filter(streak_profile, 1.0)
    streaks = np.tile(streak_profile, (size[0], 1))
    
    signal_power = np.mean(clean**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise_sigma = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_sigma, size).astype(np.float32)
    
    noisy_streaky = clean + streaks + noise
    return clean, noisy_streaky, noise_sigma

def test_stack_speed():
    # Create a stack of N frames
    N = 8 # Enough to saturate cores if parallelized over stack
    H, W = 512, 512
    
    print(f"Generating stack {N}x{H}x{W}...")
    stack_clean = np.zeros((N, H, W), dtype=np.float32)
    stack_noisy = np.zeros((N, H, W), dtype=np.float32)
    sigma = 0.0
    
    for i in range(N):
        c, n, s = generate_synthetic_sinogram(size=(H, W), snr_db=10)
        stack_clean[i] = c
        stack_noisy[i] = n
        sigma = s
        
    print(f"Processing stack (Mode=streak, Patch=8x8)...")
    start = time.time()
    denoised_stack = bm3d_ring_artifact_removal(stack_noisy, mode="streak", sigma_random=sigma, patch_size=8)
    end = time.time()

    total_time = end - start
    per_frame = total_time / N
    print(f"Total Time: {total_time:.4f} s")
    print(f"Per Frame: {per_frame:.4f} s")

    # Calculate Average PSNR
    psnrs = []
    for i in range(N):
        p = skimage.metrics.peak_signal_noise_ratio(stack_clean[i], denoised_stack[i])
        psnrs.append(p)
    avg_psnr = np.mean(psnrs)
    print(f"Average PSNR: {avg_psnr:.2f} dB")

    print(f"Processing stack (Mode=streak, Patch=8x8, max_matches=32)...")
    start = time.time()
    denoised_stack_m32 = bm3d_ring_artifact_removal(stack_noisy, mode="streak", sigma_random=sigma, patch_size=8, max_matches=32)
    end = time.time()

    total_time_m32 = end - start
    per_frame_m32 = total_time_m32 / N
    print(f"Total Time max_matches=32: {total_time_m32:.4f} s")
    print(f"Per Frame max_matches=32: {per_frame_m32:.4f} s")

    psnrs = []
    for i in range(N):
        p = skimage.metrics.peak_signal_noise_ratio(stack_clean[i], denoised_stack_m32[i])
        psnrs.append(p)
    avg_psnr_m32 = np.mean(psnrs)
    print(f"Average PSNR (max_matches=32): {avg_psnr_m32:.2f} dB")

    print(f"Processing stack (Mode=streak, Patch=8x8, step_size=3)...")
    start = time.time()
    denoised_stack_s3 = bm3d_ring_artifact_removal(stack_noisy, mode="streak", sigma_random=sigma, patch_size=8, step_size=3)
    end = time.time()

    total_time_s3 = end - start
    per_frame_s3 = total_time_s3 / N
    print(f"Total Time step_size=3: {total_time_s3:.4f} s")
    print(f"Per Frame step_size=3: {per_frame_s3:.4f} s")

    psnrs = []
    for i in range(N):
        p = skimage.metrics.peak_signal_noise_ratio(stack_clean[i], denoised_stack_s3[i])
        psnrs.append(p)
    avg_psnr_s3 = np.mean(psnrs)
    print(f"Average PSNR (step_size=3): {avg_psnr_s3:.2f} dB")

if __name__ == "__main__":
    test_stack_speed()
