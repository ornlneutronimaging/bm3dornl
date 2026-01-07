import h5py
import numpy as np
import time
import os
from bm3dornl.bm3d import bm3d_ring_artifact_removal

def verify_real_data():
    data_path = "tests/bm3dornl-data/tomostack_small.h5"
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    with h5py.File(data_path, "r") as f:
        print("Keys:", list(f.keys()))
        
        noisy_stack = f["noisy_tomostack"]
        clean_stack = f["clean_tomostack"]
        
        # Stack shape is (Angles, Y_slices, X_detectors) = (591, 540, 620)
        # We need a SINOGRAM: (Angles, X_detectors) for a fixed Y_slice.
        
        # Pick a middle Y slice
        slice_y_idx = noisy_stack.shape[1] // 2 
        
        # Slicing: [:, y, :]
        noisy_sino = noisy_stack[:, slice_y_idx, :]
        clean_sino = clean_stack[:, slice_y_idx, :]
        
        print(f"Processing Sinogram (Slice Y={slice_y_idx}), Shape: {noisy_sino.shape}")
        
        # Actual noise is low (RMSE ~0.006). Using sigma=0.005.
        start_t = time.time()
        # Enable Streak Mode with Multiscale
        denoised_sino = bm3d_ring_artifact_removal(
            noisy_sino, 
            mode="streak",
            sigma=0.002,
            filter_kwargs={
                "use_fft": True, 
                "use_dual_fft": True
            } # Dual-FFT Blending + Optimized BM3D Sigma
        ) 
        end_t = time.time()
        
        print(f"Denoising time: {end_t - start_t:.4f}s")
        
        # Compute metrics
        mse_noisy = np.mean((noisy_sino - clean_sino)**2)
        mse_denoised = np.mean((denoised_sino - clean_sino)**2)
        
        # Adjust PSNR if range is not [0,1]
        data_range = clean_sino.max() - clean_sino.min()
        print(f"Range: {data_range:.4f}")
        
        psnr_noisy = 10 * np.log10((data_range**2) / mse_noisy)
        psnr_denoised = 10 * np.log10((data_range**2) / mse_denoised)

        print(f"MSE Noisy: {mse_noisy:.6f}, PSNR: {psnr_noisy:.2f} dB")
        print(f"MSE Denoised: {mse_denoised:.6f}, PSNR: {psnr_denoised:.2f} dB")
        
        # Save plots
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(noisy_sino, cmap='gray'); plt.title(f"Noisy (PSNR {psnr_noisy:.2f}dB)")
        plt.subplot(1, 3, 2); plt.imshow(denoised_sino, cmap='gray'); plt.title(f"Denoised (Streaks, PSNR {psnr_denoised:.2f}dB)")
        plt.subplot(1, 3, 3); plt.imshow(clean_sino, cmap='gray'); plt.title("Clean Reference")
        plt.savefig("verification_real.png")
        print("Saved verification_real.png")
        
        if mse_denoised < mse_noisy:
            print("SUCCESS: Quality improved.")
        else:
            print("WARNING: Quality did not improve.")

if __name__ == "__main__":
    verify_real_data()
