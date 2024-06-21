import numpy as np
import matplotlib.pyplot as plt
from bm3dornl.denoiser import bm3d_ring_artifact_removal_ms
from bm3dornl.gpu_utils import memory_cleanup


with open("sino.npy", "rb") as f:
    sino_noisy = np.load(f)

print(sino_noisy)

memory_cleanup()

keywargs = {
    "patch_size": (8, 8),
    "stride": 3,
    "background_threshold": 1e-3,
    "cut_off_distance": (128, 128),
    "intensity_diff_threshold": 0.2,
    "num_patches_per_group": 300,
    "padding_mode": "circular",
    "gaussian_noise_estimate": 5.0,
    "wiener_filter_strength": 100.0,
    "k": 1,
}

sino_bm3dornl = bm3d_ring_artifact_removal_ms(
    sinogram=sino_noisy,
    **keywargs,
)


fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(sino_noisy, cmap='gray')
axs[0].set_title('Noisy sinogram')
axs[1].imshow(sino_bm3dornl, cmap='gray')
axs[1].set_title('BM3D denoised sinogram')
#plt.show()
fig.savefig("denoise-gpu.png")
plt.close(fig)
