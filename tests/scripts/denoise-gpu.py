import numpy as np
import matplotlib.pyplot as plt
from bm3dornl.bm3d import bm3d_ring_artifact_removal_ms
from bm3dornl.denoiser_gpu import memory_cleanup


with open("sino.npy", "rb") as f:
    sino_noisy = np.load(f)

print(sino_noisy)

memory_cleanup()

block_matching_kwargs: dict = {
    "patch_size": (8, 8),
    "stride": 3,
    "background_threshold": 0.0,
    "cut_off_distance": (64, 64),
    "num_patches_per_group": 32,
    "padding_mode": "circular",
}
filter_kwargs: dict = {
    "filter_function": "fft",
    "shrinkage_factor": 3e-2,
}
kwargs = {
    "mode": "simple",
    "k": 4,
    "block_matching_kwargs": block_matching_kwargs,
    "filter_kwargs": filter_kwargs,
}

sino_bm3dornl = bm3d_ring_artifact_removal_ms(
    sinogram=sino_noisy,
    **kwargs,
)


fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(sino_noisy, cmap="gray")
axs[0].set_title("Noisy sinogram")
axs[1].imshow(sino_bm3dornl, cmap="gray")
axs[1].set_title("BM3D denoised sinogram")
# plt.show()
fig.savefig("denoise-gpu.png")
plt.close(fig)
