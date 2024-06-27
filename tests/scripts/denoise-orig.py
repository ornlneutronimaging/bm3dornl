import numpy as np
import matplotlib.pyplot as plt
import bm3d_streak_removal as bm3dsr


with open("sino.npy", "rb") as f:
    sino_noisy = np.load(f)

print(sino_noisy)

sino_bm3d = bm3dsr.multiscale_streak_removal(
    data=sino_noisy,
    max_bin_iter_horizontal=0,
    bin_vertical=0,
    filter_strength=1.0,
    use_slices=True,
    slice_sizes=None,
    slice_step_sizes=None,
    denoise_indices=None,
)


fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(sino_noisy, cmap="gray")
axs[0].set_title("Noisy sinogram")
axs[1].imshow(sino_bm3d, cmap="gray")
axs[1].set_title("BM3D denoised sinogram")
# plt.show()
fig.savefig("denoise-orig.png")
plt.close(fig)
