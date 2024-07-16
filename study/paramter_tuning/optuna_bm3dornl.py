#!/usr/bin/env python3
"""Perform hyperparameter tuning using Optuna."""

import os
import logging
import optuna
import numpy as np
import h5py
from tqdm.auto import tqdm
from bm3dornl.bm3d import bm3d_ring_artifact_removal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
# file logging
fh = logging.FileHandler("optuna.log")
fh.setFormatter(formatter)
logger.addHandler(fh)

# configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load data
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "../../tests/bm3dornl-data")
data_file = os.path.join(data_dir, "tomostack_small.h5")
with h5py.File(data_file, "r") as f:
    tomo_stack_noisy = f["noisy_tomostack"][:]
    tomo_stack_clean = f["clean_tomostack"][:]
# select a few slices for tuning
idces = np.array([10, 53, 58, 146, 458])
test_sinos = tomo_stack_noisy[:, idces, :]
test_sinos_ref = tomo_stack_clean[:, idces, :]


def objective(trial: optuna.trial.Trial) -> float:
    """Objective function for parameter tunning."""
    patch_size = trial.suggest_int("patch_size", 4, 16)
    stride = trial.suggest_int("stride", 1, 4)
    cut_off_distance_w = trial.suggest_int("cut_off_distance", 32, 128)
    cut_off_distance_h = trial.suggest_int("cut_off_distance", 32, 128)
    num_patches_per_group = trial.suggest_int("num_patches_per_group", 8, 64)
    shrinkage_factor = trial.suggest_float("shrinkage_factor", 1e-5, 1e2)

    block_matching_kwargs = {
        "patch_size": (patch_size, patch_size),
        "stride": stride,
        "background_threshold": 0.0,
        "cut_off_distance": (cut_off_distance_w, cut_off_distance_h),
        "num_patches_per_group": num_patches_per_group,
        "padding_mode": "circular",
    }

    filter_kwargs = {
        "filter_function": "fft",
        "shrinkage_factor": shrinkage_factor,
    }

    kwargs = {
        "mode": "full",
        # "k": multi_scale,
        "block_matching_kwargs": block_matching_kwargs,
        "filter_kwargs": filter_kwargs,
    }

    logger.info(f"Parameters: {kwargs}")

    # process all the sinograms
    diff = 0.0
    for i in tqdm(range(test_sinos.shape[1])):
        sino_noisy = test_sinos[:, i, :]
        sino_denoised_ref = test_sinos_ref[:, i, :]
        try:
            sino_denoised = bm3d_ring_artifact_removal(sino_noisy, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # Raise the TrialPruned exception if CUDA memory error occurs
                raise optuna.exceptions.TrialPruned()
            else:
                # Re-raise the exception if it's not a CUDA memory error
                raise
        except MemoryError:
            # Raise the TrialPruned exception if memory error occurs
            raise optuna.exceptions.TrialPruned()
        # compute the relative difference
        diff += np.linalg.norm(sino_denoised - sino_denoised_ref) / np.linalg.norm(
            sino_denoised_ref
        )
    diff /= test_sinos.shape[1]

    logger.info(f"Average relative difference: {diff}")

    return diff


if __name__ == "__main__":
    study_name = "bm3dornl_parameter_tunning"
    db_file_path = f"{study_name}_full.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_file_path}",
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1_000)
