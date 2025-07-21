"""
experiments/embeddings/config.py

A configuration file for the embeddings module.

Contains default assignments for constants and plot styling, as well
as config maps for the encoders and datasets.

Also contains a helper function to set all seeds so that RNG is 
deterministic and reproducable for all experiments.
"""

import random

from typing import Callable
from functools import partial

import numpy as np
import torch

from experiments import shift_generator

class PlotConfig:    
    ALPHA = 0.8
    MARKER_SIZE = 40
    COLOR_PALETTE = 'tab10'
    FIGURE_SIZE = (14, 18)
    PCA_COMPONENTS = 2

class Config:
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 6

    # ---------- Config maps -------------
    FEAT_MODES: list[str] = ["final", "early", "all"]

    ENCODERS: dict[str, str] = {
            "imagenet": "encoder_imagenet.pkl",
            "simclr_imagenet": "encoder_simclr_imagenet.pkl",
            "random": "encoder_random.pkl",
    }

    DATASET_CONFIG: dict[str, dict] = {
        "Mammo": {
            "csv_files": ("val_embed.csv", "test_embed.csv"),
            "column_map": {
                "laterality": "ImageLateralityFinal",
                "view": "ViewPosition",
                "manufacturer": "ManufacturerModelName"
            },
            "plot_columns": ["class", "laterality", "view", "manufacturer"]
        },
        "Retina": {
            "csv_files": ("retina_val.csv", "retina_test.csv"),
            "column_map": {"site": "site"},
            "plot_columns": ["class", "site"]
        },
        "RSNA": {
            "csv_files": ("val_rsna.csv", "test_rsna.csv"),
            "column_map": {
                "age": "Patient Age",
                "gender": "Patient Gender",
                "view": "View Position"
            },
            "plot_columns": ["class", "gender", "view"]
        },
        "PadChest": {
            "csv_files": ("val_padchest.csv", "test_padchest.csv"),
            "column_map": {
                "age": "PatientAge",
                "view": "Projection",
                "manufacturer": "Manufacturer"
            },
            "plot_columns": ["class", "age", "view", "manufacturer"]
        }
    }

    SHIFT_REGISTRY: dict[str, dict[str, Callable]] = {
        "Mammo": {
            "acq_prev": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array([0.50, 0.00, 0.00, 0.20, 0.20, 0.10]),
                target_density_distribution=np.array([0.15, 0.35, 0.35, 0.15])
            ),
            "acq": partial(
                shift_generator.mammo_acq_prev_shift, 
                target_manufacturer_distribution=np.array([0.50, 0.00, 0.00, 0.20, 0.20, 0.10])
            ),
            "prev": partial(
                shift_generator.mammo_acq_prev_shift, 
                target_density_distribution=np.array([0.15, 0.35, 0.35, 0.15])
            ),
        },
        "Retina": {
            "acq_prev": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution = np.array([0.10, 0.20, 0.70]),
                target_prevalence = 0.5
            ),
            "acq": partial(
                shift_generator.retina_acq_prev_shift, 
                target_site_distribution = np.array([0.10, 0.20, 0.70])
            ),
            "prev": partial(
                shift_generator.retina_acq_prev_shift, 
                target_prevalence = 0.5
            ),
        },
        "RSNA": {
            "gender_prev": partial(
                shift_generator.rsna_gender_and_prev_shift,
                target_female_proportion=0.40,
                target_prevalence=0.25
            ),
            "gender": partial(
                shift_generator.rsna_gender_shift,
                target_female_proportion=0.40
            ),
            "prev": partial(
                shift_generator.rsna_prev_shift,
                target_prevalence=0.25
            ),
            "subpop": partial(
                shift_generator.rsna_subpopulation_shift,
                target_abnormal_neg=0.70
            ),
        },
        "PadChest": {
            "gender_prev": partial(
                shift_generator.padchest_gender_prev_shift,
                target_disease=0.04,
                target_female_proportion=0.50
            ),
            "gender": partial(
                shift_generator.padchest_gender_shift,
                target_female_proportion=0.50
            ),
            "sample": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.55,
                target_pneumonia=0.1
            ),
        },
    }

    # ---------------------------------------------------------
    # Helper function: Set all seeds once for deterministic RNG
    # ---------------------------------------------------------
    @staticmethod
    def set_seeds(seed: int=SEED) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
