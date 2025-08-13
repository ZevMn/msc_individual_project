"""
experiments/embeddings/config.py

A configuration file for the embeddings module.
"""

import random
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from experiments import shift_generator


class PlotConfig:
    """
    Plotting parameters used in visualise_embeddings.py.
    """

    ALPHA = 0.8
    MARKER_SIZE = 40
    COLOR_PALETTE = "tab10"
    BASE_WIDTH = 6
    BASE_HEIGHT_PER_ROW = 4

    @staticmethod
    def get_figsize(
        n_rows: int, n_cols: int = 2, permute: bool = False
    ) -> tuple[int, int]:
        if permute:
            n_cols, n_rows = n_rows, n_cols
        return (
            PlotConfig.BASE_WIDTH * n_cols + 2,
            PlotConfig.BASE_HEIGHT_PER_ROW * n_rows,
        )


class Config:
    """
    Global configuration and registries for embedding experiments.

    Attributes:
        ROOT: Project root directory (resolved from this file's path).
        SEED: Default random seed for reproducibility.
        BATCH_SIZE: Default dataloader batch size.
        NUM_WORKERS: Number of workers for PyTorch dataloaders.

        FEAT_MODES_MAP: Permissible feature extraction modes ('final', 'early', 'all').

        ENCODERS: Mapping from encoder name to filename on disk.

        DATASET_CONFIG: Per-dataset metadata containing:
            - 'csv_files': Filenames for validation/test embedding CSVs.
            - 'column_map': Mapping from friendly column names to CSV column names.
            - 'plot_columns': Ordered list of categorical columns to colour plots by.
            - 'simclr_modality_specific': Path to the SimCLR modality-specific checkpoint.
            - 'modality_encoder_path': Path to the modality-specific encoder embeddings.

        SHIFT_REGISTRY: Nested mapping from dataset name to shift name to a callable
            that, when invoked, produces indices describing a type of covariate shift.

    Methods:
        set_seeds: Set all relevant RNG seeds (Python, NumPy, PyTorch) for determinism.
        validate: Assert that DATASET_CONFIG and SHIFT_REGISTRY list the same datasets.
    """

    ROOT = Path(__file__).resolve().parent.parent.parent
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 6

    # ---------- Config maps -------------
    FEAT_MODES_MAP: dict[str, set] = {
        "final": {"final_layer"},
        "early": {"layer_1", "final_layer"},
        "all": {"after_maxpool", "layer_1", "layer_2", "layer_3", "final_layer"},
    }

    ENCODERS: dict[str, str] = {
        "imagenet": "encoder_imagenet.pkl",
        "simclr_imagenet": "encoder_simclr_imagenet.pkl",
        "random": "encoder_random.pkl",
        "simclr_modality_specific": "simclr_modality_specific",
    }

    DATASET_CONFIG: dict[str, dict] = {
        "Mammo": {
            "csv_files": ("val_embed.csv", "test_embed.csv"),
            "column_map": {
                "Laterality": "ImageLateralityFinal",
                "Imaging Protocol": "ViewPosition",
                "Manufacturer": "ManufacturerModelName",
            },
            "plot_columns": [
                "Class",
                "Laterality",
                "Imaging Protocol",
                "Manufacturer",
            ],
            "simclr_modality_specific": "run_byatk1eo/best.ckpt",
            "modality_encoder_path": "encoder_byatk1eo.pkl",
        },
        "Retina": {
            "csv_files": ("retina_val.csv", "retina_test.csv"),
            "column_map": {"Site": "site", "Diagnosis": "diagnosis"},
            "plot_columns": ["Class", "Site", "Diagnosis"],
            "simclr_modality_specific": "run_cwyi1g3d/epoch=449.ckpt",
            "modality_encoder_path": "encoder_cwyi1g3d/epoch=449.pkl",  ### ?
        },
        "RSNA": {
            "csv_files": ("val_rsna.csv", "test_rsna.csv"),
            "column_map": {
                "Age": "Patient Age",
                "Gender": "Patient Gender",
                "Imaging Protocol": "View Position",
            },
            "plot_columns": ["Class", "Gender", "Imaging Protocol"],
            "simclr_modality_specific": "run_q0kry6pk/best.ckpt",
            "modality_encoder_path": "encoder_q0kry6pk.pkl",
        },
        "PadChest": {
            "csv_files": ("val_padchest.csv", "test_padchest.csv"),
            "column_map": {
                "Gender": "PatientSex_DICOM",
                "Imaging Protocol": "Projection",
                "Manufacturer": "Manufacturer",
            },
            "plot_columns": ["Class", "Gender", "Imaging Protocol", "Manufacturer"],
            "simclr_modality_specific": "run_q0kry6pk/best.ckpt",
            "modality_encoder_path": "encoder_q0kry6pk.pkl",
        },
    }

    SHIFT_REGISTRY: dict[str, dict[str, Callable]] = {
        "Mammo": {
            "acq_moderate": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.50, 0.00, 0.00, 0.20, 0.20, 0.10]
                ),
            ),
            "prev_moderate": partial(
                shift_generator.mammo_acq_prev_shift,
                target_density_distribution=np.array([0.10, 0.40, 0.40, 0.10]),
            ),
        },
        "Retina": {
            "acq_moderate": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.10, 0.20, 0.70]),
            ),
            "prev_moderate": partial(
                shift_generator.retina_acq_prev_shift, target_prevalence=0.5
            ),
        },
        "RSNA": {
            "gender_subtle": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.40
            ),
            "gender_moderate": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.30
            ),
            "gender_extreme": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.15
            ),
            "subpop_subtle": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.70
            ),
            "subpop_moderate": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.80
            ),
            "subpop_extreme": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.90
            ),
        },
        "PadChest": {
            # Initial: Female 51%, Phillips 42%
            "gender_subtle": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.48
            ),
            "gender_moderate": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.44
            ),
            "gender_extreme": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.39
            ),
            "sample_subtle": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.40,
            ),
            "sample_moderate": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.36,
            ),
            "sample_extreme": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.28,
            ),
        },
    }


    # TBD: Expand this so that it actually simulates a wide range of shifts.
    EXTENDED_SHIFT_REGISTRY: dict[str, dict[str, Callable]] = {
        "Mammo": {
            "acq": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.50, 0.00, 0.00, 0.20, 0.20, 0.10]
                ),
            ),
            "prev": partial(
                shift_generator.mammo_acq_prev_shift,
                target_density_distribution=np.array([0.10, 0.40, 0.40, 0.10]),
            ),
        },
        "Retina": {
            "acq": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.10, 0.20, 0.70]),
            ),
            "prev": partial(
                shift_generator.retina_acq_prev_shift, target_prevalence=0.5
            ),
        },
        "RSNA": {
            "gender": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.40
            ),
            "subpop": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.70
            ),
        },
        "PadChest": {
            "gender": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.48
            ),
            "sample": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.40,
            ),
        },
    }

    # ---------------------------------------------------------
    # Helper function: Set all seeds once for deterministic RNG
    # ---------------------------------------------------------
    @staticmethod
    def set_seeds(seed: int = SEED) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------------------
    # Ensure datasets in DATASET_CONFIG are consistent with SHIFT_REGISTRY
    # --------------------------------------------------------------------
    @staticmethod
    def validate():
        assert set(Config.DATASET_CONFIG.keys()) == set(
            Config.SHIFT_REGISTRY.keys()
        ), "Mismatch between DATASET_CONFIG and SHIFT_REGISTRY datasets"
