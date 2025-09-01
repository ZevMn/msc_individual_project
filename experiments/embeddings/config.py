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
    Plotting parameters used in 'plotting_utils.py'.
    """
    ALPHA = 0.8
    MARKER_SIZE = 40
    COLOR_PALETTE = "tab10"


class Config:
    """
    Global configuration and registries for embedding experiments.

    Attributes:
        ROOT (Path): Project root directory (resolved from this file's path).
        SEED (int): Default random seed for reproducibility.
        BATCH_SIZE (int): Default dataloader batch size.
        NUM_WORKERS (int): Number of workers for PyTorch dataloaders.

        FEAT_MODES_MAP (dict[str, set[str]]): Permissible feature extraction modes:
            - "final": only 'final layer' embeddings
            - "early": 'layer 1' and 'final layer' embeddings
            - "all": all layer embeddings
    
        ENCODERS (dict[str, str]): Mapping from encoder name to saved filename.

        DATASET_CONFIG (dict[str, dict]): Dataset-specific metadata, including:
            - "csv_files": Filenames for validation/test embeddings.
            - "column_map": Maps plot labels to CSV column names.
            - "plot_columns": Ordered list of categorical variables for colouring plots.
            - "simclr_modality_specific": Path to the SimCLR modality-specific checkpoint.
            - "modality_encoder_path": Path to the modality-specific encoder embeddings.

        SHIFT_REGISTRY (dict[str, dict[str, Callable]]): Defines shift-generating
        functions for each dataset. Each callable returns indices corresponding
        to a covariate shifted subset.

        EXTENDED_SHIFT_REGISTRY (dict[str, dict[str, Callable]]): More fine-grained
            or incremental versions of shifts, used in the shift quanitification experiment.

    Methods:
        set_seeds: Set Python, NumPy, and PyTorch RNG seeds.
        validate: Assert that DATASET_CONFIG and SHIFT_REGISTRY list the same datasets.
    """

    ROOT: Path = Path(__file__).resolve().parent.parent.parent
    SEED: int = 42
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 6

    # -----------------------------
    # Encoder & feature config maps
    # -----------------------------
    FEAT_MODES_MAP: dict[str, set[str]] = {
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

    # --------------
    # Dataset config
    # --------------
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
            "column_map": {"Site": "site"},
            "plot_columns": ["Class", "Site"],
            "simclr_modality_specific": "run_cwyi1g3d/epoch=449.ckpt",
            "modality_encoder_path": "encoder_cwyi1g3d/epoch=449.pkl",
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
                "Manufacturer": "Manufacturer",
            },
            "plot_columns": ["Class", "Gender", "Manufacturer"],
            "simclr_modality_specific": "run_q0kry6pk/best.ckpt",
            "modality_encoder_path": "encoder_q0kry6pk.pkl",
        },
    }

    # ----------------
    # Shift registries
    # ----------------
    SHIFT_REGISTRY: dict[str, dict[str, Callable]] = {
        "Mammo": {
            # Original manufacturer distribution: [0.79, 0.00, 0.05, 0.04, 0.07, 0.05]
            "acq_moderate": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.50, 0.00, 0.00, 0.20, 0.20, 0.10]
                ),
            ),
            # Original density distribution: [0.07, 0.37, 0.47, 0.07]
            "prev_moderate": partial(
                shift_generator.mammo_acq_prev_shift,
                target_density_distribution=np.array([0.10, 0.40, 0.40, 0.10]),
            ),
            # # Original manufacturer distribution: [0.79, 0.00, 0.05, 0.04, 0.07, 0.05]
            # "acq_extreme": partial(
            #     shift_generator.mammo_acq_prev_shift,
            #     target_manufacturer_distribution=np.array(
            #         [0.65, 0.00, 0.00, 0.15, 0.15, 0.05]
            #     ),
            # ),
            # # Original density distribution: [0.07, 0.37, 0.47, 0.07]
            # "prev_extreme": partial(
            #     shift_generator.mammo_acq_prev_shift,
            #     target_density_distribution=np.array([0.05, 0.45, 0.45, 0.05]),
            # ),
        },
        "Retina": {
            # Original domain distribution: [0.04, 0.09, 0.87]
            "acq_moderate": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.10, 0.20, 0.70]),
            ),
            # Original disease prevalence: 0.78
            "prev_moderate": partial(
                shift_generator.retina_acq_prev_shift, target_prevalence=0.7
            ),
            # # Original domain distribution: [0.04, 0.09, 0.87]
            # "acq_extreme": partial(
            #     shift_generator.retina_acq_prev_shift,
            #     target_site_distribution=np.array([0.05, 0.10, 0.85]),
            # ),
            # # Original disease prevalence: 0.78
            # "prev_extreme": partial(
            #     shift_generator.retina_acq_prev_shift, target_prevalence=0.5
            # ),
        },
        "RSNA": {
            # Original proportion of females: 0.44
            "gender_subtle": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.40
            ),
            "gender_moderate": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.30
            ),
            "gender_extreme": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.15
            ),
            # Original prevalence: 0.23
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
            # Original proportion of females: 0.51
            "gender_subtle": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.48
            ),
            "gender_moderate": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.44
            ),
            "gender_extreme": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.39
            ),
            # Original proportion of Phillips: 0.42
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

    EXTENDED_SHIFT_REGISTRY: dict[str, dict[str, Callable]] = {
        "Mammo": {
            # Original manufacturer distribution: [0.79, 0.00, 0.05, 0.04, 0.07, 0.05]
            "acq_1": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.75, 0.00, 0.06, 0.05, 0.08, 0.06]
                ),
            ),
            "acq_2": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.70, 0.00, 0.07, 0.06, 0.10, 0.07]
                ),
            ),
            "acq_3": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.60, 0.00, 0.10, 0.10, 0.10, 0.10]
                ),
            ),
            "acq_4": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.55, 0.00, 0.11, 0.11, 0.12, 0.11]
                ),
            ),
            "acq_5": partial(
                shift_generator.mammo_acq_prev_shift,
                target_manufacturer_distribution=np.array(
                    [0.50, 0.00, 0.00, 0.20, 0.20, 0.10]
                ),
            ),
            # Original density distribution: [0.07, 0.37, 0.47, 0.07]
            "prev": partial(
                shift_generator.mammo_acq_prev_shift,
                target_density_distribution=np.array([0.10, 0.40, 0.40, 0.10]),
            ),
        },
        "Retina": {
            # Original domain distribution: [0.04, 0.09, 0.87]
            "acq_1": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.05, 0.10, 0.85]),
            ),
            "acq_2": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.06, 0.12, 0.82]),
            ),
            "acq_3": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.08, 0.14, 0.78]),
            ),
            "acq_4": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.09, 0.17, 0.74]),
            ),
            "acq_5": partial(
                shift_generator.retina_acq_prev_shift,
                target_site_distribution=np.array([0.10, 0.20, 0.70]),
            ),
            # Original disease prevalence: 0.78
            "prev": partial(
                shift_generator.retina_acq_prev_shift, target_prevalence=0.5
            ),
        },
        "RSNA": {
            # Original proportion of females: 0.44
            "gender_1": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.43
            ),
            "gender_2": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.42
            ),
            "gender_3": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.41
            ),
            "gender_4": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.40
            ),
            "gender_5": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.39
            ),
            "gender_6": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.38
            ),
            "gender_7": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.37
            ),
            "gender_8": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.36
            ),
            "gender_9": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.35
            ),
            "gender_10": partial(
                shift_generator.rsna_gender_shift, target_female_proportion=0.34
            ),
            # Original disease prevalence = 0.70
            "subpop_1": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.68
            ),
            "subpop_2": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.67
            ),
            "subpop_3": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.66
            ),
            "subpop_4": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.65
            ),
            "subpop_5": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.64
            ),
            "subpop_6": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.63
            ),
            "subpop_7": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.62
            ),
            "subpop_8": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.61
            ),
            "subpop_9": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.60
            ),
            "subpop_10": partial(
                shift_generator.rsna_subpopulation_shift, target_abnormal_neg=0.59
            ),
        },
        "PadChest": {
            # Original proportion of females: 0.51
            "gender_1": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.50
            ),
            "gender_2": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.49
            ),
            "gender_3": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.48
            ),
            "gender_4": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.47
            ),
            "gender_5": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.46
            ),
            "gender_6": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.45
            ),
            "gender_7": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.44
            ),
            "gender_8": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.43
            ),
            "gender_9": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.42
            ),
            "gender_10": partial(
                shift_generator.padchest_gender_shift, target_female_proportion=0.41
            ),
            # Original proportion of Phillips: 0.42
            "sample_1": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.41,
            ),
            "sample_2": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.40,
            ),
            "sample_3": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.39,
            ),
            "sample_4": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.38,
            ),
            "sample_5": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.37,
            ),
            "sample_6": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.36,
            ),
            "sample_7": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.35,
            ),
            "sample_8": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.34,
            ),
            "sample_9": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.33,
            ),
            "sample_10": partial(
                shift_generator.sample_shift_padchest,
                target_prev_phillips=0.32,
            ),
        },
    }

    # --------------
    # Helper methods
    # --------------
    @staticmethod
    def set_seeds(seed: int = SEED) -> None:
        """
        Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

        Args:
            seed (int): The seed value to use. Defaults to 'Config.SEED'.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    @staticmethod
    def validate() -> None:
        """
        Check consistency between DATASET_CONFIG and SHIFT_REGISTRY.

        Raises:
            AssertionError: If datasets listed in 'DATASET_CONFIG' and
            'SHIFT_REGISTRY' are not identical.
        """
        assert set(Config.DATASET_CONFIG.keys()) == set(
            Config.SHIFT_REGISTRY.keys()
        ), "Mismatch between DATASET_CONFIG and SHIFT_REGISTRY datasets"
