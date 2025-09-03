"""
experiments/embeddings/main.py

Entry point script to run experiments which analyse feature embedding representations
extracted from multiple layers of pre-trained encoders, and across several medical imaging 
datasets. The script can be launched via CLI with configurable arguments.

Command-line arguments (optional, with defaults below):
    --encoder_to_evaluate   Encoder identifier (keys of Config.ENCODERS).
    --feat_mode             Feature mode (keys of Config.FEAT_MODES_MAP).
    --dataset               Dataset name (keys of Config.DATASET_CONFIG).

Note:
    If you change 'feat_mode' after embeddings are cached, delete the
    corresponding '*.pkl' files before re-running to avoid mismatch.

Example usage:
--------------
# Use defaults declared in this file
    python experiments/embeddings/main.py

# Override settings from the CLI
    python experiments/embeddings/main.py \
        --encoder_to_evaluate imagenet \
        --feat_mode all \
        --dataset Mammo
"""

import argparse

from experiments.embeddings.config import Config
from experiments.embeddings.experiments import (
    run_visualisation_experiment,
    run_detection_rate_experiment,
    run_bootstrap_experiment,
    run_shift_quantification_experiment,
)

# --------- Defaults ----------
ENCODER_TO_EVALUATE = "imagenet"  # Options: "imagenet", "simclr_imagenet", "random", or "simclr_modality_specific"
FEAT_MODE = "all"  # Options: "final", "early", or "all"
DATASET = "Mammo"  # Options: "Mammo", "Retina", "RSNA", or "PadChest"

# --------------
# Main execution
# --------------
if __name__ == "__main__":

    # Optional: Configure global settings using CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_to_evaluate",
        default=ENCODER_TO_EVALUATE,
        choices=list(Config.ENCODERS.keys()),
    )
    parser.add_argument(
        "--feat_mode", default=FEAT_MODE, choices=list(Config.FEAT_MODES_MAP.keys())
    )
    parser.add_argument(
        "--dataset", default=DATASET, choices=list(Config.DATASET_CONFIG.keys())
    )
    args = parser.parse_args()

    # Uncomment as needed to run experiments:

    print("\nRunning visualisation experiment:\n")
    run_visualisation_experiment(
        args.encoder_to_evaluate, args.feat_mode, args.dataset, force_calculation=False
    )

    # print("Running shift quantification experiment:\n")
    # run_shift_quantification_experiment(
    #     args.encoder_to_evaluate, args.feat_mode, args.dataset, force_calculations=False
    # )

    # print("\nRunning rudimentary detection rate experiment:\n")
    # run_detection_rate_experiment(
    #     args.encoder_to_evaluate, args.feat_mode, args.dataset, force_calculations=False
    # )

    # print("Running bootstrap detection rates experiment:\n")
    # run_bootstrap_experiment(
    #     args.encoder_to_evaluate, args.feat_mode, args.dataset, force_calculations=False
    # )
