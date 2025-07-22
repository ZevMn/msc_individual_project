"""
experiments/embeddings/main.py

Entry point script to generate visualisations of feature embeddings from 
multiple layers of a pre-trained encoder across several medical imaging datasets.

The pipeline:
    1. Loads (or computes and caches) embeddings for validation/test splits.
    2. Simulates predefined covariate/prevalence shifts on the test split.
    3. Projects embeddings with PCA and t-SNE.
    4. Produces layer-wise scatter/joint plots coloured by class and other
       dataset-specific attributes.
    5. Saves all figures under `experiments/outputs/<dataset>/Plots/<encoder>/`.

Command-line arguments (all optional - fall back to in-file defaults):
    --encoder_to_evaluate   Encoder identifier: keys of 'Config.ENCODERS'.
    --feat_mode             Feature mode: One of 'final', 'early', 'all'.
    --dataset               Dataset name: keys of 'Config.DATASET_CONFIG'.

Note: If you change 'feat_mode' after embeddings have already been saved, delete 
      the existing '*.pkl' file for that dataset and encoder before re-running.

Example usage:
--------------
# Use defaults declared below
    python experiments/embeddings/main.py

# Override settings from the CLI
    python experiments/embeddings/main.py \
        --encoder_to_evaluate imagenet \
        --feat_mode all \
        --dataset Mammo
"""

import argparse

from config import Config
from driver import run_experiment

# --------- Defaults ----------
ENCODER_TO_EVALUATE = "imagenet"  # Options: "imagenet", "simclr_imagenet", or "random"
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
        "--feat_mode", 
        default=FEAT_MODE, 
        choices=Config.FEAT_MODES
    )
    parser.add_argument(
        "--dataset", 
        default=DATASET, 
        choices=list(Config.DATASET_CONFIG.keys())
    )
    args = parser.parse_args()

    run_experiment(args.encoder_to_evaluate, args.feat_mode, args.dataset)
