""" 
experiments/embeddings/main.py

This script generates visualisations of feature embeddings extracted from 
various layers of a pre-trained encoder on mammography ("Mammo"), diabetic 
retinopathy ("Retina"), and chest x-ray ("RSNA" and "PadChest") datasets.

For each layer, PCA and t-SNE are used to reduce dimensionality, allowing
visual inspection of learned representations. The resulting plots are
coloured by class as well as various other attribute labels depending on the 
dataset. These plots are saved to a specified output directory.

Global settings to configure before running (also configurable via CLI arguments):
    - ENCODER_TO_EVALUATE: encoder identifier
    - FEAT_MODE: feature mode to use
    - DATASET: dataset to be evaluated

Note: If you have previously generated an embeddings file using a different 
feature mode, you must manually delete the existing *.pkl file before rerunning 
this script. Adjust the output directory as needed.

Usage:
    # Run with the default settings declared at the top of the file
    python visualise_embeddings.py

    # Override one or more settings from the command line
    python visualise_embeddings.py \
        --encoder_type imagenet \
        --feat_mode all \
        --dataset Mammo
"""

import argparse

from config import Config
from driver import run_experiment

# --------- Defaults ----------
ENCODER_TO_EVALUATE = "imagenet" # Options: "imagenet", "simclr_imagenet", or "random"
FEAT_MODE = "all" # Options: "final", "early", or "all"
DATASET = "Mammo" # Options: "Mammo", "Retina", "RSNA", or "PadChest"

# ------------------------------------
# Main execution
# ------------------------------------
if __name__ == "__main__":

    # Optional: Configure global settings using CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_to_evaluate", 
                        default=ENCODER_TO_EVALUATE,
                        choices=list(Config.ENCODERS.keys()))
    parser.add_argument("--feat_mode", 
                        default=FEAT_MODE,
                        choices=Config.FEAT_MODES)
    parser.add_argument("--dataset", 
                        default=DATASET,
                        choices=list(Config.DATASET_CONFIG.keys()))
    args = parser.parse_args()

    run_experiment(args.encoder_to_evaluate, args.feat_mode, args.dataset)
