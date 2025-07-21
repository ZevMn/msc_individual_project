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
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.inference_utils import get_or_save_outputs

from shift_identification_detection.bbsd_tests import run_bbsd
from shift_identification_detection.mmd_test import run_mmd_permutation_test

from config import Config

from embeddings_io import load_embeddings

from visualise_embeddings import process_and_visualise_layer, aggregate_and_plot_shifted_features


ROOT = Path(__file__).resolve().parent.parent.parent

# --------- Defaults ----------
ENCODER_TO_EVALUATE = "simclr_imagenet" # Options: "imagenet", "simclr_imagenet", or "random"
FEAT_MODE = "all" # Options: "final", "early", or "all"
DATASET = "PadChest" # Options: "Mammo", "Retina", "RSNA", or "PadChest"    

# -------------------------------------------------------------
# Generate dicts of {label names: feature labels to be plotted}
# -------------------------------------------------------------
def extract_plot_labels(
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        class_labels: tuple[np.ndarray, np.ndarray],
        dataset: str
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Build two ordered dicts mapping each label category to the corresponding NumPy array 
    of labels extracted from the val and test datasets for plotting. The order follows 
    Config.DATASET_CONFIG[dataset]["plot_columns"].

    Args:
        val_df (pd.DataFrame): Dataframe for the "val" dataset.
        test_df (pd.DataFrame): Dataframe for the "test" dataset.
        class_labels (tuple[np.ndarray, np.ndarray]): A tuple of NumPy arrays containing the 
                                                    class labels for the "val" and the "test" set.
        dataset (str): The name of the dataset (e.g., "Mammo", "Retina", etc.), used to look up 
                    the column mapping and plotting order from the configuration.

    Returns:
        tuple:
            val_plot_labels (dict[str, np.ndarray]): A dictionary mapping label names to NumPy arrays 
                                                    of labels extracted from the "val" dataset.
            test_plot_labels (dict[str, np.ndarray]): A dictionary mapping label names to NumPy arrays 
                                                    of labels extracted from the "test" dataset.
    """

    val_plot_labels = {}
    test_plot_labels = {}

    ordered_columns = Config.DATASET_CONFIG[dataset]["plot_columns"]
    column_map = Config.DATASET_CONFIG[dataset]["column_map"]

    for label in ordered_columns:
        if label == "class":
            # Class labels come from the encoder output, not the csv
            val_plot_labels["class"] = class_labels[0]
            test_plot_labels["class"] = class_labels[1]
            continue

        col_name = column_map[label]

        if col_name not in val_df.columns or col_name not in test_df.columns:
            raise KeyError(f"Column '{col_name}' not found in the supplied DataFrames.")
        
        val_plot_labels[label] = val_df[col_name].to_numpy()
        test_plot_labels[label] = test_df[col_name].to_numpy()

    return val_plot_labels, test_plot_labels


# -----------------------------------------
# Wrapper to apply correct dataset function
# -----------------------------------------
def preprocess_data(dataset, val_df, test_df):
        if dataset == "Mammo":
            from data_handling.mammo import EmbedDataset as DS
        elif dataset == "Retina":
            from data_handling.retina import RetinaDataset as DS
        elif dataset == "RSNA":
            from data_handling.xray import RNSAPneumoniaDetectionDataset as DS
        else: # dataset == "PadChest"
            from data_handling.xray import PadChestDataset as DS
        
        return DS(df=val_df, transform=torch.nn.Identity(), cache=False), DS(df=test_df, transform=torch.nn.Identity(), cache=False)


# -----------------------------------
# Validate and process encoder output
# -----------------------------------
def validate_and_process_embeddings(
        encoder_output: dict[str, dict[str, torch.Tensor]], 
    ) -> tuple[list[str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Given a mapping of split ("val", "test") to a mapping of layer names to encoder outputs 
    as well as "y" to class labels, returns a list of layer names 
    and separate embeddings mappings for the "val" and "test" splits.

    Determines whether the data matches one of three expected cases:
        "final": Only "final_layer" features are present.
        "early": Both "layer_1" features and "final_layer" features are present.
        "all": Features from all layers are present.

    Args:
        encoder_output (dict): Mapping of "val" and "test" splits to the corresponding embeddings.

    Returns:
        tuple:
            layers (list[str]): Names of layers available for visualisation.
            val_embeddings (dict[str, torch.Tensor]): Mapping from layer names to feature tensors.
            test_embeddings (dict[str, torch.Tensor]): Mapping from layer names to feature tensors.

    Raises:
        ValueError: If the split is missing or the structure is not recognised.
    """

    expected_splits = {"val", "test"}
    if set(encoder_output) != expected_splits:
        raise ValueError(f"Encoder outputs must have keys {expected_splits}, found {set(encoder_output)}")
    
    for split, embeddings_mapping in encoder_output.items():
        if not embeddings_mapping:
            raise ValueError(f"No data in split '{split}'. Expected embeddings and labels.")
        
        keys = set(embeddings_mapping)
        if keys == {"y"}:
            raise ValueError(f"No embeddings in split '{split}'. Found only labels.")
        if "y" not in keys:
            raise ValueError(f"No labels in split '{split}'. Expected a 'y' key for labels.")
        
    layers = [k for k in encoder_output["val"].keys() if k != "y"]
    val_embeddings = {k: v for k, v in encoder_output["val"].items() if k != "y"}
    test_embeddings = {k: v for k, v in encoder_output["test"].items() if k != "y"}

    return layers, val_embeddings, test_embeddings


# -----------------------------------
# BBSD and MMD analysis wrapper
# -----------------------------------
def calculate_bbsd_and_mmd(
        source_distribution, # Val data
        target_distribution, # Test data with simluated shift
        layer_name: str,
        shift: str
    ) -> None:

    if run_bbsd(source_distribution, target_distribution):
        print(f"BBSD positive for {shift} shift and {layer_name}")
    else:
        print(f"BBSD negative for {shift} shift and {layer_name}")

    if run_mmd_permutation_test(source_distribution, target_distribution):
        print(f"MMD positive for {shift} shift and {layer_name}\n")
    else:
        print(f"MMD negative for {shift} shift and {layer_name}\n")


# ------------------------------------
# Main execution
# ------------------------------------
if __name__ == "__main__":
    Config.set_seeds(Config.SEED)

    # Optional: Configure global settings using CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", 
                        default=ENCODER_TO_EVALUATE,
                        choices=list(Config.ENCODERS.keys()))
    parser.add_argument("--feat_mode", 
                        default=FEAT_MODE,
                        choices=Config.FEAT_MODES)
    parser.add_argument("--dataset", 
                        default=DATASET,
                        choices=list(Config.DATASET_CONFIG.keys()))
    args = parser.parse_args()

    ENCODER_TO_EVALUATE = args.encoder_type
    FEAT_MODE = args.feat_mode
    DATASET= args.dataset
   
    # File paths
    ENCODER_PICKLE_PATH = ROOT / "experiments" / "outputs"/ DATASET / Config.ENCODERS[ENCODER_TO_EVALUATE]
    OUTPUT_DIR = ROOT / "experiments"/ "outputs" / DATASET / "Plots" / f"{ENCODER_TO_EVALUATE}/"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    print(f"\n=== SCENARIO: {DATASET.upper()} | {ENCODER_TO_EVALUATE.upper()} | {FEAT_MODE.upper()} ===")

    ### 1. Process test and val CSVs

    val_csv, test_csv = Config.DATASET_CONFIG[DATASET]["csv_files"]

    # Process the test and validation csv data
    print(f"Loading val data from '{val_csv}' and test data from '{test_csv}'...")
    val_df = pd.read_csv(ROOT / "experiments" / val_csv)
    test_df = pd.read_csv(ROOT / "experiments" / test_csv)

    # Create index column to track features
    val_df["idx_in_original"] = np.arange(len(val_df))
    test_df["idx_in_original"] = np.arange(len(test_df))


    ### 2. If feature embeddings don't already exist then generate them

    if not Path.exists(ENCODER_PICKLE_PATH):

        val_preprocessed, test_preprocessed = preprocess_data(DATASET, val_df, test_df)

        # Generate the embeddings
        print(f"Generating embeddings using '{ENCODER_TO_EVALUATE}' encoder...\n")
        get_or_save_outputs(
            model_to_evaluate=None,
            encoder_to_evaluate=ENCODER_TO_EVALUATE,
            val_loader=DataLoader(val_preprocessed, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS),
            test_loader=DataLoader(test_preprocessed, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS),
            dataset_name=DATASET,
            feat_mode=FEAT_MODE
        )

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()


    ### 3. Load and process the test and val feature embeddings to be plotted

    print(f"Loading encoder output from {ENCODER_PICKLE_PATH}...")
    encoder_output = load_embeddings(ENCODER_PICKLE_PATH)

    layers, val_embeddings, test_embeddings = validate_and_process_embeddings(encoder_output)
    print(f"Available layers for visualisation: {layers}")

    class_labels = (
        encoder_output["val"]["y"].cpu().numpy(),
        encoder_output["test"]["y"].cpu().numpy()
    )
    val_plot_labels, test_plot_labels = extract_plot_labels(val_df, test_df, class_labels, DATASET)


    ### 4. Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = {}
    for shift_name, shift_fn in Config.SHIFT_REGISTRY[DATASET].items():
        print(f"Simulating {shift_name} shift on test data...")
        df = shift_fn(test_df.copy(), random_state=Config.SEED)
        if "idx_in_original" not in df.columns:
            raise ValueError(f"Shift function '{shift_name}' must preserve 'idx_in_original' column.")
        shift_to_indices_dict[shift_name] = df["idx_in_original"].to_numpy()


    ### 5. Generate the embedding plots for each layer of the encoder

    for layer in layers:
        print(f"\n--- Processing layer: {layer} ---")

        # Reference data ("val" dataset)
        print("Processing reference data (no shift)...")
        process_and_visualise_layer(
            output_dir=OUTPUT_DIR,
            encoder_to_evaluate=ENCODER_TO_EVALUATE,
            dataset=DATASET,
            layer_name=layer,
            features=val_embeddings[layer], 
            labels=val_plot_labels,
            shift="no_shift"
        )

        # Shifted data ("test" dataset)
        for shift_name, idx_array in shift_to_indices_dict.items():
            print(f"Processing {shift_name}...")
            shifted_labels = [arr[idx_array] for arr in test_plot_labels]
            process_and_visualise_layer(
                output_dir=OUTPUT_DIR,
                encoder_to_evaluate=ENCODER_TO_EVALUATE,
                dataset=DATASET,
                layer_name=layer,
                features=test_embeddings[layer][idx_array],
                labels=shifted_labels,
                shift=shift_name
            )
            calculate_bbsd_and_mmd(
                source_distribution=val_embeddings[layer],
                target_distribution=test_embeddings[layer][idx_array],
                layer_name=layer, 
                shift=shift_name
            )

    aggregate_and_plot_shifted_features(
        output_dir=OUTPUT_DIR / "Aggregated",
        encoder_to_evaluate=ENCODER_TO_EVALUATE,
        dataset=DATASET,
        layers=layers,
        reference_features=val_embeddings,
        test_features=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
    )

    print(f"\n=== VISUALIZATION COMPLETE FOR SCENARIO: ===")
    print(f"{DATASET.upper()} | {ENCODER_TO_EVALUATE.upper()} | {FEAT_MODE.upper()}\n")