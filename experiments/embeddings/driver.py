""" 
experiments/embeddings/driver.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.inference_utils import get_or_save_outputs

from shift_identification_detection.bbsd_tests import run_bbsd
from shift_identification_detection.mmd_test import run_mmd_permutation_test

from data_handling.mammo import EmbedDataset
from data_handling.retina import RetinaDataset
from data_handling.xray import RNSAPneumoniaDetectionDataset, PadChestDataset

from config import Config

from embeddings_io import load_embeddings

import visualise_embeddings

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
            DS = EmbedDataset
        elif dataset == "Retina":
            DS = RetinaDataset
        elif dataset == "RSNA":
            DS = RNSAPneumoniaDetectionDataset
        elif dataset == "PadChest":
            DS = PadChestDataset
        else:
            raise ValueError(f"Dataset not recognised. Expected: {Config.DATASET_CONFIG.keys()}")
        
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


# --------------------------------------
# Main function called in main execution
# --------------------------------------
def run_experiment(
        encoder_to_evaluate: str,
        feat_mode: str,
        dataset: str
    ) -> None:

    Config.validate()
    Config.set_seeds()

    # File paths
    path_to_dataset = Config.ROOT / "experiments" / "outputs"/ dataset
    encoder_pickle_path =  path_to_dataset / Config.ENCODERS[encoder_to_evaluate]
    output_dir = path_to_dataset / "Plots" / encoder_to_evaluate

    print(f"\n=== {dataset.upper()} | {encoder_to_evaluate.upper()} | {feat_mode.upper()} ===\n")

    ### 1. Process test and val CSVs

    val_csv, test_csv = Config.DATASET_CONFIG[dataset]["csv_files"]

    # Process the test and validation csv data
    print(f"Loading val data from '{val_csv}' and test data from '{test_csv}'...")
    val_df = pd.read_csv(Config.ROOT / "experiments" / val_csv)
    test_df = pd.read_csv(Config.ROOT / "experiments" / test_csv)

    # Create index column to track features
    val_df["idx_in_original"] = np.arange(len(val_df))
    test_df["idx_in_original"] = np.arange(len(test_df))


    ### 2. If feature embeddings don't already exist then generate them

    if not Path.exists(encoder_pickle_path):

        val_preprocessed, test_preprocessed = preprocess_data(dataset, val_df, test_df)

        # Generate the embeddings
        print(f"Generating embeddings using '{encoder_to_evaluate}' encoder...\n")
        get_or_save_outputs(
            model_to_evaluate=None,
            encoder_to_evaluate=encoder_to_evaluate,
            val_loader=DataLoader(val_preprocessed, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS),
            test_loader=DataLoader(test_preprocessed, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS),
            dataset_name=dataset,
            feat_mode=feat_mode
        )

        # Cleanup
        torch.cuda.empty_cache()


    ### 3. Load and process the test and val feature embeddings to be plotted

    print(f"Loading encoder output from {encoder_pickle_path}...")
    encoder_output = load_embeddings(encoder_pickle_path)

    layers, val_embeddings, test_embeddings = validate_and_process_embeddings(encoder_output)
    print(f"Available layers for visualisation: {layers}\n")

    class_labels = (
        encoder_output["val"]["y"].cpu().numpy(),
        encoder_output["test"]["y"].cpu().numpy()
    )
    val_plot_labels, test_plot_labels = extract_plot_labels(val_df, test_df, class_labels, dataset)


    ### 4. Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = {}
    for shift_name, shift_fn in Config.SHIFT_REGISTRY[dataset].items():
        print(f"Simulating {shift_name} shift on test data...")
        df = shift_fn(test_df.copy(), random_state=Config.SEED)
        if "idx_in_original" not in df.columns:
            raise ValueError(f"Shift function '{shift_name}' must preserve 'idx_in_original' column.")
        shift_to_indices_dict[shift_name] = df["idx_in_original"].to_numpy()


    ### 5. Generate the embedding plots for each layer of the encoder

    for layer in layers:
        print(f"\n--- Processing layer: {layer} ---")

        # Reference data ("val" dataset)
        print("\nProcessing reference data (no shift)...")
        visualise_embeddings.plot_layer_representation_scatter(
            output_dir=output_dir / "labelled_plots",
            encoder_to_evaluate=encoder_to_evaluate,
            dataset=dataset,
            layer_name=layer,
            layer_embeddings=val_embeddings[layer], 
            labels=val_plot_labels,
            shift="no_shift"
        )

        # Shifted data ("test" dataset)
        for shift_name, idx_array in shift_to_indices_dict.items():
            print(f"\nProcessing {shift_name}...")
            shifted_labels = {k: v[idx_array] for k, v in test_plot_labels.items()}
            visualise_embeddings.plot_layer_representation_scatter(
                output_dir=output_dir / "labelled_plots",
                encoder_to_evaluate=encoder_to_evaluate,
                dataset=dataset,
                layer_name=layer,
                layer_embeddings=test_embeddings[layer][idx_array],
                labels=shifted_labels,
                shift=shift_name
            )
            # calculate_bbsd_and_mmd(
            #     source_distribution=val_embeddings[layer],
            #     target_distribution=test_embeddings[layer][idx_array],
            #     layer_name=layer, 
            #     shift=shift_name
            # )

    visualise_embeddings.plot_shift_comparison_scatter(
        output_dir=output_dir / "shift_comparison",
        encoder_to_evaluate=encoder_to_evaluate,
        dataset=dataset,
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
    )

    visualise_embeddings.plot_shift_comparison_joint(
        output_dir=output_dir / "shift_comparison",
        encoder_to_evaluate=encoder_to_evaluate,
        dataset=dataset,
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
    )

    print(f"\n=== VISUALIZATION COMPLETE ===")