"""
experiments/embeddings/driver.py

Example usage:
--------------
    run_experiment(encoder_to_evaluate="imagenet", feat_mode="all", dataset="Mammo")
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import visualise_embeddings
from config import Config
from embeddings_io import load_embeddings_pkl
from torch.utils.data import DataLoader

from data_handling.mammo import EmbedDataset
from data_handling.retina import RetinaDataset
from data_handling.xray import PadChestDataset, RNSAPneumoniaDetectionDataset
from experiments.inference_utils import get_or_save_outputs


# --------------------------------------------------------
# Generate val and test csvs into dfs and add index column
# --------------------------------------------------------
def load_csvs_and_add_idx_column(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    val_csv, test_csv = Config.DATASET_CONFIG[dataset]["csv_files"]

    print(f"Loading val data from '{val_csv}' and test data from '{test_csv}'...")
    val_df = pd.read_csv(Config.ROOT / "experiments" / val_csv)
    test_df = pd.read_csv(Config.ROOT / "experiments" / test_csv)

    # Create index column to track features
    val_df["idx_in_original"] = np.arange(len(val_df))
    test_df["idx_in_original"] = np.arange(len(test_df))

    return val_df, test_df


# -----------------------------------------
# If embeddings do not exist, generate them
# -----------------------------------------
def generate_and_load_embeddings(
    encoder_to_evaluate: str, 
    feat_mode: str, 
    dataset: str, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> dict[str, dict[str, torch.Tensor]]:

    encoder_pickle_path = (
        Config.ROOT
        / "experiments"
        / "outputs"
        / dataset
        / Config.ENCODERS[encoder_to_evaluate]
    )

    if not Path.exists(encoder_pickle_path):

        val_preprocessed, test_preprocessed = preprocess_data(dataset, val_df, test_df)

        # Generate the embeddings
        print(f"Generating embeddings using '{encoder_to_evaluate}' encoder...\n")
        get_or_save_outputs(
            model_to_evaluate=None,
            encoder_to_evaluate=encoder_to_evaluate,
            val_loader=DataLoader(
                val_preprocessed,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            ),
            test_loader=DataLoader(
                test_preprocessed,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            ),
            dataset_name=dataset,
            feat_mode=feat_mode,
        )

        # Cleanup
        torch.cuda.empty_cache()

    print(f"Loading encoder output from {encoder_pickle_path}...")

    return load_embeddings_pkl(encoder_pickle_path)


# -------------------------------------------------------------
# Generate dicts of {label names: feature labels to be plotted}
# -------------------------------------------------------------
def extract_plot_labels(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoder_output: dict[str, dict[str, torch.Tensor]],
    dataset: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Build two ordered dicts mapping each label category to the corresponding NumPy array
    of labels extracted from the val and test datasets for plotting. The order follows
    Config.DATASET_CONFIG[dataset]["plot_columns"].

    Args:
        val_df (pd.DataFrame): Dataframe for the "val" dataset.
        test_df (pd.DataFrame): Dataframe for the "test" dataset.
        encoder_output (dict[str, dict[str, torch.Tensor]]): Mapping of "val" and "test" splits 
        to the corresponding embeddings by layer.
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
            val_plot_labels["class"] = encoder_output["val"]["y"].cpu().numpy()
            test_plot_labels["class"] = encoder_output["test"]["y"].cpu().numpy()
            continue

        col_name = column_map[label]

        if col_name not in val_df.columns or col_name not in test_df.columns:
            raise KeyError(f"Column '{col_name}' not found in the supplied DataFrames.")

        val_plot_labels[label] = val_df[col_name].to_numpy()
        test_plot_labels[label] = test_df[col_name].to_numpy()

    return val_plot_labels, test_plot_labels


def simulate_shifts(dataset: str, test_df: pd.DataFrame) -> dict[str, np.ndarray]:

    shift_to_indices_dict = {}

    for shift_name, shift_fn in Config.SHIFT_REGISTRY[dataset].items():
        print(f"Simulating {shift_name} shift on test data...")
        df = shift_fn(test_df.copy(), random_state=Config.SEED)

        if "idx_in_original" not in df.columns:
            raise ValueError(
                f"Shift function '{shift_name}' must preserve 'idx_in_original' column."
            )

        shift_to_indices_dict[shift_name] = df["idx_in_original"].to_numpy()

    return shift_to_indices_dict


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
        raise ValueError(
            f"Dataset not recognised. Expected: {Config.DATASET_CONFIG.keys()}"
        )

    return DS(df=val_df, transform=torch.nn.Identity(), cache=False), DS(
        df=test_df, transform=torch.nn.Identity(), cache=False
    )


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
        raise ValueError(
            f"Encoder outputs must have keys {expected_splits}, found {set(encoder_output)}"
        )

    for split, embeddings_mapping in encoder_output.items():
        if not embeddings_mapping:
            raise ValueError(
                f"No data in split '{split}'. Expected embeddings and labels."
            )

        keys = set(embeddings_mapping)
        if keys == {"y"}:
            raise ValueError(f"No embeddings in split '{split}'. Found only labels.")
        if "y" not in keys:
            raise ValueError(
                f"No labels in split '{split}'. Expected a 'y' key for labels."
            )
        
    layers = [k for k in encoder_output["val"].keys() if k != "y"]
    val_embeddings = {k: v for k, v in encoder_output["val"].items() if k != "y"}
    test_embeddings = {k: v for k, v in encoder_output["test"].items() if k != "y"}

    print(f"Available layers for visualisation: {layers}\n")

    return layers, val_embeddings, test_embeddings


# ------------------------
# Called in main execution
# ------------------------
def run_experiment(encoder_to_evaluate: str, feat_mode: str, dataset: str) -> None:

    Config.validate()
    Config.set_seeds()

    output_dir = (
        Config.ROOT
        / "experiments"
        / "outputs"
        / dataset
        / "Plots"
        / encoder_to_evaluate
    )

    print(
        f"\n=== {dataset.upper()} | {encoder_to_evaluate.upper()} | {feat_mode.upper()} ===\n"
    )

    # Process test and val CSVs
    val_df, test_df = load_csvs_and_add_idx_column(dataset=dataset)

    # Load embeddings and generate them if they don't already exist
    encoder_output = generate_and_load_embeddings(
        encoder_to_evaluate=encoder_to_evaluate,
        feat_mode=feat_mode,
        dataset=dataset,
        val_df=val_df,
        test_df=test_df,
    )

    # Validate the embeddings output and extract layers, and val/test splits
    layers, val_embeddings, test_embeddings = validate_and_process_embeddings(
        encoder_output=encoder_output
    )

    # Extract plot labels
    val_labels, test_labels = extract_plot_labels(
        val_df=val_df, test_df=test_df, encoder_output=encoder_output, dataset=dataset
    )

    # Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = simulate_shifts(dataset=dataset, test_df=test_df)

    # Generate plots
    inputs = visualise_embeddings.PlotInputs(
        encoder_to_evaluate=encoder_to_evaluate,
        dataset=dataset,
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
    )

    visualise_embeddings.plot_all_layers_scatter_labelled(
        output_dir=output_dir / "layer_representation",
        inputs=inputs,
        val_labels=val_labels,
        test_labels=test_labels,
        run_statistical_tests=False,
    )
    visualise_embeddings.plot_shift_comparison_scatter(
        output_dir=output_dir / "shift_comparison", inputs=inputs
    )
    visualise_embeddings.plot_shift_comparison_joint(
        output_dir=output_dir / "shift_comparison", inputs=inputs
    )

    print(f"\n=== VISUALIZATION COMPLETE ===")
