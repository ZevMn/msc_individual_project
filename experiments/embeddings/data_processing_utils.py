"""
experiments/embeddings/data_processing_utils.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.embeddings.config import Config
from experiments.embeddings.statistical_utils import calculate_PCA_and_tSNE
from experiments.inference_utils import get_or_save_outputs

from data_handling.mammo import EmbedDataset
from data_handling.retina import RetinaDataset
from data_handling.xray import PadChestDataset, RNSAPneumoniaDetectionDataset


# --------------------------------------------------------
# Generate val and test csvs into dfs and add index column
# --------------------------------------------------------
def load_csvs_and_add_idx_column(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load validation and test CSVs for a dataset and to each one append an
    'idx_in_original' column used to track rows across simulated shifts.

    Args:
        dataset: Key in Config.DATASET_CONFIG.

    Returns:
        (val_df, test_df): DataFrames for validation and test splits.
    """

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
    test_df: pd.DataFrame,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Ensure embeddings exist for (dataset, encoder_to_evaluate, feat_mode).
    If the cached pickle is missing, preprocess data and call 'get_or_save_outputs'.
    Returns the loaded encoder outputs mapping for "val" and "test".

    Returns:
        dict: {"val": {layer_x: Tensor, ..., "y": Tensor}, "test": {...}}
    """

    if encoder_to_evaluate == "simclr_modality_specific":
        encoder_pickle_path = (
            Config.ROOT
            / "experiments"
            / "outputs"
            / dataset
            / Config.DATASET_CONFIG[dataset]["modality_encoder_path"]
        )
    else:
        encoder_pickle_path = (
            Config.ROOT
            / "experiments"
            / "outputs"
            / dataset
            / Config.ENCODERS[encoder_to_evaluate]
        )

    if not encoder_pickle_path.exists():

        # Fetch correct modality specifc encoder model name
        if encoder_to_evaluate == "simclr_modality_specific":
            encoder_to_evaluate = (
                "/vol/biomedic3/mb121/causal-contrastive/outputs/"
                + Config.DATASET_CONFIG[dataset]["simclr_modality_specific"]
            )

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
# Load embeddings with error handling
# -----------------------------------
def load_embeddings_pkl(file_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load a pickled file containing a mapping of val and test splits
    to the corresponding embeddings, grouped by layer of the encoder.

    Raises:
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file isn't a valid pickle.
        IOError
            For other I/O related errors.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    except pickle.UnpicklingError as e:
        raise ValueError(f"Invalid pickle file format: {e}")
    except (IOError, OSError) as e:
        raise IOError(f"Error reading pickle file: {e}")


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

    print(f"Available layers: {layers}\n")

    return layers, val_embeddings, test_embeddings


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
        if label == "Class":
            # Class labels come from the encoder output, not the csv
            val_plot_labels["Class"] = encoder_output["val"]["y"].cpu().numpy()
            test_plot_labels["Class"] = encoder_output["test"]["y"].cpu().numpy()
            continue

        col_name = column_map[label]

        if col_name not in val_df.columns or col_name not in test_df.columns:
            raise KeyError(f"Column '{col_name}' not found in the supplied DataFrames.")

        val_plot_labels[label] = val_df[col_name].to_numpy()
        test_plot_labels[label] = test_df[col_name].to_numpy()

    return val_plot_labels, test_plot_labels


def simulate_shifts(dataset: str, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Apply all registered shift functions for 'dataset' to the test DataFrame and
    return a mapping from shift name to the original row indices ('idx_in_original')
    that comprise each shifted subset.
    """

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


def concat_embeddings(
    val_embeddings_layer: torch.Tensor,
    test_embeddings_layer: torch.Tensor,
    shift_to_indices_dict: dict[str, np.ndarray],
) -> tuple[torch.Tensor, list[str]]:
    """
    Concatenate validation embeddings with each shifted subset of test embeddings.

    Args:
        val_embeddings_layer: Validation (source) embeddings for a single layer.
        test_embeddings_layer: Test (target) embeddings for the same layer.
        shift_to_indices_dict: Mapping from shift name to integer indices
            selecting rows from 'test_embeddings_layer' to form each shifted subset.

    Returns:
        A tuple ('cat_embeddings', 'shift_labels') where:
            'cat_embeddings' is a tensor formed by concatenating validation embeddings
              with each shifted subset.
            'shift_labels' is a corresponding list with the string label "no_shift" for
            validation rows and the shift name for each shifted subset row.
    """
    cat_embeddings = [val_embeddings_layer]
    shift_labels = ["no_shift"] * len(val_embeddings_layer)

    for shift_name, idx_array in shift_to_indices_dict.items():
        shift_embeddings = test_embeddings_layer[idx_array]
        cat_embeddings.append(shift_embeddings)
        shift_labels.extend([shift_name] * len(shift_embeddings))

    return torch.cat(cat_embeddings), shift_labels


def calculate_and_save_layer_pca_and_tsne(
    output_dir: Path,
    layers: list[str],
    val_embeddings: dict[str, torch.Tensor],
    test_embeddings: dict[str, torch.Tensor],
    shift_to_indices_dict: dict[str, np.ndarray],
    force_calculation: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Calculate PCA/t-SNE projections for all layers' embeddings and saves them as a csv.

    Args:
        output_dir: Directory to save the CSV files.
        layers: Ordered list of layer names to process and plot.
        val_embeddings: Mapping from layer name to a tensor of validation (source) embeddings.
        test_embeddings: Mapping from layer name to a tensor of test (target) embeddings.
        shift_to_indices_dict: Mapping from shift name to a NumPy array of
            integer indices corresponding to the subset of test embeddings belonging to
            that covariate shift.
        force_calculation: If True, recalculate PCA/t-SNE even if cached files exist.

    Returns:
        A dictionary mapping layer names to their PCA/t-SNE results as DataFrames.

    Raises:
        ValueError: If cached CSV files are missing expected columns.
        IOError: If file operations fail.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_cols = {"Shift", "PCA 1", "PCA 2", "t-SNE 1", "t-SNE 2"}
    layer_to_results_dict: dict[str, pd.DataFrame] = {}

    # Produce a separate CSV for embeddings from each layer of the encoder
    for layer in layers:

        csv_path = output_dir / f"{layer}_pca_tsne.csv"
        use_cached = csv_path.exists() and not force_calculation

        try:
            if use_cached:
                print(f"Loading cached PCA/t-SNE results for layer: {layer}")
                df = pd.read_csv(csv_path)
                if not expected_cols.issubset(df.columns):
                    raise ValueError(
                        f"Cached file '{csv_path}' missing expected columns: "
                        f"{expected_cols - set(df.columns)}"
                    )
            else:
                print(f"Generating PCA/t-SNE results for layer: {layer}")

                cat_embeddings, shift_labels = concat_embeddings(
                    val_embeddings_layer=val_embeddings[layer],
                    test_embeddings_layer=test_embeddings[layer],
                    shift_to_indices_dict=shift_to_indices_dict,
                )
                emb_pca, emb_tsne = calculate_PCA_and_tSNE(cat_embeddings)

                df = pd.DataFrame(
                    {
                        "Shift": shift_labels,
                        "PCA 1": emb_pca[:, 0],
                        "PCA 2": emb_pca[:, 1],
                        "t-SNE 1": emb_tsne[:, 0],
                        "t-SNE 2": emb_tsne[:, 1],
                    }
                )

                df.to_csv(csv_path, index=False)
                print(f"Saved results to: {csv_path}")

            layer_to_results_dict[layer] = df

        except Exception as e:
            raise IOError(f"Error processing layer {layer}: {e}")

    return layer_to_results_dict
