""" 
experiments/visualise_embeddings.py

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

import random

import os
from pathlib import Path
import pickle

import argparse

from typing import Callable, Sequence
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
import torch
import gc

from experiments import shift_generator
from experiments.inference_utils import get_or_save_outputs

from shift_identification_detection.bbsd_tests import run_bbsd
from shift_identification_detection.mmd_test import run_mmd_permutation_test


# --------- Global settings ----------
ENCODER_TO_EVALUATE = "simclr_imagenet" # Options: "imagenet", "simclr_imagenet", or "random"
FEAT_MODE = "all" # Options: "final", "early", or "all"
DATASET = "PadChest" # Options: "Mammo", "Retina", "RSNA", or "PadChest"

ROOT = Path(__file__).resolve().parent.parent

# ---------- Config maps -------------
ENCODERS = {
        "imagenet": "encoder_imagenet.pkl",
        "simclr_imagenet": "encoder_simclr_imagenet.pkl",
        "random": "encoder_random.pkl",
}

FEAT_MODES = ["final", "early", "all"]

DATASET_CONFIG = {
    "Mammo": {
        "csv_files": ("test_embed.csv", "val_embed.csv"),
        "column_map": {
            "laterality": "ImageLateralityFinal",
            "view": "ViewPosition",
            "manufacturer": "ManufacturerModelName"
        },
        "plot_columns": ["class", "laterality", "view", "manufacturer"]
    },
    "Retina": {
        "csv_files": ("retina_test.csv", "retina_val.csv"),
        "column_map": {"site": "site"},
        "plot_columns": ["class", "site"]
    },
    "RSNA": {
        "csv_files": ("test_rsna.csv", "val_rsna.csv"),
        "column_map": {
            "age": "Patient Age",
            "gender": "Patient Gender",
            "view": "View Position"
        },
        "plot_columns": ["class", "gender", "view"]
    },
    "PadChest": {
        "csv_files": ("test_padchest.csv", "val_padchest.csv"),
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
        "acq_prev": partial(shift_generator.mammo_acq_prev_shift,
                            target_manufacturer_distribution=np.array([0.50, 0.00, 0.00, 0.20, 0.20, 0.10]),
                            target_density_distribution=np.array([0.15, 0.35, 0.35, 0.15])),
        "acq": partial(shift_generator.mammo_acq_prev_shift, 
                       target_manufacturer_distribution=np.array([0.50, 0.00, 0.00, 0.20, 0.20, 0.10])),
        "prev": partial(shift_generator.mammo_acq_prev_shift, 
                        target_density_distribution=np.array([0.15, 0.35, 0.35, 0.15])),
    },
    "Retina": {
        "acq_prev": partial(shift_generator.retina_acq_prev_shift,
                            target_site_distribution = np.array([0.10, 0.20, 0.70]),
                            target_prevalence = 0.5),
        "acq": partial(shift_generator.retina_acq_prev_shift, 
                       target_site_distribution = np.array([0.10, 0.20, 0.70])),
        "prev": partial(shift_generator.retina_acq_prev_shift, 
                        target_prevalence = 0.5),
    },
    "RSNA": {
        "gender_prev": partial(shift_generator.rsna_gender_and_prev_shift,
                               target_female_proportion=0.40,
                               target_prevalence=0.25),
        "gender": partial(shift_generator.rsna_gender_shift,
                          target_female_proportion=0.40),
        "prev": partial(shift_generator.rsna_prev_shift,
                        target_prevalence=0.25),
        "subpop": partial(shift_generator.rsna_subpopulation_shift,
                          target_abnormal_neg=0.70),
    },
    "PadChest": {
        "gender_prev": partial(shift_generator.padchest_gender_prev_shift,
                               target_disease=0.04,
                               target_female_proportion=0.50),
        "gender": partial(shift_generator.padchest_gender_shift,
                          target_female_proportion=0.50),
        "sample": partial(shift_generator.sample_shift_padchest,
                            target_prev_phillips=0.55,
                            target_pneumonia=0.1),
    },
}

# -----------------------------------
# Set all seeds once helper function
# ------------------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------------
# Load embeddings with error handling
# -----------------------------------
def load_embeddings(file_path: Path):
    """
    Load a pickled embeddings file.

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
    

# ------------------------------------------------------------
# Generate dict of {label names: feature labels to be plotted}
# ------------------------------------------------------------
def build_plot_labels_df(
        full_dataset_df: pd.DataFrame, 
        columns_map: dict[str, str]
    ) -> dict[str, np.ndarray]:
    """
    Create a dictionary mapping each label category to the corresponding NumPy array 
    of labels extracted from the dataset for plotting.

    Args:
        full_dataset_df (pd.DataFrame): Dataframe from a particular "test" or "val" dataset.
        columns_map (dict[str, str]): Map of label categories to the corresponding dataframe column name.

    Returns:
        dict[str, np.ndarray]: Map of label categories to feature labels in the dataframe.

    """
    plot_labels_df = {}
    for label_name, col in columns_map.items():
        plot_labels_df[label_name] = full_dataset_df[col].to_numpy()
    return plot_labels_df


# -----------------------------------------
# Wrapper to apply correct dataset function
# -----------------------------------------
def DatasetFunctionWrapper(dataset, val_df, test_df):
        if dataset == "Mammo":
            from data_handling.mammo import EmbedDataset as DS
        elif dataset == "Retina":
            from data_handling.retina import RetinaDataset as DS
        elif dataset == "RSNA":
            from data_handling.xray import RNSAPneumoniaDetectionDataset as DS
        else: # dataset == "PadChest"
            from data_handling.xray import PadChestDataset as DS
        
        return DS(df=test_df, transform=torch.nn.Identity(), cache=False), DS(df=val_df, transform=torch.nn.Identity(), cache=False)


# -----------------------------------
# Detect scenario and get layers data
# -----------------------------------
def detect_scenario_and_process_embeddings(
        encoder_output: dict, 
        split: str
    ) -> tuple[str, list[str], dict[str, torch.Tensor]]:
    """
    Identify the type of feature extraction scenario given an encoder output and return the relevant layers and features.

    Determine whether the data matches one of three expected cases:
        "final": Only final-layer features ("feats") are present.
        "early": Both early-layer features ("early_feats") and final features ("feats") are present.
        "all": Features from multiple layers are stored in "feats_by_layer".

    Args:
        encoder_output (dict): Mapping of "train" and "test" splits to the corresponding embeddings.
        split (str): Which split to inspect.

    Returns:
        tuple:
            scenario_type (str): "final", "early", or "all"
            layers_to_visualise (list[str]): Names of layers available for visualisation.
            feats_data (dict[str, torch.Tensor]): Mapping from layer name to feature tensors.

    Raises:
        ValueError: If the split is missing or the structure is not recognised.
    """
    if split not in encoder_output:
        raise ValueError(f"Split '{split}' not found in encoder_output. Available splits: {list(encoder_output.keys())}")
    
    split_data = encoder_output[split]

    # Check for errors:
    if len(split_data.keys()) == 0:
        # Missing data
        raise ValueError(f"No data found for split '{split}'. Available keys: {list(split_data.keys())}")
    elif len(split_data.keys()) == 1 and 'y' in split_data:
        # Missing features
        raise ValueError(f"No features in split '{split}'. Available keys: {list(split_data.keys())}. Expected features data.")
    elif 'y' not in split_data:
        # Missing labels
        raise ValueError(f"Labels are missing in split '{split}'. Available keys: {list(split_data.keys())}. Expected labels.")

    # Parse data:
    elif len(split_data.keys()) == 2 and 'y' in split_data and 'feats' in split_data:
        # "final" scenario
        return "final", ["flattened"], {"flattened": split_data["feats"]}
    elif len(split_data.keys()) == 3 and 'y' in split_data and 'feats' in split_data and 'early_feats' in split_data:
        # "early" scenario
        return "early", ["layer_1", "flattened"], {"layer_1": split_data["early_feats"], "flattened": split_data["feats"]}
    else:
        # Expect "all" scenario
        all_layer_names = ["y", "after_maxpool", "layer_1", "layer_2", "layer_3", "flattened"]
        if set(split_data.keys()) == set(all_layer_names):
            feats_excluding_y = {k: v for k, v in split_data.items() if k != "y"}
            return "all", list(feats_excluding_y.keys()), feats_excluding_y
        else:
            raise ValueError(f"Unexpected layer structure in split '{split}'. Found layers: {list(split_data.keys())}. Expected: {all_layer_names}.")

# -----------------------------------
# PCA and t-SNE analysis and plots
# -----------------------------------
def process_and_visualise_layer(
        layer_name: str, 
        features: torch.Tensor, 
        labels: Sequence[torch.Tensor | np.ndarray], 
        scenario: str,
        shift: str="no_shift", 
        seed: int=42, 
        pca_components: int=2,
        num_samples: int=1000
    ) -> None:
    """
    Processes data, reduces dimensionality, and visualises layer features.

    Args:
        layer_name: The name of the layer being processed.
        features: The feature embeddings for a given layer of the encoder.
        labels: A list containing sets of labels for the data (e.g. class labels, laterality, manufacturer).
        scenario: A string identifier for the experimental scenario ("final", "early", "all").
        shifted: A string indicating the type of shift ("no_shift", "acq", "prev", or "acq_prev").
        seed: Random seed for reproducibility.
        pca_components: The number of principal components to reduce to.
        num_samples: The number of points to include in the final plot.
    """

    if features.is_cuda: # Safety check (that features is not on GPU)
        features = features.cpu()
    embeddings = features.numpy() # Convert PyTorch tensor to numpy array for processing
    print(f"[{layer_name}] Original shape: {embeddings.shape}") # Should be 2D

    # # Early subsampling
    # rng = np.random.default_rng(seed)
    # sample_idx = rng.choice(len(embeddings),
    #                         size=min(num_samples, len(embeddings)),
    #                         replace=False)
    # embeddings = embeddings[sample_idx]
    # labels = [lbl[sample_idx] for lbl in labels]

    # PCA reduction
    #pca = PCA(n_components=0.95, whiten=False, random_state=seed) # PCA embedding that preserves 95% of the variance of the input data
    pca = PCA(n_components=pca_components, whiten=False, random_state=seed)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"[{layer_name}] PCA shape: {embeddings_pca.shape}")
    print(f"[{layer_name}] PCA explained variance ratio: {pca.explained_variance_ratio_[:2]}")

    # Use the t-SNE algorithm on PCA-reduced features to obtain a 2D embedding for input data
    embeddings_tsne = TSNE(n_components=2, 
                           init='random',
                           learning_rate='auto',
                           random_state=seed).fit_transform(embeddings_pca)
    print(f"[{layer_name}] t-SNE shape: {embeddings_tsne.shape}")
    
    # Create a pandas DataFrame to process data
    feature_attributes = DATASET_CONFIG[DATASET]["plot_columns"]
    df = pd.DataFrame({label: np_array for label, np_array in zip(feature_attributes, labels)})

    # Add PCA components and t-SNE components to the DataFrame
    for i in range(pca_components):
        df[f"{layer_name} - PCA {i+1}"] = embeddings_pca[:,i]
    df[f"{layer_name} - t-SNE 1"] = embeddings_tsne[:,0]
    df[f"{layer_name} - t-SNE 2"] = embeddings_tsne[:,1]

    # Sample for plotting
    sample = df.sample(n=min(num_samples, len(df)), random_state=seed)

    # Create plots
    sns.set_theme(style="white") # For cleaner appearance

    fig, axes = plt.subplots(len(feature_attributes), 2, figsize=(14, 18), constrained_layout=True)

    alpha = 0.8
    style = 'o'
    markersize = 40
    color_palette = 'tab10'

    for i, label_type in enumerate(feature_attributes):

        # PCA plot (left column)
        ax_pca = sns.scatterplot(
            data=sample, 
            x=f"{layer_name} - PCA 1", 
            y=f"{layer_name} - PCA 2", 
            hue=label_type, 
            alpha=alpha, 
            marker=style, 
            s=markersize, 
            palette=color_palette, 
            ax=axes[i, 0])
        ax_pca.set_title(f"PCA coloured by {label_type}")

        # t-SNE plot
        ax_tsne = sns.scatterplot(
            data=sample, 
            x=f"{layer_name} - t-SNE 1", 
            y=f"{layer_name} - t-SNE 2", 
            hue=label_type, 
            alpha=alpha, 
            marker=style, 
            s=markersize, 
            palette=color_palette, 
            ax=axes[i, 1])
        ax_tsne.set_title(f"t-SNE coloured by {label_type}")

    for ax in axes.ravel():
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1,1))

    fig.suptitle(f"{DATASET} | Scenario: {shift} - {layer_name}", fontsize=16)

    # Save the figure
    file_location = OUTPUT_DIR / f"{shift}_{layer_name}_{ENCODER_TO_EVALUATE}.png"
    fig.savefig(file_location)
    plt.close(fig)

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
    set_seeds(42)

    # Optional: Configure global settings using CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", default=ENCODER_TO_EVALUATE,
                        choices=list(ENCODERS.keys()))
    parser.add_argument("--feat_mode", default=FEAT_MODE,
                        choices=FEAT_MODES)
    parser.add_argument("--dataset", default=DATASET,
                        choices=list(DATASET_CONFIG.keys()))
    args = parser.parse_args()

    ENCODER_TO_EVALUATE = args.encoder_type
    FEAT_MODE = args.feat_mode
    DATASET= args.dataset
   
    # File paths
    ENCODER_PICKLE_PATH = ROOT / "experiments" / "outputs"/ DATASET / ENCODERS[ENCODER_TO_EVALUATE]
    OUTPUT_DIR = ROOT / "experiments"/ "outputs" / DATASET / "Plots" / f"{ENCODER_TO_EVALUATE}/"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists


    ### 1. Process test and val CSVs

    test_csv, val_csv = DATASET_CONFIG[DATASET]["csv_files"]

    # Process the test data
    print(f"Loading test data from '{test_csv}'...")
    test_df = pd.read_csv(ROOT / "experiments" / test_csv)
    test_df["idx_in_original"] = np.arange(len(test_df))

    # Process the validation dat
    print(f"Loading validation data from '{val_csv}'...")
    val_df = pd.read_csv(ROOT / "experiments" / val_csv)
    val_df["idx_in_original"] = np.arange(len(val_df))


    ### 2. If embeddings don't already exist, generate them

    if not os.path.exists(ENCODER_PICKLE_PATH):

        test_dataset, val_dataset = DatasetFunctionWrapper(DATASET, val_df, test_df)

        # Generate the embeddings
        print(f"Generating embeddings using '{ENCODER_TO_EVALUATE}' encoder...\n")
        get_or_save_outputs(
            model_to_evaluate=None,
            encoder_to_evaluate=ENCODER_TO_EVALUATE,
            val_loader=DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6),
            test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6),
            dataset_name=DATASET,
            feat_mode=FEAT_MODE
        )

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()


    ### 3. Load and process the test and val feature embeddings to be plotted

    print(f"Loading encoder output from {ENCODER_PICKLE_PATH}...")
    encoder_output = load_embeddings(ENCODER_PICKLE_PATH)
    scenario, layers_to_visualise, feats_data = detect_scenario_and_process_embeddings(encoder_output, "test")
    _, _, val_feats_data = detect_scenario_and_process_embeddings(encoder_output, "val")
    

    print(f"\n=== DETECTED SCENARIO: {DATASET.upper()} | {ENCODER_TO_EVALUATE.upper()} | {scenario.upper()} ===")
    print(f"Available layers for visualisation: {layers_to_visualise}")


    columns = DATASET_CONFIG[DATASET]["column_map"]
    test_plot_labels = build_plot_labels_df(test_df, columns)
    val_plot_labels = build_plot_labels_df(val_df, columns)

    test_plot_labels["class"] = encoder_output["test"]["y"]
    val_plot_labels["class"]  = encoder_output["val"]["y"]


    ### 4. Simulate different covariate shifts on test data

    shift_to_indices_dict = {}
    for shift_name, shift_function in SHIFT_REGISTRY[DATASET].items():
        print(f"Simulating {shift_name} shift on test data...")
        shifted_test_df = shift_function(test_df.copy(), random_state=42)
        shift_to_indices_dict[shift_name] = shifted_test_df["idx_in_original"].to_numpy()

    # Re‑orders the labels to match those expected when plotting
    layer_labels_val = [val_plot_labels[k]  for k in DATASET_CONFIG[DATASET]["plot_columns"]]
    layer_labels_test = [test_plot_labels[k] for k in DATASET_CONFIG[DATASET]["plot_columns"]]


    ### 5. Generate the embedding plots for each layer of the encoder

    for layer in layers_to_visualise:
        print(f"\n--- Processing layer: {layer} ---")

        # Reference data ("val" dataset)
        print("Processing reference data (no shift)...")
        process_and_visualise_layer(
            layer_name=layer,
            features=val_feats_data[layer], 
            labels=layer_labels_val,
            scenario=scenario,
            shift="no_shift"
        )

        # Shifted data ("test" dataset)
        for shift_name, idx_array in shift_to_indices_dict.items():
            print(f"Processing {shift_name}...")
            shifted_labels = [arr[idx_array] for arr in layer_labels_test]
            process_and_visualise_layer(
                layer_name=layer,
                features=feats_data[layer][idx_array],
                labels=shifted_labels,
                scenario=scenario,
                shift=shift_name
            )
            calculate_bbsd_and_mmd(
                source_distribution=val_feats_data[layer],
                target_distribution=feats_data[layer][idx_array],
                layer_name=layer, 
                shift=shift_name
            )


    print(f"\n=== VISUALIZATION COMPLETE FOR SCENARIO: {scenario.upper()} ===\n")