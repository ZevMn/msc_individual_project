# NB: This file must be run from the root of the project

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pathlib import Path

from torch.utils.data import DataLoader
import torch
from data_handling.mammo import EmbedDataset
import gc

from experiments import shift_generator
from experiments.inference_utils import get_or_save_outputs

# Variable
ENCODER_TO_EVALUATE = "imagenet"
EMBEDDINGS_FILE_NAME = "encoder_imagenet.pkl"

# Define paths
ROOT = Path(__file__).resolve().parent.parent
ENCODER_PICKLE_PATH = ROOT / "experiments/outputs/Mammo/" / EMBEDDINGS_FILE_NAME
OUTPUT_DIR = ROOT / "experiments/outputs/Mammo/Plots/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

"""
NB: GenAI was used to generate the docstrings in this file, 
but they have all been manually reviewed and edited.
"""

# ------------------------------------
# Load embeddings with error handling
# ------------------------------------
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
            return "all", feats_excluding_y.keys(), feats_excluding_y
        else:
            raise ValueError(f"Unexpected layer structure in split '{split}'. Found layers: {layer_keys}. Expected: {all_layer_names}.")


def process_and_visualise_layer(
        layer_name: str, 
        features: torch.Tensor, 
        labels: list[torch.Tensor | np.ndarray], 
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
    scenario += f"_{shift}"

    embeddings = features.numpy() # Convert PyTorch tensor to numpy array for processing

    print(f"[{layer_name}] Original shape: {embeddings.shape}") # Should be 2D
      
    # PCA reduction
    pca = PCA(n_components=0.95, whiten=False) # PCA embedding that preserves 95% of the variance of the input data
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"[{layer_name}] PCA shape: {embeddings_pca.shape}")
    print(f"[{layer_name}] PCA explained variance ratio: {pca.explained_variance_ratio_[:2]}")

    # Use the t-SNE algorithm on PCA-reduced features to obtain a 2D embedding for input data
    embeddings_tsne = TSNE(n_components=2, init='random', learning_rate='auto', random_state=seed).fit_transform(embeddings_pca)
    print(f"[{layer_name}] t-SNE shape: {embeddings_tsne.shape}")

    # Create a pandas DataFrame to process data
    df = pd.DataFrame({
        "class": labels[0],  # Class labels
        "laterality": labels[1],  # Laterality labels
        "view_position": labels[2],  # View position labels
        "manufacturer_model": labels[3],  # Manufacturer model names
    })


    # Add PCA components and t-SNE components to the DataFrame
    for i in range(pca_components):
        df[f"{layer_name} - PCA {i+1}"] = embeddings_pca[:,i]
    df[f"{layer_name} - t-SNE 1"] = embeddings_tsne[:,0]
    df[f"{layer_name} - t-SNE 2"] = embeddings_tsne[:,1]

    # Sample for plotting
    sample = df.sample(n=min(num_samples, len(df)), random_state=seed)

    # Create plots
    sns.set_theme(style="white") # For cleaner appearance

    fig, axes = plt.subplots(len(labels), 2, figsize=(14, 18), constrained_layout=True)

    alpha = 0.8
    style = 'o'
    markersize = 40
    color_palette = 'tab10'

    for i, label_type in enumerate(["class", "laterality", "view_position", "manufacturer_model"]):

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
        sns.move_legend(ax_pca, loc="upper left", bbox_to_anchor=(1, 1))
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
        sns.move_legend(ax_tsne, loc="upper left", bbox_to_anchor=(1, 1))
        ax_tsne.set_title(f"t-SNE coloured by {label_type}")

        fig.suptitle(f"Scenario: {scenario.upper()} - {layer_name}", fontsize=16)

    # Save the figure
    file_location = OUTPUT_DIR / f"{scenario}_{layer_name}_{ENCODER_TO_EVALUATE}.png"
    fig.savefig(file_location)
    plt.close(fig)


# ------------------------------------
# Main execution
# ------------------------------------
if __name__ == "__main__":

    ### 1. Process test and val feature data

    # Process the test data
    print(f"Loading test data from 'test_embed.csv'...")
    test_df = pd.read_csv(ROOT / "experiments/test_embed.csv")
    test_df["idx_in_original"] = np.arange(len(test_df))

    # Process the validation data
    print(f"Loading validation data from 'val_embed.csv'...")
    val_df = pd.read_csv(ROOT / "experiments/val_embed.csv")
    val_df["idx_in_original"] = np.arange(len(val_df))


    ### 2. Simulate different shifts on the test data

    # Simulate acquisition shift on the test data
    print(f"Simulating acquisition shift on test data...")
    acq_shift_test_df = shift_generator.mammo_acq_prev_shift(
        test_df, 
        target_manufacturer_distribution=np.array([0.50, 0.00, 0.00, 0.20, 0.20, 0.10])
    )
    acq_shift_sampled_idx = acq_shift_test_df["idx_in_original"]
    acq_shift_idx_array = acq_shift_sampled_idx.to_numpy()

    # Simulate prevalence shift on the test data
    print(f"Simulating prevalence shift on test data...")
    prev_shift_test_df = shift_generator.mammo_acq_prev_shift(
        test_df, 
        target_density_distribution=np.array([0.15, 0.35, 0.35, 0.15])
    )
    prev_shift_sampled_idx = prev_shift_test_df["idx_in_original"]
    prev_shift_idx_array = prev_shift_sampled_idx.to_numpy()

    # Simulate acquisition + prevalence shift on the test data
    print(f"Simulating acquisition + prevalence shift on test data...")
    acq_prev_shift_test_df = shift_generator.mammo_acq_prev_shift(
        test_df,
        target_manufacturer_distribution=np.array([0.50, 0.00, 0.00, 0.20, 0.20, 0.10]),
        target_density_distribution=np.array([0.15, 0.35, 0.35, 0.15])
    )
    acq_prev_shift_sampled_idx = acq_prev_shift_test_df["idx_in_original"]
    acq_prev_shift_idx_array = acq_prev_shift_sampled_idx.to_numpy()


    ### 3 If embeddings don't exist, generate them

    if not os.path.exists(ENCODER_PICKLE_PATH):
        # Prepare test and val datasets to be loaded into the encoder
        val_dataset = EmbedDataset(df=val_df, transform=torch.nn.Identity(), cache=False)
        val_dataloader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=6
        )
        test_dataset = EmbedDataset(df=test_df, transform=torch.nn.Identity(), cache=False)
        test_dataloader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=6
        )

        # Generate the embeddings
        print(f"Generating embeddings using '{ENCODER_TO_EVALUATE}' encoder...\n")
        get_or_save_outputs(
            model_to_evaluate=None,
            encoder_to_evaluate=ENCODER_TO_EVALUATE,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            dataset_name="Mammo",
            feat_mode="all",  # options: "final", "early", "all"
        )

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()


    ### 4. Load and process the test and val feature embeddings to be plotted
    print(f"Loading encoder output from {ENCODER_PICKLE_PATH}...")
    encoder_output = load_embeddings(ENCODER_PICKLE_PATH)
    scenario, layers_to_visualise, feats_data = detect_scenario_and_process_embeddings(encoder_output, "test")
    _, _, val_feats_data = detect_scenario_and_process_embeddings(encoder_output, "val")


    print(f"\n=== DETECTED SCENARIO: {scenario.upper()} ===")
    print(f"Available layers for visualisation: {layers_to_visualise}")


    ### 5. Extract category information to visualise

    val_classes = encoder_output["val"]["y"]
    test_classes = encoder_output["test"]["y"]

    test_laterality_array = test_df["ImageLateralityFinal"].to_numpy()
    val_laterality_array = val_df["ImageLateralityFinal"].to_numpy()

    test_view_array = test_df["ViewPosition"].to_numpy()
    val_view_array = val_df["ViewPosition"].to_numpy()

    test_model_array = test_df["ManufacturerModelName"].to_numpy()
    val_model_array = val_df["ManufacturerModelName"].to_numpy()


    ### 6. Generate the embedding plots for each layer of the encoder

    for layer in layers_to_visualise:
        print(f"\n--- Processing layer: {layer} ---")
        # Reference data ("val" dataset)
        process_and_visualise_layer(
            layer_name=layer,
            features=val_feats_data[layer], 
            labels=[val_classes, val_laterality_array, val_view_array, val_model_array],
            scenario=scenario,
            shift="no_shift"
        )
        # Acquisition shift
        process_and_visualise_layer(
            layer_name=layer,
            features=feats_data[layer][acq_shift_idx_array],
            labels=[test_classes[acq_shift_idx_array], test_laterality_array[acq_shift_idx_array], test_view_array[acq_shift_idx_array], test_model_array[acq_shift_idx_array]],
            scenario=scenario, 
            shift="acq"
        )
        # Prevalence shift
        process_and_visualise_layer(
            layer_name=layer,
            features=feats_data[layer][prev_shift_idx_array],
            labels=[test_classes[prev_shift_idx_array], test_laterality_array[prev_shift_idx_array], test_view_array[prev_shift_idx_array], test_model_array[prev_shift_idx_array]],
            scenario=scenario, 
            shift="prev"
        )
        # Acquisition + Prevalence shift
        process_and_visualise_layer(
            layer_name=layer,
            features=feats_data[layer][acq_prev_shift_idx_array],
            labels=[test_classes[acq_prev_shift_idx_array], test_laterality_array[acq_prev_shift_idx_array], test_view_array[acq_prev_shift_idx_array], test_model_array[acq_prev_shift_idx_array]],
            scenario=scenario, 
            shift="acq_prev"
        )

    print(f"\n=== VISUALIZATION COMPLETE FOR SCENARIO: {scenario.upper()} ===\n")