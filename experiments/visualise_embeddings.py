# NB: This file must be run from the root of the project

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

from experiments import shift_generator
from data_handling.mammo import EmbedDataset

ROOT = Path(__file__).resolve().parent.parent
ENCODER_PICKLE_PATH = ROOT / "experiments/outputs/Mammo/encoder_simclr_imagenet.pkl"
OUTPUT_DIR = ROOT / "experiments/outputs/Mammo/Plots/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

SPLIT = "test"  # "test" or "val"

sns.set_theme(style="white") # For cleaner appearance of plots

"""
NB: GenAI was used to generate the docstrings in this file, 
but they have all been manually reviewed and edited.
"""

# ------------------------------------
# Load embeddings with error handling
# ------------------------------------
def load_embeddings(file_path: Path):
    """
    Load a pickled embeddings object.

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
def detect_scenario_and_process_embeddings(encoder_output: dict, split: str):
    """
    Identify the type of feature extraction scenario given an encoder output and return the relevant layers and features.

    Determine whether the data matches one of three expected cases:
        "final": Only final-layer features ("feats") are present.
        "early": Both early-layer features ("early_feats") and final features ("feats") are present.
        "all": Features from multiple layers are stored in "feats_by_layer".

    Args:
        encoder_output (dict): Mapping of splits (e.g. "train", "test") to their data.
        split (str): Which split to inspect.

    Returns:
        tuple:
            scenario_type (str): "final", "early", or "all"
            layers_to_visualise (list of str): Names of layers available for visualisation.
            feats_data (dict): Mapping from layer name to feature arrays.

    Raises:
        ValueError: If the split is missing or the structure is not recognised.
    """
    if split not in encoder_output:
        raise ValueError(f"Split '{split}' not found in encoder_output. Available splits: {list(encoder_output.keys())}")
    
    split_data = encoder_output[split]
    # print("First few lines of split data for debugging:")
    # for key, value in split_data.items():
    #     print(f"  {key}: {value[:5]}")

    if len(split_data.keys()) == 0:
        # Missing data
        raise ValueError(f"No data found for split '{split}'. Available keys: {list(split_data.keys())}")
    elif len(split_data.keys()) == 1 and 'y' in split_data:
        # Missing features
        raise ValueError(f"No features in split '{split}'. Available keys: {list(split_data.keys())}. Expected features data.")
    elif 'y' not in split_data:
        # Missing labels
        raise ValueError(f"Labels are missing in split '{split}'. Available keys: {list(split_data.keys())}. Expected labels.")
    elif len(split_data.keys()) == 2 and 'y' in split_data and 'feats' in split_data:
        # "final" scenario
        return "final", ["flattened"], {"flattened": split_data["feats"]}
    elif len(split_data.keys()) == 3 and 'y' in split_data and 'feats' in split_data and 'early_feats' in split_data:
        # "early" scenario
        return "early", ["layer_1", "flattened"], {"layer_1": split_data["early_feats"], "flattened": split_data["feats"]}
    elif 'feats_by_layer' in split_data:
        # "all" scenario
        all_layer_names = ["after_maxpool", "layer_1", "layer_2", "layer_3", "layer_4", "avgpool", "flattened"]
        feats_by_layer = split_data['feats_by_layer']
        layer_keys = list(feats_by_layer.keys())
        if len([k for k in layer_keys if k in all_layer_names]) == len(all_layer_names):
            return "all", layer_keys, feats_by_layer
        else:
            raise ValueError(f"Unexpected layer structure in split '{split}'. Found layers: {layer_keys}. Expected: {all_layer_names}.")
    else:
        raise ValueError(f"Unexpected data structure in split '{split}'. Available keys: {list(split_data.keys())}.")
        

def process_and_visualise_layer(
        layer_name: str, 
        features: torch.Tensor, 
        labels: torch.Tensor, 
        scenario: str,
        shifted: bool=False, 
        seed: int=42, 
        pca_components: int=2,
        num_samples: int=1000):
    """
    Processes data, reduces dimensionality, and visualises layer features.

    Args:
        layer_name: The name of the layer being processed.
        features: The feature embeddings for a given layer of the encoder.
        labels: The layer of the encoder corresponding to the feature embeddings.
        scenario: A string identifier for the experimental scenario ("final", "early", "all").
        output_dir: Directory to save the output plots.
        seed: Random seed for reproducibility.
        pca_components: The number of principal components to reduce to.
        num_samples: The number of points to include in the final plot.
    """

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
    df = pd.DataFrame(labels, columns=["class_label"])

    # Add PCA components and t-SNE components to the DataFrame
    for i in range(pca_components):
        df[f"{layer_name} - PCA {i+1}"] = embeddings_pca[:,i]
    df[f"{layer_name} - t-SNE 1"] = embeddings_tsne[:,0]
    df[f"{layer_name} - t-SNE 2"] = embeddings_tsne[:,1]

    # Sample for plotting
    sample = df.sample(n=min(num_samples, len(df)), random_state=seed)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    alpha = 0.8
    style = 'o'
    markersize = 40
    color_palette = 'tab10'

    # PCA plot
    ax_pca = sns.scatterplot(
        data=sample, 
        x=f"{layer_name} - PCA 1", 
        y=f"{layer_name} - PCA 2", 
        hue='class_label', 
        alpha=alpha, 
        marker=style, 
        s=markersize, 
        palette=color_palette, 
        ax=axes[0])
    sns.move_legend(ax_pca, loc="upper left", bbox_to_anchor=(1, 1))
    ax_pca.set_title("PCA")

    # t-SNE plot
    ax_tsne = sns.scatterplot(
        data=sample, 
        x=f"{layer_name} - t-SNE 1", 
        y=f"{layer_name} - t-SNE 2", 
        hue="class_label", 
        alpha=alpha, 
        marker=style, 
        s=markersize, 
        palette=color_palette, 
        ax=axes[1])
    sns.move_legend(ax_tsne, loc="upper left", bbox_to_anchor=(1, 1))
    ax_tsne.set_title("t-SNE")

    fig.suptitle(f"Scenario: {scenario.upper()} - {layer_name}", fontsize=14)

    if shifted:
        scenario += "_shifted"
    file_location = OUTPUT_DIR / f"{scenario}_{layer_name}_visualisation.png"
    fig.savefig(file_location)
    plt.close(fig)


# ------------------------------------
# Main execution
# ------------------------------------
if __name__ == "__main__":

    encoder_output = load_embeddings(ENCODER_PICKLE_PATH)

    # Process the test set
    print(f"Loading test set from 'test_embed.csv'...")
    test_df = pd.read_csv(ROOT / "experiments/test_embed.csv")
    test_df["idx_in_original"] = np.arange(len(test_df))

    test_dataset = EmbedDataset(df=test_df, transform=torch.nn.Identity(), cache=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=6
    )
    print(f"Test set loaded with {len(test_dataset)} samples.")

    shifted_test_df = shift_generator.mammo_acq_prev_shift(test_df)
    sampled_idx = shifted_test_df["idx_in_original"]
    idx_array = sampled_idx.to_numpy()

    scenario, layers_to_visualise, feats_data = detect_scenario_and_process_embeddings(encoder_output, SPLIT)

    print(f"=== DETECTED SCENARIO: {scenario.upper()} ===")
    print(f"Available layers for visualisation: {layers_to_visualise}")

    # Get labels
    labels = encoder_output[SPLIT]["y"]

    # Process each layer
    for layer in layers_to_visualise:
        print(f"\n--- Processing layer: {layer} ---")
        # Full data set
        process_and_visualise_layer(
            layer_name=layer,
            features=feats_data[layer], 
            labels=labels,
            scenario=scenario,
        )
        # Shifted subset
        process_and_visualise_layer(
            layer_name=layer,
            features=feats_data[layer][idx_array], 
            labels=labels[idx_array],
            scenario=scenario, 
            shifted=True
        )

    print(f"\n=== VISUALIZATION COMPLETE FOR SCENARIO: {scenario.upper()} ===\n")