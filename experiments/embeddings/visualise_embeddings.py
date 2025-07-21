""" 
experiments/embeddings/visualise_embeddings.py
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import PlotConfig, Config

def calculate_PCA_and_tSNE(
        embeddings: torch.Tensor,
        seed: int=Config.SEED, 
        pca_components: int=PlotConfig.PCA_COMPONENTS,
    ) -> Tuple[np.ndarray, np.ndarray]:

    if embeddings.is_cuda: # Ensure embeddings on CPU (and not GPU)
        embeddings = embeddings.cpu()

    # Convert PyTorch tensor to np.ndarray for processing
    embeddings_np = embeddings.numpy()

    if embeddings_np.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings_np.shape}")

    # PCA reduction
    pca = PCA(n_components=pca_components, whiten=False, random_state=seed)
    embeddings_pca = pca.fit_transform(embeddings_np)
    print(f"PCA shape: {embeddings_pca.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:2]}")

    # Use the t-SNE algorithm on PCA-reduced features to obtain a 2D embedding for input data
    embeddings_tsne = TSNE(
        n_components=pca_components, 
        init='random',
        learning_rate='auto',
        random_state=seed
    ).fit_transform(embeddings_pca)
    print(f"t-SNE shape: {embeddings_tsne.shape}")

    return embeddings_pca, embeddings_tsne
     


# -----------------------------------
# PCA and t-SNE analysis and plots
# -----------------------------------
def process_and_visualise_layer(
        output_dir: Path,
        encoder_to_evaluate: str,
        dataset: str,
        layer_name: str, 
        layer_embeddings: torch.Tensor, 
        labels: dict[str, np.ndarray], 
        shift: str="no_shift",
        seed: int=Config.SEED,
        pca_components: int=PlotConfig.PCA_COMPONENTS,
        num_samples: int=1000
    ) -> None:
    """
    Processes data, reduces dimensionality, and visualises layer features.

    Args:
        output_dir: Path to the directory where the plots will be saved.
        encoder_to_evaluate: Encoder used to generate features.
        dataset: Dataset name.
        layer_name: The layer of the encoder being processed.
        layer_embeddings: The feature embeddings for a given layer of the encoder.
        labels: A list containing sets of labels for the data (e.g. class labels, laterality, manufacturer).
        shift: A string indicating the type of shift (e.g. "no_shift", "acq", "prev", "acq_prev").
        seed: Random seed for reproducibility.
        pca_components: The number of principal components to reduce to.
        num_samples: The number of points to include in the final plot.
    """

    embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(layer_embeddings)

    # Create a pandas DataFrame to process data
    columns = Config.DATASET_CONFIG[dataset]["plot_columns"]
    df = pd.DataFrame({col: labels[col] for col in columns})

    # Add PCA components and t-SNE components to the DataFrame
    for i in range(pca_components):
        df[f"{layer_name} - PCA {i+1}"] = embeddings_pca[:,i]
    df[f"{layer_name} - t-SNE 1"] = embeddings_tsne[:,0]
    df[f"{layer_name} - t-SNE 2"] = embeddings_tsne[:,1]

    # Sample for plotting
    sample = df.sample(n=min(num_samples, len(df)), random_state=seed)

    # Create plots
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams.update({'font.family': 'serif'})

    fig, axes = plt.subplots(len(columns), 2, figsize=PlotConfig.FIGURE_SIZE, constrained_layout=True)

    for i, column in enumerate(columns):
        # PCA plot (left column)
        ax_pca = sns.scatterplot(
            data=sample, 
            x=f"{layer_name} - PCA 1", 
            y=f"{layer_name} - PCA 2", 
            hue=column, 
            alpha=PlotConfig.ALPHA, 
            marker="o", 
            s=PlotConfig.MARKER_SIZE, 
            palette=PlotConfig.COLOR_PALETTE, 
            ax=axes[i, 0])
        ax_pca.set_title(f"PCA coloured by {column}")

        # t-SNE plot
        ax_tsne = sns.scatterplot(
            data=sample, 
            x=f"{layer_name} - t-SNE 1", 
            y=f"{layer_name} - t-SNE 2", 
            hue=column, 
            alpha=PlotConfig.ALPHA, 
            marker="o", 
            s=PlotConfig.MARKER_SIZE, 
            palette=PlotConfig.COLOR_PALETTE, 
            ax=axes[i, 1])
        ax_tsne.set_title(f"t-SNE coloured by {column}")

    for ax in axes.ravel():
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1,1))

    fig.suptitle(f"{dataset} | Scenario: {shift} - {layer_name}", fontsize=16)

    # Save the figure
    output_dir.mkdir(parents=True, exist_ok=True)
    file_location = output_dir / f"{shift}_{layer_name}_{encoder_to_evaluate}.png"
    fig.savefig(file_location)
    plt.close(fig)
    print(f"[Saved] {file_location}")


def aggregate_features_and_plot_shift_comparison(
    output_dir: Path,  
    encoder_to_evaluate: str, 
    dataset: str,
    layers: list[str],
    val_embeddings: dict[str, torch.Tensor],
    test_embeddings: dict[str, torch.Tensor], 
    shift_to_indices_dict: dict[str, np.ndarray],
) -> None:
    """
    Aggregates features from reference dataset and shifted datasets and plots 
    then plots PCA and t-SNE visualisations to compare how the feature spaces and
    learnt representations differ between different layers of the encoder.

    Args:
        output_dir: Path to the directory where the plots will be saved.
        encoder_to_evaluate: Encoder used to generate features.
        dataset: Dataset name.
        layers: The layers of the encoder that embeddings have been extracted from.
        val_embeddings: Mapping of layer names to feature embeddings from the val dataset.
        test_embeddings: Mapping of layer names to feature embeddings from the test dataset.
        shift_to_indices_dict: A mapping of shift name to indices of covariate-shifted test subsets.
    """

    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams.update({'font.family': 'serif'})

    fig, axes = plt.subplots(len(layers), 2, figsize=PlotConfig.FIGURE_SIZE, constrained_layout=True)

    for i, layer in enumerate(layers):

        # Concatenate the reference features and shifted features so that they share a PCA space
        cat_embeddings = [val_embeddings[layer]]
        shift_labels = ["no_shift"] * len(val_embeddings[layer])

        for shift_name, idx_array in shift_to_indices_dict.items():
            shift_embeddings = test_embeddings[layer][idx_array]
            cat_embeddings.append(shift_embeddings)
            shift_labels.extend([shift_name] * len(shift_embeddings))

        cat_embeddings = torch.cat(cat_embeddings)

        # PCA and t-SNE
        embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(cat_embeddings)

        df = pd.DataFrame({
            "PCA 1": embeddings_pca[:,0],
            "PCA 2": embeddings_pca[:,1],
            "t-SNE 1": embeddings_tsne[:,0],
            "t-SNE 2": embeddings_tsne[:,1],
            "Shift": shift_labels,
        })

        # PCA plot (left column)
        ax_pca = sns.scatterplot(
            data=df, 
            x="PCA 1", 
            y="PCA 2", 
            hue="Shift", 
            style="Shift", 
            alpha=PlotConfig.ALPHA, 
            s=PlotConfig.MARKER_SIZE, 
            palette=PlotConfig.COLOR_PALETTE, 
            ax=axes[i, 0])
        ax_pca.set_title(f"PCA Shift Comparison for {layer}")

        # t-SNE plot (right column)
        ax_tsne = sns.scatterplot(
            data=df, 
            x="t-SNE 1", 
            y="t-SNE 2", 
            hue="Shift", 
            alpha=PlotConfig.ALPHA, 
            s=PlotConfig.MARKER_SIZE, 
            palette=PlotConfig.COLOR_PALETTE, 
            ax=axes[i, 1])
        ax_tsne.set_title(f"t-SNE Shift Comparison for {layer}")

    for ax in axes.ravel():
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1,1))

    fig.suptitle(f"{dataset} | Shift Comparisons for all layers of {encoder_to_evaluate} encoder using PCA and t-SNE analysis", fontsize=16)

    # Save the figure
    output_dir.mkdir(parents=True, exist_ok=True)
    file_location = output_dir / f"{dataset}_{encoder_to_evaluate}_shift_comparison.png"
    fig.savefig(file_location)
    plt.close(fig)
    print(f"[Saved] {file_location}")
