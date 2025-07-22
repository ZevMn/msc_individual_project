""" 
experiments/embeddings/visualise_embeddings.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from shift_identification_detection.bbsd_tests import run_bbsd
from shift_identification_detection.mmd_test import run_mmd_permutation_test

from config import PlotConfig, Config

@dataclass
class PlotInputs:
    encoder_to_evaluate: str
    dataset: str
    layers: list[str]
    val_embeddings: dict[str, torch.Tensor]
    test_embeddings: dict[str, torch.Tensor]
    shift_to_indices_dict: dict[str, np.ndarray]

# ----------------
# Helper functions
# ----------------
def set_plot_style() -> None:
    """
    Apply a consistent seaborn/matplotlib style.
    """ 
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams.update({'font.family': 'serif'})


def concat_embeddings(
    val_embeddings_layer: torch.Tensor,
    test_embeddings_layer: torch.Tensor,
    shift_to_indices_dict: dict[str, np.ndarray],
) -> tuple[torch.Tensor, list[str]]:
    """
    Concatenate validation embeddings with each shifted subset of test embeddings,
    returning the combined tensor and it's associated list of shift labels.
    """
    cat_embeddings = [val_embeddings_layer]
    shift_labels = ["no_shift"] * len(val_embeddings_layer)

    for shift_name, idx_array in shift_to_indices_dict.items():
        shift_embeddings = test_embeddings_layer[idx_array]
        cat_embeddings.append(shift_embeddings)
        shift_labels.extend([shift_name] * len(shift_embeddings))

    return torch.cat(cat_embeddings), shift_labels


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


def save_fig(
        fig: Figure, 
        file_location: Path,
        file_name: str
    ) -> None:
    """
    Create dirs, save, close and print once.
    """
    file_location.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_location / file_name, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {file_name}")


# ------------------
# Plotting functions
# ------------------
def plot_layer_representation_scatter(
        output_dir: Path,
        encoder_to_evaluate: str,
        dataset: str,
        layer_name: str, 
        layer_embeddings: torch.Tensor, 
        labels: dict[str, np.ndarray], 
        shift: str="no_shift",
        seed: int=Config.SEED,
        num_samples: int=1000,
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

    set_plot_style()

    columns = Config.DATASET_CONFIG[dataset]["plot_columns"]
    fig, axes = plt.subplots(len(columns), 2, figsize=PlotConfig.FIGURE_SIZE, constrained_layout=True)

    # PCA and t-SNE
    embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(layer_embeddings)

    df = pd.DataFrame({
        **{col: labels[col] for col in columns},
        "PCA 1":  embeddings_pca[:,0],
        "PCA 2":  embeddings_pca[:,1],
        "t-SNE 1": embeddings_tsne[:,0],
        "t-SNE 2": embeddings_tsne[:,1],
    })

    # Sample for plotting
    sample = df.sample(n=min(num_samples, len(df)), random_state=seed)

    for i, column in enumerate(columns):
        # PCA plot (left column)
        ax_pca = sns.scatterplot(
            data=sample, 
            x="PCA 1", 
            y="PCA 2", 
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
            x="t-SNE 1", 
            y="t-SNE 2", 
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

    save_fig(fig, output_dir, f"{shift}_{layer_name}_{encoder_to_evaluate}.png")


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


def plot_labelled_scatter_for_all_layers(
        output_dir: Path,
        inputs: PlotInputs,
        val_labels: dict[str, np.ndarray],
        test_labels: dict[str, np.ndarray],
        run_statistical_tests: bool=False,
    ) -> None:

    for layer in inputs.layers:
        print(f"\n--- Processing layer: {layer} ---")

        # Reference data (full val dataset)
        print("\nProcessing reference data (no shift)...")
        plot_layer_representation_scatter(
            output_dir=output_dir / "labelled_plots",
            encoder_to_evaluate=inputs.encoder_to_evaluate,
            dataset=inputs.dataset,
            layer_name=layer,
            layer_embeddings=inputs.val_embeddings[layer], 
            labels=val_labels,
            shift="no_shift"
        )

        # Shifted data (test dataset subsets)
        for shift_name, idx_array in inputs.shift_to_indices_dict.items():
            print(f"\nProcessing {shift_name}...")
            shifted_labels = {k: v[idx_array] for k, v in test_labels.items()}
            plot_layer_representation_scatter(
                output_dir=output_dir / "layer_representation",
                encoder_to_evaluate=inputs.encoder_to_evaluate,
                dataset=inputs.dataset,
                layer_name=layer,
                layer_embeddings=inputs.test_embeddings[layer][idx_array],
                labels=shifted_labels,
                shift=shift_name
            )

            if run_statistical_tests:
                calculate_bbsd_and_mmd(
                    source_distribution=inputs.val_embeddings[layer],
                    target_distribution=inputs.test_embeddings[layer][idx_array],
                    layer_name=layer, 
                    shift=shift_name
                )

    return


def plot_shift_comparison_scatter(
    output_dir: Path,  
    inputs: PlotInputs,
) -> None:
    """
    Aggregates features from reference dataset and shifted datasets and plots 
    PCA and t-SNE scatterplots to compare how the feature spaces and
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

    set_plot_style()

    fig, axes = plt.subplots(len(inputs.layers), 2, figsize=PlotConfig.FIGURE_SIZE, constrained_layout=True)

    for i, layer in enumerate(inputs.layers):

        # Concatenate the reference features and shifted features so that they share a PCA space
        cat_embeddings, shift_labels = concat_embeddings(
            val_embeddings_layer=inputs.val_embeddings[layer],
            test_embeddings_layer=inputs.test_embeddings[layer],
            shift_to_indices_dict=inputs.shift_to_indices_dict
        )

        # PCA and t-SNE
        embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(cat_embeddings)

        df = pd.DataFrame({
            "Shift": shift_labels,
            "PCA 1": embeddings_pca[:,0],
            "PCA 2": embeddings_pca[:,1],
            "t-SNE 1": embeddings_tsne[:,0],
            "t-SNE 2": embeddings_tsne[:,1],
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

    fig.suptitle(f"{inputs.dataset} | Shift Comparisons for All Layers of {inputs.encoder_to_evaluate} Encoder Using PCA and t-SNE Analysis", fontsize=16)

    save_fig(fig, output_dir, f"{inputs.dataset}_{inputs.encoder_to_evaluate}_shift_comparison_scatterplots.png")


def plot_shift_comparison_joint(
    output_dir: Path,  
    inputs: PlotInputs
) -> None:
    """
    Creates and saves seaborn jointplots (scatter + marginal densities) of PCA and t-SNE
    projections for each encoder layer, comparing reference and shifted distributions.

    Args:
        output_dir: Directory to save the jointplots.
        encoder_to_evaluate: Name of encoder.
        dataset: Name of dataset.
        layers: List of encoder layers to plot.
        val_embeddings: Embeddings from reference (validation) set.
        test_embeddings: Embeddings from test (possibly shifted) set.
        shift_to_indices_dict: Dictionary mapping shift name to indices in test set.
        seed: Random seed for reproducibility.
        num_samples: Max number of points to plot.
    """

    set_plot_style()

    for layer in inputs.layers:

        # Concatenate the reference features and shifted features so that they share a PCA space
        cat_embeddings, shift_labels = concat_embeddings(
            val_embeddings_layer=inputs.val_embeddings[layer],
            test_embeddings_layer=inputs.test_embeddings[layer],
            shift_to_indices_dict=inputs.shift_to_indices_dict
        )

        # PCA and t-SNE
        embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(cat_embeddings)

        df = pd.DataFrame({
            "Shift": shift_labels,
            "PCA 1": embeddings_pca[:, 0],
            "PCA 2": embeddings_pca[:, 1],
            "t-SNE 1": embeddings_tsne[:, 0],
            "t-SNE 2": embeddings_tsne[:, 1],
        })

        # Create PCA jointplot
        g_pca = sns.jointplot(
            data=df,
            x="PCA 1",
            y="PCA 2",
            hue="Shift",
            kind="scatter",
            height=8,
            ratio=5,
            space=0.1,
            alpha=PlotConfig.ALPHA,
            s=PlotConfig.MARKER_SIZE,
            palette=PlotConfig.COLOR_PALETTE,
            marginal_kws=dict(common_norm=False, fill=True),
        )
        if g_pca.ax_joint.legend_ is not None:
            g_pca.ax_joint.legend_.set_bbox_to_anchor((1.1, 1))
        g_pca.figure.suptitle(f"{inputs.dataset} | PCA Shift Comparison | {layer}", fontsize=14)
        g_pca.figure.tight_layout()
        g_pca.figure.subplots_adjust(top=0.9, right=0.8)

        save_fig(g_pca.figure, output_dir, f"{inputs.dataset}_{inputs.encoder_to_evaluate}_{layer}_PCA_jointplot.png")

        # Create t-SNE jointplot
        g_tsne = sns.jointplot(
            data=df,
            x="t-SNE 1",
            y="t-SNE 2",
            hue="Shift",
            kind="scatter",
            height=8,
            ratio=5,
            space=0.1,
            alpha=PlotConfig.ALPHA,
            s=PlotConfig.MARKER_SIZE,
            palette=PlotConfig.COLOR_PALETTE,
            marginal_kws=dict(common_norm=False, fill=True),
        )
        if g_tsne.ax_joint.legend_ is not None:
            g_tsne.ax_joint.legend_.set_bbox_to_anchor((1.1, 1))
        g_tsne.figure.suptitle(f"{inputs.dataset} | t-SNE Shift Comparison | {layer}", fontsize=14)
        g_tsne.figure.tight_layout()
        g_tsne.figure.subplots_adjust(top=0.9, right=0.8)

        save_fig(g_tsne.figure, output_dir, f"{inputs.dataset}_{inputs.encoder_to_evaluate}_{layer}_tSNE_jointplot.png")
