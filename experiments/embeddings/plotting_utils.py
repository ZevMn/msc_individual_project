"""
experiments/embeddings/plotting_utils.py

Utilities to visualise and statistically compare embedding spaces across encoder
layers and distribution shifts.

The module provides:
    - Helpers to concatenate embeddings and compute PCA/t-SNE projections.
    - High-level plotting routines (scatter plots and joint plots) for individual
      layers or across layers, optionally coloured by labels or shift type.
    - A thin wrapper to run BBSD and MMD permutation tests for distribution shift
      detection.

Typical workflow:
    1. Create a 'PlotInputs' instance bundling embeddings and metadata.
    2. Call 'plot_all_layers_labelled_scatter' to visualise each layer by labels.
    3. Call 'plot_shift_comparison_scatter' or 'plot_shift_comparison_joint'
       to compare reference vs shifted distributions.
    4. Optionally enable statistical tests via 'calculate_bbsd_and_mmd'.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from config import Config, PlotConfig
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.embeddings.statistical_utils import calculate_bbsd_and_mmd

@dataclass
class PlotInputs:
    """
    Container for inputs required by plotting functions.

    Attributes:
        encoder_to_evaluate: Name of the encoder used to generate features.
        dataset: Name of the dataset being analysed.
        layers: Ordered list of layer names to process and plot.
        val_embeddings: Mapping from layer name to a tensor of validation (source) embeddings.
        test_embeddings: Mapping from layer name to a tensor of test (target) embeddings.
        shift_to_indices_dict: Mapping from shift name to a NumPy array of
            integer indices corresponding to the subset of test embeddings belonging to
            that covariate shift.
    """
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
    Apply a consistent seaborn/matplotlib style to all subsequent plots.
    """
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams.update({"font.family": "serif"})


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


def calculate_PCA_and_tSNE(
    embeddings: torch.Tensor,
    seed: int = Config.SEED,
    pca_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings with PCA and t-SNE.

    The embeddings are first moved to CPU (if necessary) and converted to a
    NumPy array. PCA is applied to reduce to 2D, and then t-SNE is applied to
    the PCA output to obtain a 2D embedding.

    Args:
        embeddings: Tensor containing the features to reduce.
        seed: Random seed for reproducibility of PCA and t-SNE.

    Returns:
        A tuple (embeddings_pca, embeddings_tsne) where:
            embeddings_pca is a NumPy array.
            embeddings_tsne is a NumPy array.
    """

    if embeddings.is_cuda:  # Ensure embeddings on CPU (and not GPU)
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
        init="random",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embeddings_pca)
    print(f"t-SNE shape: {embeddings_tsne.shape}")

    return embeddings_pca, embeddings_tsne


def save_fig(fig: Figure, file_location: Path, file_name: str) -> None:
    """
    Saves a figure, closes it, and prints a confirmation.

    Args:
        fig: Matplotlib figure to save.
        file_location: Directory where the figure should be written. Created if
            it does not already exist.
        file_name: Name of figure to be saved.
    """
    file_location.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_location / file_name, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {file_name}\n")


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
    shift: str = "no_shift",
    seed: int = Config.SEED,
    num_samples: int = 1000,
) -> None:
    """
    For a single layer, this function computes PCA and t-SNE on the provided
    embeddings and produces a grid of scatter plots (one row per label categorisation).

    Args:
        output_dir: Directory where the plot PNG will be saved.
        encoder_to_evaluate: Name of the encoder used to generate features.
        dataset: Name of the dataset being analysed.
        layer_name: Name of the layer of the encoder being plotted.
        layer_embeddings: Feature embeddings tensor for this layer.
        labels: Dict mapping column name to a NumPy array with categorical labels to colour points.
        shift: String identifier for the shift scenario (e.g., "no_shift", "acq").
        seed: Random seed for reproducibility of sampling points for plotting.
        pca_components: The number of principal components to reduce to.
        num_samples: Maximum number of points to include in the plot.
    """
    set_plot_style()

    columns = Config.DATASET_CONFIG[dataset]["plot_columns"]
    fig, axes = plt.subplots(
        len(columns), 2, figsize=PlotConfig.FIGURE_SIZE, constrained_layout=True
    )

    # PCA and t-SNE
    embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(layer_embeddings)

    df = pd.DataFrame(
        {
            **{col: labels[col] for col in columns},
            "PCA 1": embeddings_pca[:, 0],
            "PCA 2": embeddings_pca[:, 1],
            "t-SNE 1": embeddings_tsne[:, 0],
            "t-SNE 2": embeddings_tsne[:, 1],
        }
    )

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
            ax=axes[i, 0],
        )
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
            ax=axes[i, 1],
        )
        ax_tsne.set_title(f"t-SNE coloured by {column}")

    for ax in axes.ravel():
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))

    fig.suptitle(f"{dataset} | Scenario: {shift} - {layer_name}", fontsize=16)

    save_fig(fig, output_dir, f"{shift}_{layer_name}_{encoder_to_evaluate}.png")


def plot_all_layers_scatter_labelled(
    output_dir: Path,
    inputs: PlotInputs,
    val_labels: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    run_statistical_tests: bool = False,
) -> None:
    """
    Plot labelled PCA/t-SNE scatterplots for each layer and each shift subset.

    For every layer in inputs.layers:
        1. Plot the full validation set (no shift).
        2. For each shift subset, plot the corresponding test embeddings.
        3. Optionally, run BBSD and MMD between validation and each shifted subset.

    Args:
        output_dir: Directory where the plot PNGs will be saved.
        inputs: A 'PlotInputs' instance.
        val_labels: Dict of label arrays (same length as validation embeddings).
        test_labels: Dict of label arrays (same length as test embeddings).
        run_statistical_tests: If True, BBSD and MMD tests are executed.
    """
    for layer in inputs.layers:
        print(f"\n--- Processing layer: {layer} ---")

        # Reference data (full val dataset)
        print("\nProcessing reference data (no shift)...")
        plot_layer_representation_scatter(
            output_dir=output_dir / "layers_representations",
            encoder_to_evaluate=inputs.encoder_to_evaluate,
            dataset=inputs.dataset,
            layer_name=layer,
            layer_embeddings=inputs.val_embeddings[layer],
            labels=val_labels,
            shift="no_shift",
        )

        # Shifted data (test dataset subsets)
        for shift_name, idx_array in inputs.shift_to_indices_dict.items():
            print(f"\nProcessing {shift_name}...")
            shifted_labels = {k: v[idx_array] for k, v in test_labels.items()}
            plot_layer_representation_scatter(
                output_dir=output_dir / "layers_representations",
                encoder_to_evaluate=inputs.encoder_to_evaluate,
                dataset=inputs.dataset,
                layer_name=layer,
                layer_embeddings=inputs.test_embeddings[layer][idx_array],
                labels=shifted_labels,
                shift=shift_name,
            )

            if run_statistical_tests:
                calculate_bbsd_and_mmd(
                    source_distribution=inputs.val_embeddings[layer],
                    target_distribution=inputs.test_embeddings[layer][idx_array],
                    layer_name=layer,
                    shift=shift_name,
                )


def plot_shift_comparison_scatter(
    output_dir: Path,
    inputs: PlotInputs,
) -> None:
    """
    Compare reference vs shifted distributions across layers with scatter plots.

    For each layer, validation and shifted subsets are concatenated to share the
    same PCA/t-SNE space. Two scatterplots (PCA and t-SNE), coloured by shift type,
    are produced per layer.

    Args:
        output_dir: Directory where the plot PNGs will be saved.
        inputs: A 'PlotInputs' instance.
    """
    set_plot_style()

    fig, axes = plt.subplots(
        len(inputs.layers), 2, figsize=PlotConfig.FIGURE_SIZE, constrained_layout=True
    )

    for i, layer in enumerate(inputs.layers):

        # Concatenate the reference features and shifted features so that they share a PCA space
        cat_embeddings, shift_labels = concat_embeddings(
            val_embeddings_layer=inputs.val_embeddings[layer],
            test_embeddings_layer=inputs.test_embeddings[layer],
            shift_to_indices_dict=inputs.shift_to_indices_dict,
        )

        # PCA and t-SNE
        embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(cat_embeddings)

        df = pd.DataFrame(
            {
                "Shift": shift_labels,
                "PCA 1": embeddings_pca[:, 0],
                "PCA 2": embeddings_pca[:, 1],
                "t-SNE 1": embeddings_tsne[:, 0],
                "t-SNE 2": embeddings_tsne[:, 1],
            }
        )

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
            ax=axes[i, 0],
        )
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
            ax=axes[i, 1],
        )
        ax_tsne.set_title(f"t-SNE Shift Comparison for {layer}")

    for ax in axes.ravel():
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))

    fig.suptitle(
        f"{inputs.dataset} | Shift Comparisons for All Layers of {inputs.encoder_to_evaluate} Encoder Using PCA and t-SNE Analysis",
        fontsize=16,
    )

    save_fig(
        fig,
        output_dir,
        f"{inputs.dataset}_{inputs.encoder_to_evaluate}_shift_comparison_scatterplots.png",
    )


def plot_shift_comparison_joint(output_dir: Path, inputs: PlotInputs) -> None:
    """
    Create seaborn jointplots (scatter + marginal densities) for PCA and t-SNE.

    For each layer, validation and shifted subsets are concatenated to share the
    same PCA/t-SNE space. Two jointplots (PCA and t-SNE), coloured by shift type,
    are produced per layer.

    Args:
        output_dir: Directory where the plot PNGs will be saved.
        inputs: A 'PlotInputs' instance.
    """
    set_plot_style()

    for layer in inputs.layers:

        # Concatenate the reference features and shifted features so that they share a PCA space
        cat_embeddings, shift_labels = concat_embeddings(
            val_embeddings_layer=inputs.val_embeddings[layer],
            test_embeddings_layer=inputs.test_embeddings[layer],
            shift_to_indices_dict=inputs.shift_to_indices_dict,
        )

        # PCA and t-SNE
        embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(cat_embeddings)

        df = pd.DataFrame(
            {
                "Shift": shift_labels,
                "PCA 1": embeddings_pca[:, 0],
                "PCA 2": embeddings_pca[:, 1],
                "t-SNE 1": embeddings_tsne[:, 0],
                "t-SNE 2": embeddings_tsne[:, 1],
            }
        )

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
        g_pca.figure.suptitle(
            f"{inputs.dataset} | PCA Shift Comparison | {layer}", fontsize=14
        )
        g_pca.figure.tight_layout()
        g_pca.figure.subplots_adjust(top=0.9, right=0.8)

        save_fig(
            g_pca.figure,
            output_dir,
            f"{inputs.dataset}_{inputs.encoder_to_evaluate}_{layer}_PCA_jointplot.png",
        )

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
        g_tsne.figure.suptitle(
            f"{inputs.dataset} | t-SNE Shift Comparison | {layer}", fontsize=14
        )
        g_tsne.figure.tight_layout()
        g_tsne.figure.subplots_adjust(top=0.9, right=0.8)

        save_fig(
            g_tsne.figure,
            output_dir,
            f"{inputs.dataset}_{inputs.encoder_to_evaluate}_{layer}_tSNE_jointplot.png",
        )
