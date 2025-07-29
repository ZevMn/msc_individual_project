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
    4. Optionally enable statistical tests via 'calculate_all_shift_metrics'.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.embeddings.config import Config, PlotConfig
from experiments.embeddings.statistical_utils import (
    calculate_all_shift_metrics,
    save_results,
)


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
    pca_components: int = 2,
    seed: int = Config.SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings with PCA and t-SNE.

    PCA is applied to reduce dimensionality (default 2D), followed by t-SNE
    on the PCA output to obtain a 2D embedding.

    Args:
        embeddings: Tensor of shape (n_samples, n_features).
        pca_components: Number of components for PCA (capped to available dims).
        seed: Random seed for reproducibility.

    Returns:
        Tuple (embeddings_pca, embeddings_tsne), both np.ndarrays.
    """
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()

    embeddings_np = embeddings.numpy()
    if embeddings_np.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings_np.shape}")

    max_components = min(embeddings_np.shape[0], embeddings_np.shape[1])
    if max_components < 2:
        raise ValueError(
            f"Too few samples or features to reduce: shape {embeddings_np.shape}"
        )

    pca_components = min(pca_components, max_components)
    pca = PCA(n_components=pca_components, whiten=False, random_state=seed)
    embeddings_pca = pca.fit_transform(embeddings_np)

    # Optional: log explained variance
    print(f"PCA shape: {embeddings_pca.shape}")
    print(
        f"PCA explained variance ratio: {pca.explained_variance_ratio_[:min(2, pca_components)]}"
    )

    # t-SNE always outputs 2D for plotting purposes
    n_samples = embeddings_np.shape[0]
    perplexity = min(30, max(1, (n_samples - 1) // 3))
    embeddings_tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embeddings_pca)

    print(f"t-SNE shape: {embeddings_tsne.shape}")

    return embeddings_pca, embeddings_tsne


def title_and_save_fig(
    title: str, fig: Figure, file_location: Path, file_name: str, fontsize: int = 16
) -> None:
    """
    Saves a figure, closes it, and prints a confirmation.

    Args:
        fig: Matplotlib figure to save.
        file_location: Directory where the figure should be written. Created if
            it does not already exist.
        file_name: Name of figure to be saved.
    """
    fig.suptitle(title, fontsize=fontsize)
    file_location.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_location / file_name, dpi=400, bbox_inches="tight")
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
    num_samples: int = 2048,
) -> None:
    """
    Plots PCA and t-SNE projections for a single layer's embeddings, coloured by
    each label column (e.g., class, view, laterality, etc).

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

    n_rows = len(columns)
    fig = plt.figure(figsize=PlotConfig.get_figsize(n_rows), constrained_layout=True)
    axes = fig.subplots(n_rows, 2)

    # PCA and t-SNE (dimensionality reduction)
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
            palette=PlotConfig.COLOR_PALETTE,
            alpha=PlotConfig.ALPHA,
            s=PlotConfig.MARKER_SIZE,
            ax=axes[i, 0],
            legend=False,
        )
        ax_pca.set_title(f"{column.title()} (PCA)", fontsize=12)
        ax_pca.set_xlabel("")
        ax_pca.set_ylabel("")

        # t-SNE plot
        ax_tsne = sns.scatterplot(
            data=sample,
            x="t-SNE 1",
            y="t-SNE 2",
            hue=column,
            palette=PlotConfig.COLOR_PALETTE,
            alpha=PlotConfig.ALPHA,
            s=PlotConfig.MARKER_SIZE,
            ax=axes[i, 1],
            legend=True,
        )
        ax_tsne.set_title(f"{column.title()} (t-SNE)", fontsize=12)
        ax_tsne.set_xlabel("")
        ax_tsne.set_ylabel("")
        sns.move_legend(
            ax_tsne, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False
        )

    title_and_save_fig(
        f"Feature Representation: {dataset} | {encoder_to_evaluate.title()} | {layer_name.replace('_', ' ').title()} | {shift.replace('_', ' ').title()} Shift",
        fig,
        output_dir,
        f"{dataset}_{encoder_to_evaluate}_{layer_name}_{shift}.png",
    )


def plot_all_layer_representations_scatter(
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
    all_results = []

    for layer in inputs.layers:
        print(f"\n--- Processing layer: {layer} ---")

        # Reference data (full val dataset)
        print("\nProcessing reference data (no shift)...")
        plot_layer_representation_scatter(
            output_dir=output_dir,
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
                output_dir=output_dir,
                encoder_to_evaluate=inputs.encoder_to_evaluate,
                dataset=inputs.dataset,
                layer_name=layer,
                layer_embeddings=inputs.test_embeddings[layer][idx_array],
                labels=shifted_labels,
                shift=shift_name,
            )

            if run_statistical_tests:
                results = calculate_all_shift_metrics(
                    source_distribution=inputs.val_embeddings[layer],
                    target_distribution=inputs.test_embeddings[layer][idx_array],
                    layer_name=layer,
                    shift=shift_name,
                )
                all_results.extend(results)

    if run_statistical_tests and all_results:
        save_results(all_results, output_dir)


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
        df = df.sample(frac=1)  # Shuffle data for unbiased visualisation

        projections = [("PCA 1", "PCA 2", "PCA"), ("t-SNE 1", "t-SNE 2", "t-SNE")]

        for x, y, title_suffix in projections:
            graph = sns.jointplot(
                data=df,
                x=x,
                y=y,
                hue="Shift",
                kind="scatter",
                height=5,
                ratio=4,
                space=0,
                palette=PlotConfig.COLOR_PALETTE,
                alpha=PlotConfig.ALPHA,
                s=PlotConfig.MARKER_SIZE,
                marginal_kws=dict(common_norm=False, fill=True),
            )

            handles, labels = graph.ax_joint.get_legend_handles_labels()

            if graph.ax_joint.legend_ is not None:
                graph.ax_joint.legend_.remove()

            graph.figure.legend(
                handles,
                labels,
                title="Shift",
                loc="upper right",
                bbox_to_anchor=(1, 0.9),
                borderaxespad=0.5,
                frameon=False,
                fontsize=8,
                title_fontsize=12,
            )

            title_and_save_fig(
                f"Shift Comparison: {inputs.dataset} | {inputs.encoder_to_evaluate.title()} | {layer.replace('_', ' ').title()} | {title_suffix}",
                graph.figure,
                output_dir,
                f"{inputs.dataset}_{inputs.encoder_to_evaluate}_{layer}_{title_suffix}_joint.png",
                fontsize=12,
            )


def plot_shift_comparison_scatter(output_dir: Path, inputs: PlotInputs) -> None:
    """
    Generate a single figure with a grid of PCA and t-SNE scatterplots per layer
    to compare embedding spaces across layers and shifts.
    """
    set_plot_style()

    n_layers = len(inputs.layers)
    fig = plt.figure(figsize=PlotConfig.get_figsize(n_layers, permute=True))
    gs = gridspec.GridSpec(2, n_layers, figure=fig, hspace=0.35, wspace=0.25)

    handles_legend, labels_legend = None, None

    for i, layer in enumerate(inputs.layers):

        cat_embeddings, shift_labels = concat_embeddings(
            val_embeddings_layer=inputs.val_embeddings[layer],
            test_embeddings_layer=inputs.test_embeddings[layer],
            shift_to_indices_dict=inputs.shift_to_indices_dict,
        )
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
        df = df.sample(frac=1)  # Shuffle data for unbiased visualisation

        clean_layer = layer.replace("_", " ").title()

        # PCA subplot
        ax_pca = fig.add_subplot(gs[0, i])
        sc = sns.scatterplot(
            data=df,
            x="PCA 1",
            y="PCA 2",
            hue="Shift",
            style="Shift",
            palette=PlotConfig.COLOR_PALETTE,
            s=PlotConfig.MARKER_SIZE,
            alpha=PlotConfig.ALPHA,
            ax=ax_pca,
            legend="brief",
        )
        ax_pca.set_title(f"{clean_layer} (PCA)", fontsize=11)
        ax_pca.set_xlabel("")
        ax_pca.set_ylabel("")

        if i == 0:
            handles_legend, labels_legend = sc.get_legend_handles_labels()
        if ax_pca.legend_:
            ax_pca.legend_.remove()

        # t-SNE subplot
        ax_tsne = fig.add_subplot(gs[1, i])
        sns.scatterplot(
            data=df,
            x="t-SNE 1",
            y="t-SNE 2",
            hue="Shift",
            style="Shift",
            palette=PlotConfig.COLOR_PALETTE,
            s=PlotConfig.MARKER_SIZE,
            alpha=PlotConfig.ALPHA,
            ax=ax_tsne,
            legend=False,
        )
        ax_tsne.set_title(f"{clean_layer} (t-SNE)", fontsize=11)
        ax_tsne.set_xlabel("")
        ax_tsne.set_ylabel("")

    if handles_legend and labels_legend and len(labels_legend) > 0:
        fig.legend(
            handles_legend,
            labels_legend,
            loc="upper center",
            ncol=len(labels_legend),
            frameon=False,
            bbox_to_anchor=(0.8, 0.99),
            bbox_transform=fig.transFigure,
        )

    title_and_save_fig(
        f"Shift Comparison: {inputs.dataset} | {inputs.encoder_to_evaluate.title()}",
        fig,
        output_dir,
        f"{inputs.dataset}_{inputs.encoder_to_evaluate}_shift_comparison_scatter.png",
    )


def plot_detection_rate_heatmap(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
) -> None:

    set_plot_style()

    try:
        detection_rate_csv = pd.read_csv(
            output_dir / f"{dataset}_{encoder_to_evaluate}_stats.csv"
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {dataset}_{encoder_to_evaluate}_stats.csv was not found in {output_dir}.")

    significance_columns = [
        "mp_mmd_is_significant",
        "layer_1_mmd_is_significant",
        "layer_2_mmd_is_significant",
        "layer_3_mmd_is_significant",
        "final_layer_mmd_is_significant",
        "bbsd_is_significant",
    ]
    significance_data = detection_rate_csv[significance_columns]
    formatted_data = significance_data.apply(lambda x: f"**{x}**" if x else f"{x}")

    fig = plt.figure(figsize=(10, 8))

    sns.heatmap(
        detection_rate_csv[significance_columns],
        annot=formatted_data,
        fmt="",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Detection Rate"},
    )

    title_and_save_fig(
        f"Shift Detection Rate Heatmap: {dataset} | {encoder_to_evaluate.title()}",
        fig,
        output_dir,
        f"{dataset}_{encoder_to_evaluate}_detection_rate_heatmap",
    )

    return
