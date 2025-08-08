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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from experiments.embeddings.config import Config, PlotConfig
from experiments.embeddings.statistical_utils import calculate_PCA_and_tSNE


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


def title_and_save_fig(
    title: str, fig: Figure, file_location: Path, file_name: str, fontsize: int = 16
) -> None:
    """
    Set a figure title, ensure 'file_location' exists, save PNG, and close.

    Args:
        title: Figure suptitle.
        fig: Matplotlib Figure to save.
        file_location: Target directory (created if needed).
        file_name: Output file name (include extension).
        fontsize: Title font size.
    """
    fig.suptitle(title, fontsize=fontsize)
    file_location.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_location / file_name, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {file_name}\n")


# ------------------
# Plotting functions
# ------------------
@dataclass
class VisPlotInputs:
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


def plot_layer_representations_scatter(
    output_dir: Path,
    inputs: VisPlotInputs,
    labels: dict[str, np.ndarray],
    shift: str = "no_shift",
    seed: int = Config.SEED,
    num_samples: int = 2000,
) -> None:
    """
    Plots PCA and t-SNE projections for all layers' embeddings, creating separate
    figures for PCA and t-SNE. Each figure has layers as rows and label types as columns.

    Args:
        output_dir: Directory where the plot PNGs will be saved.
        encoder_to_evaluate: Name of the encoder used to generate features.
        dataset: Name of the dataset being analysed.
        layers: Names of the layers of the encoder being plotted.
        layer_embeddings: Feature embeddings tensor for each layer.
        labels: Dict mapping column name to a NumPy array (of same length as 'layer_embeddings[layer]')
            with categorical labels to colour points.
        shift: String identifier for the simulated shift (or "no_shift" for reference data).
        seed: Random seed for reproducibility of sampling points for plotting.
        num_samples: Maximum number of points to include in the plot.
    """
    set_plot_style()

    output_dir = Config.ROOT / "experiments" / "collected_outputs" / inputs.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = Config.DATASET_CONFIG[inputs.dataset]["plot_columns"]
    n_rows = len(inputs.layers)
    n_cols = len(columns)

    # Calculate PCA dimensionality reductions for all layers
    embeddings_data = {}
    for layer in inputs.layers:
        print(f"Processing embeddings for layer: {layer}")
        embeddings_pca, embeddings_tsne = calculate_PCA_and_tSNE(
            inputs.val_embeddings[layer]
        )

        df = pd.DataFrame(
            {
                **{col: labels[col] for col in columns},
                "PCA 1": embeddings_pca[:, 0],
                "PCA 2": embeddings_pca[:, 1],
                "t-SNE 1": embeddings_tsne[:, 0],
                "t-SNE 2": embeddings_tsne[:, 1],
            }
        )

        if inputs.dataset == "Mammo" and "Manufacturer" in df.columns:
            # Replace manufacturer names with numbers or they spill over the legend
            df["Manufacturer"] = df["Manufacturer"].astype("category").cat.codes

        # Sample for plotting
        embeddings_data[layer] = df.sample(
            n=min(num_samples, len(df)), random_state=seed
        )

    main_title = f"{inputs.dataset} | {inputs.encoder_to_evaluate.title()} | {shift.replace('_', ' ').title()}"

    # Create PCA figure
    print("\nCreating PCA visualization...")
    fig_pca = plt.figure(
        figsize=(4 * n_cols, 3 * n_rows + 2),
        constrained_layout=True,
    )

    # Add main title for PCA
    fig_pca.suptitle(
        f"PCA Feature Representation: {main_title}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Create subplot grid for PCA
    gs_pca = fig_pca.add_gridspec(
        n_rows, n_cols, top=0.90, left=0.08, right=0.95, hspace=0.3, wspace=0.4
    )

    # Plot PCA subplots
    for row_idx, layer in enumerate(inputs.layers):
        sample = embeddings_data[layer]

        for col_idx, column in enumerate(columns):
            ax = fig_pca.add_subplot(gs_pca[row_idx, col_idx])

            sns.scatterplot(
                data=sample,
                x="PCA 1",
                y="PCA 2",
                hue=column,
                palette=PlotConfig.COLOR_PALETTE,
                alpha=PlotConfig.ALPHA,
                s=PlotConfig.MARKER_SIZE,
                ax=ax,
                legend=(row_idx == 0),
            )

            # Position legend at top of column
            if row_idx == 0 and ax.get_legend():
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.4),
                    ncol=min(len(sample[column].unique()), 4),
                    frameon=False,
                    fontsize=9,
                )

            # Set titles and labels
            if row_idx == 0:
                ax.set_title(f"{column}", fontsize=14, fontweight="bold", pad=10)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{layer.replace('_', ' ').title()}",
                    fontsize=12,
                    fontweight="bold",
                    labelpad=10,
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    plt.tight_layout()

    # Save PCA figure
    pca_filename = f"{inputs.dataset}_{inputs.encoder_to_evaluate}_pca_{shift}.png"
    fig_pca.savefig(
        output_dir / pca_filename, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_pca)

    print(f"[Saved] {pca_filename}\n")

    # Create t-SNE figure
    print("Creating t-SNE visualization...")
    fig_tsne = plt.figure(figsize=(4 * n_cols, 3 * n_rows + 2), constrained_layout=True)

    # Add main title for t-SNE
    fig_tsne.suptitle(
        f"t-SNE Feature Representation: {main_title}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Create subplot grid for t-SNE
    gs_tsne = fig_tsne.add_gridspec(
        n_rows, n_cols, top=0.90, left=0.08, right=0.95, hspace=0.3, wspace=0.4
    )

    # Plot t-SNE subplots
    for row_idx, layer in enumerate(inputs.layers):
        sample = embeddings_data[layer]

        for col_idx, column in enumerate(columns):
            ax = fig_tsne.add_subplot(gs_tsne[row_idx, col_idx])

            sns.scatterplot(
                data=sample,
                x="t-SNE 1",
                y="t-SNE 2",
                hue=column,
                palette=PlotConfig.COLOR_PALETTE,
                alpha=PlotConfig.ALPHA,
                s=PlotConfig.MARKER_SIZE,
                ax=ax,
                legend=(row_idx == 0),
            )

            # Position legend at top of column
            if row_idx == 0 and ax.get_legend():
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.4),
                    ncol=min(len(sample[column].unique()), 4),
                    frameon=False,
                    fontsize=9,
                )

            # Set titles and labels
            if row_idx == 0:
                ax.set_title(f"{column}", fontsize=14, fontweight="bold", pad=10)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{layer.replace('_', ' ').title()}",
                    fontsize=12,
                    fontweight="bold",
                    labelpad=10,
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    plt.tight_layout()

    # Save t-SNE figure
    tsne_filename = f"{inputs.dataset}_{inputs.encoder_to_evaluate}_tsne_{shift}.png"
    fig_tsne.savefig(
        output_dir / tsne_filename, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_tsne)

    print(f"[Saved] {tsne_filename}\n")


def plot_shift_comparison_scatter(
    output_dir: Path, inputs: VisPlotInputs, n_samples=2000
) -> None:
    """
    Generate a single figure with a grid of PCA and t-SNE scatterplots per layer
    to compare embedding spaces across layers and shifts.
    """
    set_plot_style()

    output_dir = Config.ROOT / "experiments" / "collected_outputs" / inputs.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(inputs.layers)
    fig = plt.figure(figsize=PlotConfig.get_figsize(n_layers, permute=True))
    gs = gridspec.GridSpec(2, n_layers, figure=fig, hspace=0.35, wspace=0.25)

    handles_legend, labels_legend = None, None

    for i, layer in enumerate(inputs.layers):

        moderate_shifts_dict = {}
        for shift in inputs.shift_to_indices_dict.keys():
            if "moderate" in shift:
                moderate_shifts_dict[shift] = inputs.shift_to_indices_dict[shift]

        cat_embeddings, shift_labels = concat_embeddings(
            val_embeddings_layer=inputs.val_embeddings[layer],
            test_embeddings_layer=inputs.test_embeddings[layer],
            shift_to_indices_dict=moderate_shifts_dict,
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
        sample = df.sample(n=min(n_samples, len(df)))

        clean_layer = layer.replace("_", " ").title()

        # PCA subplot
        ax_pca = fig.add_subplot(gs[0, i])
        sc = sns.scatterplot(
            data=sample,
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
        ax_pca.set_title(f"{clean_layer} (PCA)", fontsize=11, fontweight="bold")
        ax_pca.set_xlabel("")
        ax_pca.set_ylabel("")

        if i == 0:
            handles_legend, labels_legend = sc.get_legend_handles_labels()
        if ax_pca.legend_:
            ax_pca.legend_.remove()

        # t-SNE subplot
        ax_tsne = fig.add_subplot(gs[1, i])
        sns.scatterplot(
            data=sample,
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
        ax_tsne.set_title(f"{clean_layer} (t-SNE)", fontsize=11, fontweight="bold")
        ax_tsne.set_xlabel("")
        ax_tsne.set_ylabel("")

    if handles_legend and labels_legend and len(labels_legend) > 0:
        fig.subplots_adjust(top=0.88)
        fig.legend(
            handles_legend,
            labels_legend,
            loc="upper center",
            ncol=len(labels_legend),
            frameon=False,
            bbox_to_anchor=(0.5, 0.95),
            bbox_transform=fig.transFigure,
            fontsize=10,
            columnspacing=1.0,
            handletextpad=0.5,
        )

    title_and_save_fig(
        f"Shift Comparison: {inputs.dataset} | {inputs.encoder_to_evaluate.title()}",
        fig,
        output_dir,
        f"{inputs.dataset}_{inputs.encoder_to_evaluate}_shift_comparison_scatter.png",
    )


def plot_shift_comparison_joint(output_dir: Path, inputs: VisPlotInputs) -> None:
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
        sample = df.sample(n=min(2048, len(df)))

        projections = [("PCA 1", "PCA 2", "PCA"), ("t-SNE 1", "t-SNE 2", "t-SNE")]

        for x, y, title_suffix in projections:
            graph = sns.jointplot(
                data=sample,
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


def plot_detection_rate_heatmap(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
) -> None:
    """
    Generate a heatmap of p-values representing shift detection rates of all simulated
    shifts across encoder layers, with significance marked by an asterisk.
    Expects a CSV named f"{dataset}_{encoder}_stats.csv" in output_dir.

    Args:
        output_dir: Directory where the heatmap PNG will be saved.
        dataset: Name of the dataset being analysed.
        encoder_to_evaluate: Name of the encoder used to generate features.

    """

    set_plot_style()

    # Load the shift detection rate csv
    filepath = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found.")
    detection_rate_df = pd.read_csv(filepath, index_col=0)

    # Extract p‑value and significance columns
    pval_cols = [c for c in detection_rate_df.columns if c.endswith("_pvalue")]
    sig_cols = [c for c in detection_rate_df.columns if c.endswith("_is_significant")]

    pvals = detection_rate_df[pval_cols]
    sigs = detection_rate_df[sig_cols]

    # Format p-values and append "*" if significant
    p_arr = pvals.values
    s_arr = sigs.values.astype(bool)
    annot_arr = np.char.mod("%.2f", p_arr)
    annot_arr = np.char.add(annot_arr, np.where(s_arr, "*", ""))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pvals,
        annot=annot_arr,
        fmt="",
        cmap="Blues_r",
        linewidths=0.5,
        cbar_kws={"label": "p value"},
        ax=ax,
        vmin=0.05,
        vmax=1,
    )
    # for i, j in zip(*np.where(s_arr)):
    #     ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

    # Clean up labels
    ax.set_ylabel("Shift", fontsize=14)
    clean_idx = [idx.replace("_", " ").title() for idx in pvals.index]
    ax.set_xticklabels(pvals.columns, rotation=45, ha="right")
    ax.set_yticklabels(clean_idx, rotation=0)

    title_and_save_fig(
        f"Shift Detection Rate Heatmap: {dataset} | {encoder_to_evaluate.replace('_', ' ').title()}",
        fig,
        output_dir,
        f"{dataset}_{encoder_to_evaluate}_detection_rate_heatmap",
    )


def plot_detection_rate_linegraph(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
) -> None:
    """
    Generate a line graph of p-values representing shift detection rates of all simulated
    shifts across encoder layers, with significance marked by an asterisk.
    Expects a CSV named f"{dataset}_{encoder}_stats.csv" in output_dir.

    Args:
        output_dir: Directory where the heatmap PNG will be saved.
        dataset: Name of the dataset being analysed.
        encoder_to_evaluate: Name of the encoder used to generate features.

    """
    set_plot_style()

    # Load the shift detection rate csv
    filepath = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found.")
    detection_rate_df = pd.read_csv(filepath, index_col=0)

    # Extract p‑value and significance columns
    pval_cols = [c for c in detection_rate_df.columns if c.endswith("_pvalue")]
    sig_cols = [c for c in detection_rate_df.columns if c.endswith("_is_significant")]

    pvals = detection_rate_df[pval_cols]
    sigs = detection_rate_df[sig_cols]

    layers = [
        col.removesuffix("_pvalue").replace("_", " ").title() for col in pval_cols
    ]
    shifts = [idx.replace("_", " ").title() for idx in pvals.index]

    # Plot the line graph
    fig, ax = plt.subplots(figsize=(10, 8))
    x = list(range(len(layers)))

    for shift_name, (p_row, s_row) in zip(shifts, zip(pvals.values, sigs.values)):
        ax.plot(x, p_row, marker="o", label=shift_name)
        for xi, (yi, sig) in enumerate(zip(p_row, s_row)):
            if sig:
                ax.text(xi, yi + 0.02, "*", ha="center", va="bottom", fontsize=12)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel("p-value", fontsize=12)
    ax.set_xlabel("Encoder Layer / Test", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.05, color="red", linestyle="--", label="Significance Threshold (0.05)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    title_and_save_fig(
        f"Shift Detection Rate Line Graph: {dataset} | {encoder_to_evaluate.replace('_', ' ').title()}",
        fig,
        output_dir,
        f"{dataset}_{encoder_to_evaluate}_detection_rate_line_graph",
    )


def plot_bootstrap_detection_rates(
    output_dir: Path,
    dataset: str,
    all_results_df: pd.DataFrame,
) -> None:
    """
    Plot bootstrap detection rates for a given dataset as bar charts.

    Args:
        output_dir: Directory to save the plot
        dataset: Name of the dataset to plot results for.
        all_results_df: DataFrame containing bootstrap results with columns:
                   ['dataset', 'encoder', 'shift', 'layer', 'test_size', 'detection_rate', ...]
    """

    # Check if required columns are present
    required = {"encoder", "layer", "shift", "test_size", "detection_rate"}
    missing = required - set(all_results_df.columns)
    if missing:
        raise ValueError(f"Missing columns in all_results_df: {missing}")

    # Get unique values for organizing the plot
    encoders = sorted(all_results_df["encoder"].unique())
    layers = ["after_maxpool", "layer_1", "layer_2", "layer_3", "final_layer"]
    shifts = sorted(all_results_df["shift"].unique())
    test_sizes = sorted(all_results_df["test_size"].unique())

    # Create figure and subplots
    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(test_sizes)))
    n_encoders = len(encoders)
    n_layers = len(layers)
    fig, axes = plt.subplots(n_encoders, n_layers, figsize=(4*n_layers, 3*n_encoders + 2))

    # Handle the case where there's only one encoder - shape (1, n_layers)
    if len(encoders) == 1:
        axes = axes.reshape(1, -1)

    # Set up bar positioning
    bar_width = 0.8 / len(test_sizes)
    shift_positions = np.arange(len(shifts))

    # Plot for each encoder and layer combination
    for encoder_idx, encoder in enumerate(encoders):
        for layer_idx, layer in enumerate(layers):
            # Account for only one encoder
            ax = axes[encoder_idx, layer_idx]

            # Filter data for this encoder and layer
            subset = all_results_df[
                (all_results_df["encoder"] == encoder)
                & (all_results_df["layer"] == layer)
            ]

            available_test_sizes = sorted(subset["test_size"].unique())

            # Plot bars for each test size
            for size_idx, test_size in enumerate(available_test_sizes):
                size_data = subset[subset["test_size"] == test_size]

                # Get detection rates for each shift (in order)
                detection_rates = []
                valid_shifts_for_size = []
                for shift in shifts:
                    shift_data = size_data[size_data["shift"] == shift]
                    if len(shift_data) > 0:
                        detection_rates.append(shift_data["detection_rate"].iloc[0])
                        valid_shifts_for_size.append(shift)
                # Skip this test_size if no shifts have data for it
                if not detection_rates:
                    continue

                # Calculate bar positions
                valid_shift_positions = np.arange(len(valid_shifts_for_size))
                bar_positions = (
                    valid_shift_positions
                    + (size_idx - len(test_sizes) / 2 + 0.5) * bar_width
                )

                # Plot bars
                bars = ax.bar(
                    bar_positions,
                    detection_rates,
                    width=bar_width,
                    color=colors[size_idx],
                    alpha=0.8,
                    label=(
                        f"{test_size} samples"
                        if encoder_idx == 0 and layer_idx == 0
                        else ""
                    ),
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add value labels on bars if detection rate > 0
                for bar, rate in zip(bars, detection_rates):
                    if rate > 0.05:  # Only label if detection rate is substantial
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{rate:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=0,
                        )

            ax.set_xlim(-0.5, len(shifts) - 0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(shift_positions)
            ax.set_xticklabels(shifts, rotation=45, ha="right")
            ax.set_ylabel("Detection Rate" if layer_idx == 0 else "")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)

            # Add layer title
            if encoder_idx == 0:
                ax.set_title(
                    layer.replace("_", " ").title(), fontsize=12, fontweight="bold"
                )

            # Add encoder label on the left
            if layer_idx == 0:
                ax.text(
                    -0.15,
                    0.5,
                    encoder,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

    # Add main title
    fig.suptitle(
        f"{dataset} - Bootstrap Detection Rates", fontsize=16, fontweight="bold", y=0.98
    )

    # Add shared legend
    if len(test_sizes) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(test_sizes),
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    # Save figure if output directory is provided
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"bootstrap_detection_rates_{dataset}.png"
    filepath = output_dir / file_name
    fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {file_name}\n")


def plot_all_bootstrap_results(
    output_dir: Path,
) -> None:
    """
    Load and plot bootstrap results for all datasets found in results directory.

    Args:
        output_dir: Directory containing CSV files with bootstrap results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    file_pattern = "bootstrap_detection_rates-*.csv"
    csv_files = list(output_dir.glob(file_pattern))
    if not csv_files:
        raise ValueError(
            f"No files matching pattern '{file_pattern}' found in {output_dir}"
        )

    all_results_df = pd.DataFrame()
    for csv_file in csv_files:
        # Extract dataset and encoder from filename
        # Assuming format: bootstrap_detection_rates-{dataset}-{encoder}.csv
        filename_parts = csv_file.stem.split("-")
        dataset = filename_parts[1]
        encoder = filename_parts[2]

        try:
            results_df = pd.read_csv(csv_file)
            results_df["dataset"] = dataset
            results_df["encoder"] = encoder
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
        except Exception as e:
            print(f"! Error processing {csv_file}: {e}")
            continue

    datasets = all_results_df["dataset"].unique()
    for dataset in datasets:
        plot_bootstrap_detection_rates(
            output_dir=output_dir,
            dataset=dataset,
            all_results_df=all_results_df[all_results_df["dataset"] == dataset],
        )
