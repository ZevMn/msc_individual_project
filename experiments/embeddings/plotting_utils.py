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

import os
from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
import matplotlib.lines as mlines

from experiments.embeddings.config import Config, PlotConfig


# ----------------
# Helper functions
# ----------------
def set_plot_style() -> None:
    """
    Apply a consistent seaborn/matplotlib style to all subsequent plots.
    """
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams.update({"font.family": "serif"})


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
def plot_layer_representations_scatter(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
    layer_to_results_dict: dict[str, pd.DataFrame],
    labels: dict[str, np.ndarray],
    shift: str = "no_shift",
    seed: int = Config.SEED,
    n_samples: int = 2000,
) -> None:
    """
    Plots PCA and t-SNE projections for all layers' embeddings, creating separate
    figures for PCA and t-SNE. Each figure has layers as rows and label types as columns.

    Args:
        output_dir: Directory where the plot PNGs will be saved.
        inputs: VisPlotInputs object containing data for plotting.
        labels: Dict mapping column name to a NumPy array (of same length as 'layer_embeddings[layer]')
            with categorical labels to colour points.
        shift: String identifier for the simulated shift (or "no_shift" for reference data).
        seed: Random seed for reproducibility of sampling points for plotting.
        n_samples: Maximum number of points to include in the plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    layers = layer_to_results_dict.keys()
    columns = Config.DATASET_CONFIG[dataset]["plot_columns"]
    n_rows = len(layers)
    n_cols = len(columns)

    embeddings_data: dict[str, pd.DataFrame] = {}

    # Build PCA and t-SNE dataframe for plotting for all layers
    for layer in layers:
        layer_df = layer_to_results_dict[layer].query("Shift == @shift").copy()
        if len(layer_df) == 0:
            raise ValueError(f"No data found for layer {layer}")

        layer_df = layer_df.assign(**{col: labels[col] for col in columns})

        if dataset == "Mammo" and "Manufacturer" in layer_df.columns:
            # Replace manufacturer names with numbers or they spill over the legend
            layer_df["Manufacturer"] = (
                layer_df["Manufacturer"].astype("category").cat.codes
            )

        embeddings_data[layer] = layer_df.sample(
            n=min(n_samples, len(layer_df)), random_state=seed
        )

    main_title = (
        f"{dataset} | {encoder_to_evaluate.title()} | {shift.replace('_', ' ').title()}"
    )

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
    for row_idx, layer in enumerate(layers):
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
    pca_filename = f"{dataset}_{encoder_to_evaluate}_pca_{shift}.png"
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
    for row_idx, layer in enumerate(layers):
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
    tsne_filename = f"{dataset}_{encoder_to_evaluate}_tsne_{shift}.png"
    fig_tsne.savefig(
        output_dir / tsne_filename, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_tsne)

    print(f"[Saved] {tsne_filename}\n")


def plot_shift_comparison_scatter(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
    layer_to_results_dict: dict,
    n_samples=2000,
) -> None:
    """
    Generate a single figure with a grid of PCA and t-SNE scatterplots per layer
    to compare embedding spaces across layers and shifts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    layers = layer_to_results_dict.keys()

    n_layers = len(layers)

    print("\nCreating scatterplots for shift comparison...")
    fig = plt.figure(figsize=PlotConfig.get_figsize(n_layers, permute=True))
    gs = gridspec.GridSpec(2, n_layers, figure=fig, hspace=0.35, wspace=0.25)

    handles_legend, labels_legend = None, None

    for i, layer in enumerate(layers):

        layer_df = layer_to_results_dict[layer]

        # Filter for moderate shifts only
        moderate_shifts = [
            shift for shift in layer_df["Shift"].unique() if "moderate" in shift
        ]
        filtered_shifts_df = layer_df[layer_df["Shift"].isin(moderate_shifts)].copy()

        filtered_shifts_df = filtered_shifts_df.sample(frac=1, random_state=42)
        sample = filtered_shifts_df.sample(
            n=min(n_samples, len(filtered_shifts_df)), random_state=42
        )

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
        f"Shift Comparison: {dataset} | {encoder_to_evaluate.title()}",
        fig,
        output_dir,
        f"{dataset}_{encoder_to_evaluate}_shift_comparison_scatter.png",
    )


def plot_shift_comparison_joint(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
    layer_to_results_dict: dict[str, pd.DataFrame],
    n_samples=2000,
) -> None:
    """
    Grid of 'jointplot-style' panels (scatter + filled KDE marginals) for PCA and t-SNE.
    One row per layer, two columns: PCA (left) and t-SNE (right). A single legend is
    shown above each column (once).

    Preserves the look of the old jointplots:
      - seaborn scatter with hue="Shift"
      - filled KDE marginals with common_norm=False (per-class areas aren't normalized)
      - alpha/marker size/palette taken from PlotConfig

    Saves:  {dataset}-{encoder}-shift_comparison_jointgrid.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    layers = layer_to_results_dict.keys()
    n_rows = len(layers)
    n_cols = 2

    print("\nCreating jointplots for shift comparison...")
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows + 2), constrained_layout=False)
    fig.suptitle(
        f"Shift Comparison: {dataset} | {encoder_to_evaluate.title()}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    outer = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.04,
        wspace=0.30,
        hspace=0.40,
    )

    top_row_main_axes: List[Optional[Axes]] = [None, None]

    for i, layer in enumerate(layers):

        layer_df = layer_to_results_dict[layer].sample(frac=1)
        sample = layer_df.sample(n=min(n_samples, len(layer_df)), random_state=42)

        unique_shifts = sorted(sample["Shift"].unique())
        palette = sns.color_palette(
            PlotConfig.COLOR_PALETTE, n_colors=len(unique_shifts)
        )
        pal = {k: v for k, v in zip(unique_shifts, palette)}

        # ---- helper for one panel ----
        def joint_panel(
            cell_spec,
            x_col,
            y_col,
            title_if_top=None,
            draw_legend=False,
            show_row_label=True,
        ):
            sub = cell_spec.subgridspec(
                2, 2, height_ratios=(1, 4), width_ratios=(4, 1), wspace=0.0, hspace=0.0
            )
            ax_top = fig.add_subplot(sub[0, 0])
            ax_main = fig.add_subplot(sub[1, 0])
            ax_right = fig.add_subplot(sub[1, 1])
            fig.add_subplot(sub[0, 1]).axis("off")

            # main scatter
            sns.scatterplot(
                data=sample,
                x=x_col,
                y=y_col,
                hue="Shift",
                palette=pal,
                alpha=PlotConfig.ALPHA,
                s=PlotConfig.MARKER_SIZE,
                ax=ax_main,
                legend=False,
            )

            # KDE marginals
            for s in unique_shifts:
                d = sample[sample["Shift"] == s]
                sns.kdeplot(
                    data=d,
                    x=x_col,
                    ax=ax_top,
                    fill=True,
                    common_norm=False,
                    alpha=0.35,
                    linewidth=1,
                )
                sns.kdeplot(
                    data=d,
                    y=y_col,
                    ax=ax_right,
                    fill=True,
                    common_norm=False,
                    alpha=0.35,
                    linewidth=1,
                )

            # clean marginal axes
            for a in (ax_top, ax_right):
                a.set_xticks([])
                a.set_yticks([])
                a.set_xlabel("")
                a.set_ylabel("")
                sns.despine(ax=a, left=True, bottom=True)

            # axis labels
            ax_main.set_xlabel("")
            ax_main.set_ylabel(
                layer.replace("_", " ").title() if show_row_label else "",
                fontweight="bold",
            )

            if title_if_top:
                ax_main.set_title(title_if_top, fontsize=14, fontweight="bold", y=1.65)

            sns.despine(ax=ax_main)

            if draw_legend:
                handles, labels = [], []
                for s in unique_shifts:
                    h = mlines.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markersize=np.sqrt(PlotConfig.MARKER_SIZE),
                        markerfacecolor=pal[s],
                        markeredgecolor="none",
                        alpha=PlotConfig.ALPHA,
                    )
                    handles.append(h)
                    labels.append(s)
                leg = ax_main.legend(
                    handles,
                    labels,
                    title="Shift",
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.42),
                    ncol=min(4, len(labels)),
                    fontsize=9,
                    title_fontsize=11,
                    borderaxespad=0.0,
                )
                leg.set_in_layout(False)

            return ax_main

        # left col (PCA)
        title_left = "PCA" if i == 0 else None
        ax_main_left = joint_panel(
            outer[i, 0],
            "PCA 1",
            "PCA 2",
            title_if_top=title_left,
            draw_legend=(i == 0),
            show_row_label=True,
        )
        if i == 0:
            top_row_main_axes[0] = ax_main_left

        # right col (t-SNE)
        title_right = "t-SNE" if i == 0 else None
        ax_main_right = joint_panel(
            outer[i, 1],
            "t-SNE 1",
            "t-SNE 2",
            title_if_top=title_right,
            draw_legend=(i == 0),
            show_row_label=False,
        )
        if i == 0:
            top_row_main_axes[1] = ax_main_right

    plt.tight_layout()

    filename = f"{dataset}-{encoder_to_evaluate}-shift_comparison_jointplot.png"
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {filename}")


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
    fig, axes = plt.subplots(
        n_encoders, n_layers, figsize=(4 * n_layers, 3 * n_encoders + 2)
    )

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


def plot_kl_linegraph(
    output_dir: Path, dataset: str, encoder_to_evaluate: str, kl_divs: dict[str, float]
) -> None:
    """
    Plot a line graph of KL divergences across shifts.

    Args:
        dataset: Name of the dataset to plot results for.
        encoder_to_evaluate: Name of the encoder to plot results for.
        kl_divs: Mapping {shift_name: kl_divergence_value}.
    """

    if not kl_divs:
        raise ValueError("kl_divs is empty; nothing to plot.")

    # Preserve insertion order
    items = list(kl_divs.items())
    labels = [k for k, _ in items]
    values = [
        (
            float(v)
            if v is not None and not (isinstance(v, float) and math.isnan(v))
            else float("nan")
        )
        for _, v in items
    ]

    # Figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(values)), values, marker="o", linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"KL Divergence by Shift — {dataset} | {encoder_to_evaluate}")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Tidy layout
    fig.tight_layout()

    file_name = f"{dataset}-{encoder_to_evaluate}-kl_linegraph.png"
    filepath = output_dir / file_name
    fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {file_name}\n")
