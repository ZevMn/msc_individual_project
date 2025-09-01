"""
experiments/embeddings/plotting_utils.py

Plotting functions to visualise and analyse results from the experiments
found in 'experiments.py'.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional
from matplotlib.axes import Axes
import matplotlib.lines as mlines

from experiments.embeddings.config import Config, PlotConfig
from experiments.embeddings.statistical_utils import calculate_separability_score


# --------------------------------------------
# Helper function: Set consistent plot styling
# --------------------------------------------
def set_plot_style() -> None:
    """
    Apply a consistent seaborn/matplotlib style to all subsequent plots.
    """
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams.update({"font.family": "serif"})


# ---------------------------------------
# Plot feature embeddings representations
# ---------------------------------------
def plot_layer_representations_jointplot(
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
    Plots PCA and t-SNE projections for all layers' embeddings using jointplot-style
    visualizations (scatter + filled KDE marginals), creating separate figures for
    PCA and t-SNE. Each figure has layers as rows and label types as columns.

    Args:
        output_dir: Directory where the plot PNGs will be saved.
        dataset: Dataset name for configuration and filename.
        encoder_to_evaluate: Name of encoder being evaluated.
        layer_to_results_dict: Dict mapping layer names to DataFrames with embeddings.
        labels: Dict mapping column name to a NumPy array (of same length as 'layer_embeddings[layer]')
            with categorical labels to colour points.
        shift: String identifier for the simulated shift (or "no_shift" for reference data).
        seed: Random seed for reproducibility of sampling points for plotting.
        n_samples: Maximum number of points to include in the plot.


    Saves: f"{dataset}_{encoder_to_evaluate}_pca_jointplot_{shift}.png"
           f"{dataset}_{encoder_to_evaluate}_tsne_jointplot_{shift}.png"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    layers = list(layer_to_results_dict.keys())
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

    main_title = f"{dataset.replace('Mammo', 'EMBED')} | {encoder_to_evaluate.replace('_', ' ').title()}"

    def create_jointplot_grid(x_col: str, y_col: str, method_name: str):
        """Helper function to create a jointplot-style grid for either PCA or t-SNE"""

        fig = plt.figure(
            figsize=(6 * n_cols, 4 * n_rows + 2),
            constrained_layout=False,
        )

        # Add main title
        fig.suptitle(
            f"{method_name} Feature Representation: {main_title}",
            fontsize=28,
            fontweight="bold",
            y=0.97,
        )

        # Create outer grid
        outer = fig.add_gridspec(
            nrows=n_rows,
            ncols=n_cols,
            left=0.08,
            right=0.98,
            top=0.89,
            bottom=0.28,
            wspace=0.25,
            hspace=0.35,
        )

        # Plot each layer-column combination
        for row_idx, layer in enumerate(layers):
            sample = embeddings_data[layer]

            for col_idx, column in enumerate(columns):
                # Create subgrid for jointplot structure
                cell_spec = outer[row_idx, col_idx]
                sub = cell_spec.subgridspec(
                    2,
                    2,
                    height_ratios=(1, 4),
                    width_ratios=(4, 1),
                    wspace=0.0,
                    hspace=0.0,
                )

                # Create axes
                ax_top = fig.add_subplot(sub[0, 0])  # top marginal
                ax_main = fig.add_subplot(sub[1, 0])  # main scatter
                ax_right = fig.add_subplot(sub[1, 1])  # right marginal
                fig.add_subplot(sub[0, 1]).axis("off")  # empty corner

                # Get unique categories and create palette
                unique_cats = sorted(sample[column].unique())
                palette = sns.color_palette(
                    PlotConfig.COLOR_PALETTE, n_colors=len(unique_cats)
                )
                pal = {k: v for k, v in zip(unique_cats, palette)}

                # Main scatter plot
                sns.scatterplot(
                    data=sample,
                    x=x_col,
                    y=y_col,
                    hue=column,
                    palette=pal,
                    alpha=PlotConfig.ALPHA,
                    s=PlotConfig.MARKER_SIZE,
                    edgecolor="white",
                    linewidth=0.35,
                    ax=ax_main,
                    legend=False,
                )

                # KDE marginals for each category
                for cat in unique_cats:
                    cat_data = sample[sample[column] == cat]

                    # Top marginal (x-axis)
                    sns.kdeplot(
                        data=cat_data,
                        x=x_col,
                        ax=ax_top,
                        fill=True,
                        common_norm=False,
                        alpha=0.4,
                        linewidth=1.25,
                        color=pal[cat],
                    )

                    # Right marginal (y-axis)
                    sns.kdeplot(
                        data=cat_data,
                        y=y_col,
                        ax=ax_right,
                        fill=True,
                        common_norm=False,
                        alpha=0.4,
                        linewidth=1.25,
                        color=pal[cat],
                    )

                # Clean up marginal axes
                for ax in (ax_top, ax_right):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.grid(False)
                    sns.despine(ax=ax, left=True, bottom=True)

                # Set main axis labels
                ax_main.set_xlabel("")

                # Layer label on leftmost column
                if col_idx == 0:
                    ax_main.set_ylabel(
                        f"{layer.replace('_', ' ').title()}",
                        fontsize=20,
                        fontweight="bold",
                        labelpad=12,
                    )
                else:
                    ax_main.set_ylabel("")

                # Column title on top row
                if row_idx == 0:
                    ax_main.set_title(
                        f"{column}", fontsize=22, fontweight="bold", y=1.4
                    )

                if row_idx == len(layers) - 1:
                    # Add legend for top row
                    handles, legend_labels = [], []
                    for cat in unique_cats:
                        h = mlines.Line2D(
                            [],
                            [],
                            marker="o",
                            linestyle="",
                            markerfacecolor=pal[cat],
                            markeredgecolor="white",
                            markeredgewidth=0.5,
                            markersize=8,
                            alpha=PlotConfig.ALPHA,
                        )
                        handles.append(h)
                        legend_labels.append(str(cat))

                    leg = ax_main.legend(
                        handles,
                        legend_labels,
                        title=column,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.3),
                        ncol=min(4, len(legend_labels)),
                        fontsize=18,
                        title_fontsize=20,
                        frameon=False,
                        framealpha=0.95,
                        facecolor="white",
                        edgecolor="gray",
                        borderaxespad=0.0,
                        columnspacing=1.0,
                        handlelength=1.5,
                        handletextpad=0.5,
                    )

                sns.despine(ax=ax_main)

        plt.tight_layout()
        return fig

    # Create PCA figure
    print("\nCreating PCA jointplot visualization...")
    fig_pca = create_jointplot_grid("PCA 1", "PCA 2", "PCA")

    # Save PCA figure
    pca_filename = f"{dataset}_{encoder_to_evaluate}_pca_jointplot_{shift}.png"
    fig_pca.savefig(
        output_dir / pca_filename, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_pca)
    print(f"[Saved] {pca_filename}\n")

    # Create t-SNE figure
    print("Creating t-SNE jointplot visualization...")
    fig_tsne = create_jointplot_grid("t-SNE 1", "t-SNE 2", "t-SNE")

    # Save t-SNE figure
    tsne_filename = f"{dataset}_{encoder_to_evaluate}_tsne_jointplot_{shift}.png"
    fig_tsne.savefig(
        output_dir / tsne_filename, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_tsne)
    print(f"[Saved] {tsne_filename}\n")


# -----------------------------------------
# Plot shift representations for comparison
# -----------------------------------------
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
    shown above each column.

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
        f"Shift Comparison: {dataset.replace('Mammo', 'EMBED')} | {encoder_to_evaluate.replace('_', ' ').title()}",
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )

    outer = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.28,
        wspace=0.30,
        hspace=0.40,
    )

    top_row_main_axes: List[Optional[Axes]] = [None, None]

    for i, layer in enumerate(layers):

        layer_df = layer_to_results_dict[layer]

        # Filter for extreme shifts only (and no shift)
        extreme_shifts = [
            shift
            for shift in layer_df["Shift"].unique()
            if "extreme" in shift or shift == "no_shift"
        ]
        filtered_shifts_df = layer_df[layer_df["Shift"].isin(extreme_shifts)].copy()

        sample = filtered_shifts_df.sample(
            n=min(n_samples, len(filtered_shifts_df)), random_state=42
        )

        unique_shifts = sorted(sample["Shift"].unique())
        palette = sns.color_palette(
            PlotConfig.COLOR_PALETTE, n_colors=len(unique_shifts)
        )
        pal = {k: v for k, v in zip(unique_shifts, palette)}

        # Format legend shift labels
        display_labels = [
            shift.replace("_extreme", "").replace("extreme_", "").replace("_", " ")
            for shift in unique_shifts
        ]
        if dataset == "PadChest":
            display_labels = [
                shift.replace("sample", "acq") for shift in display_labels
            ]
        if dataset == "Retina":
            display_labels = [
                shift.replace("subpop", "prev") for shift in display_labels
            ]
        display_labels = [
            shift.replace("acq", "acquisition").replace("prev", "prevalence")
            for shift in display_labels
        ]

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
                labelpad=10,
            )

            if title_if_top:
                ax_main.set_title(title_if_top, fontsize=18, fontweight="bold", y=1.35)

            sns.despine(ax=ax_main)

            if draw_legend:
                handles, labels = [], []
                for i, s in enumerate(unique_shifts):
                    h = mlines.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markerfacecolor=pal[s],
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        alpha=1.0,
                        markersize=8,
                    )
                    handles.append(h)
                    labels.append(display_labels[i])
                leg = ax_main.legend(
                    handles,
                    labels,
                    title="Shift",
                    frameon=False,
                    framealpha=0.95,
                    facecolor="white",
                    edgecolor="gray",
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=min(4, len(labels)),
                    fontsize=12,
                    title_fontsize=14,
                    borderaxespad=0.0,
                )

            return ax_main

        # left col (PCA)
        title_left = "PCA" if i == 0 else None
        ax_main_left = joint_panel(
            outer[i, 0],
            "PCA 1",
            "PCA 2",
            title_if_top=title_left,
            draw_legend=(i == len(layers) - 1),
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
            draw_legend=(i == len(layers) - 1),
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


# ---------------------------------------------------------------
# Plots to display results from initial detection rate experiment
# ---------------------------------------------------------------
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

    Saves: f"{dataset}_{encoder_to_evaluate}_detection_rate_heatmap"
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

    # Clean up labels
    ax.set_ylabel("Shift", fontsize=14)
    clean_idx = [idx.replace("_", " ").title() for idx in pvals.index]
    ax.set_xticklabels(pvals.columns, rotation=45, ha="right")
    ax.set_yticklabels(clean_idx, rotation=0)

    fig.suptitle(
        f"Shift Detection Rate Heatmap: {dataset.replace('Mammo', 'EMBED')} | {encoder_to_evaluate.replace('_', ' ').title()}",
        fontsize=16,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{dataset}_{encoder_to_evaluate}_detection_rate_heatmap"
    fig.savefig(output_dir / file_name, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {file_name}\n")


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

    Saves: f"{dataset}_{encoder_to_evaluate}_detection_rate_line_graph"
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

    fig.suptitle(
        f"Shift Detection Rate Line Graph: {dataset.replace('Mammo', 'EMBED')} | {encoder_to_evaluate.replace('_', ' ').title()}",
        fontsize=16,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{dataset}_{encoder_to_evaluate}_detection_rate_line_graph"
    fig.savefig(output_dir / file_name, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {file_name}\n")


# -----------------------------------------------------------------
# Plots to display results from bootstrap detection rate experiment
# -----------------------------------------------------------------
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
        dataset_subset = all_results_df[all_results_df["dataset"] == dataset]

        plot_bootstrap_detection_rate_barchart(
            output_dir=output_dir,
            dataset=dataset,
            all_results_df=dataset_subset,
        )
        plot_bootstrap_detection_rate_heatmap(
            output_dir=output_dir,
            dataset=dataset,
            all_results_df=dataset_subset,
        )
        plot_bootstrap_detection_rate_heatmap_combined(
            output_dir=output_dir,
            dataset=dataset,
            all_results_df=dataset_subset,
        )


def plot_bootstrap_detection_rate_barchart(
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

    Saves: f"bootstrap_detection_rates_{dataset}.png"
    """

    # Check if required columns are present
    required = {"encoder", "layer", "shift", "test_size", "detection_rate"}
    missing = required - set(all_results_df.columns)
    if missing:
        raise ValueError(f"Missing columns in all_results_df: {missing}")

    # Get unique values for organizing the plot
    encoders = sorted(all_results_df["encoder"].unique())
    layers = ["after_maxpool", "layer_1", "layer_2", "layer_3", "final_layer"]
    test_sizes = sorted(all_results_df["test_size"].unique())

    all_results_df["shift"] = all_results_df["shift"].str.lower().str.strip()
    shift_order = ["subtle", "moderate", "extreme"]

    def sort_key(s: str):
        parts = s.split("_")
        if len(parts) == 2:
            prefix, suffix = parts
            return (prefix, shift_order.index(suffix))
        else:
            return (s, 999)

    shifts = sorted(all_results_df["shift"].unique(), key=sort_key)

    # Create figure and subplots
    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(test_sizes)))
    n_encoders = len(encoders)
    n_layers = len(layers)
    fig, axes = plt.subplots(
        n_encoders,
        n_layers,
        figsize=(4 * n_layers, 3 * n_encoders + 2),
        gridspec_kw={"hspace": 0.4, "wspace": 0.3},
    )
    plt.subplots_adjust(left=0.15, bottom=0.15)

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
                        detection_rates.append(
                            shift_data["detection_rate"].iloc[0] * 100
                        )
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
                    if rate == 100:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            "*",
                            ha="center",
                            va="bottom",
                            fontsize=6.5,
                            rotation=0,
                        )
                    if (
                        rate > 5 and rate < 100
                    ):  # Only label if detection rate is substantial
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{rate:.0f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            rotation=0,
                        )

            if len(shifts) > 0:
                ax.set_xlim(-0.5, len(shifts) - 0.5)
            else:
                ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 110)
            ax.set_xticks(shift_positions)
            if encoder_idx == n_encoders - 1:
                ax.set_xticklabels(
                    [
                        shift.replace("_", " ").replace("sample", "acq").title()
                        for shift in shifts
                    ],
                    rotation=30,
                    ha="right",
                )
                ax.set_xlabel("Shift", fontsize=12, labelpad=15)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(
                "Detection Rate (%)" if layer_idx == 0 else "", labelpad=15, fontsize=12
            )

            # Add layer title
            if encoder_idx == 0:
                ax.set_title(
                    layer.replace("_", " ").title(),
                    fontsize=16,
                    fontweight="bold",
                    pad=10,
                )

            # Add encoder label on the left
            if layer_idx == 0:
                ax.text(
                    -0.55,
                    0.5,
                    encoder.replace("_", " ").title(),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                )

    fig.suptitle(
        f"{dataset.replace('Mammo', 'EMBED')} - Bootstrap Detection Rates",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )

    # Add shared legend
    if len(test_sizes) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(test_sizes),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=14,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"bootstrap_detection_rates_{dataset}.png"
    filepath = output_dir / file_name
    fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {file_name}\n")


def plot_bootstrap_detection_rate_heatmap(
    output_dir: Path,
    dataset: str,
    all_results_df: pd.DataFrame,
) -> None:
    """
    Generate heatmaps of bootstrap detection rates for each encoder/test_size combination.
    Shows detection rates across shifts (rows) and layers (columns).

    Args:
        output_dir: Directory to save the heatmap plots
        dataset: Name of the dataset to plot results for
        all_results_df: DataFrame containing bootstrap results with columns:
                   ['dataset', 'encoder', 'shift', 'layer', 'test_size', 'detection_rate', ...]

    Saves: f"bootstrap_heatmap_{dataset}_{encoder}_{test_size}samples.png"
    """

    # Check if required columns are present
    required = {"encoder", "layer", "shift", "test_size", "detection_rate"}
    missing = required - set(all_results_df.columns)
    if missing:
        raise ValueError(f"Missing columns in all_results_df: {missing}")

    # Filter for the specific dataset
    dataset_df = all_results_df[all_results_df["dataset"] == dataset].copy()
    if dataset_df.empty:
        print(f"No data found for dataset: {dataset}")
        return

    encoders = sorted(dataset_df["encoder"].unique())
    layers = ["after_maxpool", "layer_1", "layer_2", "layer_3", "final_layer"]
    shifts = sorted(dataset_df["shift"].unique())
    test_sizes = sorted(dataset_df["test_size"].unique())

    # Create a separate heatmap for each encoder/test_size combination
    for encoder in encoders:
        for test_size in test_sizes:

            subset = dataset_df[
                (dataset_df["encoder"] == encoder)
                & (dataset_df["test_size"] == test_size)
            ]
            if subset.empty:
                continue

            # Pivot table: shifts (rows) × layers (columns)
            heatmap_data = subset.pivot_table(
                index="shift", columns="layer", values="detection_rate", fill_value=0
            )

            # Check all layers present in the correct order
            available_layers = [
                layer for layer in layers if layer in heatmap_data.columns
            ]
            heatmap_data = heatmap_data.reindex(columns=available_layers)

            # Check all shifts present
            heatmap_data = heatmap_data.reindex(index=shifts, fill_value=0)

            # Create annotation array with detection rates
            annot_arr = heatmap_data.values.copy()
            annot_str = np.empty_like(annot_arr, dtype=object)

            # Show values >= 0.05 as annotations (empty string otherwise)
            for i in range(annot_arr.shape[0]):
                for j in range(annot_arr.shape[1]):
                    val = annot_arr[i, j]
                    if val >= 0.05:
                        annot_str[i, j] = f"{val:.2f}"
                    else:
                        annot_str[i, j] = ""

            # Create the heatmap
            fig, ax = plt.subplots(
                figsize=(len(available_layers) * 1.5 + 2, len(shifts) * 0.6 + 2)
            )
            sns.heatmap(
                heatmap_data,
                annot=annot_str,
                fmt="",
                cmap="Blues",
                linewidths=0.5,
                cbar_kws={"label": "Detection Rate"},
                ax=ax,
                vmin=0,
                vmax=1,
                square=False,
            )

            ax.set_ylabel("Shift", fontsize=12)
            ax.set_xlabel("Layer", fontsize=12)

            clean_layer_labels = [
                layer.replace("_", " ").title() for layer in available_layers
            ]
            ax.set_xticklabels(clean_layer_labels, rotation=45, ha="right")
            clean_shift_labels = [str(shift) for shift in heatmap_data.index]
            ax.set_yticklabels(clean_shift_labels, rotation=0)

            title = f"Bootstrap Detection Rates: {dataset.replace('Mammo', 'EMBED')}\n{encoder.replace('_', ' ').title()} | {test_size} samples"
            ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"bootstrap_heatmap_{dataset}_{encoder}_{test_size}samples.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"[Saved] {filename}")


def plot_bootstrap_detection_rate_heatmap_combined(
    output_dir: Path,
    dataset: str,
    all_results_df: pd.DataFrame,
) -> None:
    """
    Generate a combined heatmap showing bootstrap detection rates for all test sizes.
    Creates subplots for each encoder, with test sizes as separate heatmaps.

    Args:
        output_dir: Directory to save the heatmap plots
        dataset: Name of the dataset to plot results for
        all_results_df: DataFrame containing bootstrap results

    Saves: f"bootstrap_heatmap_combined_{dataset}.png"
    """

    # Filter for the specific dataset
    dataset_df = all_results_df[all_results_df["dataset"] == dataset].copy()
    if dataset_df.empty:
        print(f"No data found for dataset: {dataset}")
        return

    # Get unique values
    encoders = sorted(dataset_df["encoder"].unique())
    layers = ["after_maxpool", "layer_1", "layer_2", "layer_3", "final_layer"]
    shifts = sorted(dataset_df["shift"].unique())
    test_sizes = sorted(dataset_df["test_size"].unique())

    # Create a figure with subplots for each encoder
    n_encoders = len(encoders)
    n_test_sizes = len(test_sizes)

    fig, axes = plt.subplots(
        n_encoders,
        n_test_sizes,
        figsize=(n_test_sizes * 6, n_encoders * 4),
        squeeze=False,
    )

    # Create a separate heatmap for each encoder/test_size combination
    for encoder_idx, encoder in enumerate(encoders):
        for size_idx, test_size in enumerate(test_sizes):
            ax = axes[encoder_idx, size_idx]

            # Pivot table: shifts (rows) × layers (columns)
            subset = dataset_df[
                (dataset_df["encoder"] == encoder)
                & (dataset_df["test_size"] == test_size)
            ]
            if subset.empty:
                ax.set_visible(False)
                continue

            heatmap_data = subset.pivot_table(
                index="shift", columns="layer", values="detection_rate", fill_value=0
            )

            # Ensure all layers are present in the correct order
            available_layers = [
                layer for layer in layers if layer in heatmap_data.columns
            ]
            heatmap_data = heatmap_data.reindex(columns=available_layers)
            heatmap_data = heatmap_data.reindex(index=shifts, fill_value=0)

            # Create annotation array
            annot_str = np.empty_like(heatmap_data.values, dtype=object)
            for i in range(heatmap_data.shape[0]):
                for j in range(heatmap_data.shape[1]):
                    val = heatmap_data.iloc[i, j]
                    annot_str[i, j] = f"{val:.2f}" if val >= 0.01 else ""

            # Create the heatmap
            sns.heatmap(
                heatmap_data,
                annot=annot_str,
                fmt="",
                cmap="Blues",
                linewidths=0.5,
                cbar=size_idx
                == n_test_sizes - 1,  # Only show colorbar on rightmost plot
                cbar_kws=(
                    {"label": "Detection Rate"}
                    if size_idx == n_test_sizes - 1
                    else None
                ),
                ax=ax,
                vmin=0,
                vmax=1,
            )

            # Labels and formatting
            if encoder_idx == n_encoders - 1:  # Bottom row
                clean_layer_labels = [
                    layer.replace("_", " ").title() for layer in available_layers
                ]
                ax.set_xticklabels(clean_layer_labels, rotation=45, ha="right")
                ax.set_xlabel("Layer", fontsize=10)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")

            if size_idx == 0:  # Leftmost column
                clean_shift_labels = [str(shift) for shift in heatmap_data.index]
                ax.set_yticklabels(clean_shift_labels, rotation=0)
                ax.set_ylabel("Shift", fontsize=10)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            title = f"{encoder.replace('_', ' ').title()}\n{test_size} samples"
            ax.set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Bootstrap Detection Rate Heatmaps: {dataset.replace('Mammo', 'EMBED')}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"bootstrap_heatmap_combined_{dataset}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {filename}")


def plot_kl_panels_multi_layer(input_dir: Path, output_dir: Path) -> None:
    """
    Create panel plots showing KL divergence across all layers for each dataset.
    Each dataset gets a 3x2 grid with 5 layer plots and separability heatmap in the 6th position.

    f"kl_divergence_layers_{dataset}_improved.png"
    """

    # Set style for better appearance
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
        }
    )

    # Data loading
    csv_files = list(input_dir.glob("kl_divergence_new-*-*-*.csv"))
    if not csv_files:
        raise ValueError(
            f"No CSV files found in {input_dir} with pattern 'kl_divergence_new-*-*-*.csv'"
        )

    data_records = []
    for csv_file in csv_files:
        name_parts = csv_file.stem.split("-")
        if len(name_parts) != 4:
            continue

        dataset = name_parts[1]
        encoder = name_parts[2]
        layer = name_parts[3]

        try:
            df_temp = pd.read_csv(csv_file)
            for _, row in df_temp.iterrows():
                data_records.append(
                    {
                        "dataset": dataset,
                        "encoder": encoder,
                        "layer": layer,
                        "shift_name": row["shift_name"],
                        "kl_divergence": (
                            float(row["kl_divergence"])
                            if pd.notna(row["kl_divergence"])
                            else float("nan")
                        ),
                    }
                )
        except Exception as e:
            print(f"Warning: Error reading {csv_file.name}: {e}")
            continue

    if not data_records:
        raise ValueError("No valid data found in CSV files")

    df = pd.DataFrame(data_records)
    datasets = sorted(df["dataset"].unique())
    encoders = sorted(df["encoder"].unique())
    layers = ["after_maxpool", "layer_1", "layer_2", "layer_3", "final_layer"]

    # Improved colorblind-friendly color palette
    colors = ["#E31A1C", "#1F78B4", "#33A02C", "#FF7F00", "#6A3D9A"]
    encoder_colors = dict(zip(encoders, colors[: len(encoders)]))

    layer_display_names = {
        "after_maxpool": "After MaxPool",
        "layer_1": "Layer 1",
        "layer_2": "Layer 2",
        "layer_3": "Layer 3",
        "final_layer": "Final Layer",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    def group_aware_sort_key(shift_name):
        """Sort by prefix first, then by number within each prefix"""
        match = re.match(r"([a-zA-Z]+)_?(\d*)", str(shift_name))
        if match:
            prefix = match.group(1)
            num_str = match.group(2)
            num = int(num_str) if num_str else 0
            return (prefix, num)
        else:
            return (str(shift_name), 0)

    def create_readable_labels(shift_names):
        """Create more readable x-axis labels"""
        readable_labels = []
        for shift in shift_names:
            label = shift.replace("_", " ").title()
            readable_labels.append(label)
        return readable_labels

    # Create panel plot for each dataset
    for dataset in datasets:
        # Use 2x3 layout instead to give more space for each subplot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        dataset_data = df[df["dataset"] == dataset]

        # Get all shifts for this dataset
        all_shifts = sorted(
            dataset_data["shift_name"].unique(), key=group_aware_sort_key
        )
        readable_labels = create_readable_labels(all_shifts)
        x_positions = list(range(len(all_shifts)))

        # Plot each layer
        for layer_idx, layer in enumerate(layers):
            row = layer_idx // 3
            col = layer_idx % 3
            ax = axes[row, col]

            layer_data = dataset_data[dataset_data["layer"] == layer]

            # Plot each encoder for this layer
            for encoder in encoders:
                encoder_data = layer_data[layer_data["encoder"] == encoder]
                if encoder_data.empty:
                    continue

                values = []
                errors_lower = []
                errors_upper = []
                actual_x_positions = []

                for shift_idx, shift in enumerate(all_shifts):
                    shift_data = encoder_data[encoder_data["shift_name"] == shift]

                    if not shift_data.empty:
                        if "kl_divergence" in shift_data.columns:
                            values.append(shift_data["kl_divergence"].iloc[0])
                            if "ci_lower" in shift_data.columns:
                                errors_lower.append(
                                    shift_data["kl_divergence"].iloc[0]
                                    - shift_data["ci_lower"].iloc[0]
                                )
                                errors_upper.append(
                                    shift_data["ci_upper"].iloc[0]
                                    - shift_data["kl_divergence"].iloc[0]
                                )

                        actual_x_positions.append(shift_idx)

                if not values:
                    continue

                if errors_lower and errors_upper:
                    ax.errorbar(
                        actual_x_positions,
                        values,
                        yerr=[errors_lower, errors_upper],
                        fmt="none",
                        ecolor=encoder_colors[encoder],
                        alpha=0.3,
                        capsize=3,
                    )

                # Enhanced scatter points
                ax.scatter(
                    actual_x_positions,
                    values,
                    s=100,
                    color=encoder_colors[encoder],
                    label=encoder,
                    edgecolor="white",
                    linewidth=2,
                    zorder=3,
                    alpha=0.9,
                )

                # Group and connect points
                groups = {}
                for pos, val, shift in zip(
                    actual_x_positions,
                    values,
                    [all_shifts[i] for i in actual_x_positions],
                ):
                    match = re.match(r"([a-zA-Z]+)_?(\d*)", shift)
                    if match:
                        prefix = match.groups()[0]
                        num_str = match.groups()[1]
                        num = int(num_str) if num_str else 0
                        groups.setdefault(prefix, []).append((pos, val, num, shift))

                for prefix, points in groups.items():
                    if len(points) > 1:
                        points.sort(key=lambda x: x[2])
                        x_coords, y_coords = zip(*[(p[0], p[1]) for p in points])
                        ax.plot(
                            x_coords,
                            y_coords,
                            color=encoder_colors[encoder],
                            linewidth=3,
                            alpha=0.7,
                            zorder=2,
                        )

            # Enhanced subplot styling
            ax.set_xticks(x_positions)
            # Rotate labels more and use smaller font
            ax.set_xticklabels(readable_labels, rotation=60, ha="right", fontsize=8)
            ax.set_ylabel("KL Divergence", fontweight="bold", labelpad=15, fontsize=16)
            ax.set_title(
                layer_display_names[layer],
                fontsize=18,
                fontweight="bold",
                pad=20,
                color="#2E2E2E",
            )

            # Enhanced grid
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=1)
            ax.set_axisbelow(True)

            # Individual scaling for each layer for better clarity
            layer_values = layer_data["kl_divergence"].dropna()
            if len(layer_values) > 0:
                layer_y_min = max(0, layer_values.min() * 0.3)
                layer_y_max = layer_values.max() * 1.05
                ax.set_ylim(layer_y_min, layer_y_max)

            # Scientific notation when appropriate (based on individual layer values)
            if len(layer_values) > 0 and (
                layer_values.max() < 0.01 or layer_values.max() > 10000
            ):
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

            # Enhanced appearance
            ax.set_facecolor("#fdfdfd")
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color("#dddddd")

        # Create separability heatmap in the 6th position (bottom right)
        ax_heatmap = axes[1, 2]

        # Calculate separability scores for all encoder-layer combinations
        separability_matrix = np.zeros((len(layers), len(encoders)))
        for layer_idx, layer in enumerate(layers):
            for encoder_idx, encoder in enumerate(encoders):
                score = calculate_separability_score(dataset_data, encoder, layer)
                separability_matrix[layer_idx, encoder_idx] = score

        # Create heatmap
        im = ax_heatmap.imshow(separability_matrix, cmap="RdYlBu_r", aspect="auto")

        # Set ticks and labels
        ax_heatmap.set_xticks(range(len(encoders)))
        ax_heatmap.set_yticks(range(len(layers)))
        ax_heatmap.set_xticklabels(
            [
                enc.replace("_", " ").replace("simclr", "SimCLR").title()
                for enc in encoders
            ],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax_heatmap.set_yticklabels(
            [layer_display_names[layer] for layer in layers], fontsize=9
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label(
            "Separability Score\n(Fisher Ratio)",
            rotation=270,
            labelpad=30,
            fontsize=14,
            fontweight="bold",
        )

        # Add text annotations with separability scores
        for layer_idx in range(len(layers)):
            for encoder_idx in range(len(encoders)):
                score = separability_matrix[layer_idx, encoder_idx]
                if not np.isnan(score):
                    if score == float("inf"):
                        text = "∞"
                    else:
                        text = f"{score:.2f}"
                    color = (
                        "white" if score > np.nanmean(separability_matrix) else "black"
                    )
                    ax_heatmap.text(
                        encoder_idx,
                        layer_idx,
                        text,
                        ha="center",
                        va="center",
                        color=color,
                        fontweight="bold",
                        fontsize=8,
                    )

        ax_heatmap.set_title(
            "Shift Type Separability",
            fontsize=16,
            fontweight="bold",
            pad=20,
            color="#2E2E2E",
        )

        # Enhanced appearance for heatmap
        ax_heatmap.set_facecolor("#fdfdfd")
        for spine in ax_heatmap.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#dddddd")

        # Create horizontal legend at the bottom
        legend_handles = []
        for encoder in encoders:
            display_name = encoder.replace("_", " ").replace("simclr", "SimCLR").title()
            handle = mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=encoder_colors[encoder],
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label=display_name,
                linestyle="-",
                linewidth=3,
                alpha=0.9,
            )
            legend_handles.append(handle)

        # Position legend at bottom of figure
        fig.legend(
            handles=legend_handles,
            title="Encoder Models",
            loc="lower center",
            ncol=len(encoders),
            bbox_to_anchor=(0.5, 0.02),
            fontsize=12,
            title_fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

        fig.suptitle(
            f"KL Divergence Across Encoder Layers: {dataset.replace('Mammo', 'EMBED')}",
            fontsize=20,
            fontweight="bold",
            y=0.95,
            color="#2E2E2E",
        )

        fig.text(
            0.5,
            0.91,
            "Dataset shift sensitivity by encoder and layer depth",
            ha="center",
            fontsize=13,
            style="italic",
            color="#666666",
        )

        # Improved layout with space for bottom legend
        plt.tight_layout(rect=(0, 0.08, 1, 0.88))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Save with high quality
        filepath = output_dir / f"kl_divergence_layers_{dataset}_improved.png"
        fig.savefig(
            filepath,
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.3,
        )
        plt.close(fig)

        print(f"[Saved] kl_divergence_layers_{dataset}_improved.png")
