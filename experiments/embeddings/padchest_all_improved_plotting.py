import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import re
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

from experiments.embeddings.config import Config


def plot_shift_identification_results(
    input_dir: Path, output_dir: Path, dataset: str = "padchest"
) -> None:
    """
    Plot shift identification results as stacked bar charts organised by shift type and strength.
    """

    shift_result_order = [
        "Prevalence shift",
        "Covariate (gender) only",
        "Covariate (acquisition) only",
        "Covariate (gender) + Prevalence",
        "Covariate (acquisition) + Prevalence",
        "No shift",
    ]

    colours = {
        "Prevalence shift": "#FF746C",
        "Covariate (gender) only": "#779ECB",
        "Covariate (acquisition) only": "#3D6A9F",
        "Covariate (gender) + Prevalence": "#9e84A1",
        "Covariate (acquisition) + Prevalence": "#755B78",
        "No shift": "#CCCCCC",
    }

    print(f"Starting plot creation for dataset: {dataset}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    file_data = parse_csv_files(input_dir, dataset)

    if not file_data:
        print(f"No CSV files found for dataset '{dataset}' in {input_dir}")
        return

    print(f"Found {len(file_data)} valid files")

    organised_data = organise_data_by_shift(file_data)
    print(f"Organised into {len(organised_data)} shift types:")
    for shift_type, strengths in organised_data.items():
        print(f"  {shift_type}: {list(strengths.keys())}")

    create_subplot_grid(organised_data, shift_result_order, colours, output_dir, dataset)


def parse_csv_files(input_dir: Path, dataset: str) -> Dict:
    """Parse all CSV files matching the expected pattern."""
    file_data = {}

    print(f"Looking for CSV files in: {input_dir}")
    print(f"Dataset pattern: new_{dataset}_*.csv")

    all_csv_files = list(input_dir.glob("*.csv"))
    print(f"Found {len(all_csv_files)} CSV files total:")
    for f in all_csv_files[:5]:
        print(f"  - {f.name}")
    if len(all_csv_files) > 5:
        print(f"  ... and {len(all_csv_files) - 5} more")

    patterns = [
        rf"new_{dataset}_acq_prev_(\d+\.?\d*)_n(\d+)_.*_d(\d+\.?\d*)_t\[(.*?)\]\.csv",
        rf"new_{dataset}_(\w+)_(\d+\.?\d*)_n(\d+)_.*_t\[(.*?)\]\.csv",
        rf"new_{dataset}_(\w+)_(\d+\.?\d*)_n(\d+)_.*\.csv",
        rf"{dataset}_(\w+)_(\d+\.?\d*)_n(\d+)_.*\.csv",
    ]

    matched_files = list(input_dir.glob(f"*{dataset}*.csv"))
    print(f"Found {len(matched_files)} files matching dataset pattern:")
    for f in matched_files:
        print(f"  - {f.name}")

    for csv_file in matched_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)
            print(f"  DataFrame shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

            required_cols = {"final_identified_shift", "n_test"}
            available_cols = set(df.columns)

            if not required_cols.issubset(available_cols):
                missing = required_cols - available_cols
                print(f"  Skipping: missing required columns {missing}")
                continue

            print(f"  Unique final_identified_shift values: {df['final_identified_shift'].unique()}")
            print(f"  Unique n_test values: {df['n_test'].unique()}")

        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
            continue

        matched = False
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, csv_file.name)
            if match:
                print(f"  Matched pattern: {pattern}")

                if i == 0:
                    shift_type = "acq_prev"
                    shift_strength = float(match.group(1))
                    n_bootstrap = int(match.group(2))
                    prevalence_strength = float(match.group(3))
                    
                    test_sizes = [100, 250, 500, 1000]
                    if len(match.groups()) >= 4 and match.group(4):
                        try:
                            test_sizes_str = match.group(4)
                            test_sizes = [int(x.strip()) for x in test_sizes_str.split(",")]
                        except:
                            pass
                    
                    file_data[csv_file.name] = {
                        "shift_type": shift_type,
                        "shift_strength": shift_strength,
                        "prevalence_strength": prevalence_strength,
                        "n_bootstrap": n_bootstrap,
                        "test_sizes": test_sizes,
                        "data": df,
                    }
                    
                else:
                    shift_type = match.group(1)
                    shift_strength = float(match.group(2))
                    n_bootstrap = int(match.group(3))

                    test_sizes = [100, 250, 500, 1000]
                    if len(match.groups()) >= 4 and match.group(4):
                        try:
                            test_sizes_str = match.group(4)
                            test_sizes = [int(x.strip()) for x in test_sizes_str.split(",")]
                        except:
                            pass

                    print(
                        f"  Parsed: shift_type={shift_type}, strength={shift_strength}, n_bootstrap={n_bootstrap}"
                    )
                    
                    file_data[csv_file.name] = {
                        "shift_type": shift_type,
                        "shift_strength": shift_strength,
                        "n_bootstrap": n_bootstrap,
                        "test_sizes": test_sizes,
                        "data": df,
                    }
                
                matched = True
                break

        if not matched:
            print(f"  No pattern matched for {csv_file.name}")
            continue

    print(f"\nTotal files processed successfully: {len(file_data)}")
    return file_data


def organise_data_by_shift(file_data: Dict) -> Dict:
    """Organise parsed data by shift type and strength."""
    organised = defaultdict(lambda: defaultdict(dict))

    for file_info in file_data.values():
        shift_type = file_info["shift_type"]
        shift_strength = file_info["shift_strength"]
        
        if shift_type == "acq_prev" and "prevalence_strength" in file_info:
            key = (shift_strength, file_info["prevalence_strength"])
        else:
            key = shift_strength
            
        organised[shift_type][key] = file_info

    return dict(organised)


def calculate_proportions(
    df: pd.DataFrame, test_size: int, shift_result_order: List[str]
) -> Dict[str, float]:
    """Calculate proportions of each shift identification result for a given test size."""
    subset = df[df["n_test"] == test_size]
    if len(subset) == 0:
        return {result: 0.0 for result in shift_result_order}

    result_counts = subset["final_identified_shift"].value_counts()
    total = len(subset)

    print(f"    Test size {test_size}: {total} samples")
    print(f"    Result counts: {dict(result_counts)}")

    proportions = {}
    for result in shift_result_order:
        proportions[result] = result_counts.get(result, 0) / total if total > 0 else 0.0

    print(f"    Proportions: {proportions}")
    return proportions


def create_subplot_grid(
    organised_data: Dict,
    shift_result_order: List[str],
    colours: Dict[str, str],
    output_dir: Path,
    dataset: str,
) -> None:
    """Create the main subplot grid plot."""

    desired_order = ["acq", "gender", "prev", "acq_prev", "gender_prev"]
    shift_types = [st for st in desired_order if st in organised_data]
    max_strengths = (
        max(len(strengths) for strengths in organised_data.values())
        if organised_data
        else 3
    )

    fig, axes = plt.subplots(
        len(shift_types),
        max_strengths,
        figsize=(4 * max_strengths, 3 * len(shift_types) + 1.8),
        gridspec_kw={"hspace": 0.5, "wspace": 0.3},
    )

    if len(shift_types) == 1:
        axes = axes.reshape(1, -1)
    elif max_strengths == 1:
        axes = axes.reshape(-1, 1)

    plt.subplots_adjust(left=0.15, bottom=0.14, top=0.85)

    test_sizes = [100, 250, 500, 1000]
    bar_width = 0.8 / len(test_sizes)

    for row_idx, shift_type in enumerate(shift_types):
        shift_strengths = sorted(organised_data[shift_type].keys())

        for col_idx in range(max_strengths):
            ax = axes[row_idx, col_idx] if len(shift_types) > 1 else axes[col_idx]

            if col_idx < len(shift_strengths):
                shift_strength = shift_strengths[col_idx]
                file_info = organised_data[shift_type][shift_strength]
                plot_single_subplot(
                    ax,
                    file_info,
                    test_sizes,
                    shift_result_order,
                    colours,
                    bar_width,
                    shift_type,
                    shift_strength,
                )
            else:
                ax.set_xlim(-0.5, len(test_sizes) - 0.5)
                ax.set_ylim(0, 1)
                ax.set_xticks(range(len(test_sizes)))
                ax.set_xticklabels([])
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    alpha=0.5,
                )

            add_subplot_labels(
                ax,
                shift_type,
                col_idx,
                len(shift_strengths),
                row_idx,
                len(shift_types),
                test_sizes,
            )

    fig.suptitle(
        f"{dataset.title()} Shift Identification Results: SimCLR ImageNet layer 2",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )

    fig.text(
        0.5,
        0.91,
        "Original proportions - Females: 51%, Phillips: 42%, Prevalence: 4%",
        ha="center",
        fontsize=12,
        style="italic",
        color="gray",
    )

    legend_elements = []
    legend_labels = []

    prevalence_items = [("Prevalence shift", "Prevalence Only")]

    covariate_items = [
        ("Covariate (gender) only", "Covariate (gender) Only"),
        ("Covariate (acquisition) only", "Covariate (acquisition) Only"),
    ]

    combined_items = [
        ("Covariate (gender) + Prevalence", "Covariate (gender) + Prevalence"),
        (
            "Covariate (acquisition) + Prevalence",
            "Covariate (acquisition) + Prevalence",
        ),
    ]

    other_items = [("No shift", "No Shift")]

    for items, group_name in [
        (prevalence_items, "Prevalence"),
        (covariate_items, "Covariate Only"),
        (combined_items, "Combined"),
        (other_items, "None"),
    ]:
        for result_key, display_name in items:
            legend_elements.append(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=colours[result_key],
                    edgecolor="black",
                    linewidth=0.5,
                )
            )
            legend_labels.append(display_name)

    fig.legend(
        legend_elements,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Identified Shift Type (Colour Groups: Red=Prevalence, Blue=Covariate, Purple=Combined)",
        title_fontsize=10,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"shift_identification_results_{dataset}_color_grouped.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {filename}")


def plot_single_subplot(
    ax,
    file_info: Dict,
    test_sizes: List[int],
    shift_result_order: List[str],
    colours: Dict[str, str],
    bar_width: float,
    shift_type: str,
    shift_strength: float | Tuple,
) -> None:
    """Plot a single subplot for a specific shift type and strength."""

    df = file_info["data"]
    available_test_sizes = sorted(df["n_test"].unique())
    plot_test_sizes = [ts for ts in test_sizes if ts in available_test_sizes]

    if not plot_test_sizes:
        ax.text(
            0.5,
            0.5,
            "No Data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.5,
        )
        return

    if shift_type == "acq":
        stacking_order = [
            "Covariate (acquisition) only",
            "Covariate (acquisition) + Prevalence",
            "Covariate (gender) only",
            "Covariate (gender) + Prevalence",
            "Prevalence shift",
            "No shift",
        ]
    elif shift_type == "gender":
        stacking_order = [
            "Covariate (gender) only",
            "Covariate (gender) + Prevalence",
            "Covariate (acquisition) only",
            "Covariate (acquisition) + Prevalence",
            "Prevalence shift",
            "No shift",
        ]
    elif shift_type == "prev":
        stacking_order = [
            "Prevalence shift",
            "Covariate (gender) + Prevalence",
            "Covariate (acquisition) + Prevalence",
            "Covariate (gender) only",
            "Covariate (acquisition) only",
            "No shift",
        ]
    elif shift_type == "gender_prev":
        stacking_order = [
            "Covariate (gender) + Prevalence",
            "Covariate (acquisition) + Prevalence",
            "Prevalence shift",
            "Covariate (gender) only",
            "Covariate (acquisition) only",
            "No shift",
        ]
    elif shift_type == "acq_prev":
        stacking_order = [
            "Covariate (acquisition) + Prevalence",
            "Covariate (gender) + Prevalence",
            "Prevalence shift",
            "Covariate (acquisition) only",
            "Covariate (gender) only",
            "No shift",
        ]
    else:
        stacking_order = [
            "Prevalence shift",
            "Covariate (gender) + Prevalence",
            "Covariate (acquisition) + Prevalence",
            "Covariate (gender) only",
            "Covariate (acquisition) only",
            "No shift",
        ]

    expected_shift = {
        "gender": "Covariate (gender) only",
        "acq": "Covariate (acquisition) only",
        "prev": "Prevalence shift",
        "gender_prev": "Covariate (gender) + Prevalence",
        "acq_prev": "Covariate (acquisition) + Prevalence",
    }

    positions = np.arange(len(plot_test_sizes))

    for pos_idx, test_size in enumerate(plot_test_sizes):
        proportions = calculate_proportions(df, test_size, shift_result_order)
        bottom = 0
        for result in stacking_order:
            percentage = proportions[result] * 100
            if percentage > 0:
                ax.bar(
                    pos_idx,
                    percentage,
                    bottom=bottom,
                    color=colours[result],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                    width=0.6,
                )
                if percentage > 5:
                    ax.text(
                        pos_idx,
                        bottom + percentage / 2,
                        f"{percentage:.0f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white" if result != "No shift" else "black",
                    )
                bottom += percentage

    ax.set_xlim(-0.5, len(plot_test_sizes) - 0.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(positions)

    if shift_type == "gender":
        title_text = f"{int(shift_strength * 100)}% Female"
    elif shift_type == "acq":
        title_text = f"{int(shift_strength * 100)}% Phillips"
    elif shift_type == "prev":
        title_text = f"{int(shift_strength * 100)}% Prevalence"
    elif shift_type == "gender_prev":
        title_text = f"{int(shift_strength * 100)}% Female\n+ 15% Prevalence"
    elif shift_type == "acq_prev":
        phillips_strength, prevalence_strength = shift_strength
        title_text = (
            f"{int(phillips_strength * 100)}% Phillips\n"
            f"+ {int(prevalence_strength * 100)}% Prevalence"
        )
    else:
        strength_str = (
            f"{shift_strength:.2f}"
            if shift_strength != int(shift_strength)
            else f"{int(shift_strength)}"
        )
        title_text = f"Strength: {strength_str}"

    ax.set_title(title_text, fontsize=10, fontweight="bold")


def add_subplot_labels(
    ax,
    shift_type: str,
    col_idx: int,
    n_cols: int,
    row_idx: int,
    n_rows: int,
    test_sizes: List[int],
) -> None:
    """Add axis labels and shift type label to subplot."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if row_idx == n_rows - 1:
        ax.set_xticklabels([str(ts) for ts in test_sizes], rotation=0)
        ax.set_xlabel("Test Size", fontsize=10, labelpad=12)
    else:
        ax.set_xticklabels([])

    if col_idx == 0:
        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.text(
            -0.3,
            0.5,
            shift_type.replace("_", " ").title(),
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.grid(True, alpha=0.3, axis="y")


def plot_confusion_matrix(
    file_data: Dict,
    output_dir: Path,
    dataset: str = "padchest",
    test_size: Optional[int] = None,
    normalize: str = "true",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
) -> None:
    """Plot confusion matrix for shift identification results."""

    shift_type_to_actual = {
        "acq": "Covariate (acquisition) only",
        "gender": "Covariate (gender) only",
        "prev": "Prevalence shift",
        "gender_prev": "Covariate (gender) + Prevalence",
        "acq_prev": "Covariate (acquisition) + Prevalence",
    }

    all_identified_shifts = [
        "Prevalence shift",
        "Covariate (gender) only",
        "Covariate (acquisition) only",
        "Covariate (gender) + Prevalence",
        "Covariate (acquisition) + Prevalence",
        "No shift",
    ]

    y_true, y_pred = [], []

    for file_info in file_data.values():
        df = file_info["data"]
        shift_type = file_info["shift_type"]
        true_shift = shift_type_to_actual.get(shift_type, "Unknown")

        df_filtered = df[df["n_test"] == test_size] if test_size is not None else df
        for _, row in df_filtered.iterrows():
            y_true.append(true_shift)
            y_pred.append(row["final_identified_shift"])

    if not y_true:
        print(f"No data found for confusion matrix (test_size={test_size})")
        return

    cm = confusion_matrix(y_true, y_pred, labels=all_identified_shifts)

    if normalize == "true":
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title_suffix = " (Normalized by True Label)"
    elif normalize == "pred":
        cm = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
        title_suffix = " (Normalized by Predicted Label)"
    elif normalize == "all":
        cm = cm.astype("float") / cm.sum()
        title_suffix = " (Normalized Overall)"
    else:
        title_suffix = ""

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        square=True,
        xticklabels=all_identified_shifts,
        yticklabels=all_identified_shifts,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
    )

    ax.set_xlabel("Predicted Shift", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Shift Type", fontsize=12, fontweight="bold")

    test_size_str = f" (Test Size: {test_size})" if test_size else " (All Test Sizes)"
    ax.set_title(
        f"{dataset.title()} - Confusion Matrix{title_suffix}{test_size_str}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_str = f"_{normalize}" if normalize else ""
    test_size_str = f"_ts{test_size}" if test_size else "_all"
    filename = f"confusion_matrix_{dataset}{normalize_str}{test_size_str}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] Confusion matrix: {filename}")


def calculate_shift_identification_metrics(
    file_data: Dict,
    output_dir: Path,
    dataset: str = "padchest",
    test_sizes: Optional[List[int]] = None,
    save_to_csv: bool = True,
    print_report: bool = True,
) -> Dict:
    """Calculate precision, recall, F1-score, and other metrics for shift identification."""

    if test_sizes is None:
        test_sizes = [100, 250, 500, 1000]

    shift_type_to_actual = {
        "acq": "Covariate (acquisition) only",
        "gender": "Covariate (gender) only",
        "prev": "Prevalence shift",
    }

    all_identified_shifts = [
        "Prevalence shift",
        "Covariate (gender) only",
        "Covariate (acquisition) only",
        "Covariate (gender) + Prevalence",
        "Covariate (acquisition) + Prevalence",
        "No shift",
    ]

    metrics_by_test_size = {}

    for test_size in test_sizes:
        y_true, y_pred, shift_strengths = [], [], []

        for file_info in file_data.values():
            df = file_info["data"]
            shift_type = file_info["shift_type"]
            shift_strength = file_info["shift_strength"]
            true_shift = shift_type_to_actual.get(shift_type, "Unknown")

            df_filtered = df[df["n_test"] == test_size]
            for _, row in df_filtered.iterrows():
                y_true.append(true_shift)
                y_pred.append(row["final_identified_shift"])
                shift_strengths.append(shift_strength)

        if not y_true:
            print(f"No data found for test size {test_size}")
            continue

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=all_identified_shifts, average=None, zero_division=0
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        accuracy = np.mean([yt == yp for yt, yp in zip(y_true, y_pred)])

        metrics_by_test_size[test_size] = {
            "per_class": {
                shift: {
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1": f1[i],
                    "support": support[i],
                }
                for i, shift in enumerate(all_identified_shifts)
            },
            "overall": {
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
                "total_samples": len(y_true),
            },
        }

        if print_report:
            print(f"\n{'='*80}")
            print(f"Classification Report - Test Size: {test_size}")
            print(f"{'='*80}")
            print(
                classification_report(
                    y_true, y_pred, labels=all_identified_shifts, zero_division=0
                )
            )
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Total samples: {len(y_true)}")

    if save_to_csv:
        save_metrics_to_csv(metrics_by_test_size, output_dir, dataset)

    return metrics_by_test_size


def save_metrics_to_csv(metrics_dict: Dict, output_dir: Path, dataset: str) -> None:
    """Save metrics to CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for test_size, metrics in metrics_dict.items():
        for shift_type, class_metrics in metrics["per_class"].items():
            rows.append(
                {
                    "test_size": test_size,
                    "shift_type": shift_type,
                    "precision": class_metrics["precision"],
                    "recall": class_metrics["recall"],
                    "f1_score": class_metrics["f1"],
                    "support": class_metrics["support"],
                }
            )

    per_class_df = pd.DataFrame(rows)
    per_class_file = output_dir / f"{dataset}_per_class_metrics.csv"
    per_class_df.to_csv(per_class_file, index=False)
    print(f"[Saved] Per-class metrics: {per_class_file.name}")

    overall_rows = []
    for test_size, metrics in metrics_dict.items():
        row = {"test_size": test_size}
        row.update(metrics["overall"])
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    overall_file = output_dir / f"{dataset}_overall_metrics.csv"
    overall_df.to_csv(overall_file, index=False)
    print(f"[Saved] Overall metrics: {overall_file.name}")


def plot_metrics_by_test_size(
    metrics_dict: Dict,
    output_dir: Path,
    dataset: str = "padchest",
    metric_type: str = "f1",
) -> None:
    """Plot how precision, recall, or F1 varies with test size."""

    test_sizes = sorted(metrics_dict.keys())
    shift_types = list(next(iter(metrics_dict.values()))["per_class"].keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    colours = {
        "Prevalence shift": "#FF746C",
        "Covariate (gender) only": "#779ECB",
        "Covariate (acquisition) only": "#3D6A9F",
        "Covariate (gender) + Prevalence": "#9e84A1",
        "Covariate (acquisition) + Prevalence": "#755B78",
        "No shift": "#CCCCCC",
    }

    for shift_type in shift_types:
        metric_values = []
        for test_size in test_sizes:
            if test_size in metrics_dict:
                metric_key = "f1" if metric_type == "f1" else metric_type
                value = metrics_dict[test_size]["per_class"][shift_type][metric_key]
                metric_values.append(value)
            else:
                metric_values.append(np.nan)

        ax.plot(
            test_sizes,
            metric_values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=shift_type,
            color=colours.get(shift_type, "gray"),
        )

    ax.set_xlabel("Test Size", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{metric_type.capitalize()} Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{dataset.title()} - {metric_type.capitalize()} Score by Test Size",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xticks(test_sizes)
    ax.set_xticklabels([str(ts) for ts in test_sizes])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}_{metric_type}_by_test_size.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] Metrics plot: {filename}")


###########################################################
######################### MAIN ############################
###########################################################

if __name__ == "__main__":
    ROOT = Config.ROOT

    input_dir = ROOT / "outputs"
    output_dir = ROOT / "experiments" / "outputs" / "ImprovedPipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = "padchest"

    # Parse the CSV files once
    print("\n" + "=" * 80)
    print("STEP 1: Parsing CSV files")
    print("=" * 80)
    file_data = parse_csv_files(input_dir, dataset)

    if not file_data:
        print(f"ERROR: No CSV files found for dataset '{dataset}' in {input_dir}")
        exit(1)

    # 1. Create original stacked bar chart plots
    print("\n" + "=" * 80)
    print("STEP 2: Creating stacked bar charts")
    print("=" * 80)
    plot_shift_identification_results(
        input_dir=input_dir, output_dir=output_dir, dataset=dataset
    )

    # 2. Create confusion matrices for different test sizes
    print("\n" + "=" * 80)
    print("STEP 3: Creating confusion matrices")
    print("=" * 80)

    # Create confusion matrix for each test size
    for test_size in [100, 250, 500, 1000]:
        plot_confusion_matrix(
            file_data=file_data,
            output_dir=output_dir,
            dataset=dataset,
            test_size=test_size,
            normalize="true",  # Normalize by true label
            figsize=(10, 8),
        )

    # Also create one for all test sizes combined
    plot_confusion_matrix(
        file_data=file_data,
        output_dir=output_dir,
        dataset=dataset,
        test_size=None,  # All test sizes
        normalize="true",
        figsize=(10, 8),
    )

    # 3. Calculate and save metrics
    print("\n" + "=" * 80)
    print("STEP 4: Calculating classification metrics")
    print("=" * 80)

    metrics_dict = calculate_shift_identification_metrics(
        file_data=file_data,
        output_dir=output_dir,
        dataset=dataset,
        test_sizes=[100, 250, 500, 1000],
        save_to_csv=True,
        print_report=True,
    )

    # 4. Plot metrics across test sizes
    print("\n" + "=" * 80)
    print("STEP 5: Creating metrics visualization plots")
    print("=" * 80)

    # Plot F1 scores
    plot_metrics_by_test_size(
        metrics_dict=metrics_dict,
        output_dir=output_dir,
        dataset=dataset,
        metric_type="f1",
    )

    # Plot precision
    plot_metrics_by_test_size(
        metrics_dict=metrics_dict,
        output_dir=output_dir,
        dataset=dataset,
        metric_type="precision",
    )

    # Plot recall
    plot_metrics_by_test_size(
        metrics_dict=metrics_dict,
        output_dir=output_dir,
        dataset=dataset,
        metric_type="recall",
    )

    # 5. Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY: Overall Performance Across Test Sizes")
    print("=" * 80)

    for test_size in sorted(metrics_dict.keys()):
        overall = metrics_dict[test_size]["overall"]
        print(f"\nTest Size: {test_size}")
        print(f"  Accuracy:           {overall['accuracy']:.4f}")
        print(f"  Macro F1:           {overall['macro_f1']:.4f}")
        print(f"  Weighted F1:        {overall['weighted_f1']:.4f}")
        print(f"  Total Samples:      {overall['total_samples']}")

    print("\n" + "=" * 80)
    print("All analyses complete! Check the output directory for results.")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
