"""
experiments/embeddings/experiment.py

Example usage:
--------------
    run_*_experiment(encoder_to_evaluate="imagenet", feat_mode="all", dataset="Mammo")
"""

"""
experiments/embeddings/experiment.py

Defines the four core experiment workflows for analysing embeddings across datasets:
    1. Visualisation of embeddings and shifts.
    2. Shift quantification using KL divergence.
    3. Initial detection rate analysis for different shifts.
    4. Bootstrap detection rate analysis for different shifts.

Example usage:
--------------
    run_*_experiment(
        encoder_to_evaluate="imagenet",
        feat_mode="all",
        dataset="Mammo",
        force_calculations=False
    )
"""

from experiments.embeddings.config import Config
from experiments.embeddings import plotting_utils as plotting
from experiments.embeddings import data_processing_utils as data_processing
from experiments.embeddings import statistical_utils as statistical


# ---------------------------
# 1. Visualisation experiment
# ---------------------------
def run_visualisation_experiment(
    encoder_to_evaluate: str,
    feat_mode: str,
    dataset: str,
    force_calculation: bool = False,
) -> None:
    """
    Runs dimensionality reduction (PCA, t-SNE) and plots embeddings representations,
    as well as comparisons of simulated shifts.

    Args:
        encoder_to_evaluate (str): Encoder key (see Config.ENCODERS).
        feat_mode (str): Feature mode (see Config.FEAT_MODES_MAP).
        dataset (str): Dataset name (see Config.DATASET_CONFIG).
        force_calculation (bool): Recalculate projections even if cached results exist.
    """

    Config.validate()
    Config.set_seeds()

    output_dir = Config.ROOT / "experiments" / "outputs" / "Visualisations" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== {dataset.upper()} | {encoder_to_evaluate.upper()} | {feat_mode.upper()} ===\n"
    )

    # Process test and val CSVs
    val_df, test_df = data_processing.load_csvs_and_add_idx_column(dataset=dataset)

    # Generate or load embeddings
    encoder_output = data_processing.generate_and_load_embeddings(
        encoder_to_evaluate=encoder_to_evaluate,
        feat_mode=feat_mode,
        dataset=dataset,
        val_df=val_df,
        test_df=test_df,
    )

    # Validate the embeddings output and extract layers, and val/test splits
    layers, val_embeddings, test_embeddings = (
        data_processing.validate_and_process_embeddings(encoder_output=encoder_output)
    )

    # Extract plot labels
    val_labels, _ = data_processing.extract_plot_labels(
        val_df=val_df, test_df=test_df, encoder_output=encoder_output, dataset=dataset
    )

    # Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = data_processing.simulate_shifts(
        dataset=dataset, test_df=test_df
    )

    layer_to_results_dict = data_processing.calculate_and_save_layer_pca_and_tsne(
        output_dir=output_dir,
        encoder_to_evaluate=encoder_to_evaluate,
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
        force_calculation=force_calculation,
    )

    # Experiment 1: Visualisation of embeddings
    plotting.plot_layer_representations_jointplot(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layer_to_results_dict=layer_to_results_dict,
        labels=val_labels,
        shift="no_shift",
    )

    # Experiment 2: Visualisation of shifts
    plotting.plot_shift_comparison_joint(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layer_to_results_dict=layer_to_results_dict,
    )

    print(f"\n=== VISUALISATION COMPLETE ===\n")


# ----------------------------------
# 2. Shift quantification experiment
# ----------------------------------
def run_shift_quantification_experiment(
    encoder_to_evaluate: str,
    feat_mode: str,
    dataset: str,
    force_calculations: bool = False,
) -> None:
    """
    Quantifies covariate shifts using KL divergence with bootstrapping.

    Args:
        encoder_to_evaluate (str): Encoder key (see Config.ENCODERS).
        feat_mode (str): Feature mode (see Config.FEAT_MODES_MAP).
        dataset (str): Dataset name (see Config.DATASET_CONFIG).
        force_calculations (bool): Recalculate statistics even if cached results exist.
    """

    Config.validate()
    Config.set_seeds()

    output_dir = Config.ROOT / "experiments" / "outputs" / "ShiftQuantification"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== {dataset.upper()} | {encoder_to_evaluate.upper()} | {feat_mode.upper()} ===\n"
    )

    # Process test and val CSVs
    val_df, test_df = data_processing.load_csvs_and_add_idx_column(dataset=dataset)

    # Generate or load embeddings
    encoder_output = data_processing.generate_and_load_embeddings(
        encoder_to_evaluate=encoder_to_evaluate,
        feat_mode=feat_mode,
        dataset=dataset,
        val_df=val_df,
        test_df=test_df,
    )

    # Validate the embeddings output and extract layers, and val/test splits
    _, val_embeddings, test_embeddings = (
        data_processing.validate_and_process_embeddings(encoder_output=encoder_output)
    )

    # Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = data_processing.simulate_wide_range_of_shifts(
        dataset=dataset, test_df=test_df
    )

    # Calculate KL divergences for all simulated shifts with bootstrap
    statistical.calculate_kl_div_for_all_shifts_all_layers(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
        force_calculations=force_calculations,
        n_bootstrap=100,
    )

    # Generate plots
    plotting.plot_kl_panels_multi_layer(input_dir=output_dir, output_dir=output_dir)

    print(f"\n=== SHIFT QUANTIFICATION COMPLETE ===\n")


# ------------------------------------
# 3. Initial detection rate experiment
# ------------------------------------
def run_detection_rate_experiment(
    encoder_to_evaluate: str,
    feat_mode: str,
    dataset: str,
    force_calculations: bool = False,
) -> None:
    """
    Estimates detection rates of different shift types and severities.

    Args:
        encoder_to_evaluate (str): Encoder key (see Config.ENCODERS).
        feat_mode (str): Feature mode (see Config.FEAT_MODES_MAP).
        dataset (str): Dataset name (see Config.DATASET_CONFIG).
        force_calculations (bool): Recalculate detection rates even if cached results exist.
    """

    Config.validate()
    Config.set_seeds()

    output_dir = Config.ROOT / "experiments" / "outputs" / "DetectionRates" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== {dataset.upper()} | {encoder_to_evaluate.upper()} | {feat_mode.upper()} ===\n"
    )

    # Process test and val CSVs
    val_df, test_df = data_processing.load_csvs_and_add_idx_column(dataset=dataset)

    # Generate or load embeddings
    encoder_output = data_processing.generate_and_load_embeddings(
        encoder_to_evaluate=encoder_to_evaluate,
        feat_mode=feat_mode,
        dataset=dataset,
        val_df=val_df,
        test_df=test_df,
    )

    # Validate the embeddings output and extract layers, and val/test splits
    layers, val_embeddings, test_embeddings = (
        data_processing.validate_and_process_embeddings(encoder_output=encoder_output)
    )

    # Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = data_processing.simulate_shifts(
        dataset=dataset, test_df=test_df
    )

    n_val = val_df.shape[0]

    # Calculate shift detection rates for the simulated shifts
    statistical.calculate_detection_rates(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layers=layers,
        n_val=n_val,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
        force_calculations=force_calculations,
    )

    # Experiment 3: Initial investigation into detection rates
    plotting.plot_detection_rate_heatmap(
        output_dir=output_dir, dataset=dataset, encoder_to_evaluate=encoder_to_evaluate
    )
    plotting.plot_detection_rate_linegraph(
        output_dir=output_dir, dataset=dataset, encoder_to_evaluate=encoder_to_evaluate
    )

    print(f"\n=== DETECTION RATE ANALYSIS COMPLETE ===\n")


# --------------------------------------
# 4. Bootstrap detection rate experiment
# --------------------------------------
def run_bootstrap_experiment(
    encoder_to_evaluate: str,
    feat_mode: str,
    dataset: str,
    force_calculations: bool = False,
) -> None:
    """
    Runs bootstrap experiments to robustly quantify detection rates for
    a range of sample sizes and shift severities.

    Args:
        encoder_to_evaluate (str): Encoder key (see Config.ENCODERS).
        feat_mode (str): Feature mode (see Config.FEAT_MODES_MAP).
        dataset (str): Dataset name (see Config.DATASET_CONFIG).
        force_calculations (bool): Recalculate detection rates even if cached results exist.
    """
    Config.validate()
    Config.set_seeds()

    output_dir = Config.ROOT / "experiments" / "outputs" / "CollectedBootstrapResults"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== {dataset.upper()} | {encoder_to_evaluate.upper()} | {feat_mode.upper()} ===\n"
    )

    # Process test and val CSVs
    val_df, test_df = data_processing.load_csvs_and_add_idx_column(dataset=dataset)

    # Generate or load embeddings
    encoder_output = data_processing.generate_and_load_embeddings(
        encoder_to_evaluate=encoder_to_evaluate,
        feat_mode=feat_mode,
        dataset=dataset,
        val_df=val_df,
        test_df=test_df,
    )

    # Validate the embeddings output and extract layers, and val/test splits
    layers, val_embeddings, test_embeddings = (
        data_processing.validate_and_process_embeddings(encoder_output=encoder_output)
    )

    # Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = data_processing.simulate_shifts(
        dataset=dataset, test_df=test_df
    )

    # Bootstrap experiment configuration
    bootstrap_config = {
        "n_bootstrap": 200,
        "n_val": 2000,
        "shift_subset_sizes": [100, 250, 500, 1000],
    }

    # Calculate bootstrap detection rates for the simulated shifts
    statistical.calculate_bootstrap_detection_rates(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
        **bootstrap_config,
        force_calculations=force_calculations,
    )

    # Plot the bootstrap results
    plotting.plot_all_bootstrap_results(output_dir)

    print(f"\n=== BOOTSTRAP ANALYSIS COMPLETE ===\n")
