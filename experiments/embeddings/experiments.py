"""
experiments/embeddings/experiment.py

Example usage:
--------------
    run_*_experiment(encoder_to_evaluate="imagenet", feat_mode="all", dataset="Mammo")
"""

from experiments.embeddings.config import Config
from experiments.embeddings import plotting_utils as plotting
from experiments.embeddings import data_processing_utils as data_processing
from experiments.embeddings import statistical_utils as statistical


# -----------------------------------------
# First experiment called in main execution
# -----------------------------------------
def run_visualisation_experiment(
    encoder_to_evaluate: str,
    feat_mode: str,
    dataset: str,
    force_calculation: bool = False,
) -> None:

    Config.validate()
    Config.set_seeds()

    output_dir = (
        Config.ROOT
        / "experiments"
        / "outputs"
        / dataset
        / "Plots"
        / encoder_to_evaluate
    )
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
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
        force_calculation=force_calculation,
    )

    # Experiment 1: Visualisation of embeddings
    plotting.plot_layer_representations_scatter(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layer_to_results_dict=layer_to_results_dict,
        labels=val_labels,
        shift="no_shift",
    )

    # Experiment 2: Visualisation of shifts
    plotting.plot_shift_comparison_scatter(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layer_to_results_dict=layer_to_results_dict,
    )
    plotting.plot_shift_comparison_joint(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layer_to_results_dict=layer_to_results_dict,
    )

    print(f"\n=== VISUALIZATION COMPLETE ===\n")


# ------------------------------------------
# Second experiment called in main execution
# ------------------------------------------
def run_stats_experiment(
    encoder_to_evaluate: str,
    feat_mode: str,
    dataset: str,
    force_calculations: bool = False,
) -> None:
    # We have a given dataset and encoder.
    # We want to loop through the different shift scenarios.
    # Ignore mixed shifts. Look at e.g. "acquisition" and "population" shifts.
    # For each type of shift, we want to analyse "subtle", "moderate" and "extreme".
    # -> Produce a table with MMD outputs from each layer of the encoder (and also BBSD on softmax outputs).
    # -> (Make sure to subsample properly - look at Mel's function).
    # -> For each row, output the correct data as a csv/json.
    # -> Create a separate script to run and collect the data and process it into a table.
    # -> Plot this table as a heat map, colour coded by detection rate.
    # Ultimately, we will produce one table for each dataset and encoder.
    # The hypothesis is that different shift types will have different detection rates at different layers.
    # Consider adding other metrics to compare.

    Config.validate()
    Config.set_seeds()

    output_dir = (
        Config.ROOT
        / "experiments"
        / "outputs"
        / dataset
        / "Stats"
        / encoder_to_evaluate
    )
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

    print(f"\n=== STAT CALCULATIONS COMPLETE ===\n")


# -----------------------------------------
# Third experiment called in main execution
# -----------------------------------------
def run_bootstrap_experiment(
    encoder_to_evaluate: str,
    feat_mode: str,
    dataset: str,
    force_calculations: bool = False,
) -> None:
    """
    Runs the bootstrap experiment to calculate detection rates for covariate shifts.
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

    # Experiment 4: Bootstrap detection rates analysis
    # Plot the bootstrap results
    plotting.plot_all_bootstrap_results(output_dir)

    print(f"\n=== BOOTSTRAP CALCULATIONS COMPLETE ===\n")
