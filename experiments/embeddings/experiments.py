"""
experiments/embeddings/experiment.py

Example usage:
--------------
    run_*_experiment(encoder_to_evaluate="imagenet", feat_mode="all", dataset="Mammo")
"""

from experiments.embeddings.config import Config
from experiments.embeddings import plotting_utils as plotting
from experiments.embeddings import data_processing_utils as data_processing
from experiments.embeddings.statistical_utils import calculate_detection_rates


# -----------------------------------------
# First experiment called in main execution
# -----------------------------------------
def run_visualisation_experiment(
    encoder_to_evaluate: str, feat_mode: str, dataset: str
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
    val_labels, test_labels = data_processing.extract_plot_labels(
        val_df=val_df, test_df=test_df, encoder_output=encoder_output, dataset=dataset
    )

    # Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = data_processing.simulate_shifts(
        dataset=dataset, test_df=test_df
    )

    # Generate plots
    inputs = plotting.PlotInputs(
        encoder_to_evaluate=encoder_to_evaluate,
        dataset=dataset,
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
    )

    plotting.plot_all_layer_representations_scatter(
        output_dir=output_dir / "layers_representations",
        inputs=inputs,
        val_labels=val_labels,
        test_labels=test_labels,
        run_statistical_tests=False,
    )
    plotting.plot_shift_comparison_joint(
        output_dir=output_dir / "shift_comparison", inputs=inputs
    )
    plotting.plot_shift_comparison_scatter(
        output_dir=output_dir / "shift_comparison", inputs=inputs
    )

    print(f"\n=== VISUALIZATION COMPLETE ===")


# ------------------------------------------
# Second experiment called in main execution
# ------------------------------------------

# We have a given dataset and encoder.
# We want to loop through the different shift scenarios.
# Ignore mixed shifts. Look at e.g. "acquisition" and "population" shifts.
# For each type of shift, we want to analyse "subtle", "moderate" and "extreme".
# -> Produce a table with MMD outputs from each layer of the encoder and also BBSD on softmax outputs.
# -> (Make sure to subsample properly - look at Mel's function).
# -> For each row, output the correct data as a csv/json.
# -> Create a separate script to run and collect the data and process it into a table.
# -> Plot this table as a heat map, colour coded by detection rate.
# Ultimately, we will produce one table for each dataset and encoder.
# The hypothesis is that different shift types will have different detection rates at different layers.
# Consider adding other metrics to compare.


def run_stats_experiment(
    encoder_to_evaluate: str, feat_mode: str, dataset: str
) -> None:

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

    # Calculate shift detection rates for the simluated shifts
    n_val = len(val_df)
    calculate_detection_rates(
        output_dir=output_dir,
        dataset=dataset,
        encoder_to_evaluate=encoder_to_evaluate,
        layers=layers,
        n_val=n_val,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
    )

    print(f"\n=== STAT CALCULATIONS COMPLETE ===")
