"""
experiments/embeddings/experiment.py

Example usage:
--------------
    run_experiment(encoder_to_evaluate="imagenet", feat_mode="all", dataset="Mammo")
"""

from experiments.embeddings.config import Config
from experiments.embeddings import plotting_utils as plotting
from experiments.embeddings import data_processing_utils as data_processing

# ------------------------
# Called in main execution
# ------------------------
def run_experiment(encoder_to_evaluate: str, feat_mode: str, dataset: str) -> None:

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

    # Load embeddings and generate them if they don't already exist
    encoder_output = data_processing.generate_and_load_embeddings(
        encoder_to_evaluate=encoder_to_evaluate,
        feat_mode=feat_mode,
        dataset=dataset,
        val_df=val_df,
        test_df=test_df,
    )

    # Validate the embeddings output and extract layers, and val/test splits
    layers, val_embeddings, test_embeddings = data_processing.validate_and_process_embeddings(
        encoder_output=encoder_output
    )

    # Extract plot labels
    val_labels, test_labels = data_processing.extract_plot_labels(
        val_df=val_df, test_df=test_df, encoder_output=encoder_output, dataset=dataset
    )

    # Generate covariate-shifted test subsets and store their original indices
    shift_to_indices_dict = data_processing.simulate_shifts(dataset=dataset, test_df=test_df)

    # Generate plots
    inputs = plotting.PlotInputs(
        encoder_to_evaluate=encoder_to_evaluate,
        dataset=dataset,
        layers=layers,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        shift_to_indices_dict=shift_to_indices_dict,
    )

    plotting.plot_all_layers_scatter_labelled(
        output_dir=output_dir / "layer_representation",
        inputs=inputs,
        val_labels=val_labels,
        test_labels=test_labels,
        run_statistical_tests=False,
    )
    plotting.plot_shift_comparison_scatter(
        output_dir=output_dir / "shift_comparison", inputs=inputs
    )
    plotting.plot_shift_comparison_joint(
        output_dir=output_dir / "shift_comparison", inputs=inputs
    )

    print(f"\n=== VISUALIZATION COMPLETE ===")
