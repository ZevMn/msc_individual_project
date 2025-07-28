"""
experiments/embeddings/experiment.py

Example usage:
--------------
    run_experiment(encoder_to_evaluate="imagenet", feat_mode="all", dataset="Mammo")
"""

from experiments.embeddings.config import Config
from experiments.embeddings import plotting_utils as plotting
from experiments.embeddings import data_processing_utils as data_processing


from dataclasses import dataclass, asdict
import json
import torch
import pandas as pd
from sklearn.decomposition import PCA
from shift_identification_detection.mmd_test import run_mmd_permutation_test
from shift_identification_detection.bbsd_tests import run_bbsd
from shift_identification_detection.shift_identification import (
    embed_patient_permutations,
)


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
@dataclass
class ShiftTestResult:
    """
    Container for storing the result of a statistical test for a shift.
    """
    shift: str=""

    mp_mmd_pvalue: float=0.0
    mp_mmd_is_significant: bool=False

    layer_1_mmd_pvalue: float=0.0
    layer_1_mmd_is_significant: bool=False

    layer_2_mmd_pvalue: float=0.0
    layer_2_mmd_is_significant: bool=False

    layer_3_mmd_pvalue: float=0.0
    layer_3_mmd_is_significant: bool=False

    final_layer_mmd_pvalue: float=0.0
    final_layer_mmd_is_significant: bool=False

    bbsd_p_value: float|None = None
    bbsd_is_significant: bool | None = False


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
    output_dir.parent.mkdir(parents=True, exist_ok=True)

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

    #######################################
    # load softmax probabilities for BBSD #
    #######################################

    alpha = 0.05
    results: list[ShiftTestResult] = []

    # Loop through all shifts
    for shift_name, idx_array in shift_to_indices_dict.items():
        print(f"\nProcessing {shift_name}...")

        shift_result = ShiftTestResult(shift=shift_name)
        n_val = len(val_df)

        # Run MMD for each layer
        for layer in layers:
            print(f"\n--- Processing layer: {layer} ---")

            # Run MMD test
            cat_embeddings = torch.concatenate([
                val_embeddings[layer],
                test_embeddings[layer][idx_array]
            ])
            n_components = min(32, cat_embeddings.shape[0], cat_embeddings.shape[1])
            pca = PCA(n_components=n_components)
            embeddings_32pca = pca.fit_transform(cat_embeddings.cpu().numpy())

            mmd_p = run_mmd_permutation_test(
                embeddings_32pca[:n_val],
                embeddings_32pca[n_val:],
                structure_permutation_fn=(
                    embed_patient_permutations if dataset == "Mammo" else None
                ),
            )
            sig = mmd_p < alpha

            # Assign results for the current layer
            if layer == "after_maxpool":
                shift_result.mp_mmd_pvalue = mmd_p
                shift_result.mp_mmd_is_significant = sig
            elif layer == "layer_1":
                shift_result.layer_1_mmd_pvalue = mmd_p
                shift_result.layer_1_mmd_is_significant = sig
            elif layer == "layer_2":
                shift_result.layer_2_mmd_pvalue = mmd_p
                shift_result.layer_2_mmd_is_significant = sig
            elif layer == "layer_3":
                shift_result.layer_3_mmd_pvalue = mmd_p
                shift_result.layer_3_mmd_is_significant = sig
            elif layer == "final_layer":
                shift_result.final_layer_mmd_pvalue = mmd_p
                shift_result.final_layer_mmd_is_significant = sig
            else:
                raise ValueError(f"Unexpected layer: {layer}")


        # # Run BBSD on softmax outputs - requires task model?
        # bbsd_sig, bbsd_p = run_bbsd(
        #     probas_val, probas_test[idx_array], return_p_value=True
        # )
        # shift_result.bbsd_is_significant = bbsd_sig
        # shift_result.bbsd_p_value = bbsd_p

        # Collect the results from all shift combinations
        results.append(shift_result)

        # Save the results
        data = [asdict(r) for r in results]
        json_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.json"
        csv_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.csv"
        with open(json_path, "w") as jf:
            json.dump(data, jf, indent=2)
        pd.DataFrame(data).to_csv(csv_path, index=False)

    print(f"\n=== STAT CALCULATIONS COMPLETE ===")


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
