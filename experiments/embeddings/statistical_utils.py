"""
experiments/embeddings/statistical_utils.py

Statistical utilities for detecting distribution shifts in embedding spaces.

This module provides implementations of several statistical metrics to compare
two distributions (reference vs shifted embeddings) across model layers.
Metrics supported:
    - MMD (Maximum Mean Discrepancy with RBF kernel)
    - KL divergence

Each function returns structured results using the ShiftTestResult dataclass.
Helper functions are provided to compute metrics individually or all at once,
and to save results in JSON and CSV format.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from shift_identification_detection.mmd_test import run_mmd_permutation_test
from shift_identification_detection.shift_identification import (
    embed_patient_permutations,
)

from experiments.embeddings.config import Config


def calculate_PCA_and_tSNE(
    embeddings: torch.Tensor,
    pca_components: int = 2,
    seed: int = Config.SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings with PCA and t-SNE.

    PCA is applied to reduce dimensionality (default 2D), followed by t-SNE
    on the PCA output to obtain a 2D embedding. Perplexity is set based on
    sample size, and is always set between 5 and 30.

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
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    embeddings_tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embeddings_pca)

    print(f"t-SNE shape: {embeddings_tsne.shape}")

    return embeddings_pca, embeddings_tsne


@dataclass
class ShiftTestResult:
    """
    Container for storing the result of a statistical test for a shift.

    Specific to ResNet-50 architecture.
    """

    shift: str = ""

    mp_mmd_pvalue: float = 1.00
    mp_mmd_is_significant: bool = False

    layer_1_mmd_pvalue: float = 1.00
    layer_1_mmd_is_significant: bool = False

    layer_2_mmd_pvalue: float = 1.00
    layer_2_mmd_is_significant: bool = False

    layer_3_mmd_pvalue: float = 1.00
    layer_3_mmd_is_significant: bool = False

    final_layer_mmd_pvalue: float = 1.00
    final_layer_mmd_is_significant: bool = False


def calculate_detection_rates(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
    layers: list[str],
    n_val: int,
    val_embeddings: dict[str, torch.Tensor],
    test_embeddings: dict[str, torch.Tensor],
    shift_to_indices_dict: dict[str, np.ndarray],
    force_calculations: bool = False,
) -> list[ShiftTestResult]:
    """
    Calculate detection rates for a number of simulated covariate shifts,
    with optional caching.

    Args:
        output_dir: Directory to save results.
        dataset: Name of the dataset.
        encoder_to_evaluate: Name of the encoder being evaluated.
        layers: List of layer names to evaluate.
        n_val: Number of validation samples.
        val_embeddings: Dictionary of validation embeddings by layer.
        test_embeddings: Dictionary of test embeddings by layer.
        shift_to_indices_dict: Dictionary mapping shift names to indices.
        force_calculations: If True, skip cache and recalculate everything.
    """

    json_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.json"
    csv_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.csv"

    # Check for cached results:
    if json_path.exists() and not force_calculations:
        print(f"Loading cached results for {dataset}/{encoder_to_evaluate}")
        try:
            with open(json_path, "r") as jf:
                cached_data = json.load(jf)

            cached_shifts = {item["shift"] for item in cached_data}
            required_shifts = set(shift_to_indices_dict.keys())

            if cached_shifts == required_shifts:
                results = [ShiftTestResult(**item) for item in cached_data]
                return results
            else:
                print("Cache is incomplete or outdated.")
                missing_shifts = required_shifts - cached_shifts
                extra_shifts = cached_shifts - required_shifts
                if missing_shifts:
                    print(f"Missing shifts in cache: {missing_shifts}")
                if extra_shifts:
                    print(f"Extra shifts in cache: {extra_shifts}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error reading cached results: {e}. Recalculating...")

    print(f"Calculating results for {dataset}/{encoder_to_evaluate}")
    alpha = 0.05
    results: list[ShiftTestResult] = []

    # Loop through all shifts
    for shift_name, idx_array in shift_to_indices_dict.items():
        print(f"\nProcessing {shift_name}...")

        shift_result = ShiftTestResult(shift=shift_name)

        # Run MMD for each layer
        for layer in layers:
            print(f"--- Calculating MMD for layer: {layer} ---")

            # Run MMD test
            cat_embeddings = torch.cat(
                [val_embeddings[layer], test_embeddings[layer][idx_array]]
            )
            n_samples = cat_embeddings.shape[0]
            n_features = cat_embeddings.shape[1]
            print(f"n_samples: {n_samples}")
            print(f"n_features: {n_features}")
            n_components = min(32, n_samples, n_features)
            pca = PCA(n_components=n_components)
            embeddings_pca = pca.fit_transform(cat_embeddings.cpu().numpy())
            print("Starting the permutation test...\n")

            val_pca = embeddings_pca[:n_val]
            test_pca = embeddings_pca[n_val:]

            mmd_p = run_mmd_permutation_test(
                val_pca,
                test_pca,
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

        # Collect the results from all shift combinations
        results.append(shift_result)

        # Save the results
        data = [asdict(r) for r in results]
        with open(json_path, "w") as jf:
            json.dump(data, jf, indent=2)
        pd.DataFrame(data).to_csv(csv_path, index=False)

    print("\nCalculations complete.")
    print(f"[Saved] JSON: {json_path}")
    print(f"[Saved] CSV: {csv_path}")

    return results


def calculate_bootstrap_detection_rates(
    output_dir: Path,
    dataset: str,
    encoder_to_evaluate: str,
    layers: list[str],
    val_embeddings: dict[str, torch.Tensor],
    test_embeddings: dict[str, torch.Tensor],
    shift_to_indices_dict: dict[str, np.ndarray],
    n_bootstrap: int,
    n_val: int,
    shift_subset_sizes: list[int],
    force_calculations: bool = False,
) -> None:
    """
    Calculate bootstrap detection rates for each shift, layer, and test subset size,
    with optional caching.

    For each combination:
    - Bootstrap samples both validation (n_val) and test (test_size) subsets
    - Applies PCA dimensionality reduction
    - Runs MMD permutation test
    - Counts significant detections (p < 0.05)
    - Calculates detection rate as proportion of significant results

    Args:
        output_dir (Path): Directory to save results.
        dataset (str): Dataset name.
        encoder_to_evaluate (str): Encoder identifier.
        layers (list[str]): List of layer names.
        val_embeddings (dict): Validation embeddings per layer.
        test_embeddings (dict): Test embeddings per layer.
        shift_to_indices_dict (dict): Mapping of shifts to test indices.
        n_bootstrap (int): Number of bootstrap iterations.
        n_val (int): Number of validation samples to use per bootstrap.
        shift_subset_sizes (list[int]): List of test subset sizes to evaluate.
        force_calculations (bool): If True, skip cache and recalculate everything.
    """

    csv_path = (
        output_dir / f"bootstrap_detection_rates-{dataset}-{encoder_to_evaluate}.csv"
    )
    json_path = (
        output_dir / f"bootstrap_detection_rates-{dataset}-{encoder_to_evaluate}.json"
    )

    existing_results = []

    if json_path.exists() and not force_calculations:
        print(f"Loading cached results for {dataset}/{encoder_to_evaluate}")
        try:
            with open(json_path, "r") as jf:
                existing_results = json.load(jf)
            completed_combinations = {
                (item["shift"], item["layer"], int(item["test_size"]))
                for item in existing_results
            }
        except Exception as e:
            print(f"Error loading cache: {e}")
            existing_results = []
            completed_combinations = set()
    else:
        completed_combinations = set()

    alpha = 0.05
    all_results = existing_results.copy()

    print(f"\n=== Starting bootstrap experiment with {n_bootstrap} iterations ===")
    print(f"Validation samples per bootstrap: {n_val}")
    print(f"Shift subset sizes: {shift_subset_sizes}")
    print(f"Significance threshold: {alpha}")

    # Get total available validation samples
    total_val_samples = len(next(iter(val_embeddings.values())))
    if n_val > total_val_samples:
        print(
            f"! Warning: Requested {n_val} val samples but only {total_val_samples} available."
        )
        n_val = total_val_samples

    total_combinations = (
        len(shift_to_indices_dict) * len(layers) * len(shift_subset_sizes)
    )
    current_combination = 0

    # Process each shift
    for shift_name, shift_indices in shift_to_indices_dict.items():
        print(f"\nProcessing shift: {shift_name}...")

        # Process each layer
        for layer in layers:
            print(f"-- Layer: {layer}")

            # Get embeddings for this layer
            val_embeddings_layer = val_embeddings[layer]
            shift_embeddings_layer = test_embeddings[layer][shift_indices]
            n_shift_samples = len(shift_embeddings_layer)

            # Process each shift subset size
            for target_shift_size in shift_subset_sizes:
                combination_key = (shift_name, layer, int(target_shift_size))
                current_combination += 1
                if combination_key in completed_combinations:
                    print(f"     Skipping cached combination: {combination_key}")
                    continue
                print(
                    f"     Shift subset size: {target_shift_size} [{current_combination}/{total_combinations}]"
                )

                # Check we have enough samples
                current_shift_size = min(target_shift_size, n_shift_samples)
                if target_shift_size > n_shift_samples:
                    print(
                        f"! Warning: Requested {target_shift_size} samples but only {n_shift_samples} available."
                    )

                # Bootstrap iterations
                p_values = []
                significant_detections = 0
                successful_bootstraps = 0

                for bootstrap_iter in range(n_bootstrap):
                    try:
                        # Bootstrap sample validation set
                        val_indices = np.random.choice(
                            total_val_samples, size=n_val, replace=False
                        )
                        val_bootstrap = val_embeddings_layer[val_indices]

                        # Bootstrap sample shift set
                        bootstrap_shift_indices = np.random.choice(
                            n_shift_samples, size=current_shift_size, replace=False
                        )
                        shift_bootstrap = shift_embeddings_layer[
                            bootstrap_shift_indices
                        ]

                        # Combine embeddings for PCA preprocessing (following working implementation)
                        cat_embeddings = torch.cat([val_bootstrap, shift_bootstrap])

                        # Apply PCA preprocessing (following the working implementation)
                        n_samples_total = cat_embeddings.shape[0]
                        n_features = cat_embeddings.shape[1]
                        n_components = min(32, n_samples_total, n_features)

                        pca = PCA(n_components=n_components)
                        embeddings_pca = pca.fit_transform(cat_embeddings.cpu().numpy())

                        # Split back into val and shift after PCA
                        val_pca = embeddings_pca[:n_val]
                        shift_pca = embeddings_pca[n_val:]

                        # Run MMD permutation test
                        mmd_p = run_mmd_permutation_test(
                            val_pca,
                            shift_pca,
                            structure_permutation_fn=(
                                embed_patient_permutations
                                if dataset == "Mammo"
                                else None
                            ),
                        )

                        p_values.append(mmd_p)
                        successful_bootstraps += 1

                        # Count significant detections
                        if mmd_p < alpha:
                            significant_detections += 1

                    except Exception as e:
                        print(
                            f"!    Error in bootstrap iteration {bootstrap_iter}: {e}"
                        )
                        continue

                # Calculate detection rate and statistics
                if successful_bootstraps > 0:
                    detection_rate = significant_detections / successful_bootstraps
                    mean_pvalue = np.mean(p_values)
                    std_pvalue = np.std(p_values)
                else:
                    detection_rate = 0.0
                    mean_pvalue = 1.0
                    std_pvalue = 0.0

                print(
                    f"     Detection rate: {detection_rate:.2f} ({significant_detections}/{successful_bootstraps})"
                )
                print(f"     Mean p-value: {mean_pvalue:.3f} ± {std_pvalue:.3f}")

                new_results = {
                    "shift": shift_name,
                    "layer": layer,
                    "test_size": target_shift_size,
                    "detection_rate": detection_rate,
                    "mean_pvalue": mean_pvalue,
                    "std_pvalue": std_pvalue,
                    "n_successful_bootstraps": successful_bootstraps,
                }
                all_results.append(new_results)

                # Incremental saving of results to CSV and JSON
                if current_combination % 4 == 0:
                    try:
                        with open(json_path, "w") as jf:
                            json.dump(all_results, jf, indent=2)
                        pd.DataFrame(all_results).to_csv(csv_path, index=False)
                    except Exception as e:
                        print(f"!    Error saving periodic backup: {e}")

    print("Finished calculations.")
    try:
        with open(json_path, "w") as jf:
            json.dump(all_results, jf, indent=2)
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"[Saved] CSV: {csv_path}")
        print(f"[Saved] JSON: {json_path}")
    except Exception as e:
        print(f"Error in final save: {e}")


def kl_divergence(reference: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute average KL divergence KL(p || q).
    """
    reference_np = reference.numpy()
    target_np = target.numpy()

    total = 0.0
    for ref, tar in zip(reference_np, target_np):
        total += float(entropy(ref, tar))
    return total / len(reference)
