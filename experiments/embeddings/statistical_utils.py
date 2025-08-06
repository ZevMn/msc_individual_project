"""
experiments/embeddings/statistical_utils.py

Statistical utilities for detecting distribution shifts in embedding spaces.

This module provides implementations of several statistical metrics to compare
two distributions (reference vs shifted embeddings) across model layers.
Metrics supported:
    - BBSD
    - MMD (Maximum Mean Discrepancy with RBF kernel)
    - Energy distance
    - KL divergence
    - Jensen-Shannon divergence

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
):

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
            print("Starting the permutation test...")
            print("\n\n\n\n\n")

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
        json_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.json"
        csv_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.csv"
        with open(json_path, "w") as jf:
            json.dump(data, jf, indent=2)
        pd.DataFrame(data).to_csv(csv_path, index=False)


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
) -> None:
    """
    Calculate bootstrap detection rates for each shift, layer, and test subset size.

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
    """

    alpha = 0.05
    results = {
        "shift": [],
        "layer": [],
        "test_size": [],
        "detection_rate": [],
        "mean_pvalue": [],
        "std_pvalue": [],
        "n_successful_bootstraps": [],
    }

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
            for shift_subset_size in shift_subset_sizes:
                print(f"     Shift subset size: {shift_subset_size}")

                # Check we have enough samples
                if shift_subset_size > n_shift_samples:
                    print(
                        f"! Warning: Requested {shift_subset_size} samples but only {n_shift_samples} available."
                    )
                    shift_subset_size = n_shift_samples

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
                        shift_indices = np.random.choice(
                            n_shift_samples, size=shift_subset_size, replace=False
                        )
                        shift_bootstrap = shift_embeddings_layer[shift_indices]

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

                # Store results
                results["shift"].append(shift_name)
                results["layer"].append(layer)
                results["test_size"].append(shift_subset_size)
                results["detection_rate"].append(detection_rate)
                results["mean_pvalue"].append(mean_pvalue)
                results["std_pvalue"].append(std_pvalue)
                results["n_successful_bootstraps"].append(successful_bootstraps)

    # Save results to CSV and JSON
    results_df = pd.DataFrame(results)

    csv_path = (
        output_dir / f"bootstrap_detection_rates-{dataset}-{encoder_to_evaluate}.csv"
    )
    json_path = (
        output_dir / f"bootstrap_detection_rates-{dataset}-{encoder_to_evaluate}.json"
    )

    results_df.to_csv(csv_path, index=False)
    results_dict = results_df.to_dict("records")
    with open(json_path, "w") as jf:
        json.dump(results_dict, jf, indent=2)

    print(f"Results saved to CSV: {csv_path}")
    print(f"Results saved to JSON: {json_path}")


# --------------------------------------------------------------


# @dataclass
# class ShiftTestResult_old:
#     """
#     Container for storing the result of a statistical shift test.

#     Attributes:
#         metric (str): Name of the statistical test (e.g., "MMD", "BBSD", "KL").
#         layer (str): Identifier for the encoder layer the test was run on.
#         shift_name (str): Name of the shift scenario (e.g., "acq", "contrast").
#         stat (float): The test statistic value.
#         p_value (float | None): The associated p-value, or None if not applicable.
#         extra (dict): Additional metadata (e.g., per-dimension stats).
#     """

#     metric: str
#     layer: str
#     shift_name: str
#     stat: float
#     p_value: float | None
#     extra: dict


# def save_results(results: list[ShiftTestResult_old], output_dir: Path) -> None:
#     """
#     Save a list of ShiftTestResult objects to disk in JSON and CSV formats.

#     Args:
#         results (list[ShiftTestResult]): The test results to save.
#         output_dir (Path): File path to save the results.
#     """
#     output_dir.parent.mkdir(parents=True, exist_ok=True)
#     data = [asdict(r) for r in results]
#     with open(output_dir.with_suffix(".json"), "w") as f:
#         json.dump(data, f, indent=2)
#     pd.DataFrame(data).to_csv(output_dir.with_suffix(".csv"), index=False)


# # -----------------------------
# # BBSD and MMD analysis wrapper
# # -----------------------------
# def calculate_bbsd_and_mmd(
#     source_distribution: torch.Tensor | np.ndarray,
#     target_distribution: torch.Tensor | np.ndarray,
#     layer_name: str,
#     shift: str,
#     apply_pca: bool = True,
#     pca_dim: int = 32,
# ) -> list[ShiftTestResult_old]:
#     """
#     Compute BBSD and MMD statistics between source and target distributions.

#     - BBSD: Only computed if inputs are softmax outputs (rows sum ≈ 1).
#     - MMD: Always computed using RBF-kernel and permutation test.
#            Optionally applies PCA to reduce dimensionality before MMD.

#     Args:
#         source_distribution (Tensor | ndarray): Source (validation) embeddings or softmax outputs.
#         target_distribution (Tensor | ndarray): Target (shifted) embeddings or softmax outputs.
#         layer_name (str): Encoder layer name or index.
#         shift (str): Identifier for the shift condition.
#         apply_pca (bool): If True, reduce dimensionality to 'pca_dim' before MMD.
#         pca_dim (int): Number of PCA components to keep if PCA is applied.

#     Returns:
#         List[ShiftTestResult]: Results for BBSD (if applicable) and MMD.
#     """
#     results: list[ShiftTestResult_old] = []

#     # Convert to NumPy
#     if isinstance(source_distribution, torch.Tensor):
#         p_np = source_distribution.detach().cpu().numpy()
#     else:
#         p_np = np.asarray(source_distribution)

#     if isinstance(target_distribution, torch.Tensor):
#         q_np = target_distribution.detach().cpu().numpy()
#     else:
#         q_np = np.asarray(target_distribution)

#     # Check if inputs are softmax probability vectors
#     is_probabilistic = np.allclose(p_np.sum(axis=1), 1.0, atol=1e-4) and np.allclose(
#         q_np.sum(axis=1), 1.0, atol=1e-4
#     )

#     # BBSD (only if probabilistic inputs)
#     if is_probabilistic:
#         alpha = 0.05
#         n_dim = p_np.shape[1]
#         threshold = alpha / n_dim

#         bbsd_output = run_bbsd(p_np, q_np, alpha=alpha, return_p_value=True)
#         if isinstance(bbsd_output, tuple):
#             bbsd_flag, all_p_values = bbsd_output
#         else:
#             bbsd_flag = bbsd_output
#             all_p_values = np.ones(n_dim)

#         pval_bbsd = float(np.min(all_p_values))

#         results.append(
#             ShiftTestResult_old(
#                 metric="BBSD",
#                 layer=layer_name,
#                 shift_name=shift,
#                 stat=pval_bbsd,
#                 p_value=pval_bbsd,
#                 extra={
#                     "all_p_values": all_p_values.tolist(),
#                     "alpha_corrected": threshold,
#                     "rejected": bbsd_flag,
#                 },
#             )
#         )

#     # MMD (always run)
#     combined = np.concatenate([p_np, q_np], axis=0)
#     if apply_pca:
#         max_components = min(combined.shape[0], combined.shape[1])
#         n_components = min(pca_dim, max_components)

#         if n_components < 2:
#             print(
#                 f"[Warning] Skipping PCA (n_components={n_components}) due to insufficient shape: {combined.shape}"
#             )
#             p_reduced = p_np
#             q_reduced = q_np
#             combined_reduced = combined
#             apply_pca = False
#         else:
#             pca = PCA(n_components=n_components)
#             combined_reduced = pca.fit_transform(combined)
#             p_reduced = combined_reduced[: p_np.shape[0]]
#             q_reduced = combined_reduced[p_np.shape[0] :]
#             pca_dim = n_components
#     else:
#         p_reduced = p_np
#         q_reduced = q_np
#         combined_reduced = combined

#     distances = pairwise_distances(combined_reduced)
#     sigma = float(np.median(distances))
#     gamma = 1.0 / max(sigma, 1e-12)
#     k_mat = rbf_kernel(combined_reduced, combined_reduced, gamma=gamma)
#     n1 = p_np.shape[0]
#     observed_mmd = float(get_mmd_from_all_distances(k_mat, n1))
#     pval_mmd = float(run_mmd_permutation_test(p_reduced, q_reduced))

#     results.append(
#         ShiftTestResult_old(
#             metric="MMD",
#             layer=layer_name,
#             shift_name=shift,
#             stat=observed_mmd,
#             p_value=pval_mmd,
#             extra={
#                 "rbf_gamma": gamma,
#                 "bandwidth": sigma,
#                 "pca_applied": apply_pca,
#                 "pca_components": pca_dim if apply_pca else None,
#                 "bbsd_skipped": not is_probabilistic,
#             },
#         )
#     )

#     return results


# # -----------------------------------------
# # Energy, KL and Jensen‑Shannon divergences
# # -----------------------------------------
# def energy_distance(
#     X: np.ndarray,
#     Y: np.ndarray,
# ) -> float:
#     """
#     Compute the energy distance between two multivariate distributions.

#     Uses Euclidean distance. Suitable for real-valued embedding comparisons.

#     Args:
#         X (ndarray): Samples from distribution 1, shape (n_samples_1, d).
#         Y (ndarray): Samples from distribution 2, shape (n_samples_2, d).

#     Returns:
#         float: Energy distance.
#     """
#     m, n = len(X), len(Y)
#     if m == 0 or n == 0:
#         return 0.0
#     d_xy = cdist(X, Y, "euclidean").mean()
#     d_xx = cdist(X, X, "euclidean").mean()
#     d_yy = cdist(Y, Y, "euclidean").mean()
#     return 2.0 * d_xy - d_xx - d_yy


# def kl_divergence(
#     p: np.ndarray,
#     q: np.ndarray,
#     eps: float = 1e-8,
# ) -> float:
#     """
#     Compute average KL divergence between two probability distributions.

#     Assumes rows of 'p' and 'q' represent discrete distributions (rows sum to 1).

#     Args:
#         p (ndarray): Source probability vectors, shape (n_samples, n_classes).
#         q (ndarray): Target probability vectors, same shape.
#         eps (float): Small constant for numerical stability (avoid log(0)).

#     Returns:
#         float: KL(p || q)
#     """
#     p = np.clip(p, eps, 1)
#     q = np.clip(q, eps, 1)
#     return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


# def js_divergence(
#     p: np.ndarray,
#     q: np.ndarray,
#     eps: float = 1e-8,
# ) -> float:
#     """
#     Compute Jensen-Shannon divergence between two probability distributions.

#     Symmetric and bounded variant of KL divergence using base-e logarithms.

#     Args:
#         p (ndarray): First probability distribution, shape (n_samples, n_classes).
#         q (ndarray): Second probability distribution, same shape.
#         eps (float): Small constant to avoid numerical instability.

#     Returns:
#         float: JS(p, q)
#     """
#     m = 0.5 * (p + q)
#     return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


# # ---------------------------------------------------------------------------
# # Convenience wrapper to obtain Energy and KL stats in ShiftTestResult format
# # ---------------------------------------------------------------------------
# def calculate_energy_and_kl(
#     source_distribution: torch.Tensor | np.ndarray,
#     target_distribution: torch.Tensor | np.ndarray,
#     layer_name: str,
#     shift: str,
# ) -> list[ShiftTestResult_old]:
#     """
#     Compute Energy, KL, and Jensen-Shannon divergences between two distributions.

#     Assumes inputs are embeddings or probability vectors (for KL/JS).
#     KL and JS are only computed if inputs sum to approx. 1 across rows.

#     Args:
#         source_distribution (Tensor | ndarray): Source (validation) embeddings.
#         target_distribution (Tensor | ndarray): Target (shifted) embeddings.
#         layer_name (str): Encoder layer name or index.
#         shift (str): Identifier for the shift condition.

#     Returns:
#         List[ShiftTestResult]: Results for Energy, and if applicable, KL and JS.
#     """
#     if isinstance(source_distribution, torch.Tensor):
#         p_np = source_distribution.detach().cpu().numpy()
#     else:
#         p_np = np.asarray(source_distribution)

#     if isinstance(target_distribution, torch.Tensor):
#         q_np = target_distribution.detach().cpu().numpy()
#     else:
#         q_np = np.asarray(target_distribution)

#     results: list[ShiftTestResult_old] = []

#     # Energy distance (embeddings or logits)
#     stat_energy = energy_distance(p_np, q_np)
#     results.append(
#         ShiftTestResult_old(
#             metric="Energy",
#             layer=layer_name,
#             shift_name=shift,
#             stat=stat_energy,
#             p_value=None,
#             extra={},
#         )
#     )

#     # KL and JS expect *probability* vectors.
#     if np.allclose(p_np.sum(axis=1), 1, atol=1e-4) and np.allclose(
#         q_np.sum(axis=1), 1, atol=1e-4
#     ):
#         stat_kl = kl_divergence(p_np, q_np)
#         stat_js = js_divergence(p_np, q_np)
#         results.extend(
#             [
#                 ShiftTestResult_old(
#                     metric="KL",
#                     layer=layer_name,
#                     shift_name=shift,
#                     stat=stat_kl,
#                     p_value=None,
#                     extra={},
#                 ),
#                 ShiftTestResult_old(
#                     metric="JS",
#                     layer=layer_name,
#                     shift_name=shift,
#                     stat=stat_js,
#                     p_value=None,
#                     extra={},
#                 ),
#             ]
#         )
#     return results


# # --------------------------
# # Helper: Gather all metrics
# # --------------------------
# def calculate_all_shift_metrics(
#     source_distribution: torch.Tensor | np.ndarray,
#     target_distribution: torch.Tensor | np.ndarray,
#     layer_name: str,
#     shift: str,
#     apply_pca: bool = True,
#     pca_dim: int = 32,
# ) -> list[ShiftTestResult_old]:
#     """
#     Convenience wrapper to compute BBSD, MMD, Energy, KL, and JS shift metrics.

#     Combines all available statistical tests into a single result list.

#     Args:
#         source_distribution (Tensor | ndarray): Source (validation) embeddings.
#         target_distribution (Tensor | ndarray): Target (shifted) embeddings.
#         layer_name (str): Encoder layer name or index.
#         shift (str): Identifier for the shift condition.
#         apply_pca (bool): Whether to reduce dimensionality before MMD.
#         pca_dim (int): Number of PCA dimensions to retain.

#     Returns:
#         List[ShiftTestResult]: All shift metric results.
#     """
#     return calculate_bbsd_and_mmd(
#         source_distribution,
#         target_distribution,
#         layer_name,
#         shift,
#         apply_pca=apply_pca,
#         pca_dim=pca_dim,
#     ) + calculate_energy_and_kl(
#         source_distribution, target_distribution, layer_name, shift
#     )
