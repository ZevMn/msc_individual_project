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

import numpy as np
import pandas as pd
import torch

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

from shift_identification_detection.bbsd_tests import run_bbsd
from shift_identification_detection.mmd_test import (
    run_mmd_permutation_test,
    get_mmd_from_all_distances,
)
from shift_identification_detection.shift_identification import (
    embed_patient_permutations,
)


@dataclass
class ShiftTestResult_old:
    """
    Container for storing the result of a statistical shift test.

    Attributes:
        metric (str): Name of the statistical test (e.g., "MMD", "BBSD", "KL").
        layer (str): Identifier for the encoder layer the test was run on.
        shift_name (str): Name of the shift scenario (e.g., "acq", "contrast").
        stat (float): The test statistic value.
        p_value (float | None): The associated p-value, or None if not applicable.
        extra (dict): Additional metadata (e.g., per-dimension stats).
    """

    metric: str
    layer: str
    shift_name: str
    stat: float
    p_value: float | None
    extra: dict


def save_results(results: list[ShiftTestResult_old], output_dir: Path) -> None:
    """
    Save a list of ShiftTestResult objects to disk in JSON and CSV formats.

    Args:
        results (list[ShiftTestResult]): The test results to save.
        output_dir (Path): File path to save the results.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(output_dir.with_suffix(".json"), "w") as f:
        json.dump(data, f, indent=2)
    pd.DataFrame(data).to_csv(output_dir.with_suffix(".csv"), index=False)


# -----------------------------
# BBSD and MMD analysis wrapper
# -----------------------------
def calculate_bbsd_and_mmd(
    source_distribution: torch.Tensor | np.ndarray,
    target_distribution: torch.Tensor | np.ndarray,
    layer_name: str,
    shift: str,
    apply_pca: bool = True,
    pca_dim: int = 32,
) -> list[ShiftTestResult_old]:
    """
    Compute BBSD and MMD statistics between source and target distributions.

    - BBSD: Only computed if inputs are softmax outputs (rows sum ≈ 1).
    - MMD: Always computed using RBF-kernel and permutation test.
           Optionally applies PCA to reduce dimensionality before MMD.

    Args:
        source_distribution (Tensor | ndarray): Source (validation) embeddings or softmax outputs.
        target_distribution (Tensor | ndarray): Target (shifted) embeddings or softmax outputs.
        layer_name (str): Encoder layer name or index.
        shift (str): Identifier for the shift condition.
        apply_pca (bool): If True, reduce dimensionality to 'pca_dim' before MMD.
        pca_dim (int): Number of PCA components to keep if PCA is applied.

    Returns:
        List[ShiftTestResult]: Results for BBSD (if applicable) and MMD.
    """
    results: list[ShiftTestResult_old] = []

    # Convert to NumPy
    if isinstance(source_distribution, torch.Tensor):
        p_np = source_distribution.detach().cpu().numpy()
    else:
        p_np = np.asarray(source_distribution)

    if isinstance(target_distribution, torch.Tensor):
        q_np = target_distribution.detach().cpu().numpy()
    else:
        q_np = np.asarray(target_distribution)

    # Check if inputs are softmax probability vectors
    is_probabilistic = np.allclose(p_np.sum(axis=1), 1.0, atol=1e-4) and np.allclose(
        q_np.sum(axis=1), 1.0, atol=1e-4
    )

    # BBSD (only if probabilistic inputs)
    if is_probabilistic:
        alpha = 0.05
        n_dim = p_np.shape[1]
        threshold = alpha / n_dim

        bbsd_output = run_bbsd(p_np, q_np, alpha=alpha, return_p_value=True)
        if isinstance(bbsd_output, tuple):
            bbsd_flag, all_p_values = bbsd_output
        else:
            bbsd_flag = bbsd_output
            all_p_values = np.ones(n_dim)

        pval_bbsd = float(np.min(all_p_values))

        results.append(
            ShiftTestResult_old(
                metric="BBSD",
                layer=layer_name,
                shift_name=shift,
                stat=pval_bbsd,
                p_value=pval_bbsd,
                extra={
                    "all_p_values": all_p_values.tolist(),
                    "alpha_corrected": threshold,
                    "rejected": bbsd_flag,
                },
            )
        )

    # MMD (always run)
    combined = np.concatenate([p_np, q_np], axis=0)
    if apply_pca:
        max_components = min(combined.shape[0], combined.shape[1])
        n_components = min(pca_dim, max_components)

        if n_components < 2:
            print(
                f"[Warning] Skipping PCA (n_components={n_components}) due to insufficient shape: {combined.shape}"
            )
            p_reduced = p_np
            q_reduced = q_np
            combined_reduced = combined
            apply_pca = False
        else:
            pca = PCA(n_components=n_components)
            combined_reduced = pca.fit_transform(combined)
            p_reduced = combined_reduced[: p_np.shape[0]]
            q_reduced = combined_reduced[p_np.shape[0] :]
            pca_dim = n_components
    else:
        p_reduced = p_np
        q_reduced = q_np
        combined_reduced = combined

    distances = pairwise_distances(combined_reduced)
    sigma = float(np.median(distances))
    gamma = 1.0 / max(sigma, 1e-12)
    k_mat = rbf_kernel(combined_reduced, combined_reduced, gamma=gamma)
    n1 = p_np.shape[0]
    observed_mmd = float(get_mmd_from_all_distances(k_mat, n1))
    pval_mmd = float(run_mmd_permutation_test(p_reduced, q_reduced))

    results.append(
        ShiftTestResult_old(
            metric="MMD",
            layer=layer_name,
            shift_name=shift,
            stat=observed_mmd,
            p_value=pval_mmd,
            extra={
                "rbf_gamma": gamma,
                "bandwidth": sigma,
                "pca_applied": apply_pca,
                "pca_components": pca_dim if apply_pca else None,
                "bbsd_skipped": not is_probabilistic,
            },
        )
    )

    return results


# -----------------------------------------
# Energy, KL and Jensen‑Shannon divergences
# -----------------------------------------
def energy_distance(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    Compute the energy distance between two multivariate distributions.

    Uses Euclidean distance. Suitable for real-valued embedding comparisons.

    Args:
        X (ndarray): Samples from distribution 1, shape (n_samples_1, d).
        Y (ndarray): Samples from distribution 2, shape (n_samples_2, d).

    Returns:
        float: Energy distance.
    """
    m, n = len(X), len(Y)
    if m == 0 or n == 0:
        return 0.0
    d_xy = cdist(X, Y, "euclidean").mean()
    d_xx = cdist(X, X, "euclidean").mean()
    d_yy = cdist(Y, Y, "euclidean").mean()
    return 2.0 * d_xy - d_xx - d_yy


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute average KL divergence between two probability distributions.

    Assumes rows of 'p' and 'q' represent discrete distributions (rows sum to 1).

    Args:
        p (ndarray): Source probability vectors, shape (n_samples, n_classes).
        q (ndarray): Target probability vectors, same shape.
        eps (float): Small constant for numerical stability (avoid log(0)).

    Returns:
        float: KL(p || q)
    """
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Symmetric and bounded variant of KL divergence using base-e logarithms.

    Args:
        p (ndarray): First probability distribution, shape (n_samples, n_classes).
        q (ndarray): Second probability distribution, same shape.
        eps (float): Small constant to avoid numerical instability.

    Returns:
        float: JS(p, q)
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


# ---------------------------------------------------------------------------
# Convenience wrapper to obtain Energy and KL stats in ShiftTestResult format
# ---------------------------------------------------------------------------
def calculate_energy_and_kl(
    source_distribution: torch.Tensor | np.ndarray,
    target_distribution: torch.Tensor | np.ndarray,
    layer_name: str,
    shift: str,
) -> list[ShiftTestResult_old]:
    """
    Compute Energy, KL, and Jensen-Shannon divergences between two distributions.

    Assumes inputs are embeddings or probability vectors (for KL/JS).
    KL and JS are only computed if inputs sum to approx. 1 across rows.

    Args:
        source_distribution (Tensor | ndarray): Source (validation) embeddings.
        target_distribution (Tensor | ndarray): Target (shifted) embeddings.
        layer_name (str): Encoder layer name or index.
        shift (str): Identifier for the shift condition.

    Returns:
        List[ShiftTestResult]: Results for Energy, and if applicable, KL and JS.
    """
    if isinstance(source_distribution, torch.Tensor):
        p_np = source_distribution.detach().cpu().numpy()
    else:
        p_np = np.asarray(source_distribution)

    if isinstance(target_distribution, torch.Tensor):
        q_np = target_distribution.detach().cpu().numpy()
    else:
        q_np = np.asarray(target_distribution)

    results: list[ShiftTestResult_old] = []

    # Energy distance (embeddings or logits)
    stat_energy = energy_distance(p_np, q_np)
    results.append(
        ShiftTestResult_old(
            metric="Energy",
            layer=layer_name,
            shift_name=shift,
            stat=stat_energy,
            p_value=None,
            extra={},
        )
    )

    # KL and JS expect *probability* vectors.
    if np.allclose(p_np.sum(axis=1), 1, atol=1e-4) and np.allclose(
        q_np.sum(axis=1), 1, atol=1e-4
    ):
        stat_kl = kl_divergence(p_np, q_np)
        stat_js = js_divergence(p_np, q_np)
        results.extend(
            [
                ShiftTestResult_old(
                    metric="KL",
                    layer=layer_name,
                    shift_name=shift,
                    stat=stat_kl,
                    p_value=None,
                    extra={},
                ),
                ShiftTestResult_old(
                    metric="JS",
                    layer=layer_name,
                    shift_name=shift,
                    stat=stat_js,
                    p_value=None,
                    extra={},
                ),
            ]
        )
    return results


# --------------------------
# Helper: Gather all metrics
# --------------------------


def calculate_all_shift_metrics(
    source_distribution: torch.Tensor | np.ndarray,
    target_distribution: torch.Tensor | np.ndarray,
    layer_name: str,
    shift: str,
    apply_pca: bool = True,
    pca_dim: int = 32,
) -> list[ShiftTestResult_old]:
    """
    Convenience wrapper to compute BBSD, MMD, Energy, KL, and JS shift metrics.

    Combines all available statistical tests into a single result list.

    Args:
        source_distribution (Tensor | ndarray): Source (validation) embeddings.
        target_distribution (Tensor | ndarray): Target (shifted) embeddings.
        layer_name (str): Encoder layer name or index.
        shift (str): Identifier for the shift condition.
        apply_pca (bool): Whether to reduce dimensionality before MMD.
        pca_dim (int): Number of PCA dimensions to retain.

    Returns:
        List[ShiftTestResult]: All shift metric results.
    """
    return calculate_bbsd_and_mmd(
        source_distribution,
        target_distribution,
        layer_name,
        shift,
        apply_pca=apply_pca,
        pca_dim=pca_dim,
    ) + calculate_energy_and_kl(
        source_distribution, target_distribution, layer_name, shift
    )


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

    bbsd_pvalue: float | None = None
    bbsd_is_significant: bool | None = None


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

    #######################################
    # load softmax probabilities for BBSD #
    #######################################

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
            print(f"cat_embeddings.shape[0]: {cat_embeddings.shape[0]}")
            print(f"cat_embeddings.shape[1]: {cat_embeddings.shape[1]}")
            n_components = min(32, cat_embeddings.shape[0], cat_embeddings.shape[1])
            pca = PCA(n_components=n_components)
            embeddings_32pca = pca.fit_transform(cat_embeddings.cpu().numpy())
            print("Starting the permutation test...")
            print("\n\n\n\n\n")

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
        # print(f"--- Calculating BBSD on softmax outputs ---")
        # bbsd_sig, bbsd_p = run_bbsd(
        #     probas_val, probas_test[idx_array], return_p_value=True
        # )
        # shift_result.bbsd_is_significant = bbsd_sig
        # shift_result.bbsd_pvalue = bbsd_p

        # Collect the results from all shift combinations
        results.append(shift_result)

        # Save the results
        data = [asdict(r) for r in results]
        json_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.json"
        csv_path = output_dir / f"{dataset}_{encoder_to_evaluate}_stats.csv"
        with open(json_path, "w") as jf:
            json.dump(data, jf, indent=2)
        pd.DataFrame(data).to_csv(csv_path, index=False)
