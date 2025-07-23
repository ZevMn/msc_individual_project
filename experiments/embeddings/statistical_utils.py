# experiments/embeddings/statistical_utils.py

import torch

from shift_identification_detection.bbsd_tests import run_bbsd
from shift_identification_detection.mmd_test import run_mmd_permutation_test

# -----------------------------
# BBSD and MMD analysis wrapper
# -----------------------------
def calculate_bbsd_and_mmd(
    source_distribution: torch.Tensor,
    target_distribution: torch.Tensor,
    layer_name: str,
    shift: str,
) -> None:
    """
    Run BBSD and MMD tests on two distributions and print the outcomes.

    Args:
        source_distribution: Reference/source embeddings (validation set)
        target_distribution: Target/test embeddings (shifted).
        layer_name: Name of the layer of the encoder being evaluated.
        shift: String identifier for the shift scenario (e.g., "no_shift", "acq").
    """

    if run_bbsd(source_distribution, target_distribution):
        print(f"BBSD positive for {shift} shift and {layer_name}")
    else:
        print(f"BBSD negative for {shift} shift and {layer_name}")

    if run_mmd_permutation_test(source_distribution, target_distribution):
        print(f"MMD positive for {shift} shift and {layer_name}\n")
    else:
        print(f"MMD negative for {shift} shift and {layer_name}\n")