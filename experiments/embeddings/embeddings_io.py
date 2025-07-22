"""
experiments/embeddings/embeddings_io.py
"""

import pickle
from pathlib import Path

import torch


# -----------------------------------
# Load embeddings with error handling
# -----------------------------------
def load_embeddings_pkl(file_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load a pickled file containing a mapping of "train" and "test" splits
    to the corresponding embeddings, grouped by layer of the encoder.

    Raises:
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file isn't a valid pickle.
        IOError
            For other I/O related errors.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    except pickle.UnpicklingError as e:
        raise ValueError(f"Invalid pickle file format: {e}")
    except (IOError, OSError) as e:
        raise IOError(f"Error reading pickle file: {e}")
