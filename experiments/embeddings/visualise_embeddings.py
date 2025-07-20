""" 
experiments/embeddings/visualise_embeddings.py
"""

from pathlib import Path

from config import Config

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from typing import Sequence


# -----------------------------------
# PCA and t-SNE analysis and plots
# -----------------------------------
def process_and_visualise_layer(
        output_dir: Path,
        encoder_to_evaluate: str,
        dataset: str,
        layer_name: str, 
        features: torch.Tensor, 
        labels: dict[str, np.ndarray], 
        shift: str="no_shift", 
        seed: int=Config.SEED, 
        pca_components: int=2,
        num_samples: int=1000
    ) -> None:
    """
    Processes data, reduces dimensionality, and visualises layer features.

    Args:
        layer_name: The name of the layer being processed.
        features: The feature embeddings for a given layer of the encoder.
        labels: A list containing sets of labels for the data (e.g. class labels, laterality, manufacturer).
        scenario: A string identifier for the experimental scenario ("final", "early", "all").
        shifted: A string indicating the type of shift ("no_shift", "acq", "prev", or "acq_prev").
        seed: Random seed for reproducibility.
        pca_components: The number of principal components to reduce to.
        num_samples: The number of points to include in the final plot.
    """

    if features.is_cuda: # Safety check (that features is not on GPU)
        features = features.cpu()
    embeddings = features.numpy() # Convert PyTorch tensor to numpy array for processing
    print(f"[{layer_name}] Original shape: {embeddings.shape}") # Should be 2D

    # # Early subsampling
    # rng = np.random.default_rng(seed)
    # sample_idx = rng.choice(len(embeddings),
    #                         size=min(num_samples, len(embeddings)),
    #                         replace=False)
    # embeddings = embeddings[sample_idx]
    # labels = [lbl[sample_idx] for lbl in labels]

    # PCA reduction
    #pca = PCA(n_components=0.95, whiten=False, random_state=seed) # PCA embedding that preserves 95% of the variance of the input data
    pca = PCA(n_components=pca_components, whiten=False, random_state=Config.SEED)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"[{layer_name}] PCA shape: {embeddings_pca.shape}")
    print(f"[{layer_name}] PCA explained variance ratio: {pca.explained_variance_ratio_[:2]}")

    # Use the t-SNE algorithm on PCA-reduced features to obtain a 2D embedding for input data
    embeddings_tsne = TSNE(n_components=2, 
                           init='random',
                           learning_rate='auto',
                           random_state=Config.SEED).fit_transform(embeddings_pca)
    print(f"[{layer_name}] t-SNE shape: {embeddings_tsne.shape}")
    
    # Create a pandas DataFrame to process data
    feature_attributes = Config.DATASET_CONFIG[dataset]["plot_columns"]
    df = pd.DataFrame({label: np_array for label, np_array in zip(feature_attributes, labels)})

    # Add PCA components and t-SNE components to the DataFrame
    for i in range(pca_components):
        df[f"{layer_name} - PCA {i+1}"] = embeddings_pca[:,i]
    df[f"{layer_name} - t-SNE 1"] = embeddings_tsne[:,0]
    df[f"{layer_name} - t-SNE 2"] = embeddings_tsne[:,1]

    # Sample for plotting
    sample = df.sample(n=min(num_samples, len(df)), random_state=seed)

    # Create plots
    sns.set_theme(style="white") # For cleaner appearance

    fig, axes = plt.subplots(len(feature_attributes), 2, figsize=Config.FIGURE_SIZE, constrained_layout=True)

    for i, label_type in enumerate(feature_attributes):

        # PCA plot (left column)
        ax_pca = sns.scatterplot(
            data=sample, 
            x=f"{layer_name} - PCA 1", 
            y=f"{layer_name} - PCA 2", 
            hue=label_type, 
            alpha=Config.ALPHA, 
            marker=Config.MARKER, 
            s=Config.MARKER_SIZE, 
            palette=Config.COLOR_PALETTE, 
            ax=axes[i, 0])
        ax_pca.set_title(f"PCA coloured by {label_type}")

        # t-SNE plot
        ax_tsne = sns.scatterplot(
            data=sample, 
            x=f"{layer_name} - t-SNE 1", 
            y=f"{layer_name} - t-SNE 2", 
            hue=label_type, 
            alpha=Config.ALPHA, 
            marker=Config.MARKER, 
            s=Config.MARKER_SIZE, 
            palette=Config.COLOR_PALETTE, 
            ax=axes[i, 1])
        ax_tsne.set_title(f"t-SNE coloured by {label_type}")

    for ax in axes.ravel():
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1,1))

    fig.suptitle(f"{dataset} | Scenario: {shift} - {layer_name}", fontsize=16)

    # Save the figure
    file_location = output_dir / f"{shift}_{layer_name}_{encoder_to_evaluate}.png"
    fig.savefig(file_location)
    plt.close(fig)


def aggregate_and_plot_shifted_features(
    output_dir: Path,  
    encoder_to_evaluate: str, 
    dataset: str,
    layer_name: str,
    reference_features: torch.Tensor, 
    shifted_features_dict: dict,
    reference_labels: list, 
    shifted_labels_dict: dict,
) -> None:
    """
    Aggregates features from reference dataset and shifted datasets and plots 
    then plots PCA and t-SNE visualisations to compare how the feature spaces and
    learnt representations differ between different layers of the encoder.

    Args:
        output_dir: Path to the directory where the plots will be saved.
        encoder_to_evaluate: Encoder used to generate features.
        dataset: Dataset name.
        layer_name: Name of the layer to be processed.
        reference_features: Feature embeddings from the reference dataset.
        shifted_features_dict: A dictionary where keys are shift names, and values are shifted features.
        reference_labels: Labels corresponding to the reference dataset features.
        shifted_labels_dict: A dictionary where keys are shift names, and values are labels for shifted features.
    """
    # Concatenate the reference features and shifted features
    all_features = [reference_features]  # Start with reference
    all_labels = [reference_labels]
    shifts = ['no_shift']  # Reference shift name
    for shift_name, shifted_features in shifted_features_dict.items():
        all_features.append(shifted_features)
        all_labels.append(shifted_labels_dict[shift_name])
        shifts.append(shift_name)

    # Convert features to numpy for PCA/t-SNE processing
    all_features = np.concatenate([features.numpy() for features in all_features], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # PCA reduction
    pca = PCA(n_components=2, whiten=False, random_state=Config.SEED)
    all_features_pca = pca.fit_transform(all_features)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, init='random', learning_rate='auto', random_state=Config.SEED)
    all_features_tsne = tsne.fit_transform(all_features_pca)

    # Create DataFrame for easy plotting
    df = pd.DataFrame({
        'PCA 1': all_features_pca[:, 0],
        'PCA 2': all_features_pca[:, 1],
        't-SNE 1': all_features_tsne[:, 0],
        't-SNE 2': all_features_tsne[:, 1],
        'Shift': np.array(shifts * [len(features) for features in all_features]).flatten(),
        'Labels': all_labels
    })

    # Create PCA and t-SNE plots
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=Config.FIGURE_SIZE, constrained_layout=True)

    # PCA plot
    sns.scatterplot(data=df, x='PCA 1', y='PCA 2', hue='Shift', style='Shift', markers={'no_shift': 'o', 'acq': 'X', 'prev': 's'}, palette='tab10', ax=axes[0], alpha=0.8, s=Config.MARKER_SIZE)
    axes[0].set_title(f"PCA - {layer_name} ({dataset})")
    axes[0].legend(title="Shift", bbox_to_anchor=(1.05, 1), loc='upper left')

    # t-SNE plot
    sns.scatterplot(data=df, x='t-SNE 1', y='t-SNE 2', hue='Shift', style='Shift', markers={'no_shift': 'o', 'acq': 'X', 'prev': 's'}, palette='tab10', ax=axes[1], alpha=0.8, s=Config.MARKER_SIZE)
    axes[1].set_title(f"t-SNE - {layer_name} ({dataset})")
    axes[1].legend(title="Shift", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the figure
    file_location = output_dir / f"shift_comparison_{layer_name}_{encoder_to_evaluate}.png"
    fig.savefig(file_location)
    plt.close(fig)