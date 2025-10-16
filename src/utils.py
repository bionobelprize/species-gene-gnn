"""
Utility functions for data processing and splitting.
"""

import numpy as np
import torch
from typing import Tuple
from sklearn.model_selection import train_test_split


def split_data(
    gene_a_ids: np.ndarray,
    gene_b_ids: np.ndarray,
    similarities: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        gene_a_ids: Array of gene A IDs
        gene_b_ids: Array of gene B IDs
        similarities: Array of similarity scores
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data), where each is
        (gene_a_ids, gene_b_ids, similarities)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Create indices
    indices = np.arange(len(similarities))
    
    # Split into train and temp (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        random_state=random_seed
    )
    
    # Split temp into val and test
    if val_ratio > 0 and test_ratio > 0:
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_size),
            random_state=random_seed
        )
    elif val_ratio > 0:
        val_idx = temp_idx
        test_idx = np.array([])
    else:
        val_idx = np.array([])
        test_idx = temp_idx
    
    # Create splits
    train_data = (
        gene_a_ids[train_idx],
        gene_b_ids[train_idx],
        similarities[train_idx]
    )
    
    val_data = (
        gene_a_ids[val_idx] if len(val_idx) > 0 else np.array([]),
        gene_b_ids[val_idx] if len(val_idx) > 0 else np.array([]),
        similarities[val_idx] if len(val_idx) > 0 else np.array([])
    )
    
    test_data = (
        gene_a_ids[test_idx] if len(test_idx) > 0 else np.array([]),
        gene_b_ids[test_idx] if len(test_idx) > 0 else np.array([]),
        similarities[test_idx] if len(test_idx) > 0 else np.array([])
    )
    
    return train_data, val_data, test_data


def negative_sampling(
    num_genes: int,
    positive_edges: np.ndarray,
    num_negative: int,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generate negative samples for link prediction.
    
    Args:
        num_genes: Total number of genes
        positive_edges: Array of positive edges [2, num_pos_edges]
        num_negative: Number of negative samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of negative edges [2, num_negative]
    """
    np.random.seed(random_seed)
    
    # Create set of positive edges for fast lookup
    positive_set = set(map(tuple, positive_edges.T))
    
    # Generate negative samples
    negative_edges = []
    while len(negative_edges) < num_negative:
        gene_a = np.random.randint(0, num_genes)
        gene_b = np.random.randint(0, num_genes)
        
        # Check if this is not a positive edge and not a self-loop
        if (gene_a, gene_b) not in positive_set and gene_a != gene_b:
            negative_edges.append([gene_a, gene_b])
    
    return np.array(negative_edges).T


def normalize_similarities(
    similarities: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize similarity scores.
    
    Args:
        similarities: Array of similarity scores
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized similarity scores
    """
    if method == 'minmax':
        min_val = similarities.min()
        max_val = similarities.max()
        if max_val - min_val > 0:
            return (similarities - min_val) / (max_val - min_val)
        else:
            return similarities
    elif method == 'zscore':
        mean_val = similarities.mean()
        std_val = similarities.std()
        if std_val > 0:
            return (similarities - mean_val) / std_val
        else:
            return similarities - mean_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
