"""
Species-Gene Association Graph Neural Network Model.

This package implements a heterogeneous GNN for learning and predicting
cross-species gene functional similarities.
"""

from .data_loader import SpeciesGeneDataLoader
from .graph_builder import HeteroGraphBuilder
from .model import SpeciesGeneGNN
from .trainer import Trainer

__all__ = [
    'SpeciesGeneDataLoader',
    'HeteroGraphBuilder',
    'SpeciesGeneGNN',
    'Trainer'
]
