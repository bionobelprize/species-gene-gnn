"""
Unit tests for Species-Gene GNN components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
import torch

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer
from src.utils import split_data, normalize_similarities


class TestDataLoader:
    """Test cases for SpeciesGeneDataLoader."""
    
    def test_load_data_from_dataframe(self):
        """Test loading data from DataFrame."""
        data = pd.DataFrame({
            'species_a': ['Human', 'Mouse'],
            'gene_a': ['BRCA1', 'Brca1'],
            'species_b': ['Mouse', 'Rat'],
            'gene_b': ['Brca1', 'Brca1'],
            'similarity': [0.89, 0.92]
        })
        
        loader = SpeciesGeneDataLoader()
        df = loader.load_data(data=data)
        
        assert len(df) == 2
        assert all(col in df.columns for col in ['species_a', 'gene_a', 'species_b', 'gene_b', 'similarity'])
    
    def test_build_mappings(self):
        """Test building species and gene mappings."""
        data = pd.DataFrame({
            'species_a': ['Human', 'Mouse', 'Human'],
            'gene_a': ['BRCA1', 'Brca1', 'TP53'],
            'species_b': ['Mouse', 'Rat', 'Mouse'],
            'gene_b': ['Brca1', 'Brca1', 'Tp53'],
            'similarity': [0.89, 0.92, 0.95]
        })
        
        loader = SpeciesGeneDataLoader()
        df = loader.load_data(data=data)
        loader.build_mappings(df)
        
        assert loader.get_num_species() == 3  # Human, Mouse, Rat
        assert loader.get_num_genes() == 4    # BRCA1, Brca1, TP53, Tp53
    
    def test_encode_data(self):
        """Test encoding species and genes to IDs."""
        data = pd.DataFrame({
            'species_a': ['Human', 'Mouse'],
            'gene_a': ['BRCA1', 'Brca1'],
            'species_b': ['Mouse', 'Rat'],
            'gene_b': ['Brca1', 'Tp53'],
            'similarity': [0.89, 0.92]
        })
        
        loader = SpeciesGeneDataLoader()
        df = loader.load_data(data=data)
        loader.build_mappings(df)
        
        species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
            loader.encode_data(df)
        
        assert len(species_a_ids) == 2
        assert len(gene_a_ids) == 2
        assert len(species_b_ids) == 2
        assert len(gene_b_ids) == 2
        assert len(similarities) == 2
        assert all(isinstance(x, (int, np.integer)) for x in species_a_ids)


class TestGraphBuilder:
    """Test cases for HeteroGraphBuilder."""
    
    def test_build_graph(self):
        """Test building heterogeneous graph."""
        builder = HeteroGraphBuilder()
        
        species_a_ids = np.array([0, 1])
        gene_a_ids = np.array([0, 1])
        species_b_ids = np.array([1, 0])
        gene_b_ids = np.array([1, 2])
        similarities = np.array([0.89, 0.92])
        
        data = builder.build_graph(
            species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
            num_species=2, num_genes=3
        )
        
        assert 'species' in data.node_types
        assert 'gene' in data.node_types
        assert ('species', 'has_gene', 'gene') in data.edge_types
        assert ('gene', 'belongs_to', 'species') in data.edge_types
        assert ('gene', 'similar_to', 'gene') in data.edge_types
    
    def test_get_edge_label_index(self):
        """Test getting edge label indices."""
        builder = HeteroGraphBuilder()
        
        gene_a_ids = np.array([0, 1, 2])
        gene_b_ids = np.array([1, 2, 0])
        
        edge_index = builder.get_edge_label_index(gene_a_ids, gene_b_ids)
        
        assert edge_index.shape == (2, 3)
        assert edge_index.dtype == torch.long


class TestModel:
    """Test cases for SpeciesGeneGNN model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = SpeciesGeneGNN(
            num_species=5,
            num_genes=100,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2
        )
        
        assert model.num_species == 5
        assert model.num_genes == 100
        assert model.embedding_dim == 32
        assert model.hidden_dim == 64
    
    def test_model_forward(self):
        """Test forward pass."""
        from torch_geometric.data import HeteroData
        
        model = SpeciesGeneGNN(
            num_species=2,
            num_genes=3,
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1
        )
        
        # Create minimal graph
        data = HeteroData()
        data['species'].num_nodes = 2
        data['gene'].num_nodes = 3
        
        # Add some edges
        data['species', 'has_gene', 'gene'].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        data['gene', 'belongs_to', 'species'].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        data['gene', 'similar_to', 'gene'].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        # Forward pass
        x_dict = model.forward(data)
        
        assert 'species' in x_dict
        assert 'gene' in x_dict
        assert x_dict['species'].shape == (2, 32)
        assert x_dict['gene'].shape == (3, 32)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_split_data(self):
        """Test data splitting."""
        gene_a_ids = np.arange(100)
        gene_b_ids = np.arange(100, 200)
        similarities = np.random.rand(100)
        
        train_data, val_data, test_data = split_data(
            gene_a_ids, gene_b_ids, similarities,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        train_gene_a, train_gene_b, train_sim = train_data
        val_gene_a, val_gene_b, val_sim = val_data
        test_gene_a, test_gene_b, test_sim = test_data
        
        # Allow for rounding differences in split sizes
        assert len(train_gene_a) in [69, 70, 71]
        assert len(val_gene_a) in [14, 15, 16]
        assert len(test_gene_a) in [14, 15, 16]
        assert len(train_gene_a) + len(val_gene_a) + len(test_gene_a) == 100
    
    def test_normalize_similarities(self):
        """Test similarity normalization."""
        similarities = np.array([0.1, 0.5, 0.9])
        
        # Test minmax normalization
        normalized = normalize_similarities(similarities, method='minmax')
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        
        # Test zscore normalization
        normalized = normalize_similarities(similarities, method='zscore')
        assert abs(normalized.mean()) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
