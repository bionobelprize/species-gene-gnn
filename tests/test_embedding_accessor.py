"""
Unit tests for EmbeddingAccessor.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.embedding_accessor import EmbeddingAccessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'species_a': ['Human', 'Mouse', 'Human', 'Rat'],
        'gene_a': ['BRCA1', 'Brca1', 'TP53', 'Tp53'],
        'species_b': ['Mouse', 'Rat', 'Mouse', 'Human'],
        'gene_b': ['Brca1', 'Brca1', 'Tp53', 'TP53'],
        'similarity': [0.89, 0.92, 0.95, 0.88]
    })
    return data


@pytest.fixture
def data_loader(sample_data):
    """Create and initialize data loader."""
    loader = SpeciesGeneDataLoader()
    df = loader.load_data(data=sample_data)
    loader.build_mappings(df)
    return loader


@pytest.fixture
def hetero_data(sample_data, data_loader):
    """Create heterogeneous graph."""
    df = data_loader.load_data(data=sample_data)
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    builder = HeteroGraphBuilder()
    data = builder.build_graph(
        species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
        data_loader.get_num_species(), data_loader.get_num_genes()
    )
    return data


@pytest.fixture
def model(data_loader):
    """Create a simple model."""
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1
    )
    return model


@pytest.fixture
def accessor(model, data_loader):
    """Create embedding accessor."""
    return EmbeddingAccessor(model, data_loader, device='cpu')


class TestEmbeddingAccessor:
    """Test cases for EmbeddingAccessor."""
    
    def test_initialization(self, accessor, model, data_loader):
        """Test accessor initialization."""
        assert accessor.model == model
        assert accessor.data_loader == data_loader
        assert accessor.device == 'cpu'
    
    def test_get_raw_species_embedding_by_name(self, accessor):
        """Test getting raw species embedding by name."""
        embedding = accessor.get_raw_species_embedding('Human')
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (16,)  # embedding_dim
        assert not torch.isnan(embedding).any()
    
    def test_get_raw_species_embedding_by_id(self, accessor):
        """Test getting raw species embedding by ID."""
        embedding = accessor.get_raw_species_embedding(0)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (16,)
    
    def test_get_raw_species_embedding_as_numpy(self, accessor):
        """Test getting raw species embedding as numpy array."""
        embedding = accessor.get_raw_species_embedding('Human', as_numpy=True)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (16,)
    
    def test_get_raw_species_embedding_invalid_name(self, accessor):
        """Test getting raw species embedding with invalid name."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_raw_species_embedding('InvalidSpecies')
    
    def test_get_raw_species_embedding_invalid_id(self, accessor):
        """Test getting raw species embedding with invalid ID."""
        with pytest.raises(ValueError, match="out of range"):
            accessor.get_raw_species_embedding(100)
    
    def test_get_raw_gene_embedding_by_name(self, accessor):
        """Test getting raw gene embedding by name."""
        embedding = accessor.get_raw_gene_embedding('BRCA1')
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (16,)
        assert not torch.isnan(embedding).any()
    
    def test_get_raw_gene_embedding_by_id(self, accessor):
        """Test getting raw gene embedding by ID."""
        embedding = accessor.get_raw_gene_embedding(0)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (16,)
    
    def test_get_raw_gene_embedding_as_numpy(self, accessor):
        """Test getting raw gene embedding as numpy array."""
        embedding = accessor.get_raw_gene_embedding('BRCA1', as_numpy=True)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (16,)
    
    def test_get_raw_gene_embedding_invalid_name(self, accessor):
        """Test getting raw gene embedding with invalid name."""
        with pytest.raises(ValueError, match="not found"):
            accessor.get_raw_gene_embedding('InvalidGene')
    
    def test_get_raw_gene_embedding_invalid_id(self, accessor):
        """Test getting raw gene embedding with invalid ID."""
        with pytest.raises(ValueError, match="out of range"):
            accessor.get_raw_gene_embedding(100)
    
    def test_get_transformed_species_embedding(self, accessor, hetero_data):
        """Test getting transformed species embedding."""
        embedding = accessor.get_transformed_species_embedding('Human', hetero_data)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (32,)  # hidden_dim
        assert not torch.isnan(embedding).any()
    
    def test_get_transformed_gene_embedding(self, accessor, hetero_data):
        """Test getting transformed gene embedding."""
        embedding = accessor.get_transformed_gene_embedding('BRCA1', hetero_data)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (32,)  # hidden_dim
        assert not torch.isnan(embedding).any()
    
    def test_get_transformed_embedding_cache(self, accessor, hetero_data):
        """Test that transformed embeddings are cached."""
        # First call should compute
        emb1 = accessor.get_transformed_gene_embedding('BRCA1', hetero_data)
        
        # Second call should use cache (same result)
        emb2 = accessor.get_transformed_gene_embedding('BRCA1', hetero_data)
        
        assert torch.allclose(emb1, emb2)
    
    def test_get_batch_raw_embeddings_species(self, accessor):
        """Test getting batch of raw species embeddings."""
        species_list = ['Human', 'Mouse', 'Rat']
        embeddings = accessor.get_batch_raw_embeddings('species', species_list)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (3, 16)  # 3 species, embedding_dim=16
    
    def test_get_batch_raw_embeddings_genes(self, accessor):
        """Test getting batch of raw gene embeddings."""
        gene_list = ['BRCA1', 'TP53']
        embeddings = accessor.get_batch_raw_embeddings('gene', gene_list)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (2, 16)  # 2 genes, embedding_dim=16
    
    def test_get_batch_raw_embeddings_as_numpy(self, accessor):
        """Test getting batch embeddings as numpy."""
        species_list = ['Human', 'Mouse']
        embeddings = accessor.get_batch_raw_embeddings('species', species_list, as_numpy=True)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 16)
    
    def test_get_batch_raw_embeddings_invalid_type(self, accessor):
        """Test getting batch embeddings with invalid type."""
        with pytest.raises(ValueError, match="Unknown entity_type"):
            accessor.get_batch_raw_embeddings('invalid', ['Human'])
    
    def test_get_batch_transformed_embeddings_species(self, accessor, hetero_data):
        """Test getting batch of transformed species embeddings."""
        species_list = ['Human', 'Mouse']
        embeddings = accessor.get_batch_transformed_embeddings('species', species_list, hetero_data)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (2, 32)  # 2 species, hidden_dim=32
    
    def test_get_batch_transformed_embeddings_genes(self, accessor, hetero_data):
        """Test getting batch of transformed gene embeddings."""
        gene_list = ['BRCA1', 'TP53']
        embeddings = accessor.get_batch_transformed_embeddings('gene', gene_list, hetero_data)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (2, 32)  # 2 genes, hidden_dim=32
    
    def test_compute_similarity_cosine(self, accessor):
        """Test computing cosine similarity."""
        emb1 = accessor.get_raw_gene_embedding('BRCA1')
        emb2 = accessor.get_raw_gene_embedding('TP53')
        
        similarity = accessor.compute_similarity(emb1, emb2, method='cosine')
        
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1  # Cosine similarity range
    
    def test_compute_similarity_euclidean(self, accessor):
        """Test computing euclidean similarity."""
        emb1 = accessor.get_raw_gene_embedding('BRCA1')
        emb2 = accessor.get_raw_gene_embedding('TP53')
        
        similarity = accessor.compute_similarity(emb1, emb2, method='euclidean')
        
        assert isinstance(similarity, float)
        assert similarity <= 0  # Negative distance
    
    def test_compute_similarity_dot(self, accessor):
        """Test computing dot product similarity."""
        emb1 = accessor.get_raw_gene_embedding('BRCA1')
        emb2 = accessor.get_raw_gene_embedding('TP53')
        
        similarity = accessor.compute_similarity(emb1, emb2, method='dot')
        
        assert isinstance(similarity, float)
    
    def test_compute_similarity_with_numpy(self, accessor):
        """Test computing similarity with numpy arrays."""
        emb1 = accessor.get_raw_gene_embedding('BRCA1', as_numpy=True)
        emb2 = accessor.get_raw_gene_embedding('TP53', as_numpy=True)
        
        similarity = accessor.compute_similarity(emb1, emb2, method='cosine')
        
        assert isinstance(similarity, float)
    
    def test_compute_similarity_invalid_method(self, accessor):
        """Test computing similarity with invalid method."""
        emb1 = accessor.get_raw_gene_embedding('BRCA1')
        emb2 = accessor.get_raw_gene_embedding('TP53')
        
        with pytest.raises(ValueError, match="Unknown similarity method"):
            accessor.compute_similarity(emb1, emb2, method='invalid')
    
    def test_find_nearest_neighbors_genes(self, accessor, hetero_data):
        """Test finding nearest neighbors for a gene."""
        neighbors = accessor.find_nearest_neighbors(
            'BRCA1',
            entity_type='gene',
            data=hetero_data,
            k=2,
            use_transformed=True
        )
        
        assert len(neighbors) == 2
        assert all(isinstance(name, str) and isinstance(score, float) for name, score in neighbors)
        assert neighbors[0][1] >= neighbors[1][1]  # Sorted by similarity
        
        # Check that BRCA1 is not in its own neighbors
        neighbor_names = [name for name, _ in neighbors]
        assert 'BRCA1' not in neighbor_names
    
    def test_find_nearest_neighbors_species(self, accessor, hetero_data):
        """Test finding nearest neighbors for a species."""
        neighbors = accessor.find_nearest_neighbors(
            'Human',
            entity_type='species',
            data=hetero_data,
            k=2,
            use_transformed=True
        )
        
        assert len(neighbors) == 2
        neighbor_names = [name for name, _ in neighbors]
        assert 'Human' not in neighbor_names
    
    def test_find_nearest_neighbors_raw_embeddings(self, accessor, hetero_data):
        """Test finding nearest neighbors using raw embeddings."""
        neighbors = accessor.find_nearest_neighbors(
            'BRCA1',
            entity_type='gene',
            data=hetero_data,
            k=2,
            use_transformed=False
        )
        
        assert len(neighbors) == 2
    
    def test_find_nearest_neighbors_with_id(self, accessor, hetero_data):
        """Test finding nearest neighbors using entity ID."""
        neighbors = accessor.find_nearest_neighbors(
            0,  # First gene ID
            entity_type='gene',
            data=hetero_data,
            k=2
        )
        
        assert len(neighbors) == 2
    
    def test_get_all_embeddings_species_raw(self, accessor):
        """Test getting all raw species embeddings."""
        embeddings = accessor.get_all_embeddings('species', use_transformed=False)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (accessor.model.num_species, 16)
    
    def test_get_all_embeddings_genes_raw(self, accessor):
        """Test getting all raw gene embeddings."""
        embeddings = accessor.get_all_embeddings('gene', use_transformed=False)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (accessor.model.num_genes, 16)
    
    def test_get_all_embeddings_species_transformed(self, accessor, hetero_data):
        """Test getting all transformed species embeddings."""
        embeddings = accessor.get_all_embeddings('species', data=hetero_data, use_transformed=True)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (accessor.model.num_species, 32)
    
    def test_get_all_embeddings_genes_transformed(self, accessor, hetero_data):
        """Test getting all transformed gene embeddings."""
        embeddings = accessor.get_all_embeddings('gene', data=hetero_data, use_transformed=True)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (accessor.model.num_genes, 32)
    
    def test_get_all_embeddings_as_numpy(self, accessor):
        """Test getting all embeddings as numpy."""
        embeddings = accessor.get_all_embeddings('gene', use_transformed=False, as_numpy=True)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (accessor.model.num_genes, 16)
    
    def test_get_all_embeddings_transformed_without_data(self, accessor):
        """Test getting transformed embeddings without providing data."""
        with pytest.raises(ValueError, match="data must be provided"):
            accessor.get_all_embeddings('gene', use_transformed=True)
    
    def test_get_all_embeddings_invalid_type(self, accessor):
        """Test getting all embeddings with invalid type."""
        with pytest.raises(ValueError, match="Unknown entity_type"):
            accessor.get_all_embeddings('invalid', use_transformed=False)
    
    def test_get_embedding_info(self, accessor, data_loader):
        """Test getting embedding information."""
        info = accessor.get_embedding_info()
        
        assert 'num_species' in info
        assert 'num_genes' in info
        assert 'raw_embedding_dim' in info
        assert 'transformed_embedding_dim' in info
        assert 'species_names' in info
        assert 'gene_names' in info
        
        assert info['num_species'] == data_loader.get_num_species()
        assert info['num_genes'] == data_loader.get_num_genes()
        assert info['raw_embedding_dim'] == 16
        assert info['transformed_embedding_dim'] == 32
        assert isinstance(info['species_names'], list)
        assert isinstance(info['gene_names'], list)
    
    def test_cache_invalidation(self, accessor, hetero_data):
        """Test cache invalidation."""
        # Compute embeddings (caches them)
        emb1 = accessor.get_transformed_gene_embedding('BRCA1', hetero_data)
        
        # Invalidate cache
        accessor._invalidate_cache()
        
        # Compute again (should recompute)
        emb2 = accessor.get_transformed_gene_embedding('BRCA1', hetero_data)
        
        # Results should be the same
        assert torch.allclose(emb1, emb2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
