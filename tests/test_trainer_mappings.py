"""
Unit tests for Trainer save/load with species and gene mappings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
import torch
import tempfile
from pathlib import Path

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer


class TestTrainerMappings:
    """Test cases for saving and loading species/gene mappings with model checkpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'species_a': ['Human', 'Mouse', 'Human', 'Rat'],
            'gene_a': ['BRCA1', 'Brca1', 'TP53', 'Tp53'],
            'species_b': ['Mouse', 'Rat', 'Mouse', 'Human'],
            'gene_b': ['Brca1', 'Brca1', 'Tp53', 'TP53'],
            'similarity': [0.89, 0.92, 0.95, 0.88]
        })
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_model_with_mappings(self):
        """Test that model saves with species and gene mappings."""
        # Setup data loader and build mappings
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data=self.sample_data)
        data_loader.build_mappings(df)
        
        # Create a simple model
        model = SpeciesGeneGNN(
            num_species=data_loader.get_num_species(),
            num_genes=data_loader.get_num_genes(),
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        # Create trainer with data_loader
        trainer = Trainer(model=model, data_loader=data_loader)
        
        # Save model
        model_path = self.temp_path / "test_model.pt"
        trainer.save_model(str(model_path))
        
        # Verify file was created
        assert model_path.exists()
        
        # Load checkpoint and verify mappings are present
        checkpoint = torch.load(str(model_path))
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'id_to_species' in checkpoint
        assert 'id_to_gene' in checkpoint
        
        # Verify mappings content
        assert len(checkpoint['id_to_species']) == data_loader.get_num_species()
        assert len(checkpoint['id_to_gene']) == data_loader.get_num_genes()
        
        # Verify specific mappings
        assert checkpoint['id_to_species'] == data_loader.id_to_species
        assert checkpoint['id_to_gene'] == data_loader.id_to_gene
    
    def test_save_model_without_data_loader(self):
        """Test that model saves without mappings when data_loader is not provided."""
        # Create a simple model
        model = SpeciesGeneGNN(
            num_species=3,
            num_genes=4,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        # Create trainer without data_loader
        trainer = Trainer(model=model)
        
        # Save model
        model_path = self.temp_path / "test_model_no_loader.pt"
        trainer.save_model(str(model_path))
        
        # Verify file was created
        assert model_path.exists()
        
        # Load checkpoint and verify mappings are not present
        checkpoint = torch.load(str(model_path))
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'id_to_species' not in checkpoint
        assert 'id_to_gene' not in checkpoint
    
    def test_load_model_with_mappings(self):
        """Test that model loads correctly and returns mappings."""
        # Setup and save a model with mappings
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data=self.sample_data)
        data_loader.build_mappings(df)
        
        model = SpeciesGeneGNN(
            num_species=data_loader.get_num_species(),
            num_genes=data_loader.get_num_genes(),
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        trainer = Trainer(model=model, data_loader=data_loader)
        model_path = self.temp_path / "test_model.pt"
        trainer.save_model(str(model_path))
        
        # Create a new model and trainer to load into
        new_model = SpeciesGeneGNN(
            num_species=data_loader.get_num_species(),
            num_genes=data_loader.get_num_genes(),
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        new_trainer = Trainer(model=new_model)
        id_to_species, id_to_gene = new_trainer.load_model(str(model_path))
        
        # Verify mappings were returned
        assert id_to_species is not None
        assert id_to_gene is not None
        
        # Verify mappings content
        assert len(id_to_species) == data_loader.get_num_species()
        assert len(id_to_gene) == data_loader.get_num_genes()
        
        # Verify specific mappings match original
        assert id_to_species == data_loader.id_to_species
        assert id_to_gene == data_loader.id_to_gene
    
    def test_load_model_without_mappings(self):
        """Test that model loads correctly when no mappings are in checkpoint."""
        # Create and save a model without data_loader
        model = SpeciesGeneGNN(
            num_species=3,
            num_genes=4,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        trainer = Trainer(model=model)
        model_path = self.temp_path / "test_model_no_mappings.pt"
        trainer.save_model(str(model_path))
        
        # Load model
        new_model = SpeciesGeneGNN(
            num_species=3,
            num_genes=4,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        new_trainer = Trainer(model=new_model)
        id_to_species, id_to_gene = new_trainer.load_model(str(model_path))
        
        # Verify that None is returned for mappings
        assert id_to_species is None
        assert id_to_gene is None
    
    def test_mappings_correctness(self):
        """Test that saved mappings correctly identify species and genes."""
        # Setup data with known species and genes
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data=self.sample_data)
        data_loader.build_mappings(df)
        
        # Use data_loader's existing mappings for expected values
        expected_species = sorted(data_loader.species_to_id.keys())
        expected_genes = sorted(data_loader.gene_to_id.keys())
        
        # Create and save model
        model = SpeciesGeneGNN(
            num_species=len(expected_species),
            num_genes=len(expected_genes),
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        trainer = Trainer(model=model, data_loader=data_loader)
        model_path = self.temp_path / "test_model_correct.pt"
        trainer.save_model(str(model_path))
        
        # Load and verify
        checkpoint = torch.load(str(model_path))
        id_to_species = checkpoint['id_to_species']
        id_to_gene = checkpoint['id_to_gene']
        
        # Verify all expected species are in the mapping
        saved_species = set(id_to_species.values())
        assert saved_species == set(expected_species)
        
        # Verify all expected genes are in the mapping
        saved_genes = set(id_to_gene.values())
        assert saved_genes == set(expected_genes)
        
        # Verify ID ordering (should be sorted)
        assert [id_to_species[i] for i in sorted(id_to_species.keys())] == expected_species
        assert [id_to_gene[i] for i in sorted(id_to_gene.keys())] == expected_genes
    
    def test_embedding_matrix_alignment(self):
        """Test that embedding matrix rows align with saved mappings."""
        # Setup
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data=self.sample_data)
        data_loader.build_mappings(df)
        
        num_species = data_loader.get_num_species()
        num_genes = data_loader.get_num_genes()
        
        model = SpeciesGeneGNN(
            num_species=num_species,
            num_genes=num_genes,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1
        )
        
        trainer = Trainer(model=model, data_loader=data_loader)
        model_path = self.temp_path / "test_alignment.pt"
        trainer.save_model(str(model_path))
        
        # Load checkpoint
        checkpoint = torch.load(str(model_path))
        
        # Verify embedding dimensions match mapping counts
        species_embedding_weight = checkpoint['model_state_dict']['species_embedding.weight']
        gene_embedding_weight = checkpoint['model_state_dict']['gene_embedding.weight']
        
        assert species_embedding_weight.shape[0] == len(checkpoint['id_to_species'])
        assert gene_embedding_weight.shape[0] == len(checkpoint['id_to_gene'])
        
        # Verify each ID in mapping corresponds to a valid row in the embedding matrix
        for species_id in checkpoint['id_to_species'].keys():
            assert species_id < species_embedding_weight.shape[0], \
                f"Species ID {species_id} exceeds embedding matrix size {species_embedding_weight.shape[0]}"
        
        for gene_id in checkpoint['id_to_gene'].keys():
            assert gene_id < gene_embedding_weight.shape[0], \
                f"Gene ID {gene_id} exceeds embedding matrix size {gene_embedding_weight.shape[0]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
