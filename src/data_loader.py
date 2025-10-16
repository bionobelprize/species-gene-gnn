"""
Data loading and preprocessing module for Species-Gene GNN.

This module handles loading 5-dimensional input data and preprocessing it
for the heterogeneous graph neural network.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import torch


class SpeciesGeneDataLoader:
    """
    Loads and preprocesses species-gene association data.
    
    Input format: 5-dimensional vectors
    (species_a, gene_a, species_b, gene_b, protein_similarity)
    """
    
    def __init__(self):
        self.species_to_id = {}
        self.gene_to_id = {}
        self.id_to_species = {}
        self.id_to_gene = {}
        
    def load_data(self, data_path: str = None, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load data from CSV file or DataFrame.
        
        Args:
            data_path: Path to CSV file
            data: DataFrame with columns [species_a, gene_a, species_b, gene_b, similarity]
            
        Returns:
            Processed DataFrame
        """
        if data_path is not None:
            df = pd.read_csv(data_path)
        elif data is not None:
            df = data
        else:
            raise ValueError("Either data_path or data must be provided")
        
        # Validate columns
        required_cols = ['species_a', 'gene_a', 'species_b', 'gene_b', 'similarity']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        return df
    
    def build_mappings(self, df: pd.DataFrame) -> None:
        """
        Build mappings from species/gene names to integer IDs.
        
        Args:
            df: DataFrame with species and gene information
        """
        # Collect all unique species and genes
        all_species = set(df['species_a'].unique()) | set(df['species_b'].unique())
        all_genes = set(df['gene_a'].unique()) | set(df['gene_b'].unique())
        
        # Create mappings
        self.species_to_id = {species: idx for idx, species in enumerate(sorted(all_species))}
        self.gene_to_id = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
        
        # Create reverse mappings
        self.id_to_species = {idx: species for species, idx in self.species_to_id.items()}
        self.id_to_gene = {idx: gene for gene, idx in self.gene_to_id.items()}
    
    def encode_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode species and gene names to integer IDs.
        
        Args:
            df: DataFrame with species and gene information
            
        Returns:
            Tuple of (species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities)
        """
        species_a_ids = df['species_a'].map(self.species_to_id).values
        gene_a_ids = df['gene_a'].map(self.gene_to_id).values
        species_b_ids = df['species_b'].map(self.species_to_id).values
        gene_b_ids = df['gene_b'].map(self.gene_to_id).values
        similarities = df['similarity'].values
        
        return species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities
    
    def get_num_species(self) -> int:
        """Return the number of unique species."""
        return len(self.species_to_id)
    
    def get_num_genes(self) -> int:
        """Return the number of unique genes."""
        return len(self.gene_to_id)
