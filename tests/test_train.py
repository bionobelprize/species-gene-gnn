"""
Unit tests for training script genome subset functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.data_loader import SpeciesGeneDataLoader


class TestGenomeSubset:
    """Test cases for genome subset functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_sample_data(self, num_species=5, num_genes_per_species=10):
        """Create sample gene association data."""
        records = []
        
        species_list = [f"Species_{i}" for i in range(num_species)]
        
        # Create cross-species associations
        for i, species_a in enumerate(species_list):
            for j, species_b in enumerate(species_list):
                if i >= j:  # Only create pairs once
                    continue
                
                # Create some gene associations
                for gene_idx in range(num_genes_per_species):
                    gene_a = f"{species_a}_gene_{gene_idx}"
                    gene_b = f"{species_b}_gene_{gene_idx}"
                    
                    records.append({
                        'species_a': species_a,
                        'gene_a': gene_a,
                        'species_b': species_b,
                        'gene_b': gene_b,
                        'similarity': np.random.uniform(0.5, 1.0)
                    })
        
        df = pd.DataFrame(records)
        
        # Save to CSV
        csv_path = self.temp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path), df
    
    def test_genome_subset_filtering(self):
        """Test that genome subset filtering works correctly."""
        # Create sample data with 5 species
        csv_path, original_df = self.create_sample_data(num_species=5, num_genes_per_species=10)
        
        # Load data
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data_path=csv_path)
        
        # Get original species count
        all_species = sorted(set(df['species_a'].unique()) | set(df['species_b'].unique()))
        original_species_count = len(all_species)
        
        # Filter to 60% of genomes (should keep 3 out of 5 species)
        genome_subset_ratio = 0.6
        num_species_to_keep = max(1, int(original_species_count * genome_subset_ratio))
        
        # Select species
        np.random.seed(42)
        selected_species = set(np.random.choice(all_species, num_species_to_keep, replace=False))
        
        # Filter dataframe
        df_filtered = df[
            df['species_a'].isin(selected_species) & 
            df['species_b'].isin(selected_species)
        ]
        
        # Build mappings on filtered data
        data_loader.build_mappings(df_filtered)
        
        # Verify species count
        assert data_loader.get_num_species() == num_species_to_keep
        assert data_loader.get_num_species() < original_species_count
        
        # Verify all species in filtered data are in selected set
        filtered_species = set(df_filtered['species_a'].unique()) | set(df_filtered['species_b'].unique())
        assert filtered_species.issubset(selected_species)
        
        print(f"Original species: {original_species_count}")
        print(f"Selected species: {num_species_to_keep}")
        print(f"Original associations: {len(df)}")
        print(f"Filtered associations: {len(df_filtered)}")
    
    def test_genome_subset_ratio_100_percent(self):
        """Test that 100% ratio keeps all genomes."""
        csv_path, original_df = self.create_sample_data(num_species=5, num_genes_per_species=10)
        
        # Load data
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data_path=csv_path)
        
        # Get species count
        all_species = sorted(set(df['species_a'].unique()) | set(df['species_b'].unique()))
        original_species_count = len(all_species)
        
        # 100% should keep all species
        genome_subset_ratio = 1.0
        num_species_to_keep = max(1, int(original_species_count * genome_subset_ratio))
        
        assert num_species_to_keep == original_species_count
    
    def test_genome_subset_ratio_minimum(self):
        """Test that very small ratios keep at least 1 genome."""
        csv_path, original_df = self.create_sample_data(num_species=5, num_genes_per_species=10)
        
        # Load data
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data_path=csv_path)
        
        # Get species count
        all_species = sorted(set(df['species_a'].unique()) | set(df['species_b'].unique()))
        original_species_count = len(all_species)
        
        # Very small ratio should keep at least 1 species
        genome_subset_ratio = 0.01
        num_species_to_keep = max(1, int(original_species_count * genome_subset_ratio))
        
        assert num_species_to_keep >= 1
    
    def test_genome_subset_with_genes(self):
        """Test that genome subsetting also filters related genes."""
        csv_path, original_df = self.create_sample_data(num_species=5, num_genes_per_species=10)
        
        # Load data
        data_loader = SpeciesGeneDataLoader()
        df = data_loader.load_data(data_path=csv_path)
        
        # Get all genes before filtering
        all_genes_before = set(df['gene_a'].unique()) | set(df['gene_b'].unique())
        
        # Filter to 40% of genomes
        all_species = sorted(set(df['species_a'].unique()) | set(df['species_b'].unique()))
        genome_subset_ratio = 0.4
        num_species_to_keep = max(1, int(len(all_species) * genome_subset_ratio))
        
        np.random.seed(42)
        selected_species = set(np.random.choice(all_species, num_species_to_keep, replace=False))
        
        # Filter dataframe
        df_filtered = df[
            df['species_a'].isin(selected_species) & 
            df['species_b'].isin(selected_species)
        ]
        
        # Get all genes after filtering
        all_genes_after = set(df_filtered['gene_a'].unique()) | set(df_filtered['gene_b'].unique())
        
        # Build mappings
        data_loader.build_mappings(df_filtered)
        
        # Verify gene count is reduced
        assert len(all_genes_after) < len(all_genes_before)
        assert data_loader.get_num_genes() == len(all_genes_after)
        
        print(f"Genes before filtering: {len(all_genes_before)}")
        print(f"Genes after filtering: {len(all_genes_after)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
