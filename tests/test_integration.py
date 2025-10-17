"""
Integration test demonstrating the complete data preparation to model training workflow.

This test creates sample alignment data, generates the five-dimensional CSV,
and trains a simple model on it.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import json
from pathlib import Path
import pandas as pd

from src.data_preparation import DataPreparation
from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN


def test_complete_workflow():
    """Test the complete data preparation and model workflow."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    print("=" * 70)
    print("Integration Test: Complete Workflow")
    print("=" * 70)
    print(f"\nTemp directory: {temp_dir}\n")
    
    # Step 1: Create sample catalog
    print("Step 1: Creating sample dataset catalog...")
    catalog = {
        "apiVersion": "V2",
        "assemblies": [
            {
                "accession": "GCF_TEST001",
                "files": [
                    {
                        "filePath": str(temp_path / "genome1.faa"),
                        "fileType": "PROTEIN_FASTA",
                        "uncompressedLengthBytes": "1000"
                    }
                ]
            },
            {
                "accession": "GCF_TEST002",
                "files": [
                    {
                        "filePath": str(temp_path / "genome2.faa"),
                        "fileType": "PROTEIN_FASTA",
                        "uncompressedLengthBytes": "2000"
                    }
                ]
            }
        ]
    }
    
    catalog_path = temp_path / "dataset_catalog.json"
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f)
    print(f"  Created catalog: {catalog_path}")
    
    # Step 2: Create sample alignment data (simulating Diamond output)
    print("\nStep 2: Creating sample alignment data...")
    output_dir = temp_path / "output"
    alignment_dir = output_dir / "alignments"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create alignment: GCF_TEST001 vs GCF_TEST002
    alignment_data_1 = [
        "gene1\tgene3\t95.5\t100\t2\t0\t1\t100\t1\t100\t1e-50\t200.5",
        "gene2\tgene4\t90.0\t100\t5\t0\t1\t100\t1\t100\t1e-40\t180.0",
        "gene5\tgene6\t85.0\t100\t8\t0\t1\t100\t1\t100\t1e-30\t160.0"
    ]
    alignment_file_1 = alignment_dir / "GCF_TEST001_vs_GCF_TEST002.tsv"
    with open(alignment_file_1, 'w') as f:
        f.write('\n'.join(alignment_data_1))
    
    # Create alignment: GCF_TEST002 vs GCF_TEST001
    alignment_data_2 = [
        "gene3\tgene1\t95.5\t100\t2\t0\t1\t100\t1\t100\t1e-50\t200.5",
        "gene4\tgene2\t90.0\t100\t5\t0\t1\t100\t1\t100\t1e-40\t180.0",
        "gene6\tgene5\t85.0\t100\t8\t0\t1\t100\t1\t100\t1e-30\t160.0"
    ]
    alignment_file_2 = alignment_dir / "GCF_TEST002_vs_GCF_TEST001.tsv"
    with open(alignment_file_2, 'w') as f:
        f.write('\n'.join(alignment_data_2))
    
    print(f"  Created alignment files:")
    print(f"    {alignment_file_1}")
    print(f"    {alignment_file_2}")
    
    # Step 3: Generate five-dimensional CSV
    print("\nStep 3: Generating five-dimensional CSV...")
    prep = DataPreparation(str(catalog_path), str(output_dir))
    
    alignment_files = [str(alignment_file_1), str(alignment_file_2)]
    output_csv = temp_path / "gene_associations.csv"
    
    df = prep.generate_input_csv(alignment_files, str(output_csv), normalize_scores=True)
    
    print(f"  Generated CSV with {len(df)} records")
    print(f"  Saved to: {output_csv}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Step 4: Load data with SpeciesGeneDataLoader
    print("\nStep 4: Loading data with SpeciesGeneDataLoader...")
    loader = SpeciesGeneDataLoader()
    df_loaded = loader.load_data(str(output_csv))
    loader.build_mappings(df_loaded)
    
    print(f"  Loaded {len(df_loaded)} records")
    print(f"  Number of species: {loader.get_num_species()}")
    print(f"  Number of genes: {loader.get_num_genes()}")
    
    # Step 5: Encode data
    print("\nStep 5: Encoding data...")
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        loader.encode_data(df_loaded)
    
    print(f"  Encoded {len(gene_a_ids)} gene pairs")
    
    # Step 6: Build graph
    print("\nStep 6: Building heterogeneous graph...")
    builder = HeteroGraphBuilder()
    hetero_data = builder.build_graph(
        species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
        loader.get_num_species(), loader.get_num_genes()
    )
    
    print(f"  Graph created with:")
    print(f"    Node types: {hetero_data.node_types}")
    print(f"    Edge types: {hetero_data.edge_types}")
    print(f"    Species nodes: {hetero_data['species'].num_nodes}")
    print(f"    Gene nodes: {hetero_data['gene'].num_nodes}")
    
    # Step 7: Create model
    print("\nStep 7: Creating GNN model...")
    model = SpeciesGeneGNN(
        num_species=loader.get_num_species(),
        num_genes=loader.get_num_genes(),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1
    )
    
    print(f"  Model created with:")
    print(f"    Species: {model.num_species}")
    print(f"    Genes: {model.num_genes}")
    print(f"    Embedding dim: {model.embedding_dim}")
    print(f"    Hidden dim: {model.hidden_dim}")
    
    # Step 8: Test forward pass
    print("\nStep 8: Testing forward pass...")
    x_dict = model.forward(hetero_data)
    
    print(f"  Forward pass successful!")
    print(f"    Species embeddings shape: {x_dict['species'].shape}")
    print(f"    Gene embeddings shape: {x_dict['gene'].shape}")
    
    # Step 9: Test prediction
    print("\nStep 9: Testing prediction...")
    import torch
    gene_a_tensor = torch.tensor(gene_a_ids, dtype=torch.long)
    gene_b_tensor = torch.tensor(gene_b_ids, dtype=torch.long)
    predictions = model.predict(hetero_data, gene_a_tensor, gene_b_tensor)
    
    print(f"  Predictions generated!")
    print(f"    Shape: {predictions.shape}")
    print(f"    Min: {predictions.min():.4f}")
    print(f"    Max: {predictions.max():.4f}")
    print(f"    Mean: {predictions.mean():.4f}")
    
    print("\n" + "=" * 70)
    print("Integration Test: PASSED")
    print("=" * 70)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == '__main__':
    test_complete_workflow()
