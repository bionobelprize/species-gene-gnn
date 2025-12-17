"""
Example script showing how to access embeddings with their species/gene identifiers.

This script demonstrates how to:
1. Load a trained model checkpoint
2. Extract the species and gene ID mappings
3. Access embeddings and identify what they represent
4. Export embeddings with their labels for downstream analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import torch
import numpy as np
import tempfile
from pathlib import Path

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer


def train_and_save_model():
    """Train a simple model and save it with mappings."""
    print("Training a sample model...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'species_a': ['Human', 'Mouse', 'Human', 'Rat', 'Dog', 'Mouse'],
        'gene_a': ['BRCA1', 'Brca1', 'TP53', 'Tp53', 'BRCA1', 'TP53'],
        'species_b': ['Mouse', 'Rat', 'Mouse', 'Human', 'Human', 'Rat'],
        'gene_b': ['Brca1', 'Brca1', 'Tp53', 'TP53', 'BRCA1', 'Tp53'],
        'similarity': [0.89, 0.92, 0.95, 0.88, 0.87, 0.93]
    })
    
    # Load and process data
    data_loader = SpeciesGeneDataLoader()
    df = data_loader.load_data(data=sample_data)
    data_loader.build_mappings(df)
    
    # Build graph
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    graph_builder = HeteroGraphBuilder()
    hetero_data = graph_builder.build_graph(
        species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
        data_loader.get_num_species(), data_loader.get_num_genes()
    )
    
    # Create and train model
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2
    )
    
    trainer = Trainer(
        model=model,
        learning_rate=0.01,
        data_loader=data_loader  # Important: pass data_loader to save mappings
    )
    
    train_edge_index = graph_builder.get_edge_label_index(gene_a_ids, gene_b_ids)
    train_labels = graph_builder.get_edge_labels(similarities)
    
    trainer.train(
        data=hetero_data,
        train_edge_index=train_edge_index,
        train_labels=train_labels,
        num_epochs=20,
        verbose=False
    )
    
    # Save model
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "trained_model.pt"
    trainer.save_model(str(model_path))
    
    print(f"Model trained and saved to: {model_path}\n")
    return str(model_path), temp_dir


def load_and_extract_embeddings(model_path):
    """Load model and extract embeddings with their identifiers."""
    print("=" * 80)
    print("Loading Model and Extracting Embeddings with Identifiers")
    print("=" * 80)
    print()
    
    # Load checkpoint
    print("Step 1: Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Checkpoint loaded successfully")
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    print()
    
    # Extract mappings
    print("Step 2: Extracting species and gene mappings...")
    id_to_species = checkpoint.get('id_to_species')
    id_to_gene = checkpoint.get('id_to_gene')
    
    if id_to_species is None or id_to_gene is None:
        print("ERROR: No mappings found in checkpoint!")
        print("Make sure the model was saved with data_loader parameter.")
        return
    
    print(f"Found mappings:")
    print(f"  - {len(id_to_species)} species")
    print(f"  - {len(id_to_gene)} genes")
    print()
    
    # Extract embedding weights
    print("Step 3: Extracting embedding matrices...")
    species_embeddings = checkpoint['model_state_dict']['species_embedding.weight']
    gene_embeddings = checkpoint['model_state_dict']['gene_embedding.weight']
    
    print(f"Species embeddings shape: {species_embeddings.shape}")
    print(f"Gene embeddings shape: {gene_embeddings.shape}")
    print()
    
    # Display species embeddings with their labels
    print("Step 4: Species Embeddings with Labels")
    print("-" * 80)
    print(f"{'ID':<5} {'Species Name':<20} {'Embedding (first 5 dims)':<40}")
    print("-" * 80)
    
    species_data = []
    for species_id in sorted(id_to_species.keys()):
        species_name = id_to_species[species_id]
        embedding = species_embeddings[species_id].numpy()
        embedding_preview = ', '.join([f'{v:.4f}' for v in embedding[:5]])
        
        print(f"{species_id:<5} {species_name:<20} [{embedding_preview}...]")
        
        species_data.append({
            'id': species_id,
            'name': species_name,
            'embedding': embedding
        })
    print()
    
    # Display gene embeddings with their labels
    print("Step 5: Gene Embeddings with Labels")
    print("-" * 80)
    print(f"{'ID':<5} {'Gene Name':<20} {'Embedding (first 5 dims)':<40}")
    print("-" * 80)
    
    gene_data = []
    for gene_id in sorted(id_to_gene.keys()):
        gene_name = id_to_gene[gene_id]
        embedding = gene_embeddings[gene_id].numpy()
        embedding_preview = ', '.join([f'{v:.4f}' for v in embedding[:5]])
        
        print(f"{gene_id:<5} {gene_name:<20} [{embedding_preview}...]")
        
        gene_data.append({
            'id': gene_id,
            'name': gene_name,
            'embedding': embedding
        })
    print()
    
    # Export to CSV
    print("Step 6: Exporting embeddings to CSV files...")
    
    # Species embeddings CSV
    species_df = pd.DataFrame([
        {
            'species_id': d['id'],
            'species_name': d['name'],
            **{f'dim_{i}': d['embedding'][i] for i in range(len(d['embedding']))}
        }
        for d in species_data
    ])
    
    species_csv = Path(model_path).parent / "species_embeddings.csv"
    species_df.to_csv(species_csv, index=False)
    print(f"  - Species embeddings saved to: {species_csv}")
    
    # Gene embeddings CSV
    gene_df = pd.DataFrame([
        {
            'gene_id': d['id'],
            'gene_name': d['name'],
            **{f'dim_{i}': d['embedding'][i] for i in range(len(d['embedding']))}
        }
        for d in gene_data
    ])
    
    gene_csv = Path(model_path).parent / "gene_embeddings.csv"
    gene_df.to_csv(gene_csv, index=False)
    print(f"  - Gene embeddings saved to: {gene_csv}")
    print()
    
    # Show CSV preview
    print("Step 7: Preview of exported CSV files")
    print("-" * 80)
    print("Species embeddings CSV (first 3 rows):")
    print(species_df.head(3))
    print()
    print("Gene embeddings CSV (first 3 rows):")
    print(gene_df.head(3))
    print()
    
    # Demonstrate finding specific embeddings
    print("Step 8: Finding specific embeddings by name")
    print("-" * 80)
    
    # Find Human species embedding
    human_row = species_df[species_df['species_name'] == 'Human']
    if not human_row.empty:
        print("Human species embedding:")
        print(f"  ID: {human_row['species_id'].values[0]}")
        embedding_cols = [col for col in human_row.columns if col.startswith('dim_')]
        embedding_values = human_row[embedding_cols].values[0][:5]
        print(f"  Embedding (first 5 dims): [{', '.join([f'{v:.4f}' for v in embedding_values])}...]")
    print()
    
    # Find BRCA1 gene embedding
    brca1_row = gene_df[gene_df['gene_name'] == 'BRCA1']
    if not brca1_row.empty:
        print("BRCA1 gene embedding:")
        print(f"  ID: {brca1_row['gene_id'].values[0]}")
        embedding_cols = [col for col in brca1_row.columns if col.startswith('dim_')]
        embedding_values = brca1_row[embedding_cols].values[0][:5]
        print(f"  Embedding (first 5 dims): [{', '.join([f'{v:.4f}' for v in embedding_values])}...]")
    print()
    
    print("=" * 80)
    print("Summary:")
    print("  ✓ Successfully loaded model checkpoint with mappings")
    print("  ✓ Extracted species and gene embeddings with their identifiers")
    print("  ✓ Exported embeddings to CSV files for downstream analysis")
    print("  ✓ Demonstrated how to query embeddings by name")
    print()
    print("The exported CSV files include:")
    print("  - species_id/gene_id: Numeric ID used in the model")
    print("  - species_name/gene_name: Original name/identifier")
    print("  - dim_0, dim_1, ...: Embedding dimensions")
    print("=" * 80)


def main():
    """Main function."""
    # Train and save a model
    model_path, temp_dir = train_and_save_model()
    
    # Load and extract embeddings
    load_and_extract_embeddings(model_path)
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    print("\nTemporary files cleaned up.")


if __name__ == '__main__':
    main()
