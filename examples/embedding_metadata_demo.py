"""
Demo script showing how to save and load models with species/gene mappings.

This script demonstrates the new functionality where model checkpoints include
the id_to_species and id_to_gene mappings, allowing users to identify what each
row in the embedding matrices represents.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import torch
import tempfile
from pathlib import Path

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer


def main():
    print("=" * 80)
    print("Species-Gene GNN: Embedding Metadata Demo")
    print("=" * 80)
    print()
    print("This demo shows how model checkpoints now include species and gene mappings")
    print("to identify what each row in the embedding matrices represents.")
    print()
    
    # Create sample data
    print("Step 1: Creating sample data...")
    sample_data = pd.DataFrame({
        'species_a': ['Human', 'Mouse', 'Human', 'Rat', 'Dog'],
        'gene_a': ['BRCA1', 'Brca1', 'TP53', 'Tp53', 'BRCA1'],
        'species_b': ['Mouse', 'Rat', 'Mouse', 'Human', 'Human'],
        'gene_b': ['Brca1', 'Brca1', 'Tp53', 'TP53', 'BRCA1'],
        'similarity': [0.89, 0.92, 0.95, 0.88, 0.87]
    })
    print(f"Created {len(sample_data)} gene associations")
    print(f"Species: {sorted(set(sample_data['species_a'].unique()) | set(sample_data['species_b'].unique()))}")
    print(f"Genes: {sorted(set(sample_data['gene_a'].unique()) | set(sample_data['gene_b'].unique()))}")
    print()
    
    # Load and process data
    print("Step 2: Loading and encoding data...")
    data_loader = SpeciesGeneDataLoader()
    df = data_loader.load_data(data=sample_data)
    data_loader.build_mappings(df)
    
    print(f"Number of unique species: {data_loader.get_num_species()}")
    print(f"Number of unique genes: {data_loader.get_num_genes()}")
    print()
    
    print("Species ID Mapping:")
    for species_id, species_name in sorted(data_loader.id_to_species.items()):
        print(f"  ID {species_id} -> {species_name}")
    print()
    
    print("Gene ID Mapping (first 5):")
    for gene_id, gene_name in sorted(data_loader.id_to_gene.items())[:5]:
        print(f"  ID {gene_id} -> {gene_name}")
    print()
    
    # Encode data and build graph
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    graph_builder = HeteroGraphBuilder()
    hetero_data = graph_builder.build_graph(
        species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
        data_loader.get_num_species(), data_loader.get_num_genes()
    )
    
    # Create and train a simple model
    print("Step 3: Creating and training model...")
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2
    )
    
    # Create trainer WITH data_loader to enable mapping saving
    trainer = Trainer(
        model=model,
        learning_rate=0.01,
        data_loader=data_loader  # Pass data_loader to save mappings!
    )
    
    # Train for a few epochs
    train_edge_index = graph_builder.get_edge_label_index(gene_a_ids, gene_b_ids)
    train_labels = graph_builder.get_edge_labels(similarities)
    
    trainer.train(
        data=hetero_data,
        train_edge_index=train_edge_index,
        train_labels=train_labels,
        num_epochs=10,
        verbose=False
    )
    print("Model trained for 10 epochs")
    print()
    
    # Save model to temporary file
    print("Step 4: Saving model with mappings...")
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "model_with_mappings.pt"
    trainer.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Check what's in the checkpoint
    checkpoint = torch.load(str(model_path))
    print(f"\nCheckpoint contains:")
    for key in checkpoint.keys():
        if key in ['id_to_species', 'id_to_gene']:
            print(f"  - {key}: {len(checkpoint[key])} entries")
        else:
            print(f"  - {key}")
    print()
    
    # Load model and mappings
    print("Step 5: Loading model and extracting mappings...")
    new_model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2
    )
    
    new_trainer = Trainer(model=new_model)
    id_to_species, id_to_gene = new_trainer.load_model(str(model_path))
    
    print(f"Successfully loaded mappings:")
    print(f"  - Species mapping: {len(id_to_species)} entries")
    print(f"  - Gene mapping: {len(id_to_gene)} entries")
    print()
    
    # Demonstrate using the mappings
    print("Step 6: Using mappings to identify embeddings...")
    print("\nSpecies Embeddings:")
    species_embedding_weight = checkpoint['model_state_dict']['species_embedding.weight']
    print(f"  Shape: {species_embedding_weight.shape}")
    print(f"  Rows: {species_embedding_weight.shape[0]}")
    print()
    
    for i in range(min(3, species_embedding_weight.shape[0])):
        species_name = id_to_species[i]
        embedding_vector = species_embedding_weight[i]
        print(f"  Row {i} = {species_name}")
        print(f"    Embedding: [{', '.join([f'{v:.4f}' for v in embedding_vector[:5].tolist()])}...]")
    print()
    
    print("Gene Embeddings:")
    gene_embedding_weight = checkpoint['model_state_dict']['gene_embedding.weight']
    print(f"  Shape: {gene_embedding_weight.shape}")
    print(f"  Rows: {gene_embedding_weight.shape[0]}")
    print()
    
    for i in range(min(3, gene_embedding_weight.shape[0])):
        gene_name = id_to_gene[i]
        embedding_vector = gene_embedding_weight[i]
        print(f"  Row {i} = {gene_name}")
        print(f"    Embedding: [{', '.join([f'{v:.4f}' for v in embedding_vector[:5].tolist()])}...]")
    print()
    
    # Demonstrate backward compatibility
    print("Step 7: Demonstrating backward compatibility...")
    print("Creating a model WITHOUT data_loader...")
    model_no_loader = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2
    )
    
    # Trainer without data_loader
    trainer_no_loader = Trainer(model=model_no_loader)
    model_path_no_loader = Path(temp_dir) / "model_without_mappings.pt"
    trainer_no_loader.save_model(str(model_path_no_loader))
    
    checkpoint_no_loader = torch.load(str(model_path_no_loader))
    print(f"Checkpoint without data_loader contains:")
    for key in checkpoint_no_loader.keys():
        print(f"  - {key}")
    
    if 'id_to_species' not in checkpoint_no_loader:
        print("  ✓ Mappings NOT saved (as expected)")
    print()
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    print("=" * 80)
    print("Demo completed successfully!")
    print()
    print("Key takeaways:")
    print("  1. Pass data_loader to Trainer to enable saving of species/gene mappings")
    print("  2. Mappings are saved in the checkpoint as 'id_to_species' and 'id_to_gene'")
    print("  3. load_model() returns the mappings (or None if not present)")
    print("  4. Each row in embedding matrices can now be identified by its ID")
    print("  5. Backward compatibility is maintained for models without data_loader")
    print("=" * 80)


if __name__ == '__main__':
    main()
