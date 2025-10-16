"""
Example script demonstrating the Species-Gene GNN model.

This script shows how to:
1. Load or create sample data
2. Build a heterogeneous graph
3. Train the GNN model
4. Make predictions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer


def create_sample_data(num_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample species-gene association data for demonstration.
    
    Args:
        num_samples: Number of sample data points
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)
    
    # Define some sample species and genes
    species_list = ['Human', 'Mouse', 'Rat', 'Zebrafish', 'Fruit_Fly']
    
    # Generate gene names (in practice, these would be actual gene IDs)
    gene_pool = [f'Gene_{i:04d}' for i in range(200)]
    
    # Create random associations
    data = []
    for _ in range(num_samples):
        species_a = np.random.choice(species_list)
        species_b = np.random.choice(species_list)
        
        gene_a = np.random.choice(gene_pool)
        gene_b = np.random.choice(gene_pool)
        
        # Simulate protein similarity (higher for same genes, lower for different)
        if gene_a == gene_b:
            similarity = np.random.uniform(0.8, 1.0)
        else:
            # Add some correlation based on gene indices (simulated)
            gene_a_idx = int(gene_a.split('_')[1])
            gene_b_idx = int(gene_b.split('_')[1])
            base_sim = max(0, 1 - abs(gene_a_idx - gene_b_idx) / 200.0)
            similarity = base_sim * np.random.uniform(0.3, 0.7)
        
        data.append({
            'species_a': species_a,
            'gene_a': gene_a,
            'species_b': species_b,
            'gene_b': gene_b,
            'similarity': similarity
        })
    
    return pd.DataFrame(data)


def main():
    """Main function to run the example."""
    print("=" * 70)
    print("Species-Gene Association GNN - Example")
    print("=" * 70)
    print()
    
    # Step 1: Create or load data
    print("Step 1: Creating sample data...")
    df = create_sample_data(num_samples=1000)
    print(f"Created {len(df)} sample associations")
    print(f"Sample data:\n{df.head()}")
    print()
    
    # Step 2: Load and preprocess data
    print("Step 2: Loading and preprocessing data...")
    data_loader = SpeciesGeneDataLoader()
    df = data_loader.load_data(data=df)
    data_loader.build_mappings(df)
    
    print(f"Number of unique species: {data_loader.get_num_species()}")
    print(f"Number of unique genes: {data_loader.get_num_genes()}")
    print()
    
    # Encode data
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    # Split data into train and validation sets
    indices = np.arange(len(similarities))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Step 3: Build heterogeneous graph
    print("Step 3: Building heterogeneous graph...")
    graph_builder = HeteroGraphBuilder()
    hetero_data = graph_builder.build_graph(
        species_a_ids=species_a_ids,
        gene_a_ids=gene_a_ids,
        species_b_ids=species_b_ids,
        gene_b_ids=gene_b_ids,
        similarities=similarities,
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes()
    )
    
    print(f"Graph structure:")
    print(f"  Node types: {hetero_data.node_types}")
    print(f"  Edge types: {hetero_data.edge_types}")
    for edge_type in hetero_data.edge_types:
        print(f"    {edge_type}: {hetero_data[edge_type].edge_index.shape[1]} edges")
    print()
    
    # Prepare training and validation data
    train_edge_index = graph_builder.get_edge_label_index(
        gene_a_ids[train_idx],
        gene_b_ids[train_idx]
    )
    train_labels = graph_builder.get_edge_labels(similarities[train_idx])
    
    val_edge_index = graph_builder.get_edge_label_index(
        gene_a_ids[val_idx],
        gene_b_ids[val_idx]
    )
    val_labels = graph_builder.get_edge_labels(similarities[val_idx])
    
    # Step 4: Create and train model
    print("Step 4: Creating GNN model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Step 5: Train the model
    print("Step 5: Training the model...")
    trainer = Trainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-5,
        device=device
    )
    
    history = trainer.train(
        data=hetero_data,
        train_edge_index=train_edge_index,
        train_labels=train_labels,
        val_edge_index=val_edge_index,
        val_labels=val_labels,
        num_epochs=50,
        verbose=True
    )
    print()
    
    # Step 6: Final evaluation
    print("Step 6: Final evaluation...")
    val_metrics = trainer.evaluate(hetero_data, val_edge_index, val_labels)
    print(f"Validation Metrics:")
    print(f"  MSE: {val_metrics['mse']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  MAE: {val_metrics['mae']:.4f}")
    print()
    
    # Step 7: Make sample predictions
    print("Step 7: Making sample predictions...")
    model.eval()
    with torch.no_grad():
        # Take first 5 validation examples
        sample_indices = val_idx[:5]
        sample_gene_a = torch.tensor(gene_a_ids[sample_indices], dtype=torch.long)
        sample_gene_b = torch.tensor(gene_b_ids[sample_indices], dtype=torch.long)
        sample_true = similarities[sample_indices]
        
        # Move to device
        hetero_data = hetero_data.to(device)
        sample_gene_a = sample_gene_a.to(device)
        sample_gene_b = sample_gene_b.to(device)
        
        # Predict
        predictions = model.predict(hetero_data, sample_gene_a, sample_gene_b)
        predictions = predictions.cpu().numpy()
        
        print("Sample predictions:")
        print(f"{'Gene A':<12} {'Gene B':<12} {'True':<8} {'Predicted':<10} {'Error':<8}")
        print("-" * 58)
        for i, idx in enumerate(sample_indices):
            gene_a_name = data_loader.id_to_gene[gene_a_ids[idx]]
            gene_b_name = data_loader.id_to_gene[gene_b_ids[idx]]
            true_val = sample_true[i]
            pred_val = predictions[i]
            error = abs(true_val - pred_val)
            print(f"{gene_a_name:<12} {gene_b_name:<12} {true_val:<8.4f} {pred_val:<10.4f} {error:<8.4f}")
    
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
