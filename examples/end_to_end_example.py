"""
End-to-end example notebook for Species-Gene GNN.

This script demonstrates the complete workflow from data generation
to training and evaluation.
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
from src.utils import split_data, normalize_similarities


def plot_training_history(history):
    """Plot training history."""
    try:
        import matplotlib.pyplot as plt
        
        train_losses = history['train_loss']
        
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation metrics if available
        if history['val_metrics']:
            val_losses = [m['loss'] for m in history['val_metrics']]
            val_rmse = [m['rmse'] for m in history['val_metrics']]
            
            plt.subplot(1, 2, 2)
            plt.plot(val_losses, label='Val Loss')
            plt.plot(val_rmse, label='Val RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.title('Validation Metrics')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("Training history plot saved to: training_history.png")
        
    except ImportError:
        print("matplotlib not installed, skipping plot generation")


def main():
    """Main function demonstrating end-to-end workflow."""
    
    print("=" * 80)
    print("SPECIES-GENE GNN - END-TO-END EXAMPLE")
    print("=" * 80)
    print()
    
    # ========================================================================
    # PART 1: DATA GENERATION
    # ========================================================================
    print("PART 1: DATA GENERATION")
    print("-" * 80)
    
    print("Generating synthetic species-gene association data...")
    
    # Define parameters
    num_species = 8
    num_genes = 150
    num_samples = 1500
    
    # Generate species and gene names
    species_names = [f'Species_{chr(65+i)}' for i in range(num_species)]
    gene_names = [f'Gene_{i:04d}' for i in range(num_genes)]
    
    # Create synthetic data
    np.random.seed(42)
    data = []
    
    for _ in range(num_samples):
        species_a = np.random.choice(species_names)
        species_b = np.random.choice(species_names)
        gene_a = np.random.choice(gene_names)
        gene_b = np.random.choice(gene_names)
        
        # Simulate protein similarity
        if gene_a == gene_b:
            similarity = np.random.uniform(0.85, 1.0)
        else:
            gene_a_idx = int(gene_a.split('_')[1])
            gene_b_idx = int(gene_b.split('_')[1])
            distance = abs(gene_a_idx - gene_b_idx)
            base_sim = max(0, 1 - (distance / num_genes))
            similarity = base_sim * np.random.uniform(0.2, 0.7)
        
        data.append({
            'species_a': species_a,
            'gene_a': gene_a,
            'species_b': species_b,
            'gene_b': gene_b,
            'similarity': similarity
        })
    
    df = pd.DataFrame(data)
    
    print(f"✓ Generated {len(df)} associations")
    print(f"  - Species: {len(species_names)}")
    print(f"  - Genes: {len(gene_names)}")
    print(f"  - Similarity range: [{df['similarity'].min():.3f}, {df['similarity'].max():.3f}]")
    print()
    
    # ========================================================================
    # PART 2: DATA PREPROCESSING
    # ========================================================================
    print("PART 2: DATA PREPROCESSING")
    print("-" * 80)
    
    print("Loading and preprocessing data...")
    
    # Initialize data loader
    data_loader = SpeciesGeneDataLoader()
    df = data_loader.load_data(data=df)
    data_loader.build_mappings(df)
    
    # Encode data
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    print(f"✓ Data encoded successfully")
    print(f"  - Unique species: {data_loader.get_num_species()}")
    print(f"  - Unique genes: {data_loader.get_num_genes()}")
    print()
    
    # Split data
    train_data, val_data, test_data = split_data(
        gene_a_ids, gene_b_ids, similarities,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    print(f"✓ Data split into train/val/test sets")
    print(f"  - Train: {len(train_data[0])} samples")
    print(f"  - Val: {len(val_data[0])} samples")
    print(f"  - Test: {len(test_data[0])} samples")
    print()
    
    # ========================================================================
    # PART 3: GRAPH CONSTRUCTION
    # ========================================================================
    print("PART 3: HETEROGENEOUS GRAPH CONSTRUCTION")
    print("-" * 80)
    
    print("Building heterogeneous graph...")
    
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
    
    print(f"✓ Graph constructed successfully")
    print(f"  - Node types: {hetero_data.node_types}")
    print(f"  - Edge types:")
    for edge_type in hetero_data.edge_types:
        num_edges = hetero_data[edge_type].edge_index.shape[1]
        print(f"    • {edge_type}: {num_edges} edges")
    print()
    
    # Prepare edge indices and labels for training
    train_edge_index = graph_builder.get_edge_label_index(train_data[0], train_data[1])
    train_labels = graph_builder.get_edge_labels(train_data[2])
    
    val_edge_index = graph_builder.get_edge_label_index(val_data[0], val_data[1])
    val_labels = graph_builder.get_edge_labels(val_data[2])
    
    test_edge_index = graph_builder.get_edge_label_index(test_data[0], test_data[1])
    test_labels = graph_builder.get_edge_labels(test_data[2])
    
    # ========================================================================
    # PART 4: MODEL CREATION
    # ========================================================================
    print("PART 4: MODEL CREATION")
    print("-" * 80)
    
    print("Creating GNN model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  - Device: {device}")
    
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Embedding dim: 64")
    print(f"  - Hidden dim: 128")
    print(f"  - Num layers: 2")
    print()
    
    # ========================================================================
    # PART 5: TRAINING
    # ========================================================================
    print("PART 5: MODEL TRAINING")
    print("-" * 80)
    
    print("Training model...")
    
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
    print(f"✓ Training completed")
    print(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_metrics']:
        final_val = history['val_metrics'][-1]
        print(f"  - Final val RMSE: {final_val['rmse']:.4f}")
        print(f"  - Final val MAE: {final_val['mae']:.4f}")
    print()
    
    # Plot training history
    plot_training_history(history)
    print()
    
    # ========================================================================
    # PART 6: EVALUATION
    # ========================================================================
    print("PART 6: MODEL EVALUATION")
    print("-" * 80)
    
    print("Evaluating on test set...")
    
    test_metrics = trainer.evaluate(hetero_data, test_edge_index, test_labels)
    
    print(f"✓ Test evaluation completed")
    print(f"  - MSE: {test_metrics['mse']:.4f}")
    print(f"  - RMSE: {test_metrics['rmse']:.4f}")
    print(f"  - MAE: {test_metrics['mae']:.4f}")
    print()
    
    # ========================================================================
    # PART 7: INFERENCE
    # ========================================================================
    print("PART 7: MAKING PREDICTIONS")
    print("-" * 80)
    
    print("Making predictions on test set...")
    
    model.eval()
    with torch.no_grad():
        test_gene_a = torch.tensor(test_data[0], dtype=torch.long).to(device)
        test_gene_b = torch.tensor(test_data[1], dtype=torch.long).to(device)
        hetero_data = hetero_data.to(device)
        
        predictions = model.predict(hetero_data, test_gene_a, test_gene_b)
        predictions = predictions.cpu().numpy()
    
    print(f"✓ Generated {len(predictions)} predictions")
    print()
    
    # Show sample predictions
    print("Sample predictions:")
    print(f"{'Gene A':<12} {'Gene B':<12} {'True':<8} {'Predicted':<10} {'Error':<8}")
    print("-" * 58)
    
    for i in range(min(10, len(test_data[0]))):
        gene_a_name = data_loader.id_to_gene[test_data[0][i]]
        gene_b_name = data_loader.id_to_gene[test_data[1][i]]
        true_val = test_data[2][i]
        pred_val = predictions[i]
        error = abs(true_val - pred_val)
        print(f"{gene_a_name:<12} {gene_b_name:<12} {true_val:<8.4f} {pred_val:<10.4f} {error:<8.4f}")
    
    print()
    
    # ========================================================================
    # PART 8: SAVING MODEL
    # ========================================================================
    print("PART 8: SAVING MODEL")
    print("-" * 80)
    
    model_path = 'checkpoints/example_model.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)
    
    print(f"✓ Model saved to: {model_path}")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ Successfully completed end-to-end example")
    print(f"  - Data: {len(df)} associations, {num_species} species, {num_genes} genes")
    print(f"  - Model: {num_params:,} parameters")
    print(f"  - Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  - Test MAE: {test_metrics['mae']:.4f}")
    print()
    print("Next steps:")
    print("  1. Try with your own data")
    print("  2. Experiment with different hyperparameters")
    print("  3. Add more GNN layers or change architecture")
    print("  4. Implement advanced features (attention, graph pooling, etc.)")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
