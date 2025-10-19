"""
Training script for Species-Gene GNN model.

This script provides a command-line interface for training the model
on custom data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Species-Gene GNN model'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of training data (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of validation data (default: 0.15)'
    )
    parser.add_argument(
        '--genome-subset-ratio',
        type=float,
        default=1.0,
        help='Ratio of genomes (species) to use for training (default: 1.0, range: 0.0-1.0)'
    )
    
    # Model arguments
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=64,
        help='Dimension of initial embeddings (default: 64)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Dimension of hidden layers (default: 128)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of GNN layers (default: 2)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )
    
    # Training arguments
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay (default: 1e-5)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda) (default: auto)'
    )
    
    # Output arguments
    parser.add_argument(
        '--model-save-path',
        type=str,
        default='checkpoints/model.pt',
        help='Path to save trained model (default: checkpoints/model.pt)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed training progress'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("Species-Gene GNN Training")
    print("=" * 70)
    print()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    data_loader = SpeciesGeneDataLoader()
    df = data_loader.load_data(data_path=args.data_path)
    print(f"Loaded {len(df)} associations")
    
    # Filter by genome subset if specified
    if args.genome_subset_ratio < 1.0:
        print(f"\nFiltering to {args.genome_subset_ratio * 100:.1f}% of genomes...")
        
        # Get all unique species
        all_species = sorted(set(df['species_a'].unique()) | set(df['species_b'].unique()))
        num_species = len(all_species)
        
        # Calculate number of species to keep
        num_species_to_keep = max(1, int(num_species * args.genome_subset_ratio))
        
        # Randomly select species to keep (with fixed seed for reproducibility)
        np.random.seed(42)
        selected_species = set(np.random.choice(all_species, num_species_to_keep, replace=False))
        
        # Filter dataframe to only include selected species
        df = df[
            df['species_a'].isin(selected_species) & 
            df['species_b'].isin(selected_species)
        ]
        
        print(f"Selected {num_species_to_keep} out of {num_species} species")
        print(f"Remaining associations: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("No associations remaining after genome filtering. Try a higher genome-subset-ratio.")
    
    # Build mappings
    data_loader.build_mappings(df)
    print(f"Number of unique species: {data_loader.get_num_species()}")
    print(f"Number of unique genes: {data_loader.get_num_genes()}")
    print()
    
    # Step 2: Encode data
    print("Step 2: Encoding data...")
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    # Split data
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    indices = np.arange(len(similarities))
    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - args.train_ratio), random_state=42
    )
    
    if args.val_ratio > 0:
        val_size = args.val_ratio / (args.val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_size), random_state=42
        )
    else:
        val_idx = []
        test_idx = temp_idx
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print()
    
    # Step 3: Build graph
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
    
    print(f"Graph built with {hetero_data.num_nodes} nodes")
    print(f"Edge types: {hetero_data.edge_types}")
    print()
    
    # Prepare training data
    train_edge_index = graph_builder.get_edge_label_index(
        gene_a_ids[train_idx],
        gene_b_ids[train_idx]
    )
    train_labels = graph_builder.get_edge_labels(similarities[train_idx])
    
    val_edge_index = None
    val_labels = None
    if len(val_idx) > 0:
        val_edge_index = graph_builder.get_edge_label_index(
            gene_a_ids[val_idx],
            gene_b_ids[val_idx]
        )
        val_labels = graph_builder.get_edge_labels(similarities[val_idx])
    
    # Step 4: Create model
    print("Step 4: Creating model...")
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print()
    
    # Step 5: Train model
    print("Step 5: Training model...")
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device
    )
    
    history = trainer.train(
        data=hetero_data,
        train_edge_index=train_edge_index,
        train_labels=train_labels,
        val_edge_index=val_edge_index,
        val_labels=val_labels,
        num_epochs=args.num_epochs,
        verbose=args.verbose
    )
    print()
    
    # Step 6: Evaluate on test set
    if len(test_idx) > 0:
        print("Step 6: Evaluating on test set...")
        test_edge_index = graph_builder.get_edge_label_index(
            gene_a_ids[test_idx],
            gene_b_ids[test_idx]
        )
        test_labels = graph_builder.get_edge_labels(similarities[test_idx])
        
        test_metrics = trainer.evaluate(hetero_data, test_edge_index, test_labels)
        
        print(f"Test Metrics:")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print()
    
    # Step 7: Save model
    print("Step 7: Saving model...")
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    trainer.save_model(args.model_save_path)
    print(f"Model saved to: {args.model_save_path}")
    print()
    
    print("=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
