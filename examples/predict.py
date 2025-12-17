"""
Inference script for Species-Gene GNN model.

This script loads a trained model and makes predictions on new data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import pandas as pd

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make predictions with trained Species-Gene GNN model'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to CSV data file (should have same structure as training data)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='predictions.csv',
        help='Path to save predictions (default: predictions.csv)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda) (default: auto)'
    )
    
    # Model configuration (must match training)
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
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    print("=" * 70)
    print("Species-Gene GNN Inference")
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
    
    # Build mappings
    data_loader.build_mappings(df)
    print(f"Number of unique species: {data_loader.get_num_species()}")
    print(f"Number of unique genes: {data_loader.get_num_genes()}")
    print()
    
    # Step 2: Encode data
    print("Step 2: Encoding data...")
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
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
    print(f"Graph built successfully")
    print()
    
    # Step 4: Load model
    print("Step 4: Loading trained model...")
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    trainer = Trainer(model=model, device=device)
    id_to_species, id_to_gene = trainer.load_model(args.model_path)
    print(f"Model loaded from: {args.model_path}")
    
    if id_to_species is not None and id_to_gene is not None:
        print(f"Loaded mappings: {len(id_to_species)} species, {len(id_to_gene)} genes")
    else:
        print("Warning: No species/gene mappings found in checkpoint")
    print()
    
    # Step 5: Make predictions
    print("Step 5: Making predictions...")
    model.eval()
    with torch.no_grad():
        gene_a_tensor = torch.tensor(gene_a_ids, dtype=torch.long).to(device)
        gene_b_tensor = torch.tensor(gene_b_ids, dtype=torch.long).to(device)
        hetero_data = hetero_data.to(device)
        
        predictions = model.predict(hetero_data, gene_a_tensor, gene_b_tensor)
        predictions = predictions.cpu().numpy()
    
    print(f"Generated {len(predictions)} predictions")
    print()
    
    # Step 6: Save predictions
    print("Step 6: Saving predictions...")
    results_df = df.copy()
    results_df['predicted_similarity'] = predictions
    results_df['error'] = abs(results_df['similarity'] - results_df['predicted_similarity'])
    
    results_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to: {args.output_path}")
    print()
    
    # Step 7: Display statistics
    print("Step 7: Prediction statistics...")
    print(f"Mean absolute error: {results_df['error'].mean():.4f}")
    print(f"Median absolute error: {results_df['error'].median():.4f}")
    print(f"Max error: {results_df['error'].max():.4f}")
    print()
    
    # Show sample predictions
    print("Sample predictions:")
    print(results_df[['species_a', 'gene_a', 'species_b', 'gene_b', 'similarity', 'predicted_similarity', 'error']].head(10))
    print()
    
    print("=" * 70)
    print("Inference completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
