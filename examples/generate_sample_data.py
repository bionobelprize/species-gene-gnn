"""
Sample data generator for testing and demonstration.

This script generates synthetic species-gene association data
in the correct format for the Species-Gene GNN model.
"""

import pandas as pd
import numpy as np
import argparse
import os


def generate_sample_data(
    num_species: int = 5,
    num_genes: int = 100,
    num_samples: int = 1000,
    output_path: str = 'data/sample_data.csv',
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic species-gene association data.
    
    Args:
        num_species: Number of unique species
        num_genes: Number of unique genes  
        num_samples: Number of association samples to generate
        output_path: Path to save the CSV file
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with generated data
    """
    np.random.seed(random_seed)
    
    # Create species names
    species_names = [f'Species_{i:02d}' for i in range(num_species)]
    
    # Create gene names
    gene_names = [f'Gene_{i:04d}' for i in range(num_genes)]
    
    # Generate associations
    data = []
    for _ in range(num_samples):
        species_a = np.random.choice(species_names)
        species_b = np.random.choice(species_names)
        
        gene_a = np.random.choice(gene_names)
        gene_b = np.random.choice(gene_names)
        
        # Generate similarity score
        # Higher similarity for same genes, lower for different genes
        if gene_a == gene_b:
            # Same gene: high similarity
            similarity = np.random.uniform(0.8, 1.0)
        else:
            # Different genes: similarity based on gene index proximity
            gene_a_idx = int(gene_a.split('_')[1])
            gene_b_idx = int(gene_b.split('_')[1])
            
            # Calculate base similarity (closer genes have higher similarity)
            distance = abs(gene_a_idx - gene_b_idx)
            max_distance = num_genes
            base_sim = max(0, 1 - (distance / max_distance))
            
            # Add some randomness
            similarity = base_sim * np.random.uniform(0.3, 0.8)
            
            # Ensure similarity is in [0, 1]
            similarity = max(0.0, min(1.0, similarity))
        
        data.append({
            'species_a': species_a,
            'gene_a': gene_a,
            'species_b': species_b,
            'gene_b': gene_b,
            'similarity': similarity
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples")
    print(f"Data saved to: {output_path}")
    print(f"\nData statistics:")
    print(f"  Number of unique species: {df['species_a'].nunique() + df['species_b'].nunique() - len(set(df['species_a']) & set(df['species_b']))}")
    print(f"  Number of unique genes: {df['gene_a'].nunique() + df['gene_b'].nunique() - len(set(df['gene_a']) & set(df['gene_b']))}")
    print(f"  Similarity range: [{df['similarity'].min():.4f}, {df['similarity'].max():.4f}]")
    print(f"  Similarity mean: {df['similarity'].mean():.4f}")
    print(f"  Similarity std: {df['similarity'].std():.4f}")
    
    return df


def main():
    """Main function to run the data generator."""
    parser = argparse.ArgumentParser(
        description='Generate sample species-gene association data'
    )
    parser.add_argument(
        '--num-species',
        type=int,
        default=5,
        help='Number of unique species (default: 5)'
    )
    parser.add_argument(
        '--num-genes',
        type=int,
        default=100,
        help='Number of unique genes (default: 100)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of association samples (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/sample_data.csv',
        help='Output CSV file path (default: data/sample_data.csv)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Species-Gene Association Data Generator")
    print("=" * 70)
    print()
    
    generate_sample_data(
        num_species=args.num_species,
        num_genes=args.num_genes,
        num_samples=args.num_samples,
        output_path=args.output,
        random_seed=args.seed
    )
    
    print()
    print("=" * 70)
    print("Sample data generation completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
