"""
Example script demonstrating the EmbeddingAccessor functionality.

This script shows how to:
1. Train a simple model
2. Access embeddings for specific species and genes
3. Compute similarities between embeddings
4. Find nearest neighbors
5. Batch retrieve embeddings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch

from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer
from src.embedding_accessor import EmbeddingAccessor


def create_sample_data(num_samples: int = 500) -> pd.DataFrame:
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    species_list = ['Human', 'Mouse', 'Rat', 'Zebrafish', 'Fruit_Fly']
    gene_pool = [f'Gene_{i:04d}' for i in range(100)]
    
    data = []
    for _ in range(num_samples):
        species_a = np.random.choice(species_list)
        species_b = np.random.choice(species_list)
        gene_a = np.random.choice(gene_pool)
        gene_b = np.random.choice(gene_pool)
        
        if gene_a == gene_b:
            similarity = np.random.uniform(0.8, 1.0)
        else:
            gene_a_idx = int(gene_a.split('_')[1])
            gene_b_idx = int(gene_b.split('_')[1])
            base_sim = max(0, 1 - abs(gene_a_idx - gene_b_idx) / 100.0)
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
    """Main function to run the embedding demonstration."""
    print("=" * 80)
    print("Species-Gene GNN - Embedding Accessor Demonstration")
    print("=" * 80)
    print()
    
    # Step 1: Create and prepare data
    print("Step 1: Creating sample data...")
    df = create_sample_data(num_samples=500)
    
    data_loader = SpeciesGeneDataLoader()
    df = data_loader.load_data(data=df)
    data_loader.build_mappings(df)
    
    print(f"Number of species: {data_loader.get_num_species()}")
    print(f"Number of genes: {data_loader.get_num_genes()}")
    print()
    
    # Encode data
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    # Step 2: Build graph
    print("Step 2: Building heterogeneous graph...")
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
    print(f"Graph created with {hetero_data.num_nodes} nodes")
    print()
    
    # Step 3: Create and train model
    print("Step 3: Training a simple model...")
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2
    )
    
    # Prepare training data
    train_edge_index = graph_builder.get_edge_label_index(gene_a_ids, gene_b_ids)
    train_labels = graph_builder.get_edge_labels(similarities)
    
    trainer = Trainer(model=model, learning_rate=0.01)
    trainer.train(
        data=hetero_data,
        train_edge_index=train_edge_index,
        train_labels=train_labels,
        num_epochs=20,
        verbose=False
    )
    print("Model trained!")
    print()
    
    # Step 4: Create embedding accessor
    print("Step 4: Creating embedding accessor...")
    accessor = EmbeddingAccessor(model, data_loader, device='cpu')
    print()
    
    # Step 5: Demonstrate embedding retrieval
    print("=" * 80)
    print("DEMONSTRATION: Embedding Retrieval")
    print("=" * 80)
    print()
    
    # Get embedding info
    print("Embedding Information:")
    info = accessor.get_embedding_info()
    print(f"  Raw embedding dimension: {info['raw_embedding_dim']}")
    print(f"  Transformed embedding dimension: {info['transformed_embedding_dim']}")
    print(f"  Number of species: {info['num_species']}")
    print(f"  Number of genes: {info['num_genes']}")
    print()
    
    # Get raw embedding for a specific species
    print("Example 1: Get raw embedding for 'Human' species")
    human_raw_emb = accessor.get_raw_species_embedding('Human', as_numpy=True)
    print(f"  Shape: {human_raw_emb.shape}")
    print(f"  First 5 values: {human_raw_emb[:5]}")
    print()
    
    # Get transformed embedding for a specific gene
    print("Example 2: Get transformed embedding for 'Gene_0001'")
    gene_trans_emb = accessor.get_transformed_gene_embedding('Gene_0001', hetero_data, as_numpy=True)
    print(f"  Shape: {gene_trans_emb.shape}")
    print(f"  First 5 values: {gene_trans_emb[:5]}")
    print()
    
    # Get batch embeddings
    print("Example 3: Get batch raw embeddings for multiple species")
    species_batch = ['Human', 'Mouse', 'Rat']
    batch_emb = accessor.get_batch_raw_embeddings('species', species_batch, as_numpy=True)
    print(f"  Shape: {batch_emb.shape}")
    print(f"  Species: {species_batch}")
    print()
    
    # Step 6: Demonstrate similarity computation
    print("=" * 80)
    print("DEMONSTRATION: Similarity Computation")
    print("=" * 80)
    print()
    
    print("Example 4: Compute similarity between two genes")
    gene1 = 'Gene_0001'
    gene2 = 'Gene_0002'
    
    emb1 = accessor.get_transformed_gene_embedding(gene1, hetero_data)
    emb2 = accessor.get_transformed_gene_embedding(gene2, hetero_data)
    
    cosine_sim = accessor.compute_similarity(emb1, emb2, method='cosine')
    euclidean_sim = accessor.compute_similarity(emb1, emb2, method='euclidean')
    dot_sim = accessor.compute_similarity(emb1, emb2, method='dot')
    
    print(f"  {gene1} vs {gene2}:")
    print(f"    Cosine similarity: {cosine_sim:.4f}")
    print(f"    Euclidean similarity: {euclidean_sim:.4f}")
    print(f"    Dot product similarity: {dot_sim:.4f}")
    print()
    
    # Step 7: Demonstrate nearest neighbor search
    print("=" * 80)
    print("DEMONSTRATION: Nearest Neighbor Search")
    print("=" * 80)
    print()
    
    print("Example 5: Find 5 nearest neighbors for 'Gene_0010'")
    target_gene = 'Gene_0010'
    neighbors = accessor.find_nearest_neighbors(
        target_gene,
        entity_type='gene',
        data=hetero_data,
        k=5,
        use_transformed=True,
        similarity_method='cosine'
    )
    
    print(f"  Target gene: {target_gene}")
    print(f"  Top 5 nearest neighbors (by cosine similarity):")
    for i, (neighbor_name, similarity) in enumerate(neighbors, 1):
        print(f"    {i}. {neighbor_name}: {similarity:.4f}")
    print()
    
    print("Example 6: Find 3 nearest neighbors for 'Human' species")
    target_species = 'Human'
    neighbors = accessor.find_nearest_neighbors(
        target_species,
        entity_type='species',
        data=hetero_data,
        k=3,
        use_transformed=True,
        similarity_method='cosine'
    )
    
    print(f"  Target species: {target_species}")
    print(f"  Top 3 nearest neighbors:")
    for i, (neighbor_name, similarity) in enumerate(neighbors, 1):
        print(f"    {i}. {neighbor_name}: {similarity:.4f}")
    print()
    
    # Step 8: Demonstrate getting all embeddings
    print("=" * 80)
    print("DEMONSTRATION: Batch Retrieval of All Embeddings")
    print("=" * 80)
    print()
    
    print("Example 7: Get all raw gene embeddings")
    all_gene_emb = accessor.get_all_embeddings('gene', use_transformed=False, as_numpy=True)
    print(f"  Shape: {all_gene_emb.shape}")
    print(f"  ({all_gene_emb.shape[0]} genes, {all_gene_emb.shape[1]} dimensions)")
    print()
    
    print("Example 8: Get all transformed species embeddings")
    all_species_emb = accessor.get_all_embeddings(
        'species',
        data=hetero_data,
        use_transformed=True,
        as_numpy=True
    )
    print(f"  Shape: {all_species_emb.shape}")
    print(f"  ({all_species_emb.shape[0]} species, {all_species_emb.shape[1]} dimensions)")
    print()
    
    # Step 9: Practical use case - comparing gene groups
    print("=" * 80)
    print("PRACTICAL USE CASE: Comparing Gene Groups")
    print("=" * 80)
    print()
    
    print("Example 9: Compare similarity within two gene groups")
    group1 = ['Gene_0001', 'Gene_0002', 'Gene_0003']
    group2 = ['Gene_0050', 'Gene_0051', 'Gene_0052']
    
    # Get embeddings for both groups
    emb_group1 = accessor.get_batch_transformed_embeddings('gene', group1, hetero_data)
    emb_group2 = accessor.get_batch_transformed_embeddings('gene', group2, hetero_data)
    
    # Compute average embedding for each group
    avg_emb1 = emb_group1.mean(dim=0)
    avg_emb2 = emb_group2.mean(dim=0)
    
    # Compute similarity between group averages
    group_similarity = accessor.compute_similarity(avg_emb1, avg_emb2, method='cosine')
    
    print(f"  Group 1: {group1}")
    print(f"  Group 2: {group2}")
    print(f"  Similarity between group centroids: {group_similarity:.4f}")
    print()
    
    # Compute within-group similarities
    within_group1_sims = []
    for i in range(len(group1)):
        for j in range(i+1, len(group1)):
            sim = accessor.compute_similarity(emb_group1[i], emb_group1[j], method='cosine')
            within_group1_sims.append(sim)
    
    within_group2_sims = []
    for i in range(len(group2)):
        for j in range(i+1, len(group2)):
            sim = accessor.compute_similarity(emb_group2[i], emb_group2[j], method='cosine')
            within_group2_sims.append(sim)
    
    print(f"  Average within-group similarity (Group 1): {np.mean(within_group1_sims):.4f}")
    print(f"  Average within-group similarity (Group 2): {np.mean(within_group2_sims):.4f}")
    print()
    
    print("=" * 80)
    print("Demonstration completed successfully!")
    print("=" * 80)
    print()
    print("Key features demonstrated:")
    print("  ✓ Retrieving raw and transformed embeddings")
    print("  ✓ Batch retrieval of embeddings")
    print("  ✓ Computing similarities (cosine, euclidean, dot product)")
    print("  ✓ Finding nearest neighbors")
    print("  ✓ Getting all embeddings for analysis")
    print("  ✓ Practical use case: comparing gene groups")
    print()


if __name__ == '__main__':
    main()
