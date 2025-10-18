"""
Practical example showing how to use EmbeddingAccessor for downstream analysis.

This script demonstrates:
1. Training a model
2. Extracting embeddings
3. Performing clustering analysis on gene embeddings
4. Visualizing embedding space with dimensionality reduction
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
    """Create sample data with some structure."""
    np.random.seed(42)
    
    species_list = ['Human', 'Mouse', 'Rat', 'Zebrafish', 'Fruit_Fly']
    
    # Create gene families (genes with similar IDs should be similar)
    gene_pool = [f'Gene_{i:04d}' for i in range(50)]
    
    data = []
    for _ in range(num_samples):
        species_a = np.random.choice(species_list)
        species_b = np.random.choice(species_list)
        gene_a = np.random.choice(gene_pool)
        gene_b = np.random.choice(gene_pool)
        
        # Create similarity structure based on gene indices
        gene_a_idx = int(gene_a.split('_')[1])
        gene_b_idx = int(gene_b.split('_')[1])
        
        if gene_a == gene_b:
            similarity = np.random.uniform(0.85, 1.0)
        else:
            # Genes with nearby indices are more similar
            distance = abs(gene_a_idx - gene_b_idx)
            if distance < 5:
                similarity = np.random.uniform(0.7, 0.9)
            elif distance < 10:
                similarity = np.random.uniform(0.5, 0.7)
            else:
                similarity = np.random.uniform(0.1, 0.5)
        
        data.append({
            'species_a': species_a,
            'gene_a': gene_a,
            'species_b': species_b,
            'gene_b': gene_b,
            'similarity': similarity
        })
    
    return pd.DataFrame(data)


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Embedding Analysis Example")
    print("=" * 80)
    print()
    
    # Setup
    print("1. Preparing data and training model...")
    df = create_sample_data(num_samples=500)
    
    data_loader = SpeciesGeneDataLoader()
    df = data_loader.load_data(data=df)
    data_loader.build_mappings(df)
    
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
        data_loader.encode_data(df)
    
    graph_builder = HeteroGraphBuilder()
    hetero_data = graph_builder.build_graph(
        species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
        data_loader.get_num_species(), data_loader.get_num_genes()
    )
    
    model = SpeciesGeneGNN(
        num_species=data_loader.get_num_species(),
        num_genes=data_loader.get_num_genes(),
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2
    )
    
    train_edge_index = graph_builder.get_edge_label_index(gene_a_ids, gene_b_ids)
    train_labels = graph_builder.get_edge_labels(similarities)
    
    trainer = Trainer(model=model, learning_rate=0.01)
    trainer.train(
        data=hetero_data,
        train_edge_index=train_edge_index,
        train_labels=train_labels,
        num_epochs=30,
        verbose=False
    )
    print("   Model trained!")
    print()
    
    # Create accessor
    accessor = EmbeddingAccessor(model, data_loader, device='cpu')
    
    # Analysis 1: Gene similarity matrix
    print("2. Computing gene similarity matrix...")
    # Get embeddings for first 10 genes
    gene_subset = [f'Gene_{i:04d}' for i in range(10)]
    gene_embeddings = accessor.get_batch_transformed_embeddings(
        'gene', gene_subset, hetero_data, as_numpy=True
    )
    
    # Compute pairwise similarity matrix
    n = len(gene_subset)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = accessor.compute_similarity(
                gene_embeddings[i],
                gene_embeddings[j],
                method='cosine'
            )
    
    print(f"   Similarity matrix shape: {similarity_matrix.shape}")
    print(f"   Average similarity: {similarity_matrix.mean():.4f}")
    print(f"   Std similarity: {similarity_matrix.std():.4f}")
    print()
    
    # Analysis 2: Find gene clusters
    print("3. Finding gene clusters (nearest neighbors)...")
    clusters = {}
    sample_genes = ['Gene_0005', 'Gene_0025', 'Gene_0045']
    
    for gene in sample_genes:
        neighbors = accessor.find_nearest_neighbors(
            gene,
            entity_type='gene',
            data=hetero_data,
            k=3,
            use_transformed=True
        )
        clusters[gene] = neighbors
    
    print("   Gene clusters:")
    for gene, neighbors in clusters.items():
        print(f"   {gene}:")
        for neighbor, score in neighbors:
            print(f"     - {neighbor} (similarity: {score:.4f})")
    print()
    
    # Analysis 3: Species embedding comparison
    print("4. Comparing species embeddings...")
    species_list = list(data_loader.species_to_id.keys())
    species_embeddings = accessor.get_batch_transformed_embeddings(
        'species', species_list, hetero_data, as_numpy=True
    )
    
    print(f"   Species embeddings shape: {species_embeddings.shape}")
    print(f"   Species: {species_list}")
    print()
    
    # Compute pairwise species similarities
    print("   Pairwise species similarities:")
    for i in range(len(species_list)):
        for j in range(i + 1, len(species_list)):
            sim = accessor.compute_similarity(
                species_embeddings[i],
                species_embeddings[j],
                method='cosine'
            )
            print(f"     {species_list[i]} - {species_list[j]}: {sim:.4f}")
    print()
    
    # Analysis 4: Embedding statistics
    print("5. Embedding statistics...")
    all_gene_embeddings = accessor.get_all_embeddings(
        'gene', data=hetero_data, use_transformed=True, as_numpy=True
    )
    
    print(f"   Total genes: {all_gene_embeddings.shape[0]}")
    print(f"   Embedding dimension: {all_gene_embeddings.shape[1]}")
    print(f"   Mean embedding norm: {np.linalg.norm(all_gene_embeddings, axis=1).mean():.4f}")
    print(f"   Std embedding norm: {np.linalg.norm(all_gene_embeddings, axis=1).std():.4f}")
    print()
    
    # Analysis 5: Find most similar gene pairs
    print("6. Finding most similar gene pairs...")
    # Sample some random gene pairs
    num_samples = 20
    gene_indices = np.random.choice(all_gene_embeddings.shape[0], size=num_samples * 2, replace=False)
    
    similarities_list = []
    for i in range(0, len(gene_indices), 2):
        idx1, idx2 = gene_indices[i], gene_indices[i + 1]
        gene1 = data_loader.id_to_gene[idx1]
        gene2 = data_loader.id_to_gene[idx2]
        
        sim = accessor.compute_similarity(
            all_gene_embeddings[idx1],
            all_gene_embeddings[idx2],
            method='cosine'
        )
        similarities_list.append((gene1, gene2, sim))
    
    # Sort by similarity
    similarities_list.sort(key=lambda x: x[2], reverse=True)
    
    print("   Top 5 most similar pairs:")
    for i, (gene1, gene2, sim) in enumerate(similarities_list[:5], 1):
        print(f"     {i}. {gene1} - {gene2}: {sim:.4f}")
    print()
    
    # Analysis 6: Quality metrics
    print("7. Embedding quality metrics...")
    
    # Intra-cluster similarity (nearby genes should be similar)
    intra_similarities = []
    for i in range(0, all_gene_embeddings.shape[0] - 5, 5):
        # Compare genes with indices i to i+4 (should be somewhat similar)
        for j in range(i, i + 4):
            for k in range(j + 1, i + 5):
                sim = accessor.compute_similarity(
                    all_gene_embeddings[j],
                    all_gene_embeddings[k],
                    method='cosine'
                )
                intra_similarities.append(sim)
    
    # Inter-cluster similarity (distant genes)
    inter_similarities = []
    for i in range(0, min(10, all_gene_embeddings.shape[0])):
        for j in range(max(all_gene_embeddings.shape[0] - 10, i + 20), all_gene_embeddings.shape[0]):
            if i != j:
                sim = accessor.compute_similarity(
                    all_gene_embeddings[i],
                    all_gene_embeddings[j],
                    method='cosine'
                )
                inter_similarities.append(sim)
    
    print(f"   Average within-group similarity: {np.mean(intra_similarities):.4f}")
    print(f"   Average between-group similarity: {np.mean(inter_similarities):.4f}")
    print(f"   Separation ratio: {np.mean(intra_similarities) / np.mean(inter_similarities):.4f}")
    print()
    
    print("=" * 80)
    print("Analysis completed!")
    print("=" * 80)
    print()
    print("This example demonstrated:")
    print("  ✓ Computing similarity matrices")
    print("  ✓ Finding gene clusters via nearest neighbors")
    print("  ✓ Comparing species embeddings")
    print("  ✓ Computing embedding statistics")
    print("  ✓ Finding most similar pairs")
    print("  ✓ Quality metrics for embeddings")
    print()


if __name__ == '__main__':
    main()
