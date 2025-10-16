"""
Heterogeneous graph construction module for Species-Gene GNN.

This module builds a heterogeneous graph with species and gene nodes,
and various edge types representing their relationships.
"""

import torch
from torch_geometric.data import HeteroData
import numpy as np
from typing import Tuple, Dict


class HeteroGraphBuilder:
    """
    Builds heterogeneous graph from species-gene association data.
    
    Node types:
    - species: Species nodes
    - gene: Gene nodes
    
    Edge types:
    - (species, has_gene, gene): Species contains gene
    - (gene, belongs_to, species): Gene belongs to species
    - (gene, similar_to, gene): Gene similarity relationships
    """
    
    def __init__(self):
        self.data = None
        
    def build_graph(
        self,
        species_a_ids: np.ndarray,
        gene_a_ids: np.ndarray,
        species_b_ids: np.ndarray,
        gene_b_ids: np.ndarray,
        similarities: np.ndarray,
        num_species: int,
        num_genes: int
    ) -> HeteroData:
        """
        Build heterogeneous graph from encoded data.
        
        Args:
            species_a_ids: Array of species A IDs
            gene_a_ids: Array of gene A IDs
            species_b_ids: Array of species B IDs
            gene_b_ids: Array of gene B IDs
            similarities: Array of protein similarities
            num_species: Total number of unique species
            num_genes: Total number of unique genes
            
        Returns:
            HeteroData object representing the graph
        """
        data = HeteroData()
        
        # Create node features (initially using identity embeddings)
        data['species'].num_nodes = num_species
        data['gene'].num_nodes = num_genes
        
        # Build species-gene edges (species has gene)
        species_has_gene_edges = []
        gene_belongs_to_species_edges = []
        
        for species_id, gene_id in zip(species_a_ids, gene_a_ids):
            species_has_gene_edges.append([species_id, gene_id])
            gene_belongs_to_species_edges.append([gene_id, species_id])
        
        for species_id, gene_id in zip(species_b_ids, gene_b_ids):
            species_has_gene_edges.append([species_id, gene_id])
            gene_belongs_to_species_edges.append([gene_id, species_id])
        
        # Remove duplicates
        species_has_gene_edges = list(set(map(tuple, species_has_gene_edges)))
        gene_belongs_to_species_edges = list(set(map(tuple, gene_belongs_to_species_edges)))
        
        if species_has_gene_edges:
            species_has_gene_edges = np.array(species_has_gene_edges).T
            data['species', 'has_gene', 'gene'].edge_index = torch.tensor(
                species_has_gene_edges, dtype=torch.long
            )
        
        if gene_belongs_to_species_edges:
            gene_belongs_to_species_edges = np.array(gene_belongs_to_species_edges).T
            data['gene', 'belongs_to', 'species'].edge_index = torch.tensor(
                gene_belongs_to_species_edges, dtype=torch.long
            )
        
        # Build gene-gene similarity edges
        gene_similarity_edges = []
        gene_similarity_weights = []
        
        for gene_a, gene_b, sim in zip(gene_a_ids, gene_b_ids, similarities):
            gene_similarity_edges.append([gene_a, gene_b])
            gene_similarity_weights.append(sim)
            # Add reverse edge for undirected graph
            gene_similarity_edges.append([gene_b, gene_a])
            gene_similarity_weights.append(sim)
        
        if gene_similarity_edges:
            gene_similarity_edges = np.array(gene_similarity_edges).T
            data['gene', 'similar_to', 'gene'].edge_index = torch.tensor(
                gene_similarity_edges, dtype=torch.long
            )
            data['gene', 'similar_to', 'gene'].edge_attr = torch.tensor(
                gene_similarity_weights, dtype=torch.float
            ).unsqueeze(1)
        
        self.data = data
        return data
    
    def get_edge_label_index(
        self,
        gene_a_ids: np.ndarray,
        gene_b_ids: np.ndarray
    ) -> torch.Tensor:
        """
        Get edge label index for link prediction task.
        
        Args:
            gene_a_ids: Array of gene A IDs
            gene_b_ids: Array of gene B IDs
            
        Returns:
            Edge index tensor for supervision
        """
        edges = np.stack([gene_a_ids, gene_b_ids], axis=0)
        return torch.tensor(edges, dtype=torch.long)
    
    def get_edge_labels(self, similarities: np.ndarray) -> torch.Tensor:
        """
        Get edge labels (similarities) for supervision.
        
        Args:
            similarities: Array of protein similarities
            
        Returns:
            Similarity tensor for supervision
        """
        return torch.tensor(similarities, dtype=torch.float)
