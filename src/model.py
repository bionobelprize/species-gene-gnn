"""
Heterogeneous Graph Neural Network model for Species-Gene associations.

This module implements the core GNN architecture for learning embeddings
of species and genes, and predicting gene-gene similarities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple


class SpeciesGeneGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for Species-Gene associations.
    
    This model learns embeddings for species and genes through message passing,
    and predicts similarity scores for gene-gene pairs.
    """
    
    def __init__(
        self,
        num_species: int,
        num_genes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the heterogeneous GNN model.
        
        Args:
            num_species: Number of unique species
            num_genes: Number of unique genes
            embedding_dim: Dimension of initial embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(SpeciesGeneGNN, self).__init__()
        
        self.num_species = num_species
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Learnable embeddings for species and genes
        self.species_embedding = nn.Embedding(num_species, embedding_dim)
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        
        # Input projection layers
        self.species_lin = Linear(embedding_dim, hidden_dim)
        self.gene_lin = Linear(embedding_dim, hidden_dim)
        
        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('species', 'has_gene', 'gene'): SAGEConv(hidden_dim, hidden_dim),
                ('gene', 'belongs_to', 'species'): SAGEConv(hidden_dim, hidden_dim),
                ('gene', 'similar_to', 'gene'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='mean')
            self.convs.append(conv)
        
        # Output layers for link prediction
        self.link_predictor = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.species_embedding.weight)
        nn.init.xavier_uniform_(self.gene_embedding.weight)
    
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the heterogeneous GNN.
        
        Args:
            data: HeteroData object containing the graph
            
        Returns:
            Dictionary of node embeddings for each node type
        """
        # Determine device from model parameters
        device = next(self.parameters()).device
        
        # Get initial embeddings
        species_x = self.species_embedding(
            torch.arange(self.num_species, device=device)
        )
        gene_x = self.gene_embedding(
            torch.arange(self.num_genes, device=device)
        )
        
        # Project to hidden dimension
        x_dict = {
            'species': self.species_lin(species_x),
            'gene': self.gene_lin(gene_x)
        }
        
        # Apply GNN layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        return x_dict
    
    def predict_link(
        self,
        gene_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict similarity scores for gene pairs.
        
        Args:
            gene_embeddings: Embeddings for gene nodes
            edge_index: Edge indices for gene pairs [2, num_edges]
            
        Returns:
            Predicted similarity scores
        """
        # Get embeddings for source and target genes
        src_embeddings = gene_embeddings[edge_index[0]]
        dst_embeddings = gene_embeddings[edge_index[1]]
        
        # Concatenate embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        
        # Predict similarity
        predictions = self.link_predictor(edge_embeddings).squeeze(-1)
        
        return predictions
    
    def predict(
        self,
        data: HeteroData,
        gene_a_ids: torch.Tensor,
        gene_b_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        End-to-end prediction for gene-gene similarities.
        
        Args:
            data: HeteroData object containing the graph
            gene_a_ids: Tensor of gene A IDs
            gene_b_ids: Tensor of gene B IDs
            
        Returns:
            Predicted similarity scores
        """
        # Get embeddings
        x_dict = self.forward(data)
        gene_embeddings = x_dict['gene']
        
        # Create edge index
        edge_index = torch.stack([gene_a_ids, gene_b_ids], dim=0)
        
        # Predict similarities
        predictions = self.predict_link(gene_embeddings, edge_index)
        
        return predictions
