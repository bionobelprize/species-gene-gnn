"""
Embedding accessor module for Species-Gene GNN.

This module provides utilities to access and manipulate embeddings from the trained model,
including retrieving embeddings for specific entities, computing similarities, and finding
nearest neighbors.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple, Dict, Optional
from torch_geometric.data import HeteroData
from .model import SpeciesGeneGNN
from .data_loader import SpeciesGeneDataLoader


class EmbeddingAccessor:
    """
    Provides access to embeddings and embedding-related operations for the Species-Gene GNN model.
    
    This class allows users to:
    - Retrieve embeddings for specific species or genes
    - Compute similarities between entities
    - Find nearest neighbors
    - Batch retrieve embeddings
    """
    
    def __init__(
        self,
        model: SpeciesGeneGNN,
        data_loader: SpeciesGeneDataLoader,
        device: str = 'cpu'
    ):
        """
        Initialize the embedding accessor.
        
        Args:
            model: Trained SpeciesGeneGNN model
            data_loader: Data loader with mappings
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.data_loader = data_loader
        self.device = device
        
        # Cache for computed embeddings
        self._cached_embeddings = {}
        self._cache_valid = False
    
    def _invalidate_cache(self):
        """Invalidate the embedding cache."""
        self._cached_embeddings = {}
        self._cache_valid = False
    
    def _get_or_compute_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Get cached embeddings or compute them if cache is invalid.
        
        Args:
            data: HeteroData object containing the graph
            
        Returns:
            Dictionary of node embeddings for each node type
        """
        if not self._cache_valid:
            with torch.no_grad():
                data = data.to(self.device)
                self._cached_embeddings = self.model.forward(data)
                self._cache_valid = True
        return self._cached_embeddings
    
    def get_raw_species_embedding(
        self,
        species: Union[str, int],
        as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the raw (initial) embedding for a species.
        
        Args:
            species: Species name (str) or ID (int)
            as_numpy: Whether to return as numpy array instead of tensor
            
        Returns:
            Embedding vector for the species
        """
        # Convert name to ID if necessary
        if isinstance(species, str):
            if species not in self.data_loader.species_to_id:
                raise ValueError(f"Species '{species}' not found in data")
            species_id = self.data_loader.species_to_id[species]
        else:
            species_id = species
            if species_id < 0 or species_id >= self.model.num_species:
                raise ValueError(f"Species ID {species_id} out of range [0, {self.model.num_species})")
        
        with torch.no_grad():
            species_id_tensor = torch.tensor([species_id], device=self.device)
            embedding = self.model.species_embedding(species_id_tensor).squeeze(0)
        
        if as_numpy:
            return embedding.cpu().numpy()
        return embedding
    
    def get_raw_gene_embedding(
        self,
        gene: Union[str, int],
        as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the raw (initial) embedding for a gene.
        
        Args:
            gene: Gene name (str) or ID (int)
            as_numpy: Whether to return as numpy array instead of tensor
            
        Returns:
            Embedding vector for the gene
        """
        # Convert name to ID if necessary
        if isinstance(gene, str):
            if gene not in self.data_loader.gene_to_id:
                raise ValueError(f"Gene '{gene}' not found in data")
            gene_id = self.data_loader.gene_to_id[gene]
        else:
            gene_id = gene
            if gene_id < 0 or gene_id >= self.model.num_genes:
                raise ValueError(f"Gene ID {gene_id} out of range [0, {self.model.num_genes})")
        
        with torch.no_grad():
            gene_id_tensor = torch.tensor([gene_id], device=self.device)
            embedding = self.model.gene_embedding(gene_id_tensor).squeeze(0)
        
        if as_numpy:
            return embedding.cpu().numpy()
        return embedding
    
    def get_transformed_species_embedding(
        self,
        species: Union[str, int],
        data: HeteroData,
        as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the transformed (post-GNN) embedding for a species.
        
        Args:
            species: Species name (str) or ID (int)
            data: HeteroData object containing the graph
            as_numpy: Whether to return as numpy array instead of tensor
            
        Returns:
            Transformed embedding vector for the species
        """
        # Convert name to ID if necessary
        if isinstance(species, str):
            if species not in self.data_loader.species_to_id:
                raise ValueError(f"Species '{species}' not found in data")
            species_id = self.data_loader.species_to_id[species]
        else:
            species_id = species
            if species_id < 0 or species_id >= self.model.num_species:
                raise ValueError(f"Species ID {species_id} out of range [0, {self.model.num_species})")
        
        # Get embeddings through forward pass
        x_dict = self._get_or_compute_embeddings(data)
        embedding = x_dict['species'][species_id]
        
        if as_numpy:
            return embedding.cpu().numpy()
        return embedding
    
    def get_transformed_gene_embedding(
        self,
        gene: Union[str, int],
        data: HeteroData,
        as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get the transformed (post-GNN) embedding for a gene.
        
        Args:
            gene: Gene name (str) or ID (int)
            data: HeteroData object containing the graph
            as_numpy: Whether to return as numpy array instead of tensor
            
        Returns:
            Transformed embedding vector for the gene
        """
        # Convert name to ID if necessary
        if isinstance(gene, str):
            if gene not in self.data_loader.gene_to_id:
                raise ValueError(f"Gene '{gene}' not found in data")
            gene_id = self.data_loader.gene_to_id[gene]
        else:
            gene_id = gene
            if gene_id < 0 or gene_id >= self.model.num_genes:
                raise ValueError(f"Gene ID {gene_id} out of range [0, {self.model.num_genes})")
        
        # Get embeddings through forward pass
        x_dict = self._get_or_compute_embeddings(data)
        embedding = x_dict['gene'][gene_id]
        
        if as_numpy:
            return embedding.cpu().numpy()
        return embedding
    
    def get_batch_raw_embeddings(
        self,
        entity_type: str,
        entities: List[Union[str, int]],
        as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get raw embeddings for a batch of entities.
        
        Args:
            entity_type: Type of entities ('species' or 'gene')
            entities: List of entity names (str) or IDs (int)
            as_numpy: Whether to return as numpy array instead of tensor
            
        Returns:
            Batch of embedding vectors [num_entities, embedding_dim]
        """
        if entity_type == 'species':
            embeddings = [self.get_raw_species_embedding(e, as_numpy=False) for e in entities]
        elif entity_type == 'gene':
            embeddings = [self.get_raw_gene_embedding(e, as_numpy=False) for e in entities]
        else:
            raise ValueError(f"Unknown entity_type: {entity_type}. Must be 'species' or 'gene'")
        
        batch = torch.stack(embeddings)
        
        if as_numpy:
            return batch.cpu().numpy()
        return batch
    
    def get_batch_transformed_embeddings(
        self,
        entity_type: str,
        entities: List[Union[str, int]],
        data: HeteroData,
        as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get transformed embeddings for a batch of entities.
        
        Args:
            entity_type: Type of entities ('species' or 'gene')
            entities: List of entity names (str) or IDs (int)
            data: HeteroData object containing the graph
            as_numpy: Whether to return as numpy array instead of tensor
            
        Returns:
            Batch of transformed embedding vectors [num_entities, hidden_dim]
        """
        if entity_type == 'species':
            embeddings = [self.get_transformed_species_embedding(e, data, as_numpy=False) for e in entities]
        elif entity_type == 'gene':
            embeddings = [self.get_transformed_gene_embedding(e, data, as_numpy=False) for e in entities]
        else:
            raise ValueError(f"Unknown entity_type: {entity_type}. Must be 'species' or 'gene'")
        
        batch = torch.stack(embeddings)
        
        if as_numpy:
            return batch.cpu().numpy()
        return batch
    
    def compute_similarity(
        self,
        embedding1: Union[torch.Tensor, np.ndarray],
        embedding2: Union[torch.Tensor, np.ndarray],
        method: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity metric ('cosine', 'euclidean', or 'dot')
            
        Returns:
            Similarity score
        """
        # Convert to tensors if numpy arrays
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1).to(self.device)
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2).to(self.device)
        
        with torch.no_grad():
            if method == 'cosine':
                similarity = F.cosine_similarity(
                    embedding1.unsqueeze(0),
                    embedding2.unsqueeze(0)
                ).item()
            elif method == 'euclidean':
                # Return negative distance (higher is more similar)
                distance = torch.norm(embedding1 - embedding2).item()
                similarity = -distance
            elif method == 'dot':
                similarity = torch.dot(embedding1, embedding2).item()
            else:
                raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity
    
    def find_nearest_neighbors(
        self,
        entity: Union[str, int],
        entity_type: str,
        data: HeteroData,
        k: int = 5,
        use_transformed: bool = True,
        similarity_method: str = 'cosine'
    ) -> List[Tuple[Union[str, int], float]]:
        """
        Find k nearest neighbors for a given entity.
        
        Args:
            entity: Entity name (str) or ID (int)
            entity_type: Type of entity ('species' or 'gene')
            data: HeteroData object containing the graph
            k: Number of nearest neighbors to return
            use_transformed: Whether to use transformed embeddings (True) or raw (False)
            similarity_method: Similarity metric ('cosine', 'euclidean', or 'dot')
            
        Returns:
            List of (entity_name, similarity_score) tuples, sorted by similarity (descending)
        """
        # Get the target embedding
        if use_transformed:
            if entity_type == 'species':
                target_emb = self.get_transformed_species_embedding(entity, data, as_numpy=False)
            elif entity_type == 'gene':
                target_emb = self.get_transformed_gene_embedding(entity, data, as_numpy=False)
            else:
                raise ValueError(f"Unknown entity_type: {entity_type}")
        else:
            if entity_type == 'species':
                target_emb = self.get_raw_species_embedding(entity, as_numpy=False)
            elif entity_type == 'gene':
                target_emb = self.get_raw_gene_embedding(entity, as_numpy=False)
            else:
                raise ValueError(f"Unknown entity_type: {entity_type}")
        
        # Convert entity to ID if necessary
        if isinstance(entity, str):
            if entity_type == 'species':
                entity_id = self.data_loader.species_to_id[entity]
            else:
                entity_id = self.data_loader.gene_to_id[entity]
        else:
            entity_id = entity
        
        # Get all embeddings of the same type
        if entity_type == 'species':
            num_entities = self.model.num_species
            id_to_name = self.data_loader.id_to_species
            if use_transformed:
                x_dict = self._get_or_compute_embeddings(data)
                all_embeddings = x_dict['species']
            else:
                all_ids = torch.arange(num_entities, device=self.device)
                all_embeddings = self.model.species_embedding(all_ids)
        else:  # gene
            num_entities = self.model.num_genes
            id_to_name = self.data_loader.id_to_gene
            if use_transformed:
                x_dict = self._get_or_compute_embeddings(data)
                all_embeddings = x_dict['gene']
            else:
                all_ids = torch.arange(num_entities, device=self.device)
                all_embeddings = self.model.gene_embedding(all_ids)
        
        # Compute similarities
        similarities = []
        with torch.no_grad():
            for i in range(num_entities):
                if i == entity_id:
                    continue  # Skip self
                
                emb = all_embeddings[i]
                sim = self.compute_similarity(target_emb, emb, method=similarity_method)
                similarities.append((i, sim))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        # Convert IDs to names
        result = [(id_to_name[idx], score) for idx, score in top_k]
        
        return result
    
    def get_all_embeddings(
        self,
        entity_type: str,
        data: Optional[HeteroData] = None,
        use_transformed: bool = True,
        as_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Get all embeddings for a given entity type.
        
        Args:
            entity_type: Type of entities ('species' or 'gene')
            data: HeteroData object (required if use_transformed=True)
            use_transformed: Whether to use transformed embeddings (True) or raw (False)
            as_numpy: Whether to return as numpy array instead of tensor
            
        Returns:
            Matrix of embeddings [num_entities, embedding_dim or hidden_dim]
        """
        if entity_type not in ['species', 'gene']:
            raise ValueError(f"Unknown entity_type: {entity_type}. Must be 'species' or 'gene'")
        
        with torch.no_grad():
            if use_transformed:
                if data is None:
                    raise ValueError("data must be provided when use_transformed=True")
                x_dict = self._get_or_compute_embeddings(data)
                embeddings = x_dict[entity_type]
            else:
                if entity_type == 'species':
                    num_entities = self.model.num_species
                    all_ids = torch.arange(num_entities, device=self.device)
                    embeddings = self.model.species_embedding(all_ids)
                else:  # gene
                    num_entities = self.model.num_genes
                    all_ids = torch.arange(num_entities, device=self.device)
                    embeddings = self.model.gene_embedding(all_ids)
        
        if as_numpy:
            return embeddings.cpu().numpy()
        return embeddings
    
    def get_embedding_info(self) -> Dict[str, any]:
        """
        Get information about the embeddings.
        
        Returns:
            Dictionary containing embedding dimensions and entity counts
        """
        return {
            'num_species': self.model.num_species,
            'num_genes': self.model.num_genes,
            'raw_embedding_dim': self.model.embedding_dim,
            'transformed_embedding_dim': self.model.hidden_dim,
            'species_names': list(self.data_loader.species_to_id.keys()),
            'gene_names': list(self.data_loader.gene_to_id.keys()),
        }
