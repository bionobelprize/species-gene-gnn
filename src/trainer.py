"""
Training pipeline for Species-Gene GNN.

This module implements the training loop, loss functions, and evaluation metrics
for the link prediction task.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .model import SpeciesGeneGNN


class Trainer:
    """
    Trainer class for Species-Gene GNN model.
    
    Handles training, validation, and evaluation of the link prediction model.
    """
    
    def __init__(
        self,
        model: SpeciesGeneGNN,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cpu'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SpeciesGeneGNN model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
    def train_epoch(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        edge_labels: torch.Tensor
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            data: HeteroData object containing the graph
            edge_label_index: Edge indices for supervision [2, num_edges]
            edge_labels: Ground truth similarity scores
            
        Returns:
            Average training loss
        """
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        edge_label_index = edge_label_index.to(self.device)
        edge_labels = edge_labels.to(self.device)
        
        # Forward pass
        x_dict = self.model.forward(data)
        gene_embeddings = x_dict['gene']
        
        # Predict similarities
        predictions = self.model.predict_link(gene_embeddings, edge_label_index)
        
        # Compute loss
        loss = self.criterion(predictions, edge_labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        edge_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            data: HeteroData object containing the graph
            edge_label_index: Edge indices for evaluation [2, num_edges]
            edge_labels: Ground truth similarity scores
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Move data to device
        data = data.to(self.device)
        edge_label_index = edge_label_index.to(self.device)
        edge_labels = edge_labels.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            x_dict = self.model.forward(data)
            gene_embeddings = x_dict['gene']
            
            # Predict similarities
            predictions = self.model.predict_link(gene_embeddings, edge_label_index)
            
            # Compute loss
            loss = self.criterion(predictions, edge_labels)
            
            # Move to CPU for sklearn metrics
            predictions_np = predictions.cpu().numpy()
            labels_np = edge_labels.cpu().numpy()
            
            # Compute metrics
            mse = mean_squared_error(labels_np, predictions_np)
            mae = mean_absolute_error(labels_np, predictions_np)
            rmse = np.sqrt(mse)
            
        return {
            'loss': loss.item(),
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def train(
        self,
        data: HeteroData,
        train_edge_index: torch.Tensor,
        train_labels: torch.Tensor,
        val_edge_index: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            data: HeteroData object containing the graph
            train_edge_index: Training edge indices
            train_labels: Training similarity labels
            val_edge_index: Validation edge indices (optional)
            val_labels: Validation similarity labels (optional)
            num_epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(data, train_edge_index, train_labels)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_edge_index is not None and val_labels is not None:
                val_metrics = self.evaluate(data, val_edge_index, val_labels)
                history['val_metrics'].append(val_metrics)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}")
                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val Loss: {val_metrics['loss']:.4f}")
                    print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
                    print(f"  Val MAE: {val_metrics['mae']:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        return history
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
