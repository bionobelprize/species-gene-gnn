# Species-Gene Association Graph Neural Network Model

**物种基因关联图神经网络模型** - A heterogeneous Graph Neural Network (GNN) model for learning and predicting cross-species gene functional similarities.

## 📋 Overview

This project implements a heterogeneous graph neural network model designed to learn and predict gene function similarities across different species. The model maps discrete entities (species and genes) to continuous vector spaces (embeddings), transforming sparse five-dimensional relationships into dense, deep learning-compatible data structures.

### Core Objectives

- **Input Format**: Five-dimensional vectors (species_a, gene_a, species_b, gene_b, protein_similarity)
- **Task**: Link prediction for gene-gene relationships
- **Goal**: Learn species and gene embeddings to accurately predict similarity scores between gene pairs
- **Approach**: Heterogeneous GNN with multiple node types and edge types

## 🏗️ Architecture

### Graph Structure

The heterogeneous graph consists of:

**Node Types:**
- `species`: Species nodes
- `gene`: Gene nodes

**Edge Types:**
- `(species, has_gene, gene)`: Species contains gene
- `(gene, belongs_to, species)`: Gene belongs to species
- `(gene, similar_to, gene)`: Gene similarity relationships (prediction target)

### Model Components

1. **Learnable Embeddings**: Initial embeddings for species and genes
2. **Heterogeneous Convolution Layers**: Message passing across different edge types
3. **Link Prediction Head**: Predicts similarity scores for gene pairs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bionobelprize/species-gene-gnn.git
cd species-gene-gnn

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import pandas as pd
from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN
from src.trainer import Trainer

# 1. Load data
data_loader = SpeciesGeneDataLoader()
df = data_loader.load_data('path/to/data.csv')
data_loader.build_mappings(df)

# 2. Encode data
species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
    data_loader.encode_data(df)

# 3. Build graph
graph_builder = HeteroGraphBuilder()
hetero_data = graph_builder.build_graph(
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
    data_loader.get_num_species(), data_loader.get_num_genes()
)

# 4. Create model
model = SpeciesGeneGNN(
    num_species=data_loader.get_num_species(),
    num_genes=data_loader.get_num_genes(),
    embedding_dim=64,
    hidden_dim=128,
    num_layers=2
)

# 5. Train model
trainer = Trainer(model, learning_rate=0.001)
history = trainer.train(
    data=hetero_data,
    train_edge_index=train_edge_index,
    train_labels=train_labels,
    num_epochs=100
)

# 6. Make predictions
predictions = model.predict(hetero_data, gene_a_ids, gene_b_ids)
```

### Run Example

```bash
python examples/demo.py
```

## 📊 Data Format

### Input CSV Format

Your data should be in CSV format with the following columns:

| species_a | gene_a | species_b | gene_b | similarity |
|-----------|--------|-----------|--------|------------|
| Human     | BRCA1  | Mouse     | Brca1  | 0.89       |
| Human     | TP53   | Rat       | Tp53   | 0.92       |
| ...       | ...    | ...       | ...    | ...        |

### Data Description

- `species_a`: Name/ID of the first species
- `gene_a`: Name/ID of a gene from species A
- `species_b`: Name/ID of the second species
- `gene_b`: Name/ID of a gene from species B
- `similarity`: Protein similarity score (0-1 range recommended)

## 🔧 Model Configuration

### SpeciesGeneGNN Parameters

```python
model = SpeciesGeneGNN(
    num_species=10,        # Number of unique species
    num_genes=1000,        # Number of unique genes
    embedding_dim=64,      # Dimension of initial embeddings
    hidden_dim=128,        # Dimension of hidden layers
    num_layers=2,          # Number of GNN layers
    dropout=0.1            # Dropout rate
)
```

### Training Parameters

```python
trainer = Trainer(
    model=model,
    learning_rate=0.001,   # Learning rate
    weight_decay=1e-5,     # Weight decay for regularization
    device='cpu'           # 'cpu' or 'cuda'
)
```

## 📈 Evaluation Metrics

The model reports the following metrics:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and ground truth
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **MAE (Mean Absolute Error)**: Average absolute difference
- **Loss**: Training/validation loss

## 🔬 Model Details

### Embedding Strategy

The model learns continuous embeddings for both species and genes through:

1. **Initial Embeddings**: Learnable embedding tables for species and genes
2. **Message Passing**: Heterogeneous convolution layers aggregate information from neighbors
3. **Link Prediction**: Concatenate gene embeddings and pass through MLP to predict similarity

### Training Objective

The model minimizes Mean Squared Error (MSE) between predicted and actual similarity scores:

```
Loss = MSE(predicted_similarity, actual_similarity)
```

## 📁 Project Structure

```
species-gene-gnn/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── graph_builder.py     # Heterogeneous graph construction
│   ├── model.py             # GNN model architecture
│   ├── trainer.py           # Training pipeline
│   └── utils.py             # Utility functions
├── examples/
│   └── demo.py              # Example usage script
├── tests/
│   └── (test files)         # Unit tests
├── data/
│   └── (data files)         # Data directory
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🧪 Testing

Run tests with:

```bash
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@software{species_gene_gnn,
  title={Species-Gene Association Graph Neural Network Model},
  author={BioNobel Prize},
  year={2025},
  url={https://github.com/bionobelprize/species-gene-gnn}
}
```

## 🔗 Related Work

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graph Neural Networks: https://distill.pub/2021/gnn-intro/
- Link Prediction: https://pytorch-geometric.readthedocs.io/en/latest/notes/link_pred.html

## 📞 Contact

For questions or issues, please open an issue on GitHub.
