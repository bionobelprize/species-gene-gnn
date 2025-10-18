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

# Optional: Install Diamond for data preparation
# Ubuntu/Debian
sudo apt-get install diamond-aligner

# Or download from: https://github.com/bbuchfink/diamond
```

### Data Preparation (Optional)

If you have genome assembly data from NCBI Datasets, you can use the built-in data preparation module to generate the input CSV:

```python
from src.data_preparation import DataPreparation

# Initialize data preparation
prep = DataPreparation(
    catalog_path='path/to/dataset_catalog.json',
    output_dir='output'
)

# Run the complete pipeline
df = prep.prepare_data(
    max_target_seqs=5,
    output_csv='data/gene_associations.csv'
)
```

Or use the command-line interface:

```bash
python -m src.data_preparation \
    --catalog path/to/dataset_catalog.json \
    --output-dir output \
    --output-csv data/gene_associations.csv \
    --max-target-seqs 5
```

The data preparation pipeline will:
1. Read `dataset_catalog.json` to extract protein FASTA files
2. Create Diamond databases for each genome
3. Perform all-vs-all BLASTP alignments (excluding self-alignments)
4. Parse alignment results and extract best hits
5. Generate a five-dimensional CSV file (species_a, gene_a, species_b, gene_b, similarity)

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
# Run the complete demo
python examples/demo.py

# Prepare data from genome assemblies
python examples/prepare_data.py

# Generate sample data
python examples/generate_sample_data.py --num-samples 1000 --output data/sample_data.csv

# Train a model
python examples/train.py --data-path data/sample_data.csv --num-epochs 100 --verbose

# Make predictions
python examples/predict.py --model-path checkpoints/model.pt --data-path data/sample_data.csv
```

## 📊 Data Format

### dataset_catalog.json Format (for Data Preparation)

The `dataset_catalog.json` file from NCBI Datasets has the following structure:

```json
{
  "apiVersion": "V2",
  "assemblies": [
    {
      "files": [
        {
          "filePath": "data_summary.tsv",
          "fileType": "DATA_TABLE",
          "uncompressedLengthBytes": "2142"
        }
      ]
    },
    {
      "accession": "GCF_000157895.3",
      "files": [
        {
          "filePath": "GCF_000157895.3/protein.faa",
          "fileType": "PROTEIN_FASTA",
          "uncompressedLengthBytes": "2320001"
        }
      ]
    }
  ]
}
```

The data preparation module extracts:
- `accession`: Genome assembly identifier (used as species name)
- `filePath`: Path to protein FASTA file (PROTEIN_FASTA type)

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

## 🛠️ Command-Line Tools

### Generate Sample Data

```bash
python examples/generate_sample_data.py \
    --num-species 10 \
    --num-genes 200 \
    --num-samples 2000 \
    --output data/my_data.csv \
    --seed 42
```

### Train Model

```bash
python examples/train.py \
    --data-path data/my_data.csv \
    --embedding-dim 64 \
    --hidden-dim 128 \
    --num-layers 2 \
    --num-epochs 100 \
    --learning-rate 0.001 \
    --model-save-path checkpoints/best_model.pt \
    --verbose
```

### Make Predictions

```bash
python examples/predict.py \
    --model-path checkpoints/best_model.pt \
    --data-path data/new_data.csv \
    --output-path predictions.csv
```

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

## 🧬 Embedding Accessor

The `EmbeddingAccessor` class provides a comprehensive API for accessing and manipulating embeddings from trained models.

### Features

- **Retrieve Embeddings**: Get raw or transformed embeddings for specific species/genes
- **Batch Operations**: Retrieve embeddings for multiple entities at once
- **Similarity Computation**: Calculate cosine, euclidean, or dot product similarities
- **Nearest Neighbors**: Find k-most similar entities
- **Flexible Output**: Return embeddings as PyTorch tensors or NumPy arrays

### Usage Example

```python
from src.embedding_accessor import EmbeddingAccessor

# Create accessor (after training a model)
accessor = EmbeddingAccessor(model, data_loader, device='cpu')

# Get raw embedding for a species
human_emb = accessor.get_raw_species_embedding('Human')

# Get transformed embedding for a gene
gene_emb = accessor.get_transformed_gene_embedding('BRCA1', hetero_data)

# Batch retrieve embeddings
species_list = ['Human', 'Mouse', 'Rat']
batch_emb = accessor.get_batch_raw_embeddings('species', species_list)

# Compute similarity between two embeddings
emb1 = accessor.get_transformed_gene_embedding('BRCA1', hetero_data)
emb2 = accessor.get_transformed_gene_embedding('TP53', hetero_data)
similarity = accessor.compute_similarity(emb1, emb2, method='cosine')

# Find nearest neighbors
neighbors = accessor.find_nearest_neighbors(
    'BRCA1',
    entity_type='gene',
    data=hetero_data,
    k=5,
    use_transformed=True
)

# Get all embeddings for a given type
all_gene_emb = accessor.get_all_embeddings('gene', data=hetero_data, use_transformed=True)

# Get embedding information
info = accessor.get_embedding_info()
print(f"Embedding dimensions: {info['raw_embedding_dim']} -> {info['transformed_embedding_dim']}")
```

### Available Methods

**Single Entity Retrieval:**
- `get_raw_species_embedding(species)` - Get initial species embedding
- `get_raw_gene_embedding(gene)` - Get initial gene embedding
- `get_transformed_species_embedding(species, data)` - Get post-GNN species embedding
- `get_transformed_gene_embedding(gene, data)` - Get post-GNN gene embedding

**Batch Retrieval:**
- `get_batch_raw_embeddings(entity_type, entities)` - Get multiple raw embeddings
- `get_batch_transformed_embeddings(entity_type, entities, data)` - Get multiple transformed embeddings
- `get_all_embeddings(entity_type, data, use_transformed)` - Get all embeddings for a type

**Analysis:**
- `compute_similarity(emb1, emb2, method)` - Calculate similarity (cosine/euclidean/dot)
- `find_nearest_neighbors(entity, entity_type, data, k)` - Find k nearest neighbors
- `get_embedding_info()` - Get metadata about embeddings

### Run Embedding Demo

```bash
python examples/embedding_demo.py
```

This demonstrates all embedding accessor functionality including retrieval, similarity computation, and nearest neighbor search.

## 📁 Project Structure

```
species-gene-gnn/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── data_preparation.py  # Genome data preparation pipeline
│   ├── graph_builder.py     # Heterogeneous graph construction
│   ├── model.py             # GNN model architecture
│   ├── trainer.py           # Training pipeline
│   ├── embedding_accessor.py # Embedding access and utilities
│   └── utils.py             # Utility functions
├── examples/
│   ├── demo.py              # Complete demo script
│   ├── embedding_demo.py    # Embedding accessor demo
│   ├── generate_sample_data.py  # Sample data generator
│   ├── prepare_data.py      # Data preparation example
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── tests/
│   ├── test_model.py        # Unit tests for model components
│   ├── test_embedding_accessor.py  # Unit tests for embedding accessor
│   └── test_data_preparation.py  # Unit tests for data preparation
├── data/
│   └── (data files)         # Data directory
├── checkpoints/
│   └── (model files)        # Saved models
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
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
