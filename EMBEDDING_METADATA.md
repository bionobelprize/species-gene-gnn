# Embedding Metadata: Species and Gene Mappings

## Overview

Model checkpoints now include metadata that maps embedding matrix row indices to their corresponding species and gene identifiers. This allows you to know exactly what each row in the `species_embedding` and `gene_embedding` weight matrices represents.

## The Problem

Previously, when you saved a trained model, the checkpoint contained the embedding weights:
- `species_embedding.weight`: A matrix of shape `[num_species, embedding_dim]`
- `gene_embedding.weight`: A matrix of shape `[num_genes, embedding_dim]`

However, there was no way to know which row corresponded to which species or gene. For example, if row 5 of the gene embedding matrix had a particular vector, you couldn't determine if it represented "BRCA1", "TP53", or some other gene.

## The Solution

When you pass a `data_loader` to the `Trainer`, the checkpoint now includes two additional dictionaries:
- `id_to_species`: Maps row indices to species names (e.g., `{0: 'Human', 1: 'Mouse', ...}`)
- `id_to_gene`: Maps row indices to gene names (e.g., `{0: 'BRCA1', 1: 'TP53', ...}`)

This means:
- Row 0 in `species_embedding.weight` corresponds to `id_to_species[0]`
- Row 0 in `gene_embedding.weight` corresponds to `id_to_gene[0]`

## How to Use

### Saving a Model with Mappings

When creating a `Trainer`, pass the `data_loader` that contains the mappings:

```python
from src.data_loader import SpeciesGeneDataLoader
from src.model import SpeciesGeneGNN
from src.trainer import Trainer

# Load and prepare data
data_loader = SpeciesGeneDataLoader()
df = data_loader.load_data('data.csv')
data_loader.build_mappings(df)

# Create model
model = SpeciesGeneGNN(
    num_species=data_loader.get_num_species(),
    num_genes=data_loader.get_num_genes()
)

# Create trainer WITH data_loader
trainer = Trainer(
    model=model,
    data_loader=data_loader  # This enables saving mappings!
)

# Train and save
trainer.train(...)
trainer.save_model('model.pt')  # Mappings are automatically saved
```

### Loading a Model and Accessing Mappings

When loading a model, you can retrieve the mappings:

```python
from src.trainer import Trainer
from src.model import SpeciesGeneGNN

# Create model (dimensions must match)
model = SpeciesGeneGNN(num_species=10, num_genes=100)
trainer = Trainer(model=model)

# Load model and get mappings
id_to_species, id_to_gene = trainer.load_model('model.pt')

if id_to_species is not None:
    print(f"Species at row 0: {id_to_species[0]}")
    print(f"Gene at row 0: {id_to_gene[0]}")
```

### Accessing Embeddings with Their Labels

You can load the checkpoint directly and access both embeddings and mappings:

```python
import torch

# Load checkpoint
checkpoint = torch.load('model.pt')

# Get mappings
id_to_species = checkpoint['id_to_species']
id_to_gene = checkpoint['id_to_gene']

# Get embeddings
species_embeddings = checkpoint['model_state_dict']['species_embedding.weight']
gene_embeddings = checkpoint['model_state_dict']['gene_embedding.weight']

# Access specific embeddings
for species_id, species_name in id_to_species.items():
    embedding = species_embeddings[species_id]
    print(f"{species_name}: {embedding}")
```

### Exporting Embeddings to CSV

You can export embeddings with their labels for downstream analysis:

```python
import pandas as pd
import torch

checkpoint = torch.load('model.pt')
id_to_gene = checkpoint['id_to_gene']
gene_embeddings = checkpoint['model_state_dict']['gene_embedding.weight'].numpy()

# Create DataFrame with gene names and embeddings
data = []
for gene_id, gene_name in id_to_gene.items():
    row = {'gene_name': gene_name}
    for i, val in enumerate(gene_embeddings[gene_id]):
        row[f'dim_{i}'] = val
    data.append(row)

df = pd.DataFrame(data)
df.to_csv('gene_embeddings.csv', index=False)
```

## Backward Compatibility

The new functionality is fully backward compatible:

1. **Old models**: Models saved without `data_loader` will load normally, but `load_model()` will return `(None, None)` for the mappings.

2. **Old code**: If you don't pass `data_loader` to `Trainer`, the model will save without mappings (same as before).

3. **New models, old code**: If you save a model with mappings but load it without checking the return value, everything still works:
   ```python
   trainer.load_model('model.pt')  # Works fine, mappings are just not retrieved
   ```

## Examples

### Full Example Scripts

The repository includes several example scripts demonstrating this feature:

1. **`examples/embedding_metadata_demo.py`**: Comprehensive demo showing save/load with mappings
2. **`examples/access_embeddings_with_ids.py`**: Extract and export embeddings with their identifiers
3. **`examples/train.py`**: Updated to save mappings by default
4. **`examples/predict.py`**: Updated to load and display mapping information

### Quick Start

```bash
# Run the embedding metadata demo
python examples/embedding_metadata_demo.py

# Run the access embeddings example
python examples/access_embeddings_with_ids.py

# Train a model (will save with mappings)
python examples/train.py --data-path data/sample_data.csv --num-epochs 50

# Load and use the model
python examples/predict.py --model-path checkpoints/model.pt --data-path data/test_data.csv
```

## Technical Details

### Checkpoint Structure

A checkpoint saved with mappings has the following structure:

```python
{
    'model_state_dict': {
        'species_embedding.weight': Tensor(...),  # [num_species, embedding_dim]
        'gene_embedding.weight': Tensor(...),     # [num_genes, embedding_dim]
        # ... other model parameters
    },
    'optimizer_state_dict': {...},
    'id_to_species': {0: 'Human', 1: 'Mouse', ...},  # New!
    'id_to_gene': {0: 'BRCA1', 1: 'TP53', ...}       # New!
}
```

### Mapping Guarantees

- **Sorted keys**: Species and gene names are sorted alphabetically when building mappings
- **Consistent IDs**: The same species/gene always gets the same ID within a dataset
- **Complete coverage**: Every row in the embedding matrix has a corresponding entry in the mapping

### Memory Overhead

The mappings add minimal overhead to checkpoint files:
- Each mapping entry is approximately 50-100 bytes (depends on name length)
- For 1000 species and 10000 genes: ~1 MB additional storage
- This is negligible compared to the model weights (typically 10-100+ MB)

## Testing

The feature includes comprehensive tests in `tests/test_trainer_mappings.py`:

```bash
# Run the mapping tests
pytest tests/test_trainer_mappings.py -v

# Run all tests
pytest tests/ -v
```

Tests cover:
- Saving models with and without mappings
- Loading models and retrieving mappings
- Mapping correctness and alignment with embeddings
- Backward compatibility

## Use Cases

### 1. Exploratory Analysis
```python
# Which species are most similar in the embedding space?
checkpoint = torch.load('model.pt')
species_embeddings = checkpoint['model_state_dict']['species_embedding.weight']
id_to_species = checkpoint['id_to_species']

# Compute similarities and identify the species
# ...
```

### 2. Transfer Learning
```python
# Load embeddings from a pre-trained model for specific genes
checkpoint = torch.load('pretrained_model.pt')
gene_embeddings = checkpoint['model_state_dict']['gene_embedding.weight']
id_to_gene = checkpoint['id_to_gene']

# Find the embedding for BRCA1
gene_name_to_id = {name: id for id, name in id_to_gene.items()}
brca1_id = gene_name_to_id['BRCA1']
brca1_embedding = gene_embeddings[brca1_id]
```

### 3. Visualization
```python
# Create a t-SNE plot with labels
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

checkpoint = torch.load('model.pt')
gene_embeddings = checkpoint['model_state_dict']['gene_embedding.weight'].numpy()
id_to_gene = checkpoint['id_to_gene']

# Reduce dimensions
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(gene_embeddings)

# Plot with labels
plt.figure(figsize=(12, 8))
for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y)
    plt.annotate(id_to_gene[i], (x, y))
plt.show()
```

## FAQ

**Q: Do I need to change my existing training code?**
A: Only if you want to save mappings. Add `data_loader=data_loader` when creating the `Trainer`.

**Q: Will old model checkpoints still work?**
A: Yes! Loading old checkpoints returns `(None, None)` for mappings, but everything else works normally.

**Q: Can I add mappings to an existing checkpoint?**
A: Yes, you can manually add `id_to_species` and `id_to_gene` dictionaries to the checkpoint file.

**Q: What if I rename a species or gene?**
A: The mappings reflect the names at training time. If you rename entities, you'll need to update the mappings manually or retrain.

**Q: How do I know if a checkpoint has mappings?**
A: Check if `'id_to_species'` is in the loaded checkpoint dictionary, or check if `load_model()` returns non-None values.

## Related Documentation

- [README.md](README.md) - Main project documentation
- [Embedding Accessor](README.md#-embedding-accessor) - API for accessing embeddings during inference
- [examples/](examples/) - Example scripts using this feature
- [tests/test_trainer_mappings.py](tests/test_trainer_mappings.py) - Test suite for this feature
