# Data Preparation Quick Reference

This guide shows how to prepare data from genome assemblies for the Species-Gene GNN model.

## Prerequisites

1. Install Diamond BLASTP aligner:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install diamond-aligner
   
   # Or download from: https://github.com/bbuchfink/diamond
   ```

2. Download genome assemblies from NCBI Datasets with the `dataset_catalog.json` file.

## Quick Start

### Command Line Interface

```bash
python -m src.data_preparation \
    --catalog path/to/dataset_catalog.json \
    --output-dir output \
    --output-csv data/gene_associations.csv \
    --max-target-seqs 5
```

### Python API

```python
from src.data_preparation import DataPreparation

# Initialize
prep = DataPreparation(
    catalog_path='path/to/dataset_catalog.json',
    output_dir='output'
)

# Run complete pipeline
df = prep.prepare_data(
    max_target_seqs=5,
    output_csv='data/gene_associations.csv',
    normalize_scores=True
)
```

## Pipeline Steps

The data preparation pipeline automatically:

1. **Reads dataset_catalog.json** - Extracts protein FASTA files for each genome assembly
2. **Creates Diamond databases** - Builds searchable databases from protein sequences
3. **Runs BLASTP alignments** - Performs all-vs-all protein comparisons (excluding self-alignments)
4. **Parses results** - Extracts best hits based on bitscore
5. **Generates CSV** - Creates five-dimensional input file for the model

## Input Format: dataset_catalog.json

The catalog file should have this structure:

```json
{
  "apiVersion": "V2",
  "assemblies": [
    {
      "accession": "GCF_000001405.40",
      "files": [
        {
          "filePath": "GCF_000001405.40/protein.faa",
          "fileType": "PROTEIN_FASTA",
          "uncompressedLengthBytes": "23200001"
        }
      ]
    }
  ]
}
```

## Output Format: gene_associations.csv

The output CSV has five columns:

| Column | Description |
|--------|-------------|
| species_a | Source genome accession |
| gene_a | Source gene/protein ID |
| species_b | Target genome accession |
| gene_b | Target gene/protein ID |
| similarity | Normalized similarity score (0-1) |

Example:

```csv
species_a,gene_a,species_b,gene_b,similarity
GCF_000001405.40,BRCA1,GCF_000001635.27,Brca1,0.89
GCF_000001405.40,TP53,GCF_000002985.6,Tp53,0.92
```

## Parameters

- `--catalog`: Path to dataset_catalog.json (required)
- `--output-dir`: Directory for intermediate files (default: `output`)
- `--output-csv`: Path to output CSV file (default: `output/gene_associations.csv`)
- `--max-target-seqs`: Maximum hits per query (default: 5)
- `--no-normalize`: Skip score normalization (scores remain as bitscores)

## Using with the Model

Once you have the CSV file, use it with the model:

```python
from src.data_loader import SpeciesGeneDataLoader
from src.graph_builder import HeteroGraphBuilder
from src.model import SpeciesGeneGNN

# Load data
loader = SpeciesGeneDataLoader()
df = loader.load_data('data/gene_associations.csv')
loader.build_mappings(df)

# Encode data
species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \
    loader.encode_data(df)

# Build graph
builder = HeteroGraphBuilder()
hetero_data = builder.build_graph(
    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,
    loader.get_num_species(), loader.get_num_genes()
)

# Create and train model
model = SpeciesGeneGNN(
    num_species=loader.get_num_species(),
    num_genes=loader.get_num_genes(),
    embedding_dim=64,
    hidden_dim=128,
    num_layers=2
)
```

## Troubleshooting

### Diamond not found
```
Error: Diamond is not installed
```
**Solution**: Install Diamond following the instructions in Prerequisites.

### Empty alignment files
```
Warning: Empty alignment file: ...
```
**Solution**: This is normal if genomes have no similar proteins. The pipeline will continue with other alignments.

### No alignment records found
```
Warning: No alignment records found
```
**Solution**: Check that your protein FASTA files are not empty and contain valid sequences.

## Advanced Usage

### Skip Normalization

To keep raw bitscores instead of normalized [0, 1] scores:

```bash
python -m src.data_preparation \
    --catalog dataset_catalog.json \
    --output-csv data/gene_associations.csv \
    --no-normalize
```

### Adjust Sensitivity

Increase `--max-target-seqs` to keep more alignment hits per query:

```bash
python -m src.data_preparation \
    --catalog dataset_catalog.json \
    --max-target-seqs 10
```

### Process Only Part of the Pipeline

You can use individual methods for more control:

```python
from src.data_preparation import DataPreparation

prep = DataPreparation('dataset_catalog.json', 'output')

# Load catalog
prep.load_catalog()

# Create databases only
for accession in prep.assemblies:
    protein_file = prep.protein_files[accession]
    db_path = prep.create_diamond_database(accession, protein_file)

# Run specific alignment
alignment_file = prep.run_diamond_blastp(
    'GCF_001', 'genome1.faa',
    'GCF_002', 'genome2.dmnd',
    max_target_seqs=5
)

# Parse specific file
df = prep.parse_alignment_file(alignment_file)
```

## See Also

- [README.md](README.md) - Full documentation
- [examples/prepare_data.py](examples/prepare_data.py) - Example script
- [tests/test_data_preparation.py](tests/test_data_preparation.py) - Unit tests
- [tests/test_integration.py](tests/test_integration.py) - Integration test
