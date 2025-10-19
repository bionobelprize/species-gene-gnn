# New Features Changelog

## Summary

This update adds two major features to the Species-Gene GNN pipeline:

1. **Self-Alignment Switch** in `data_preparation.py` - Enables within-species protein alignments
2. **Genome Subset Feature** in `train.py` - Allows using partial genome data to reduce memory usage

## Feature 1: Self-Alignment Switch

### Overview
The data preparation module now supports within-species all-against-all protein alignments (self-alignments) through a new optional parameter. When enabled, this feature performs alignments between proteins within the same species, while automatically filtering out meaningless gene self-comparisons (a gene compared to itself).

### Changes Made

#### `src/data_preparation.py`

1. **New Parameter**: `include_self_alignment` (bool, default=False)
   - Added to `__init__()` method
   - Controls whether within-species alignments are performed

2. **Modified Methods**:
   - `perform_all_alignments()`: Now includes self-alignments when flag is enabled
   - `generate_input_csv()`: Automatically filters out gene self-comparisons in self-alignments

3. **CLI Addition**: `--include-self-alignment` flag

### Usage

#### Python API:
```python
from src.data_preparation import DataPreparation

prep = DataPreparation(
    catalog_path='dataset_catalog.json',
    output_dir='output',
    include_self_alignment=True  # Enable self-alignments
)
df = prep.prepare_data()
```

#### Command Line:
```bash
python -m src.data_preparation \
    --catalog dataset_catalog.json \
    --output-dir output \
    --include-self-alignment
```

### Behavior

- **Default (False)**: Only cross-species alignments are performed (original behavior)
- **Enabled (True)**: 
  - Performs both cross-species AND within-species alignments
  - Automatically filters gene self-comparisons (gene_a == gene_b in same species)
  - Keeps meaningful within-species gene-gene relationships

### Example

For 2 species with self-alignment enabled:
- Without flag: 2 alignments (A→B, B→A)
- With flag: 4 alignments (A→B, B→A, A→A, B→B)

Gene self-comparisons are automatically excluded:
```
# Alignment file for Species_A vs Species_A might contain:
gene1 → gene1  (self-comparison, FILTERED OUT)
gene1 → gene2  (kept)
gene2 → gene1  (kept)
gene2 → gene2  (self-comparison, FILTERED OUT)

# Result: Only gene1↔gene2 relationships are kept
```

## Feature 2: Genome Subset Feature

### Overview
The training script now supports using only a subset of genomes (species) for model training. This feature is designed to reduce memory usage when working with large datasets on systems with limited GPU memory.

### Changes Made

#### `examples/train.py`

1. **New Parameter**: `--genome-subset-ratio` (float, default=1.0)
   - Range: 0.0 to 1.0
   - Ratio of genomes to use for training

2. **Implementation**:
   - Randomly selects specified percentage of species (with fixed seed for reproducibility)
   - Filters all gene associations to only include selected species
   - Updates mappings to reflect reduced dataset

### Usage

#### Command Line:
```bash
# Use all genomes (default)
python examples/train.py \
    --data-path data/gene_associations.csv

# Use 50% of genomes
python examples/train.py \
    --data-path data/gene_associations.csv \
    --genome-subset-ratio 0.5

# Use 30% of genomes (for very limited memory)
python examples/train.py \
    --data-path data/gene_associations.csv \
    --genome-subset-ratio 0.3
```

### Behavior

1. **Species Selection**: Randomly selects N% of unique species from dataset
2. **Association Filtering**: Only keeps associations where both species are in selected set
3. **Gene Filtering**: Automatically filters genes that only appear in excluded species
4. **Reproducibility**: Uses fixed random seed (42) for consistent results

### Memory Reduction Example

Original dataset:
- 10 species, 100 genes per species = 1,000 genes
- ~4,500 cross-species associations
- Memory: 100%

With `--genome-subset-ratio 0.5`:
- 5 species, ~500 genes
- ~1,125 associations (75% reduction)
- Memory: ~50% of original

## Testing

### New Tests Added

#### `tests/test_data_preparation.py`
- `test_self_alignment_enabled`: Verifies flag initialization
- `test_self_alignment_disabled`: Verifies default behavior
- `test_generate_input_csv_excludes_gene_self_comparison`: Tests filtering logic
- `test_generate_input_csv_cross_species_allows_same_gene_name`: Ensures cross-species isn't affected

#### `tests/test_train.py` (New file)
- `test_genome_subset_filtering`: Tests species and gene filtering
- `test_genome_subset_ratio_100_percent`: Verifies 100% keeps all data
- `test_genome_subset_ratio_minimum`: Verifies minimum of 1 species kept
- `test_genome_subset_with_genes`: Tests gene filtering with species subset

### Test Results
- **Total tests**: 62
- **Passed**: 62
- **Failed**: 0

## Backward Compatibility

Both features are fully backward compatible:

1. **Self-alignment**: Default is `False` (original behavior)
2. **Genome subset**: Default is `1.0` (use all genomes)

Existing code and scripts will continue to work without modification.

## Implementation Details

### Self-Alignment Gene Filtering

The filtering happens BEFORE extracting best alignments to ensure:
1. Gene self-comparisons don't interfere with best hit selection
2. Meaningful within-species relationships are preserved
3. Only true self-comparisons (same gene ID) are filtered

### Genome Subset Selection

The selection process:
1. Collects all unique species from both `species_a` and `species_b` columns
2. Calculates number to keep: `max(1, int(num_species * ratio))`
3. Randomly selects species using `np.random.choice()` with seed=42
4. Filters DataFrame keeping only rows where BOTH species are in selected set
5. Rebuilds mappings on filtered data

## Examples

### Example 1: Enable self-alignment for more comprehensive data

```bash
# Prepare data with self-alignments
python -m src.data_preparation \
    --catalog my_catalog.json \
    --output-dir output \
    --include-self-alignment \
    --output-csv gene_associations_with_self.csv

# Train normally
python examples/train.py \
    --data-path gene_associations_with_self.csv \
    --num-epochs 100
```

### Example 2: Train with limited memory

```bash
# Use only 30% of genomes to fit in GPU memory
python examples/train.py \
    --data-path data/large_dataset.csv \
    --genome-subset-ratio 0.3 \
    --num-epochs 100 \
    --verbose
```

### Example 3: Combined usage

```bash
# 1. Prepare data with self-alignments
python -m src.data_preparation \
    --catalog catalog.json \
    --include-self-alignment \
    --output-csv full_data.csv

# 2. Train on subset for faster iteration
python examples/train.py \
    --data-path full_data.csv \
    --genome-subset-ratio 0.5 \
    --num-epochs 50 \
    --model-save-path models/subset_model.pt
```

## Files Modified

1. `src/data_preparation.py` - Added self-alignment functionality
2. `examples/train.py` - Added genome subset feature
3. `tests/test_data_preparation.py` - Added self-alignment tests
4. `tests/test_train.py` - Added genome subset tests (new file)

## Performance Considerations

### Self-Alignment
- **Time**: Increases alignment time (N² instead of N²-N comparisons)
- **Space**: Minimal additional space for filtered results
- **Benefit**: More comprehensive within-species gene relationships

### Genome Subset
- **Time**: Linear reduction based on subset ratio
- **Space**: Proportional memory reduction
- **Trade-off**: Less data may reduce model accuracy

## Future Enhancements

Potential improvements for future versions:

1. **Self-alignment**: Add parameter to control minimum similarity threshold
2. **Genome subset**: Add options for stratified sampling or specific species selection
3. **Both**: Add support for incremental training with subset increases
