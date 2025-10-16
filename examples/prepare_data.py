"""
Example script demonstrating the data preparation pipeline.

This script shows how to use the DataPreparation class to process
genome assemblies and generate input data for the Species-Gene GNN model.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path
from src.data_preparation import DataPreparation


def create_sample_catalog(output_path='data/dataset_catalog.json'):
    """
    Create a sample dataset_catalog.json for demonstration.
    
    This is for demonstration purposes. In real use cases,
    you would have this file provided by NCBI Datasets or similar.
    """
    catalog = {
        "apiVersion": "V2",
        "assemblies": [
            {
                "files": [
                    {
                        "filePath": "data_summary.tsv",
                        "fileType": "DATA_TABLE",
                        "uncompressedLengthBytes": "2142"
                    },
                    {
                        "filePath": "assembly_data_report.jsonl",
                        "fileType": "DATA_REPORT",
                        "uncompressedLengthBytes": "40657"
                    }
                ]
            },
            {
                "accession": "GCF_000001405.40",
                "files": [
                    {
                        "filePath": "GCF_000001405.40/protein.faa",
                        "fileType": "PROTEIN_FASTA",
                        "uncompressedLengthBytes": "23200001"
                    }
                ]
            },
            {
                "accession": "GCF_000001635.27",
                "files": [
                    {
                        "filePath": "GCF_000001635.27/protein.faa",
                        "fileType": "PROTEIN_FASTA",
                        "uncompressedLengthBytes": "18500000"
                    }
                ]
            },
            {
                "accession": "GCF_000002985.6",
                "files": [
                    {
                        "filePath": "GCF_000002985.6/protein.faa",
                        "fileType": "PROTEIN_FASTA",
                        "uncompressedLengthBytes": "19800000"
                    }
                ]
            }
        ]
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Created sample catalog: {output_path}")
    return output_path


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Data Preparation Pipeline Example")
    print("=" * 70)
    print()
    
    # Step 1: Create a sample catalog (in real use, you'd have this already)
    print("Step 1: Creating sample dataset catalog...")
    catalog_path = create_sample_catalog('data/dataset_catalog.json')
    print()
    
    # Step 2: Initialize DataPreparation
    print("Step 2: Initializing data preparation...")
    prep = DataPreparation(
        catalog_path=catalog_path,
        output_dir='data/prepared'
    )
    print()
    
    # Step 3: Show what the pipeline would do
    print("Step 3: Pipeline overview...")
    print()
    print("The data preparation pipeline will:")
    print("  1. Read dataset_catalog.json")
    print("  2. Extract protein FASTA files for each genome")
    print("  3. Create Diamond databases for each genome")
    print("  4. Perform all-vs-all BLASTP alignments (excluding self)")
    print("  5. Parse alignment results")
    print("  6. Generate five-dimensional CSV file")
    print()
    
    # Step 4: Load catalog to see what assemblies we have
    print("Step 4: Loading catalog...")
    prep.load_catalog()
    print()
    
    # Step 5: Explain the full pipeline (without running Diamond)
    print("Step 5: To run the full pipeline:")
    print()
    print("First, ensure you have Diamond installed:")
    print("  # Ubuntu/Debian")
    print("  sudo apt-get install diamond-aligner")
    print()
    print("  # Or download from: https://github.com/bbuchfink/diamond")
    print()
    print("Then run:")
    print("  python -m src.data_preparation \\")
    print("    --catalog data/dataset_catalog.json \\")
    print("    --output-dir data/prepared \\")
    print("    --output-csv data/gene_associations.csv \\")
    print("    --max-target-seqs 5")
    print()
    
    # Step 6: Show expected output format
    print("Step 6: Expected output format:")
    print()
    print("The output CSV will have the following columns:")
    print("  - species_a: Source species/genome accession")
    print("  - gene_a: Source gene ID")
    print("  - species_b: Target species/genome accession")
    print("  - gene_b: Target gene ID")
    print("  - similarity: Normalized protein similarity score [0, 1]")
    print()
    print("Example rows:")
    print("  species_a,gene_a,species_b,gene_b,similarity")
    print("  GCF_000001405.40,BRCA1,GCF_000001635.27,Brca1,0.89")
    print("  GCF_000001405.40,TP53,GCF_000002985.6,Tp53,0.92")
    print()
    
    # Step 7: Integration with the model
    print("Step 7: Using the prepared data with the model:")
    print()
    print("from src.data_loader import SpeciesGeneDataLoader")
    print("from src.graph_builder import HeteroGraphBuilder")
    print("from src.model import SpeciesGeneGNN")
    print()
    print("# Load prepared data")
    print("loader = SpeciesGeneDataLoader()")
    print("df = loader.load_data('data/gene_associations.csv')")
    print("loader.build_mappings(df)")
    print()
    print("# Encode and build graph")
    print("species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities = \\")
    print("    loader.encode_data(df)")
    print()
    print("builder = HeteroGraphBuilder()")
    print("hetero_data = builder.build_graph(")
    print("    species_a_ids, gene_a_ids, species_b_ids, gene_b_ids, similarities,")
    print("    loader.get_num_species(), loader.get_num_genes()")
    print(")")
    print()
    print("# Create and train model")
    print("model = SpeciesGeneGNN(")
    print("    num_species=loader.get_num_species(),")
    print("    num_genes=loader.get_num_genes()")
    print(")")
    print()
    
    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
