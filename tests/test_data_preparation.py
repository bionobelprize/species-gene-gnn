"""
Unit tests for data preparation module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import json
import pandas as pd
import tempfile
from pathlib import Path

from src.data_preparation import DataPreparation


class TestDataPreparation:
    """Test cases for DataPreparation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_sample_catalog(self):
        """Create a sample dataset_catalog.json for testing."""
        catalog = {
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
                    "accession": "GCF_000001",
                    "files": [
                        {
                            "filePath": "GCF_000001/protein.faa",
                            "fileType": "PROTEIN_FASTA",
                            "uncompressedLengthBytes": "1000000"
                        }
                    ]
                },
                {
                    "accession": "GCF_000002",
                    "files": [
                        {
                            "filePath": "GCF_000002/protein.faa",
                            "fileType": "PROTEIN_FASTA",
                            "uncompressedLengthBytes": "2000000"
                        }
                    ]
                }
            ]
        }
        
        catalog_path = self.temp_path / "dataset_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f)
        
        return str(catalog_path)
    
    def test_load_catalog(self):
        """Test loading dataset catalog."""
        catalog_path = self.create_sample_catalog()
        output_dir = self.temp_path / "output"
        
        prep = DataPreparation(catalog_path, str(output_dir))
        prep.load_catalog()
        
        assert len(prep.assemblies) == 2
        assert "GCF_000001" in prep.assemblies
        assert "GCF_000002" in prep.assemblies
        assert prep.protein_files["GCF_000001"] == "GCF_000001/protein.faa"
        assert prep.protein_files["GCF_000002"] == "GCF_000002/protein.faa"
    
    def test_parse_alignment_file(self):
        """Test parsing alignment file."""
        catalog_path = self.create_sample_catalog()
        output_dir = self.temp_path / "output"
        
        prep = DataPreparation(catalog_path, str(output_dir))
        
        # Create a sample alignment file
        alignment_data = [
            "gene1\tgene2\t95.5\t100\t2\t0\t1\t100\t1\t100\t1e-50\t200.5",
            "gene1\tgene3\t90.0\t100\t5\t0\t1\t100\t1\t100\t1e-40\t180.0",
            "gene2\tgene1\t95.5\t100\t2\t0\t1\t100\t1\t100\t1e-50\t200.5"
        ]
        
        alignment_file = self.temp_path / "test_alignment.tsv"
        with open(alignment_file, 'w') as f:
            f.write('\n'.join(alignment_data))
        
        df = prep.parse_alignment_file(str(alignment_file))
        
        assert len(df) == 3
        assert list(df.columns) == [
            'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
            'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'
        ]
        assert df.iloc[0]['qseqid'] == 'gene1'
        assert df.iloc[0]['sseqid'] == 'gene2'
        assert df.iloc[0]['bitscore'] == 200.5
    
    def test_extract_best_alignments(self):
        """Test extracting best alignments."""
        catalog_path = self.create_sample_catalog()
        output_dir = self.temp_path / "output"
        
        prep = DataPreparation(catalog_path, str(output_dir))
        
        # Create test DataFrame
        data = {
            'qseqid': ['gene1', 'gene1', 'gene2', 'gene2'],
            'sseqid': ['target1', 'target2', 'target3', 'target4'],
            'pident': [95.5, 90.0, 88.0, 92.0],
            'length': [100, 100, 100, 100],
            'mismatch': [2, 5, 6, 4],
            'gapopen': [0, 0, 0, 0],
            'qstart': [1, 1, 1, 1],
            'qend': [100, 100, 100, 100],
            'sstart': [1, 1, 1, 1],
            'send': [100, 100, 100, 100],
            'evalue': [1e-50, 1e-40, 1e-35, 1e-45],
            'bitscore': [200.5, 180.0, 170.0, 190.0]
        }
        df = pd.DataFrame(data)
        
        df_best = prep.extract_best_alignments(df)
        
        assert len(df_best) == 2
        assert df_best.iloc[0]['qseqid'] == 'gene1'
        assert df_best.iloc[0]['bitscore'] == 200.5
        assert df_best.iloc[1]['qseqid'] == 'gene2'
        assert df_best.iloc[1]['bitscore'] == 190.0
    
    def test_generate_input_csv_with_sample_data(self):
        """Test generating input CSV from sample alignment files."""
        catalog_path = self.create_sample_catalog()
        output_dir = self.temp_path / "output"
        
        prep = DataPreparation(catalog_path, str(output_dir))
        
        # Create sample alignment files
        alignment_dir = self.temp_path / "alignments"
        alignment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create alignment file: GCF_000001 vs GCF_000002
        alignment_data_1 = [
            "gene1\tgene2\t95.5\t100\t2\t0\t1\t100\t1\t100\t1e-50\t200.5",
            "gene3\tgene4\t90.0\t100\t5\t0\t1\t100\t1\t100\t1e-40\t180.0"
        ]
        alignment_file_1 = alignment_dir / "GCF_000001_vs_GCF_000002.tsv"
        with open(alignment_file_1, 'w') as f:
            f.write('\n'.join(alignment_data_1))
        
        # Create alignment file: GCF_000002 vs GCF_000001
        alignment_data_2 = [
            "gene2\tgene1\t95.5\t100\t2\t0\t1\t100\t1\t100\t1e-50\t200.5",
            "gene4\tgene3\t90.0\t100\t5\t0\t1\t100\t1\t100\t1e-40\t180.0"
        ]
        alignment_file_2 = alignment_dir / "GCF_000002_vs_GCF_000001.tsv"
        with open(alignment_file_2, 'w') as f:
            f.write('\n'.join(alignment_data_2))
        
        alignment_files = [str(alignment_file_1), str(alignment_file_2)]
        
        output_csv = self.temp_path / "output.csv"
        df = prep.generate_input_csv(alignment_files, str(output_csv), normalize_scores=True)
        
        assert len(df) == 4
        assert all(col in df.columns for col in ['species_a', 'gene_a', 'species_b', 'gene_b', 'similarity'])
        
        # Check normalization
        assert df['similarity'].min() >= 0.0
        assert df['similarity'].max() <= 1.0
        
        # Check output file was created
        assert output_csv.exists()
        
        # Verify CSV content
        df_read = pd.read_csv(output_csv)
        assert len(df_read) == 4
    
    def test_generate_input_csv_without_normalization(self):
        """Test generating input CSV without normalization."""
        catalog_path = self.create_sample_catalog()
        output_dir = self.temp_path / "output"
        
        prep = DataPreparation(catalog_path, str(output_dir))
        
        # Create sample alignment file
        alignment_dir = self.temp_path / "alignments"
        alignment_dir.mkdir(parents=True, exist_ok=True)
        
        alignment_data = [
            "gene1\tgene2\t95.5\t100\t2\t0\t1\t100\t1\t100\t1e-50\t200.5"
        ]
        alignment_file = alignment_dir / "GCF_000001_vs_GCF_000002.tsv"
        with open(alignment_file, 'w') as f:
            f.write('\n'.join(alignment_data))
        
        df = prep.generate_input_csv([str(alignment_file)], normalize_scores=False)
        
        assert len(df) == 1
        assert df.iloc[0]['similarity'] == 200.5
    
    def test_empty_alignment_file(self):
        """Test handling empty alignment file."""
        catalog_path = self.create_sample_catalog()
        output_dir = self.temp_path / "output"
        
        prep = DataPreparation(catalog_path, str(output_dir))
        
        # Create empty alignment file
        alignment_file = self.temp_path / "empty.tsv"
        alignment_file.touch()
        
        df = prep.parse_alignment_file(str(alignment_file))
        
        assert df.empty
        assert list(df.columns) == [
            'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
            'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'
        ]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
