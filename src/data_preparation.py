"""
Data preparation module for Species-Gene GNN.

This module handles reading dataset_catalog.json, running Diamond BLASTP
alignments, and generating the five-dimensional input CSV for the model.
"""

import json
import os
import subprocess
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import tempfile


class DataPreparation:
    """
    Prepares data for Species-Gene GNN from genome assemblies.
    
    This class reads a dataset_catalog.json file, extracts protein sequences,
    performs all-vs-all Diamond BLASTP alignments (excluding self-alignments),
    and generates a five-dimensional CSV file for model input.
    """
    
    def __init__(self, catalog_path: str, output_dir: str = "output", include_self_alignment: bool = False):
        """
        Initialize DataPreparation.
        
        Args:
            catalog_path: Path to dataset_catalog.json
            output_dir: Directory for intermediate and output files
            include_self_alignment: Whether to include within-species alignments (all-against-all)
        """
        self.catalog_path = catalog_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_dir = self.output_dir / "databases"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self.alignment_dir = self.output_dir / "alignments"
        self.alignment_dir.mkdir(parents=True, exist_ok=True)
        
        self.assemblies = []
        self.protein_files = {}
        self.include_self_alignment = include_self_alignment
        
    def load_catalog(self) -> None:
        """
        Load and parse dataset_catalog.json.
        
        Extracts protein FASTA files for each genome assembly.
        """
        with open(self.catalog_path, 'r') as f:
            catalog = json.load(f)
        
        # Extract assemblies with protein files
        for assembly in catalog.get('assemblies', []):
            accession = assembly.get('accession')
            if not accession:
                # Skip entries without accession (like summary files)
                continue
                
            files = assembly.get('files', [])
            for file_info in files:
                if file_info.get('fileType') == 'PROTEIN_FASTA':
                    file_path = file_info.get('filePath')
                    if file_path:
                        self.assemblies.append(accession)
                        self.protein_files[accession] = file_path
                        break
        
        print(f"Loaded {len(self.assemblies)} assemblies with protein sequences")
        for accession in self.assemblies:
            print(f"  {accession}: {self.protein_files[accession]}")
    
    def create_diamond_database(self, accession: str, protein_fasta: str) -> str:
        """
        Create a Diamond database from protein FASTA file.
        
        Args:
            accession: Genome accession ID
            protein_fasta: Path to protein FASTA file
            
        Returns:
            Path to created Diamond database
        """
        db_path = self.db_dir / f"{accession}.dmnd"
        
        if db_path.exists():
            print(f"Database already exists: {db_path}")
            return str(db_path)
        
        print(f"Creating Diamond database for {accession}...")
        
        cmd = [
            'diamond', 'makedb',
            '--in', protein_fasta,
            '--db', str(db_path.with_suffix(''))  # Diamond adds .dmnd
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Database created: {db_path}")
            return str(db_path)
        except subprocess.CalledProcessError as e:
            print(f"  Error creating database: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "Diamond is not installed. Please install Diamond: "
                "https://github.com/bbuchfink/diamond"
            )
    
    def run_diamond_blastp(
        self,
        query_accession: str,
        query_fasta: str,
        target_accession: str,
        target_db: str,
        max_target_seqs: int = 5
    ) -> str:
        """
        Run Diamond BLASTP alignment.
        
        Args:
            query_accession: Query genome accession
            query_fasta: Path to query protein FASTA
            target_accession: Target genome accession
            target_db: Path to target Diamond database
            max_target_seqs: Maximum number of target sequences per query
            
        Returns:
            Path to output tabular file
        """
        output_file = self.alignment_dir / f"{query_accession}_vs_{target_accession}.tsv"
        
        if output_file.exists():
            print(f"Alignment already exists: {output_file}")
            return str(output_file)
        
        print(f"Running BLASTP: {query_accession} vs {target_accession}...")
        
        cmd = [
            'diamond', 'blastp',
            '--db', target_db,
            '--query', query_fasta,
            '--outfmt', '6',
            '--out', str(output_file),
            '--max-target-seqs', str(max_target_seqs)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Alignment complete: {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            print(f"  Error running BLASTP: {e.stderr}")
            raise
    
    def perform_all_alignments(self, max_target_seqs: int = 5) -> List[str]:
        """
        Perform all-vs-all alignments.
        
        Args:
            max_target_seqs: Maximum number of target sequences per query
            
        Returns:
            List of alignment output file paths
        """
        alignment_files = []
        
        # Create databases for all assemblies
        databases = {}
        for accession in self.assemblies:
            protein_file = self.protein_files[accession]
            db_path = self.create_diamond_database(accession, protein_file)
            databases[accession] = db_path
        
        print(f"\nPerforming alignments...")
        # Perform all-vs-all alignments
        for query_acc in self.assemblies:
            query_fasta = self.protein_files[query_acc]
            for target_acc in self.assemblies:
                # Skip self-alignment unless explicitly enabled
                if query_acc == target_acc and not self.include_self_alignment:
                    continue
                
                target_db = databases[target_acc]
                output_file = self.run_diamond_blastp(
                    query_acc, query_fasta, target_acc, target_db, max_target_seqs
                )
                alignment_files.append(output_file)
        
        print(f"\nCompleted {len(alignment_files)} alignments")
        return alignment_files
    
    def parse_alignment_file(self, alignment_file: str) -> pd.DataFrame:
        """
        Parse Diamond BLASTP tabular output.
        
        Args:
            alignment_file: Path to alignment TSV file
            
        Returns:
            DataFrame with columns: qseqid, sseqid, pident, length, mismatch,
            gapopen, qstart, qend, sstart, send, evalue, bitscore
        """
        # Diamond default outfmt 6 columns
        columns = [
            'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
            'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'
        ]
        
        if not os.path.exists(alignment_file):
            print(f"Warning: Alignment file not found: {alignment_file}")
            return pd.DataFrame(columns=columns)
        
        try:
            df = pd.read_csv(alignment_file, sep='\t', header=None, names=columns)
            return df
        except pd.errors.EmptyDataError:
            print(f"Warning: Empty alignment file: {alignment_file}")
            return pd.DataFrame(columns=columns)
    
    def extract_best_alignments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract best alignment for each query gene based on bitscore.
        
        Args:
            df: DataFrame with alignment results
            
        Returns:
            DataFrame with best alignment per query gene
        """
        if df.empty:
            return df
        
        # Sort by qseqid and bitscore (descending)
        df_sorted = df.sort_values(['qseqid', 'bitscore'], ascending=[True, False])
        
        # Keep only the best alignment for each query
        df_best = df_sorted.groupby('qseqid').first().reset_index()
        
        return df_best
    
    def generate_input_csv(
        self,
        alignment_files: List[str],
        output_path: str = None,
        normalize_scores: bool = True
    ) -> pd.DataFrame:
        """
        Generate five-dimensional input CSV from alignment files.
        
        Args:
            alignment_files: List of alignment output file paths
            output_path: Path to save output CSV (optional)
            normalize_scores: Whether to normalize pident to [0, 1] (by dividing by 100)
        
        Returns:
            DataFrame with columns: species_a, gene_a, species_b, gene_b, similarity
        """
        print("\nGenerating five-dimensional input CSV...")
        
        records = []
        
        for alignment_file in alignment_files:
            # Extract species names from filename: query_vs_target.tsv
            filename = Path(alignment_file).stem
            parts = filename.split('_vs_')
            if len(parts) != 2:
                print(f"Warning: Unexpected filename format: {filename}")
                continue
            
            species_a = parts[0]
            species_b = parts[1]
            
            # Parse alignment file
            df = self.parse_alignment_file(alignment_file)
            
            if df.empty:
                continue
            
            # Check if this is a self-alignment
            is_self_alignment = (species_a == species_b)
            
            # Filter out gene self-comparisons in self-alignments BEFORE extracting best
            if is_self_alignment:
                df = df[df['qseqid'] != df['sseqid']]
                if df.empty:
                    continue
            
            # Extract best alignments
            df_best = self.extract_best_alignments(df)
            
            # Create records for CSV, use pident as similarity
            for _, row in df_best.iterrows():
                records.append({
                    'species_a': species_a,
                    'gene_a': row['qseqid'],
                    'species_b': species_b,
                    'gene_b': row['sseqid'],
                    'similarity': row['pident']
                })
        
        if not records:
            print("Warning: No alignment records found")
            return pd.DataFrame(columns=['species_a', 'gene_a', 'species_b', 'gene_b', 'similarity'])
        
        result_df = pd.DataFrame(records)
        
        # Normalize similarity scores: pident is 0-100, so divide by 100 if requested
        if normalize_scores and len(result_df) > 0:
            result_df['similarity'] = result_df['similarity'] / 100.0
        
        print(f"Generated {len(result_df)} gene-gene associations")
        print(f"  Species: {result_df['species_a'].nunique() + result_df['species_b'].nunique() - len(set(result_df['species_a']) & set(result_df['species_b']))}")
        print(f"  Genes: {result_df['gene_a'].nunique() + result_df['gene_b'].nunique() - len(set(result_df['gene_a']) & set(result_df['gene_b']))}")
        print(f"  Similarity range: [{result_df['similarity'].min():.4f}, {result_df['similarity'].max():.4f}]")
        
        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
        
        return result_df
    
    def prepare_data(
        self,
        max_target_seqs: int = 5,
        output_csv: str = None,
        normalize_scores: bool = True
    ) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            max_target_seqs: Maximum number of target sequences per query
            output_csv: Path to save output CSV (optional)
            normalize_scores: Whether to normalize bitscore to [0, 1]
            
        Returns:
            DataFrame with five-dimensional input data
        """
        print("=" * 70)
        print("Data Preparation Pipeline")
        print("=" * 70)
        
        # Load catalog
        self.load_catalog()
        
        # Perform alignments
        alignment_files = self.perform_all_alignments(max_target_seqs)
        
        # Generate input CSV
        if output_csv is None:
            output_csv = str(self.output_dir / "gene_associations.csv")
        
        result_df = self.generate_input_csv(
            alignment_files,
            output_path=output_csv,
            normalize_scores=normalize_scores
        )
        
        print("=" * 70)
        print("Data preparation completed!")
        print("=" * 70)
        
        return result_df


def main():
    """Command-line interface for data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare data for Species-Gene GNN from genome assemblies'
    )
    parser.add_argument(
        '--catalog',
        type=str,
        required=True,
        help='Path to dataset_catalog.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for intermediate and final files (default: output)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Path to output CSV file (default: output/gene_associations.csv)'
    )
    parser.add_argument(
        '--max-target-seqs',
        type=int,
        default=5,
        help='Maximum number of target sequences per query (default: 5)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Do not normalize similarity scores'
    )
    parser.add_argument(
        '--include-self-alignment',
        action='store_true',
        help='Include within-species all-vs-all alignments (excluding gene self-comparisons)'
    )
    
    args = parser.parse_args()
    
    prep = DataPreparation(
        args.catalog, 
        args.output_dir, 
        include_self_alignment=args.include_self_alignment
    )
    prep.prepare_data(
        max_target_seqs=args.max_target_seqs,
        output_csv=args.output_csv,
        normalize_scores=not args.no_normalize
    )


if __name__ == '__main__':
    main()
