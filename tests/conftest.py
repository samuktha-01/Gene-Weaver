"""Shared pytest fixtures for GeneWeaver test suites."""

import gzip
from pathlib import Path
from typing import Generator
import pytest


@pytest.fixture
def temp_fasta_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for generating test FASTA files."""
    d = tmp_path / "fasta_fixtures"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_valid_fasta(temp_fasta_dir: Path) -> Path:
    """Creates a standard single-record FASTA file."""
    file_path = temp_fasta_dir / "sample.fasta"
    content = (
        ">chr21 Homo sapiens chromosome 21, GRCh38\n"
        "NNNNNNNNNNNNNNNNNNNN\n"
        "GAGTCCGAGCAGAAGAAGAAAGG\n"
        "acgtacgtacgt\n"
        "UGCAUGCA\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_multi_fasta(temp_fasta_dir: Path) -> Path:
    """Creates a multi-record FASTA file with different chromosomes."""
    file_path = temp_fasta_dir / "multi_genome.fa"
    content = (
        ">chr1 Chromosome 1\n"
        "ATGCATGCATGC\n"
        "GCATGCATGCAT\n"
        ">chr2 Chromosome 2 scaffold\n"
        "CCGGAATTCCGG\n"
        ">chrX Chromosome X sex contig\n"
        "TTTTAAAAGGGGCCCC\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_gzipped_fasta(temp_fasta_dir: Path) -> Path:
    """Creates a valid gzip-compressed FASTA file."""
    file_path = temp_fasta_dir / "compressed.fa.gz"
    content = (
        ">chrM Mitochondrion complete genome\n"
        "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCAT\n"
        "TTGGTATTTTCGTCTGGGGGGTGTGCACGCGATAGCATTGCGAGACGCTG\n"
    ).encode("utf-8")
    with gzip.open(file_path, "wb") as gz:
        gz.write(content)
    return file_path
