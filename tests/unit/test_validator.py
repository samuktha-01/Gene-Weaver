"""Unit tests for SequenceValidator."""

from pathlib import Path
import pytest

from geneweaver.genomic_parser.exceptions import (
    EmptyFastaFileError,
    InvalidFileFormatError,
    InvalidSequenceError,
    MalformedHeaderError,
)
from geneweaver.genomic_parser.validator import SequenceValidator


class TestSequenceValidator:
    """Tests validation of file paths, headers, and CRISPR guides."""

    def test_validate_header_valid(self):
        id_, desc = SequenceValidator.validate_header(">chr1 Homo sapiens chromosome 1")
        assert id_ == "chr1"
        assert desc == "Homo sapiens chromosome 1"

    def test_validate_header_no_description(self):
        id_, desc = SequenceValidator.validate_header(">scaffold_42")
        assert id_ == "scaffold_42"
        assert desc == ""

    def test_validate_header_missing_gt(self):
        with pytest.raises(MalformedHeaderError, match="must start with '>'"):
            SequenceValidator.validate_header("chr1 Homo sapiens")

    def test_validate_header_empty_id(self):
        with pytest.raises(MalformedHeaderError, match="contains no identifier after '>'"):
            SequenceValidator.validate_header(">   ")

    def test_validate_fasta_path_valid(self, tmp_path: Path):
        f = tmp_path / "valid.fa"
        f.write_text(">seq1\nACGT\n")
        validated = SequenceValidator.validate_fasta_path(f)
        assert validated == f.resolve()

    def test_validate_fasta_path_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.fa"
        f.write_text("")
        with pytest.raises(EmptyFastaFileError, match="FASTA file is empty"):
            SequenceValidator.validate_fasta_path(f)

    def test_validate_fasta_path_nonexistent(self, tmp_path: Path):
        f = tmp_path / "missing.fasta"
        with pytest.raises(FileNotFoundError, match="FASTA file not found"):
            SequenceValidator.validate_fasta_path(f)

    def test_validate_fasta_path_unsupported_extension(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("header,val\n")
        with pytest.raises(InvalidFileFormatError, match="Unsupported file format"):
            SequenceValidator.validate_fasta_path(f)

    def test_validate_guide_canonical(self):
        guide = "GAGTCCGAGCAGAAGAAGAANGG"
        validated = SequenceValidator.validate_guide(guide, expected_len=23)
        assert validated == "GAGTCCGAGCAGAAGAAGAANGG"

    def test_validate_guide_lowercase_and_rna(self):
        guide = "gaguccgagcagaagaagaangg"
        validated = SequenceValidator.validate_guide(guide, expected_len=23)
        assert validated == "GAGTCCGAGCAGAAGAAGAANGG"

    def test_validate_guide_invalid_length(self):
        with pytest.raises(ValueError, match="Invalid CRISPR guide length: got 20 nt, expected 23 nt"):
            SequenceValidator.validate_guide("GAGTCCGAGCAGAAGAAGAA", expected_len=23)

    def test_validate_guide_invalid_characters(self):
        with pytest.raises(InvalidSequenceError, match="Invalid nucleotide in CRISPR guide"):
            SequenceValidator.validate_guide("GAGTCCGAGCAGAAGAAG12NGG", expected_len=23)
