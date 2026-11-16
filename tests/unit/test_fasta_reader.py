"""Unit tests for FastaReader genomic file parsing and validation."""

from pathlib import Path
import pytest

from geneweaver.genomic_parser.exceptions import (
    EmptyFastaFileError,
    EmptySequenceError,
    InvalidFileFormatError,
    InvalidSequenceError,
    MalformedHeaderError,
)
from geneweaver.genomic_parser.fasta_reader import FastaReader
from geneweaver.genomic_parser.records import FastaRecord


class TestFastaReaderValidFiles:
    """Tests parsing valid single-record, multi-record, and gzipped FASTA files."""

    def test_read_valid_single_record(self, sample_valid_fasta: Path):
        reader = FastaReader(sample_valid_fasta)
        records = reader.read_records()

        assert len(records) == 1
        record = records[0]
        assert isinstance(record, FastaRecord)
        assert record.identifier == "chr21"
        assert record.description == "Homo sapiens chromosome 21, GRCh38"

        # Verify case normalization and U -> T conversion
        assert "NNNNNNNNNNNNNNNNNNNN" in record.sequence
        assert "ACGTACGTACGT" in record.sequence
        assert "TGCATGCA" in record.sequence  # UGCAUGCA converted to TGCATGCA
        assert record.sequence.isupper()
        assert record.length == len(record.sequence)
        assert len(record) == record.length

    def test_read_valid_multi_records(self, sample_multi_fasta: Path):
        reader = FastaReader(sample_multi_fasta)
        records = reader.read_records()

        assert len(records) == 3
        ids = [r.identifier for r in records]
        assert ids == ["chr1", "chr2", "chrX"]

        assert records[0].sequence == "ATGCATGCATGCGCATGCATGCAT"
        assert records[1].sequence == "CCGGAATTCCGG"
        assert records[2].sequence == "TTTTAAAAGGGGCCCC"

    def test_read_gzipped_fasta(self, sample_gzipped_fasta: Path):
        reader = FastaReader(sample_gzipped_fasta)
        records = reader.read_records()

        assert len(records) == 1
        assert records[0].identifier == "chrM"
        assert records[0].length == 100
        assert records[0].sequence.startswith("GATCACAGGT")

    def test_multiline_wrapped_sequence(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "wrapped.fasta"
        file_path.write_text(
            ">seq1 wrapped across short lines\n"
            "AAAA\n"
            "CCCC\n"
            "GGGG\n"
            "TTTT\n",
            encoding="utf-8",
        )
        reader = FastaReader(file_path)
        records = reader.read_records()

        assert len(records) == 1
        assert records[0].sequence == "AAAACCCCGGGGTTTT"

    def test_header_without_description(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "nodesc.fa"
        file_path.write_text(">scaffold_101\nACGTACGT\n", encoding="utf-8")
        reader = FastaReader(file_path)
        records = reader.read_records()

        assert records[0].identifier == "scaffold_101"
        assert records[0].description == ""

    def test_stream_records_is_lazy_generator(self, sample_multi_fasta: Path):
        reader = FastaReader(sample_multi_fasta)
        stream = reader.stream_records()

        # Generator should not be a list
        assert hasattr(stream, "__next__")
        first = next(stream)
        assert first.identifier == "chr1"
        second = next(stream)
        assert second.identifier == "chr2"
        third = next(stream)
        assert third.identifier == "chrX"
        with pytest.raises(StopIteration):
            next(stream)

    def test_stream_chromosomes_chunking(self, sample_multi_fasta: Path):
        reader = FastaReader(sample_multi_fasta)
        chrom_stream = reader.stream_chromosomes(chunk_size=6)

        chrom_list = list(chrom_stream)
        assert len(chrom_list) == 3

        # chr1 has 24 bases: chunks should be 6 + 6 + 6 + 6
        chr1_id, chr1_chunks_gen = chrom_list[0]
        assert chr1_id == "chr1"
        chr1_chunks = list(chr1_chunks_gen)
        assert chr1_chunks == ["ATGCAT", "GCATGC", "GCATGC", "ATGCAT"]
        assert "".join(chr1_chunks) == "ATGCATGCATGCGCATGCATGCAT"

    def test_get_chromosome_lengths(self, sample_multi_fasta: Path):
        reader = FastaReader(sample_multi_fasta)
        lengths = reader.get_chromosome_lengths()

        assert lengths == {
            "chr1": 24,
            "chr2": 12,
            "chrX": 16,
        }

    def test_iupac_ambiguity_characters_allowed(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "iupac.fa"
        file_path.write_text(">iupac_seq\nACGT RYSWKM BDHVN\n", encoding="utf-8")
        reader = FastaReader(file_path, allow_ambiguity=True)
        records = reader.read_records()

        assert records[0].sequence == "ACGTRYSWKMBDHVN"

    def test_disallow_ambiguity_raises_on_n(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "strict_canonical.fa"
        file_path.write_text(">strict\nACGTNACGT\n", encoding="utf-8")
        reader = FastaReader(file_path, allow_ambiguity=False)

        with pytest.raises(InvalidSequenceError, match="Invalid sequence in record 'strict'"):
            reader.read_records()


class TestFastaReaderMalformedAndEdgeCases:
    """Tests error handling for invalid files, malformed headers, and empty records."""

    def test_empty_file_raises_error(self, temp_fasta_dir: Path):
        empty_file = temp_fasta_dir / "empty.fasta"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(EmptyFastaFileError, match="FASTA file is empty"):
            FastaReader(empty_file)

    def test_file_with_only_whitespace_raises_error(self, temp_fasta_dir: Path):
        whitespace_file = temp_fasta_dir / "whitespace.fa"
        whitespace_file.write_text("   \n\n\t\n  \n", encoding="utf-8")
        reader = FastaReader(whitespace_file)

        with pytest.raises(EmptyFastaFileError, match="contains no sequence records"):
            reader.read_records()

    def test_file_with_only_comments_raises_error(self, temp_fasta_dir: Path):
        comment_file = temp_fasta_dir / "comments.fa"
        comment_file.write_text("# This is a comment\n; Another comment\n", encoding="utf-8")
        reader = FastaReader(comment_file)

        with pytest.raises(EmptyFastaFileError, match="contains no sequence records"):
            reader.read_records()

    def test_missing_header_raises_malformed_header_error(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "no_header.fasta"
        file_path.write_text("ATGCATGCATGC\n", encoding="utf-8")
        reader = FastaReader(file_path)

        with pytest.raises(MalformedHeaderError, match="must begin with a record header starting with '>'"):
            reader.read_records()

    def test_empty_header_id_raises_malformed_header_error(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "empty_header.fa"
        file_path.write_text(">\nATGCATGC\n", encoding="utf-8")
        reader = FastaReader(file_path)

        with pytest.raises(MalformedHeaderError, match="contains no identifier after '>'"):
            reader.read_records()

    def test_empty_sequence_before_next_header_raises_error(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "consecutive_headers.fa"
        file_path.write_text(
            ">chr1\n"
            ">chr2\n"
            "ACGTACGT\n",
            encoding="utf-8",
        )
        reader = FastaReader(file_path)

        with pytest.raises(EmptySequenceError, match="FASTA record 'chr1' contains no sequence data"):
            reader.read_records()

    def test_empty_sequence_at_eof_raises_error(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "trailing_header.fa"
        file_path.write_text(">chr1\n", encoding="utf-8")
        reader = FastaReader(file_path)

        with pytest.raises(EmptySequenceError, match="at end of file contains no sequence data"):
            reader.read_records()

    def test_invalid_characters_in_sequence_raises_error(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "invalid_chars.fasta"
        file_path.write_text(">chr1\nACGT123ACGT\n", encoding="utf-8")
        reader = FastaReader(file_path)

        with pytest.raises(InvalidSequenceError, match="Invalid sequence in record 'chr1'"):
            reader.read_records()

    def test_invalid_symbol_character_raises_error(self, temp_fasta_dir: Path):
        file_path = temp_fasta_dir / "symbol_chars.fasta"
        file_path.write_text(">chr1\nACGT@ACGT\n", encoding="utf-8")
        reader = FastaReader(file_path)

        with pytest.raises(InvalidSequenceError, match="Invalid nucleotide character '@'"):
            reader.read_records()

    def test_file_not_found_raises_error(self, temp_fasta_dir: Path):
        non_existent = temp_fasta_dir / "does_not_exist.fa"
        with pytest.raises(FileNotFoundError, match="FASTA file not found"):
            FastaReader(non_existent)

    def test_directory_path_raises_is_a_directory_error(self, temp_fasta_dir: Path):
        with pytest.raises(IsADirectoryError, match="is a directory, not a file"):
            FastaReader(temp_fasta_dir)

    def test_unsupported_file_extension_raises_error(self, temp_fasta_dir: Path):
        bad_ext = temp_fasta_dir / "genome.txt"
        bad_ext.write_text(">chr1\nACGT\n", encoding="utf-8")

        with pytest.raises(InvalidFileFormatError, match="Unsupported file format"):
            FastaReader(bad_ext)
