"""Unit tests for GenomicChunker and ChunkDescriptor."""

import numpy as np
import pytest

from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder


class TestGenomicChunkerBasic:
    """Tests basic chunking behavior with strings, lengths, and edge cases."""

    def test_empty_sequence_returns_empty_list(self):
        chunker = GenomicChunker(chunk_size=100, halo_size=20)
        chunks = chunker.chunk("")
        assert chunks == []
        assert list(chunker.stream_chunks("")) == []

    def test_sequence_shorter_than_chunk_size(self):
        chunker = GenomicChunker(chunk_size=1000, halo_size=22)
        seq = "A" * 250
        chunks = chunker.chunk(seq, chromosome="chr1")

        assert len(chunks) == 1
        chunk = chunks[0]

        assert chunk.chunk_id == "chr1_chunk_0000"
        assert chunk.chromosome == "chr1"
        assert chunk.chunk_index == 0
        assert chunk.global_start == 0
        assert chunk.global_end == 250
        assert chunk.actual_length == 250
        assert chunk.core_start == 0
        assert chunk.core_end == 250
        assert chunk.core_length == 250
        assert chunk.halo_start == 250
        assert chunk.halo_end == 250
        assert chunk.halo_length == 0
        assert chunk.has_halo is False
        assert chunk.sequence_data == seq

    def test_sequence_exactly_equal_to_chunk_size(self):
        chunker = GenomicChunker(chunk_size=500, halo_size=22)
        seq = "C" * 500
        chunks = chunker.chunk(seq, chromosome="chr2")

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.actual_length == 500
        assert chunk.core_length == 500
        assert chunk.halo_length == 0
        assert chunk.has_halo is False
        assert chunk.global_start == 0
        assert chunk.global_end == 500
        assert chunk.sequence_data == seq

    def test_sequence_larger_than_chunk_size_two_chunks(self):
        # Chunk size 100, halo 20, total length 150
        # Chunk 0: core [0, 100), halo [100, 120), actual 120
        # Chunk 1: core [100, 150), halo [150, 150), actual 50
        chunker = GenomicChunker(chunk_size=100, halo_size=20)
        seq = "G" * 150
        chunks = chunker.chunk(seq, chromosome="chr3")

        assert len(chunks) == 2

        # First chunk
        c0 = chunks[0]
        assert c0.chunk_id == "chr3_chunk_0000"
        assert c0.core_bounds == (0, 100)
        assert c0.core_length == 100
        assert c0.halo_bounds == (100, 120)
        assert c0.halo_length == 20
        assert c0.has_halo is True
        assert c0.actual_length == 120
        assert len(c0.sequence_data) == 120

        # Second chunk (terminal)
        c1 = chunks[1]
        assert c1.chunk_id == "chr3_chunk_0001"
        assert c1.core_bounds == (100, 150)
        assert c1.core_length == 50
        assert c1.halo_bounds == (150, 150)
        assert c1.halo_length == 0
        assert c1.has_halo is False
        assert c1.actual_length == 50
        assert len(c1.sequence_data) == 50


class TestGenomicChunkerHaloOverlapAndCoordinates:
    """Tests the 22-base halo overlap mechanics and global coordinate reconstruction."""

    def test_halo_content_matches_next_chunk_core(self):
        # Use an enumerated sequence string where each base reflects its position
        chunk_size = 50
        halo_size = 10
        total_len = 125
        # Create unique sequence
        bases = "ACGT"
        seq = "".join(bases[i % 4] for i in range(total_len))

        chunker = GenomicChunker(chunk_size=chunk_size, halo_size=halo_size)
        chunks = chunker.chunk(seq, chromosome="chrTest")

        assert len(chunks) == 3

        # Chunk 0 halo should be exactly equal to the first 10 bases of Chunk 1 core
        c0_halo = chunks[0].sequence_data[chunks[0].core_length :]
        c1_core_start = chunks[1].sequence_data[:halo_size]
        assert c0_halo == c1_core_start
        assert c0_halo == seq[50:60]

        # Chunk 1 halo should be exactly equal to the first 10 bases of Chunk 2 core
        c1_halo = chunks[1].sequence_data[chunks[1].core_length :]
        c2_core_start = chunks[2].sequence_data[:halo_size]
        assert c1_halo == c2_core_start
        assert c1_halo == seq[100:110]

        # Chunk 2 is terminal, halo length is 0
        assert chunks[2].halo_length == 0

    def test_no_missing_or_duplicated_core_positions(self):
        # Test across multiple non-trivial sizes (e.g. 100,003 bp with chunk_size 10,000)
        total_len = 100_003
        chunk_size = 10_000
        halo_size = 22
        seq = "A" * total_len

        chunker = GenomicChunker(chunk_size=chunk_size, halo_size=halo_size)
        chunks = chunker.chunk(seq, chromosome="chrVerify")

        # Verify contiguous partition of core regions
        covered_positions = []
        for chunk in chunks:
            covered_positions.extend(range(chunk.core_start, chunk.core_end))

        # Check that covered positions exactly equal [0, 1, ..., total_len - 1]
        assert covered_positions == list(range(total_len))
        assert len(set(covered_positions)) == total_len

    def test_global_coordinate_translation_and_helpers(self):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        seq = "ACGT" * 50  # 200 bp
        chunks = chunker.chunk(seq, chromosome="chrX")

        c1 = chunks[1]  # Core [100, 200), global_start = 100
        assert c1.global_start == 100
        assert c1.global_start_bp == 100

        # Local offset 0 is global position 100 (0-based) -> 101 (1-based)
        assert c1.to_1based_coordinate(0) == 101
        assert c1.to_1based_coordinate(50) == 151
        assert c1.to_1based_coordinate(99) == 200

        # Out of bounds offset raises IndexError
        with pytest.raises(IndexError, match="out of bounds"):
            c1.to_1based_coordinate(150)
        with pytest.raises(IndexError, match="out of bounds"):
            c1.to_1based_coordinate(-1)

        # Helper predicates
        assert c1.is_in_core(0) is True
        assert c1.is_in_core(99) is True
        assert c1.is_in_core(100) is False  # halo
        assert c1.is_in_halo(100) is False  # c1 is terminal so actual_len = 100

        # Check c0 helper predicates
        c0 = chunks[0]  # Core [0, 100), Halo [100, 122), Actual 122
        assert c0.is_in_core(99) is True
        assert c0.is_in_core(100) is False
        assert c0.is_in_halo(100) is True
        assert c0.is_in_halo(121) is True
        assert c0.is_in_halo(122) is False

        assert c0.local_core_bounds == (0, 100)
        assert c0.local_halo_bounds == (100, 122)


class TestGenomicChunkerBoundaryCases:
    """Tests delicate boundary cases around chunk_size and halo_size."""

    def test_sequence_length_chunk_size_minus_one(self):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        seq = "T" * 99
        chunks = chunker.chunk(seq)
        assert len(chunks) == 1
        assert chunks[0].core_length == 99
        assert chunks[0].halo_length == 0

    def test_sequence_length_chunk_size_plus_one(self):
        # Chunk 0: core 100, halo min(22, 1) = 1, actual 101
        # Chunk 1: core 1, halo 0, actual 1
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        seq = "T" * 101
        chunks = chunker.chunk(seq)
        assert len(chunks) == 2

        assert chunks[0].core_length == 100
        assert chunks[0].halo_length == 1
        assert chunks[0].actual_length == 101

        assert chunks[1].core_length == 1
        assert chunks[1].halo_length == 0
        assert chunks[1].actual_length == 1

    def test_sequence_length_chunk_size_plus_halo_size(self):
        # Total 122 bp, chunk 100, halo 22
        # Chunk 0: core 100, halo min(22, 22) = 22, actual 122
        # Chunk 1: core 22, halo 0, actual 22
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        seq = "G" * 122
        chunks = chunker.chunk(seq)
        assert len(chunks) == 2

        assert chunks[0].core_length == 100
        assert chunks[0].halo_length == 22
        assert chunks[0].actual_length == 122

        assert chunks[1].core_length == 22
        assert chunks[1].halo_length == 0
        assert chunks[1].actual_length == 22

    def test_sequence_length_chunk_size_plus_halo_size_plus_one(self):
        # Total 123 bp
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        seq = "C" * 123
        chunks = chunker.chunk(seq)
        assert len(chunks) == 2

        assert chunks[0].core_length == 100
        assert chunks[0].halo_length == 22
        assert chunks[0].actual_length == 122

        assert chunks[1].core_length == 23
        assert chunks[1].halo_length == 0
        assert chunks[1].actual_length == 23


class TestGenomicChunkerMultipleRecords:
    """Tests chunking across multiple independent FASTA records."""

    def test_chunk_records_independent_coordinates(self):
        rec1 = FastaRecord(identifier="chr1", sequence="A" * 150)
        rec2 = FastaRecord(identifier="chr2", sequence="C" * 220)

        chunker = GenomicChunker(chunk_size=100, halo_size=20)
        chunks = chunker.chunk_records([rec1, rec2])

        # rec1: 2 chunks (chr1_chunk_0000, chr1_chunk_0001)
        # rec2: 3 chunks (chr2_chunk_0000, chr2_chunk_0001, chr2_chunk_0002)
        assert len(chunks) == 5

        # Check rec1 chunks
        assert chunks[0].chunk_id == "chr1_chunk_0000"
        assert chunks[0].chromosome == "chr1"
        assert chunks[0].global_start == 0

        assert chunks[1].chunk_id == "chr1_chunk_0001"
        assert chunks[1].chromosome == "chr1"
        assert chunks[1].global_start == 100

        # Check rec2 chunks (coordinates reset!)
        assert chunks[2].chunk_id == "chr2_chunk_0000"
        assert chunks[2].chromosome == "chr2"
        assert chunks[2].global_start == 0  # Not 150!

        assert chunks[3].chunk_id == "chr2_chunk_0001"
        assert chunks[3].chromosome == "chr2"
        assert chunks[3].global_start == 100

        assert chunks[4].chunk_id == "chr2_chunk_0002"
        assert chunks[4].chromosome == "chr2"
        assert chunks[4].global_start == 200

    def test_stream_records_lazy_generation(self):
        recs = [
            FastaRecord(identifier="chrA", sequence="A" * 100),
            FastaRecord(identifier="chrB", sequence="T" * 100),
        ]
        chunker = GenomicChunker(chunk_size=50, halo_size=10)
        stream = chunker.stream_records(recs)
        first_chunk = next(stream)
        assert first_chunk.chunk_id == "chrA_chunk_0000"
        second_chunk = next(stream)
        assert second_chunk.chunk_id == "chrA_chunk_0001"
        third_chunk = next(stream)
        assert third_chunk.chunk_id == "chrB_chunk_0000"


class TestGenomicChunkerCompatibilityAndTypes:
    """Tests compatibility with NumPy uint8 arrays, EncodedTwoBit, and FastaRecord."""

    def test_chunking_unpacked_numpy_array(self):
        seq_str = "ACGT" * 25  # 100 bp
        unpacked_2bit = TwoBitEncoder.encode(seq_str)
        assert isinstance(unpacked_2bit, np.ndarray)

        chunker = GenomicChunker(chunk_size=40, halo_size=10)
        chunks = chunker.chunk(unpacked_2bit, chromosome="chrNumPy")

        assert len(chunks) == 3
        c0 = chunks[0]
        assert isinstance(c0.sequence_data, np.ndarray)
        assert c0.sequence_data.dtype == np.uint8
        assert len(c0.sequence_data) == 50  # 40 core + 10 halo
        # Verify decoding the chunk slice matches the original string slice
        assert TwoBitEncoder.decode(c0.sequence_data) == seq_str[0:50]

    def test_chunking_encoded_twobit_container(self):
        seq_str = "ACGT" * 50  # 200 bp
        encoded = TwoBitEncoder.encode_packed(seq_str)
        assert isinstance(encoded, EncodedTwoBit)

        chunker = GenomicChunker(chunk_size=80, halo_size=15)
        chunks = chunker.chunk(encoded, chromosome="chrTwoBit")

        assert len(chunks) == 3
        c0 = chunks[0]
        assert isinstance(c0.sequence_data, EncodedTwoBit)
        assert c0.actual_length == 95  # 80 core + 15 halo
        assert len(c0.sequence_data) == 95
        assert TwoBitEncoder.decode_packed(c0.sequence_data) == seq_str[0:95]

    def test_chunking_direct_fasta_record(self):
        rec = FastaRecord(identifier="chrMito", sequence="ACGT" * 30, description="Mitochondria")
        chunker = GenomicChunker(chunk_size=50, halo_size=10)
        chunks = chunker.chunk(rec)

        assert len(chunks) == 3
        assert chunks[0].chromosome == "chrMito"
        assert chunks[0].chunk_id == "chrMito_chunk_0000"

    def test_architectural_partition_alias(self):
        chunker = GenomicChunker(chunk_size=50, halo_size=10)
        chunks = chunker.partition("chrTest", "ACGT" * 20)
        assert len(chunks) == 2


class TestGenomicChunkerValidation:
    """Tests parameter validation for chunk_size and halo_size."""

    def test_zero_or_negative_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            GenomicChunker(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            GenomicChunker(chunk_size=-100)

    def test_negative_halo_size_raises_value_error(self):
        with pytest.raises(ValueError, match="halo_size cannot be negative"):
            GenomicChunker(chunk_size=100, halo_size=-1)

    def test_chunk_size_less_than_or_equal_to_halo_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size .* must be greater than halo_size"):
            GenomicChunker(chunk_size=20, halo_size=20)
        with pytest.raises(ValueError, match="chunk_size .* must be greater than halo_size"):
            GenomicChunker(chunk_size=15, halo_size=22)

    def test_unsupported_input_type_raises_type_error(self):
        chunker = GenomicChunker()
        with pytest.raises(TypeError, match="Unsupported sequence type"):
            chunker.chunk(12345)  # type: ignore
