"""Comprehensive unit tests for CPUAligner NumPy baseline alignment engine."""

import numpy as np
import pytest

from geneweaver.alignment import (
    AlignmentConfig,
    AlignmentResult,
    BaseSequenceAligner,
    CPUAligner,
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    SequenceAligner,
    Strand,
)
from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder


@pytest.fixture
def aligner() -> CPUAligner:
    """Provides a default CPUAligner instance."""
    return CPUAligner()


@pytest.fixture
def guide_20() -> str:
    """Standard 20 nt guide sequence for tests."""
    return "GAGTCCGAGCAGAAGAAGAA"


@pytest.fixture
def guide_23() -> str:
    """Standard 23 nt guide sequence with PAM for tests."""
    return "GAGTCCGAGCAGAAGAAGAAAGG"


class TestCPUAlignerProtocolAndBasics:
    """Tests SequenceAligner protocol conformance and base properties."""

    def test_protocol_conformance(self, aligner: CPUAligner):
        assert isinstance(aligner, SequenceAligner)
        assert isinstance(aligner, BaseSequenceAligner)
        assert aligner.backend_name == "cpu-numpy"

    def test_custom_config(self):
        cfg = AlignmentConfig(max_mismatches=2, batch_size=1024)
        aligner = CPUAligner(config=cfg)
        assert aligner.config.max_mismatches == 2
        assert aligner.config.batch_size == 1024


class TestCPUAlignerSequenceMatches:
    """Tests exact matches, single mismatches, and multiple mismatches."""

    def test_exact_sequence_match(self, aligner: CPUAligner, guide_20: str):
        # Place exact guide match at offset 10 with PAM 'AGG'
        seq = "A" * 10 + guide_20 + "AGG" + "T" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert isinstance(result, AlignmentResult)
        assert result.hit_count == 1
        assert len(result.hits) == 1

        hit = result.hits[0]
        assert hit.local_offset == 10
        assert hit.global_start == 10
        assert hit.global_end == 33  # 10 + 23 = 33
        assert hit.genomic_position == 11
        assert hit.strand == Strand.FORWARD
        assert hit.strand_symbol == "+"
        assert hit.mismatch_count == 0
        assert hit.mismatch_bitmask == 0
        assert hit.mismatch_info.positions == ()
        assert hit.mismatch_info.seed_count == 0
        assert hit.target_sequence[:20] == guide_20
        assert hit.pam_sequence == "AGG"

    def test_one_mismatch(self, aligner: CPUAligner, guide_20: str):
        # 1 mismatch at position 5 (1-based, index 4): 'T' -> 'C'
        mutated_guide = guide_20[:4] + ("C" if guide_20[4] != "C" else "A") + guide_20[5:]
        seq = "C" * 5 + mutated_guide + "TGG" + "G" * 5
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.mismatch_count == 1
        assert hit.mismatch_info.positions == (5,)
        assert hit.mismatch_bitmask == (1 << 4)
        assert hit.mismatch_info.seed_count == 0

    def test_multiple_mismatches(self, aligner: CPUAligner, guide_20: str):
        # 3 mismatches: pos 2 (index 1), pos 11 (index 10, seed), pos 18 (index 17, seed)
        mut_list = list(guide_20)
        mut_list[1] = "C" if mut_list[1] != "C" else "T"
        mut_list[10] = "A" if mut_list[10] != "A" else "G"
        mut_list[17] = "T" if mut_list[17] != "T" else "C"
        mut_guide = "".join(mut_list)

        seq = "T" * 8 + mut_guide + "CGG" + "A" * 8
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr2")[0]

        cfg = AlignmentConfig(max_mismatches=3)
        result = aligner.align(chunk=chunk, guide=guide_20, config=cfg)

        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.mismatch_count == 3
        assert hit.mismatch_info.positions == (2, 11, 18)
        assert hit.mismatch_info.seed_count == 2
        expected_bitmask = (1 << 1) | (1 << 10) | (1 << 17)
        assert hit.mismatch_bitmask == expected_bitmask

    def test_maximum_mismatch_threshold(self, aligner: CPUAligner, guide_20: str):
        # Site 1 has 2 mismatches, Site 2 has 4 mismatches
        mut2 = list(guide_20)
        mut2[0] = "C" if mut2[0] != "C" else "T"
        mut2[1] = "C" if mut2[1] != "C" else "T"
        site2 = "".join(mut2)

        mut4 = list(guide_20)
        for i in range(4):
            mut4[i] = "C" if mut4[i] != "C" else "T"
        site4 = "".join(mut4)

        seq = site2 + "TTTT" + site4 + "TTTT"
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        # Max mismatches = 1: neither passes
        r1 = aligner.align(
            chunk=chunk,
            guide=guide_20,
            config=AlignmentConfig(max_mismatches=1, seed_max_mismatches=1),
        )
        assert r1.hit_count == 0

        # Max mismatches = 2: only site2 passes
        r2 = aligner.align(chunk=chunk, guide=guide_20, config=AlignmentConfig(max_mismatches=2))
        assert r2.hit_count == 1
        assert r2.hits[0].local_offset == 0
        assert r2.hits[0].mismatch_count == 2

        # Max mismatches = 4: both pass
        r4 = aligner.align(chunk=chunk, guide=guide_20, config=AlignmentConfig(max_mismatches=4))
        assert r4.hit_count == 2
        assert r4.hits[0].local_offset == 0
        assert r4.hits[1].local_offset == 24

    def test_no_matching_candidates(self, aligner: CPUAligner, guide_20: str):
        # Completely discordant sequence (all C when guide is mostly A/G/T)
        seq = "C" * 60
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk=chunk, guide=guide_20, config=AlignmentConfig(max_mismatches=2))
        assert result.hit_count == 0
        assert len(result.hits) == 0
        assert result.total_windows_evaluated == 60 - 20 + 1

    def test_multiple_candidate_matches(self, aligner: CPUAligner, guide_20: str):
        # 3 identical guide copies separated by spacers
        seq = guide_20 + "AAAA" + guide_20 + "CCCC" + guide_20
        chunker = GenomicChunker(chunk_size=150, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 3
        offsets = [h.local_offset for h in result.hits]
        assert offsets == [0, 24, 48]
        assert all(h.mismatch_count == 0 for h in result.hits)


class TestCPUAlignerBoundaryAndEdgeCases:
    """Tests edge cases: empty inputs, short sequences, first/final positions, and halos."""

    def test_empty_genomic_sequence(self, aligner: CPUAligner, guide_20: str):
        chunk = ChunkDescriptor(
            chunk_id="chr1_chunk_0000",
            chromosome="chr1",
            chunk_index=0,
            global_start=0,
            global_end=0,
            actual_length=0,
            core_start=0,
            core_end=0,
            halo_start=0,
            halo_end=0,
            sequence_data="",
        )
        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 0
        assert len(result.hits) == 0
        assert result.total_windows_evaluated == 0

    def test_sequence_shorter_than_guide(self, aligner: CPUAligner, guide_20: str):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk("ACGTACGT", chromosome="chr1")[0]  # 8 nt < 20 nt
        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 0
        assert result.total_windows_evaluated == 0

    def test_candidate_at_first_valid_position(self, aligner: CPUAligner, guide_20: str):
        # Match starts immediately at local_offset = 0
        seq = guide_20 + "AGG" + "TTTTTTTT"
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chrTest")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.local_offset == 0
        assert hit.global_start == 0
        assert hit.genomic_position == 1
        assert hit.target_sequence[:20] == guide_20

    def test_candidate_at_final_valid_position(self, aligner: CPUAligner, guide_20: str):
        # In a sequence of length 50, the last 20-mer starts at index 50 - 20 = 30
        prefix = "T" * 30
        seq = prefix + guide_20
        assert len(seq) == 50

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chrTest")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.local_offset == 30
        assert hit.global_start == 30
        assert hit.genomic_position == 31
        assert hit.target_sequence == guide_20

    def test_correct_global_coordinate_mapping(self, aligner: CPUAligner, guide_20: str):
        # Explicit chunk descriptor with arbitrary global offset
        global_base = 5_000_000
        seq = "C" * 42 + guide_20 + "GGG" + "C" * 10
        chunk = ChunkDescriptor(
            chunk_id="chr7_chunk_0005",
            chromosome="chr7",
            chunk_index=5,
            global_start=global_base,
            global_end=global_base + len(seq),
            actual_length=len(seq),
            core_start=global_base,
            core_end=global_base + len(seq),
            halo_start=global_base + len(seq),
            halo_end=global_base + len(seq),
            sequence_data=seq,
        )

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.chromosome == "chr7"
        assert hit.local_offset == 42
        assert hit.global_start == global_base + 42
        assert hit.global_end == global_base + 42 + 23
        assert hit.genomic_position == global_base + 42 + 1

    def test_halo_region_candidate_filtering(self, aligner: CPUAligner, guide_20: str):
        # Chunk has core length 50 and halo length 22 (actual length 72)
        # Place match 1 in core at offset 10
        # Place match 2 in halo at offset 51
        seq_list = ["T"] * 72
        seq_list[10:30] = list(guide_20)
        seq_list[51:71] = list(guide_20)
        seq = "".join(seq_list)

        chunk = ChunkDescriptor(
            chunk_id="chr1_chunk_0000",
            chromosome="chr1",
            chunk_index=0,
            global_start=0,
            global_end=72,
            actual_length=72,
            core_start=0,
            core_end=50,
            halo_start=50,
            halo_end=72,
            sequence_data=seq,
        )

        result = aligner.align(chunk=chunk, guide=guide_20)
        # Only the candidate in core (offset 10) must be returned; offset 51 in halo is dropped
        assert result.hit_count == 1
        assert result.hits[0].local_offset == 10
        assert chunk.is_in_core(result.hits[0].local_offset)


class TestCPUAlignerMultiChunkAndMultiRecord:
    """Tests multi-chunk partitioning, halo deduplication across chunks, and multi-FASTA records."""

    def test_multiple_genomic_chunks_deduplication(self, aligner: CPUAligner, guide_20: str):
        # Total sequence length 200 bp partitioned into 2 chunks with chunk_size=100, halo_size=22
        # Place hits:
        # Hit 1: global offset 20 (Chunk 0 core)
        # Hit 2: global offset 85 (straddles boundary: starts in Chunk 0 core at 85, extends into halo to 105)
        # Hit 3: global offset 110 (starts in Chunk 0 halo at 110, starts in Chunk 1 core at 10)
        # Hit 4: global offset 150 (Chunk 1 core)
        seq_chars = ["A"] * 200

        def inject_target(pos: int):
            for i, char in enumerate(guide_20):
                seq_chars[pos + i] = char

        inject_target(20)
        inject_target(85)
        inject_target(110)
        inject_target(150)
        full_seq = "".join(seq_chars)

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunks = chunker.chunk(full_seq, chromosome="chr1")
        assert len(chunks) == 2

        # Align both chunks
        results = aligner.align_chunks(chunks=chunks, guide=guide_20)
        assert len(results) == 2

        all_hits = [hit for res in results for hit in res.hits]
        hit_starts = [hit.global_start for hit in all_hits]

        # Verify exact hits detected without any duplicates:
        # Pos 20 -> Chunk 0
        # Pos 85 -> Chunk 0 (starts in Chunk 0 core; not present in Chunk 1)
        # Pos 110 -> Chunk 1 (starts in Chunk 0 halo so filtered out by Chunk 0; reported by Chunk 1)
        # Pos 150 -> Chunk 1
        assert sorted(hit_starts) == [20, 85, 110, 150]
        assert len(all_hits) == 4

        # Verify Chunk 0 reported positions 20 and 85
        c0_starts = [h.global_start for h in results[0].hits]
        assert c0_starts == [20, 85]

        # Verify Chunk 1 reported positions 110 and 150
        c1_starts = [h.global_start for h in results[1].hits]
        assert c1_starts == [110, 150]

    def test_multiple_fasta_records(self, aligner: CPUAligner, guide_20: str):
        # Two distinct records: chr1 and chr2
        rec1 = FastaRecord(identifier="chr1", sequence="T" * 20 + guide_20 + "AGG" + "T" * 10)
        rec2 = FastaRecord(identifier="chr2", sequence="C" * 30 + guide_20 + "TGG" + "C" * 10)

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunks1 = chunker.chunk(rec1)
        chunks2 = chunker.chunk(rec2)

        r1 = aligner.align(chunks1[0], guide=guide_20)
        r2 = aligner.align(chunks2[0], guide=guide_20)

        assert r1.chromosome == "chr1"
        assert r1.hits[0].chromosome == "chr1"
        assert r1.hits[0].local_offset == 20
        assert r1.hits[0].global_start == 20

        assert r2.chromosome == "chr2"
        assert r2.hits[0].chromosome == "chr2"
        assert r2.hits[0].local_offset == 30
        assert r2.hits[0].global_start == 30


class TestCPUAlignerValidationAndTypes:
    """Tests validation of invalid guides, invalid genomes, and NumPy input types."""

    def test_invalid_guide_length_raises_value_error(self, aligner: CPUAligner):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk("A" * 50)[0]

        # Too short
        with pytest.raises(ValueError, match="Invalid CRISPR guide length"):
            aligner.align(chunk, guide="GAGTCCGAGCAGAAGAA")  # 17 nt

        # Too long
        with pytest.raises(ValueError, match="Invalid CRISPR guide length"):
            aligner.align(chunk, guide="GAGTCCGAGCAGAAGAAGAAAGGAAT")  # 26 nt

    def test_invalid_guide_characters_raises_sequence_error(self, aligner: CPUAligner):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk("A" * 50)[0]

        with pytest.raises(InvalidSequenceError):
            aligner.align(chunk, guide="GAGTCCGAGCAGAAGAA123")  # Digits

    def test_invalid_genomic_sequence_raises_sequence_error(self, aligner: CPUAligner, guide_20: str):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        # Sequence with ambiguous 'N'
        chunk = chunker.chunk("ACGTACGTNNNNACGTACGT", chromosome="chr1")[0]

        with pytest.raises(InvalidSequenceError, match="Unsupported or ambiguous nucleotide character"):
            aligner.align(chunk, guide=guide_20)

    def test_invalid_numpy_genomic_array_raises_sequence_error(self, aligner: CPUAligner, guide_20: str):
        # NumPy array with values > 3 and not valid ASCII
        invalid_arr = np.array([0, 1, 2, 4, 255], dtype=np.uint8)
        chunk = ChunkDescriptor(
            chunk_id="chrTest",
            chromosome="chrTest",
            chunk_index=0,
            global_start=0,
            global_end=5,
            actual_length=5,
            core_start=0,
            core_end=5,
            halo_start=5,
            halo_end=5,
            sequence_data=invalid_arr,
        )
        with pytest.raises(InvalidSequenceError):
            aligner.align(chunk, guide=guide_20)

    def test_numpy_unpacked_array_input_compatibility(self, aligner: CPUAligner, guide_20: str):
        # Sequence encoded as unpacked 2-bit uint8 array
        seq_str = "C" * 15 + guide_20 + "AGG" + "A" * 10
        seq_unpacked = TwoBitEncoder.encode(seq_str)

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq_unpacked, chromosome="chrNumPy")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.local_offset == 15
        assert hit.target_sequence[:20] == guide_20
        assert hit.pam_sequence == "AGG"

    def test_numpy_guide_input_compatibility(self, aligner: CPUAligner, guide_20: str):
        guide_unpacked = TwoBitEncoder.encode(guide_20)
        seq_str = "A" * 12 + guide_20 + "TGG" + "T" * 10

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq_str, chromosome="chr1")[0]

        result = aligner.align(chunk, guide=guide_unpacked)
        assert result.hit_count == 1
        assert result.hits[0].local_offset == 12

    def test_encoded_twobit_container_compatibility(self, aligner: CPUAligner, guide_20: str):
        # Sequence packed as EncodedTwoBit
        seq_str = "T" * 25 + guide_20 + "CGG" + "G" * 15
        packed = TwoBitEncoder.encode_packed(seq_str)

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(packed, chromosome="chrPacked")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        assert result.hits[0].local_offset == 25
        assert result.hits[0].target_sequence[:20] == guide_20

    def test_guide_target_23nt_input_compatibility(self, aligner: CPUAligner, guide_23: str):
        # Passing 23 nt GuideTarget directly
        guide_target = GuideTarget.from_string(guide_23)
        seq_str = "G" * 10 + guide_23 + "A" * 10

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq_str, chromosome="chr1")[0]

        result = aligner.align(chunk, guide=guide_target)
        assert result.hit_count == 1
        assert result.hits[0].local_offset == 10


class TestCPUAlignerDeterministicAndBatching:
    """Tests determinism of results and configurable batch sizes."""

    def test_deterministic_result_ordering(self, aligner: CPUAligner, guide_20: str):
        # Sequence with multiple candidate hits with various mismatches
        mut1 = guide_20[:5] + ("C" if guide_20[5] != "C" else "T") + guide_20[6:]
        mut2 = guide_20[:12] + ("A" if guide_20[12] != "A" else "G") + guide_20[13:]
        seq = guide_20 + "TT" + mut1 + "CC" + mut2

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        r1 = aligner.align(chunk, guide=guide_20)
        r2 = aligner.align(chunk, guide=guide_20)

        assert r1.hit_count == r2.hit_count == 3
        for h1, h2 in zip(r1.hits, r2.hits):
            assert h1.local_offset == h2.local_offset
            assert h1.global_start == h2.global_start
            assert h1.mismatch_count == h2.mismatch_count
            assert h1.mismatch_bitmask == h2.mismatch_bitmask
            assert h1.target_sequence == h2.target_sequence

        # Verify strict ascending order of local offsets
        offsets = [h.local_offset for h in r1.hits]
        assert offsets == sorted(offsets)

    def test_batch_size_invariant(self, aligner: CPUAligner, guide_20: str):
        # Generate sequence with multiple matches and run with batch_size=5 vs batch_size=1024
        seq = (guide_20 + "ACGT") * 5
        chunker = GenomicChunker(chunk_size=200, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        cfg_small = AlignmentConfig(batch_size=5, max_mismatches=2)
        cfg_large = AlignmentConfig(batch_size=1024, max_mismatches=2)

        r_small = aligner.align(chunk, guide=guide_20, config=cfg_small)
        r_large = aligner.align(chunk, guide=guide_20, config=cfg_large)

        assert r_small.hit_count == r_large.hit_count == 5
        for h_s, h_l in zip(r_small.hits, r_large.hits):
            assert h_s.local_offset == h_l.local_offset
            assert h_s.global_start == h_l.global_start
            assert h_s.mismatch_count == h_l.mismatch_count
            assert h_s.mismatch_bitmask == h_l.mismatch_bitmask
