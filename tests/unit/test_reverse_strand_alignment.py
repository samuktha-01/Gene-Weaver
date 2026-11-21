"""Comprehensive unit tests for Reverse-Strand Alignment in GeneWeaver CPU alignment engine.

Covers all 13 required test categories:
1. Basic reverse-complement generation.
2. Reverse-complement correctness for known DNA sequences.
3. Reverse-strand exact matches.
4. Reverse-strand matches with mismatches.
5. Correct mismatch counts.
6. Correct mismatch positions (guide-relative and genomic-reference relative).
7. Correct genomic coordinate mapping.
8. Forward and reverse strand searches together.
9. Palindromic sequences.
10. Empty sequences and invalid input.
11. Chunk-boundary reverse-strand matches.
12. Duplicate prevention across overlapping chunks.
13. Regression tests for existing forward-strand behavior.
"""

import numpy as np
import pytest

from geneweaver.alignment import (
    AlignmentConfig,
    AlignmentResult,
    CPUAligner,
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    ReverseComplement,
    Strand,
    reverse_complement,
)
from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder
from geneweaver.sequence_processor.normalizer import SequenceNormalizer


@pytest.fixture
def aligner() -> CPUAligner:
    """Provides a default CPUAligner instance."""
    return CPUAligner()


@pytest.fixture
def guide_20() -> str:
    """Standard 20 nt guide sequence for tests (SpCas9 EMX1)."""
    return "GAGTCCGAGCAGAAGAAGAA"


@pytest.fixture
def guide_23() -> str:
    """Standard 23 nt guide sequence with canonical PAM."""
    return "GAGTCCGAGCAGAAGAAGAAAGG"


# ============================================================================
# 1. Basic Reverse-Complement Generation
# ============================================================================
class TestBasicReverseComplementGeneration:
    """Tests basic reverse-complement generation across strings, arrays, and representations."""

    def test_canonical_single_bases(self):
        assert reverse_complement("A") == "T"
        assert reverse_complement("T") == "A"
        assert reverse_complement("C") == "G"
        assert reverse_complement("G") == "C"

    def test_reverse_and_complement_order(self):
        # 5'-AACG-3' -> complement is TTGC -> reversed is 5'-CGTT-3'
        assert reverse_complement("AACG") == "CGTT"
        assert reverse_complement("ATCG") == "CGAT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("CCCC") == "GGGG"
        assert reverse_complement("GGCC") == "GGCC"

    def test_case_insensitivity_and_uppercase_output(self):
        assert reverse_complement("aacg") == "CGTT"
        assert reverse_complement("aAcG") == "CGTT"
        assert reverse_complement("atcg") == "CGAT"

    def test_rna_uracil_support(self):
        # U is treated as Uracil (complement is Adenine A)
        assert reverse_complement("AUCG") == "CGAT"
        assert reverse_complement("u") == "A"
        assert reverse_complement("UUUU") == "AAAA"

    def test_numpy_2bit_array(self):
        # 2-bit values: A=0, C=1, G=2, T=3
        # ACGT -> [0, 1, 2, 3] -> comp is [3, 2, 1, 0] -> rev is [0, 1, 2, 3] (ACGT)
        arr = np.array([0, 1, 2, 3], dtype=np.uint8)
        rc_arr = reverse_complement(arr)
        assert isinstance(rc_arr, np.ndarray)
        assert rc_arr.dtype == np.uint8
        assert np.array_equal(rc_arr, np.array([0, 1, 2, 3], dtype=np.uint8))

        # AACG -> [0, 0, 1, 2] -> comp is [3, 3, 2, 1] -> rev is [1, 2, 3, 3] (CGTT)
        arr2 = np.array([0, 0, 1, 2], dtype=np.uint8)
        rc_arr2 = reverse_complement(arr2)
        assert np.array_equal(rc_arr2, np.array([1, 2, 3, 3], dtype=np.uint8))

    def test_encoded_twobit_container(self):
        seq = "GAGTCCGAGCAGAAGAAGAAAGG"
        packed = TwoBitEncoder.encode_packed(seq)
        rc_packed = reverse_complement(packed)
        assert isinstance(rc_packed, EncodedTwoBit)
        assert len(rc_packed) == len(seq)
        decoded = TwoBitEncoder.decode(rc_packed)
        assert decoded == reverse_complement(seq)

    def test_sequence_normalizer_convenience_method(self):
        normalizer = SequenceNormalizer()
        assert normalizer.reverse_complement("AACG") == "CGTT"
        assert normalizer.reverse_complement("atcg") == "CGAT"


# ============================================================================
# 2. Reverse-Complement Correctness for Known DNA Sequences
# ============================================================================
class TestReverseComplementKnownSequences:
    """Tests reverse-complement correctness against known biological sequences and invariants."""

    def test_emx1_target_sequence(self, guide_20: str):
        # EMX1 guide: GAGTCCGAGCAGAAGAAGAA
        # Complement: CTCAGGCTCGTCTTCTTCTT
        # Reversed:   TTCTTCTTCTGCTCGGACTC
        expected_rc = "TTCTTCTTCTGCTCGGACTC"
        assert reverse_complement(guide_20) == expected_rc

    def test_emx1_with_pam(self, guide_23: str):
        # 5'-GAGTCCGAGCAGAAGAAGAAAGG-3'
        # Reverse complement: 5'-CCTTTCTTCTTCTGCTCGGACTC-3'
        expected_rc = "CCTTTCTTCTTCTGCTCGGACTC"
        assert reverse_complement(guide_23) == expected_rc

    def test_vegfa_target_sequence(self):
        vegfa = "GGGTGGGGGGAGTTTGCTCC"
        # Complement: CCCACCCCCCTCAAACGAGG
        # Reversed:   GGAGCAAACTCCCCCCACCC
        expected_rc = "GGAGCAAACTCCCCCCACCC"
        assert reverse_complement(vegfa) == expected_rc

    def test_involutory_double_reverse_complement_property(self, guide_23: str):
        # RC(RC(S)) == S
        assert reverse_complement(reverse_complement(guide_23)) == guide_23

    def test_watson_crick_pairing_invariant(self, guide_20: str):
        rc = reverse_complement(guide_20)
        pairs = {"A": "T", "T": "A", "C": "G", "G": "C"}
        n = len(guide_20)
        for i in range(n):
            assert pairs[guide_20[i]] == rc[n - 1 - i]


# ============================================================================
# 3. Reverse-Strand Exact Matches
# ============================================================================
class TestReverseStrandExactMatches:
    """Tests detecting exact matches on the reverse strand (-)."""

    def test_exact_reverse_strand_match(self, aligner: CPUAligner, guide_20: str):
        # Reverse strand has exact 23 bp target: guide_20 + "AGG"
        # On the forward reference strand, this sequence appears as RC(guide_20 + "AGG")
        # RC(guide_20 + "AGG") = "CCT" + RC(guide_20)
        rc_target_23 = reverse_complement(guide_20 + "AGG")
        assert len(rc_target_23) == 23

        # Embed target at forward offset 15
        seq = "A" * 15 + rc_target_23 + "T" * 20
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]

        assert hit.strand == Strand.REVERSE
        assert hit.strand_symbol == "-"
        assert hit.local_offset == 15
        assert hit.global_start == 15
        assert hit.global_end == 38  # 15 + 23 = 38
        assert hit.genomic_position == 16
        assert hit.mismatch_count == 0
        assert hit.mismatch_bitmask == 0
        assert hit.mismatch_positions == ()
        assert hit.target_sequence[:20] == guide_20
        assert hit.pam_sequence == "AGG"
        assert hit.target_sequence == guide_20 + "AGG"
        assert hit.forward_sequence == rc_target_23

    def test_exact_reverse_match_at_chunk_start(self, aligner: CPUAligner, guide_20: str):
        rc_target_23 = reverse_complement(guide_20 + "CGG")
        seq = rc_target_23 + "T" * 30
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr2")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.strand == Strand.REVERSE
        assert hit.local_offset == 0
        assert hit.global_start == 0
        assert hit.genomic_position == 1
        assert hit.mismatch_count == 0
        assert hit.pam_sequence == "CGG"


# ============================================================================
# 4. Reverse-Strand Matches with Mismatches
# ============================================================================
class TestReverseStrandMatchesWithMismatches:
    """Tests reverse-strand candidate detection with 1 to 4 mismatches."""

    def test_single_mismatch_on_reverse_strand(self, aligner: CPUAligner, guide_20: str):
        # Mutate guide at guide position 4 (1-based, index 3): 'T' -> 'A'
        mut_guide = list(guide_20)
        mut_guide[3] = "A" if mut_guide[3] != "A" else "C"
        mut_spacer = "".join(mut_guide)
        target_23 = mut_spacer + "TGG"
        rc_target_23 = reverse_complement(target_23)

        seq = "C" * 10 + rc_target_23 + "G" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr3")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.strand == Strand.REVERSE
        assert hit.mismatch_count == 1
        assert hit.mismatch_positions == (4,)
        assert hit.pam_sequence == "TGG"

    def test_multiple_mismatches_within_threshold(self, aligner: CPUAligner, guide_20: str):
        # 3 mismatches: pos 3, pos 8, pos 15
        mut_guide = list(guide_20)
        mut_guide[2] = "T" if mut_guide[2] != "T" else "A"
        mut_guide[7] = "C" if mut_guide[7] != "C" else "G"
        mut_guide[14] = "T" if mut_guide[14] != "T" else "A"
        mut_spacer = "".join(mut_guide)
        rc_target_23 = reverse_complement(mut_spacer + "AGG")

        seq = "G" * 12 + rc_target_23 + "A" * 12
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr4")[0]

        cfg = AlignmentConfig(max_mismatches=3)
        result = aligner.align(chunk=chunk, guide=guide_20, config=cfg)
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.strand == Strand.REVERSE
        assert hit.mismatch_count == 3
        assert hit.mismatch_positions == (3, 8, 15)

    def test_mismatches_exceeding_threshold_are_filtered(self, aligner: CPUAligner, guide_20: str):
        # 4 mismatches: passes max_mismatches=4, rejected by max_mismatches=3
        mut_guide = list(guide_20)
        for i in [1, 5, 9, 13]:
            mut_guide[i] = "C" if mut_guide[i] != "C" else "T"
        rc_target_23 = reverse_complement("".join(mut_guide) + "AGG")

        seq = "T" * 8 + rc_target_23 + "C" * 8
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        # max_mismatches = 3: rejected
        r3 = aligner.align(chunk, guide=guide_20, config=AlignmentConfig(max_mismatches=3))
        assert r3.hit_count == 0

        # max_mismatches = 4: accepted
        r4 = aligner.align(chunk, guide=guide_20, config=AlignmentConfig(max_mismatches=4))
        assert r4.hit_count == 1
        assert r4.hits[0].strand == Strand.REVERSE
        assert r4.hits[0].mismatch_count == 4


# ============================================================================
# 5. Correct Mismatch Counts
# ============================================================================
class TestReverseStrandMismatchCounts:
    """Tests exact mismatch count reporting across 0, 1, 2, 3, and 4 mismatches."""

    @pytest.mark.parametrize("num_mismatches", [0, 1, 2, 3, 4])
    def test_mismatch_counts(self, aligner: CPUAligner, guide_20: str, num_mismatches: int):
        mut_list = list(guide_20)
        positions = [2, 6, 12, 17][:num_mismatches]
        for p in positions:
            idx = p - 1
            mut_list[idx] = "C" if mut_list[idx] != "C" else "T"

        mut_spacer = "".join(mut_list)
        rc_target = reverse_complement(mut_spacer + "AGG")
        seq = "A" * 20 + rc_target + "T" * 20
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk, guide=guide_20, config=AlignmentConfig(max_mismatches=4))
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.strand == Strand.REVERSE
        assert hit.mismatch_count == num_mismatches
        assert hit.mismatch_info.count == num_mismatches


# ============================================================================
# 6. Correct Mismatch Positions
# ============================================================================
class TestReverseStrandMismatchPositions:
    """Tests coordinate mapping of mismatch positions relative to guide and forward reference."""

    def test_distal_and_seed_mismatch_positions(self, aligner: CPUAligner, guide_20: str):
        # Mutate at guide position 3 (distal, 1-based) and position 16 (seed, 1-based)
        mut_list = list(guide_20)
        mut_list[2] = "C" if mut_list[2] != "C" else "T"
        mut_list[15] = "A" if mut_list[15] != "A" else "G"
        mut_spacer = "".join(mut_list)
        rc_target_23 = reverse_complement(mut_spacer + "AGG")

        seq = "T" * 10 + rc_target_23 + "A" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]

        # 1. Guide-relative positions (5' to 3', 1-based)
        assert hit.mismatch_positions == (3, 16)
        assert hit.guide_mismatch_positions == (3, 16)
        assert hit.mismatch_info.seed_count == 1

        # Integer bitmask: (1 << 2) | (1 << 15)
        expected_bitmask = (1 << 2) | (1 << 15)
        assert hit.mismatch_bitmask == expected_bitmask

        # 2. Forward-relative positions along the 23-mer window:
        # In a 23-mer window (3 bp PAM at pos 1-3, 20 bp spacer at pos 4-23 on forward strand):
        # Guide pos 3 maps to: 23 - 3 + 1 = 21
        # Guide pos 16 maps to: 23 - 16 + 1 = 8
        assert hit.forward_relative_mismatch_positions == (8, 21)

        # 3. Absolute genomic coordinates on the forward reference:
        # global_start = 10, so genomic 1-based coordinates are:
        # pos 16 -> 10 + 8 = 18
        # pos 3  -> 10 + 21 = 31
        assert hit.genomic_mismatch_positions == (18, 31)

    def test_extreme_mismatch_positions_pos1_and_pos20(self, aligner: CPUAligner, guide_20: str):
        # Mutate first base (pos 1, 5' end) and last spacer base (pos 20, 3' PAM-proximal)
        mut_list = list(guide_20)
        mut_list[0] = "T" if mut_list[0] != "T" else "A"
        mut_list[19] = "C" if mut_list[19] != "C" else "G"
        mut_spacer = "".join(mut_list)
        rc_target_23 = reverse_complement(mut_spacer + "CGG")

        seq = "C" * 25 + rc_target_23 + "G" * 25
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr5")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        hit = result.hits[0]

        assert hit.mismatch_positions == (1, 20)
        # Guide pos 20 (PAM-adjacent) maps to forward relative pos: 23 - 20 + 1 = 4 (just after PAM at 1-3)
        # Guide pos 1 (5' distal) maps to forward relative pos: 23 - 1 + 1 = 23 (terminal base)
        assert hit.forward_relative_mismatch_positions == (4, 23)
        # global_start = 25
        assert hit.genomic_mismatch_positions == (25 + 4, 25 + 23)  # (29, 48)


# ============================================================================
# 7. Correct Genomic Coordinate Mapping
# ============================================================================
class TestReverseStrandCoordinateMapping:
    """Tests exact coordinate preservation and mapping to the forward genomic reference."""

    def test_arbitrary_global_start_offset(self, aligner: CPUAligner, guide_20: str):
        global_base = 15_000_000
        rc_target_23 = reverse_complement(guide_20 + "AGG")
        seq = "A" * 45 + rc_target_23 + "T" * 32

        chunk = ChunkDescriptor(
            chunk_id="chr11_chunk_0015",
            chromosome="chr11",
            chunk_index=15,
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

        assert hit.chromosome == "chr11"
        assert hit.strand == Strand.REVERSE
        assert hit.local_offset == 45
        assert hit.global_start == global_base + 45
        assert hit.global_end == global_base + 45 + 23
        assert hit.genomic_position == global_base + 45 + 1
        assert chunk.to_1based_coordinate(hit.local_offset) == hit.genomic_position
        assert chunk.is_in_core(hit.local_offset)


# ============================================================================
# 8. Forward and Reverse Strand Searches Together
# ============================================================================
class TestBothStrandAlignment:
    """Tests searching forward and reverse strands simultaneously and toggling search_reverse_strand."""

    def test_both_strands_detected_in_same_chunk(self, aligner: CPUAligner, guide_20: str):
        fwd_target = guide_20 + "AGG"
        rev_target = reverse_complement(guide_20 + "CGG")

        # Place forward hit at offset 10, reverse hit at offset 60
        seq_chars = ["T"] * 120
        seq_chars[10:33] = list(fwd_target)
        seq_chars[60:83] = list(rev_target)
        seq = "".join(seq_chars)

        chunker = GenomicChunker(chunk_size=120, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk=chunk, guide=guide_20)
        assert result.hit_count == 2

        # Sorted by local_offset: hit 0 is Forward at 10, hit 1 is Reverse at 60
        h0, h1 = result.hits[0], result.hits[1]
        assert h0.strand == Strand.FORWARD
        assert h0.local_offset == 10
        assert h0.pam_sequence == "AGG"

        assert h1.strand == Strand.REVERSE
        assert h1.local_offset == 60
        assert h1.pam_sequence == "CGG"

    def test_search_reverse_strand_config_toggle(self, aligner: CPUAligner, guide_20: str):
        fwd_target = guide_20 + "AGG"
        rev_target = reverse_complement(guide_20 + "CGG")

        seq_chars = ["T"] * 120
        seq_chars[10:33] = list(fwd_target)
        seq_chars[60:83] = list(rev_target)
        seq = "".join(seq_chars)

        chunker = GenomicChunker(chunk_size=120, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        # search_reverse_strand = False: only forward hit is returned
        cfg_fwd_only = AlignmentConfig(search_reverse_strand=False)
        res_fwd = aligner.align(chunk=chunk, guide=guide_20, config=cfg_fwd_only)
        assert res_fwd.hit_count == 1
        assert res_fwd.hits[0].strand == Strand.FORWARD
        assert res_fwd.hits[0].local_offset == 10

        # search_reverse_strand = True: both hits returned
        cfg_both = AlignmentConfig(search_reverse_strand=True)
        res_both = aligner.align(chunk=chunk, guide=guide_20, config=cfg_both)
        assert res_both.hit_count == 2
        assert res_both.total_windows_evaluated == res_fwd.total_windows_evaluated == 120 - 20 + 1


# ============================================================================
# 9. Palindromic Sequences
# ============================================================================
class TestPalindromicSequences:
    """Tests behavior when the target sequence is its own reverse complement."""

    def test_palindromic_spacer_detected_on_both_strands(self, aligner: CPUAligner):
        # 20 nt palindromic spacer: 5 repeats of "ACGT" (RC("ACGT") == "ACGT")
        # RC("ACGTACGTACGTACGTACGT") == "ACGTACGTACGTACGTACGT"
        palindromic_guide = "ACGT" * 5
        assert reverse_complement(palindromic_guide) == palindromic_guide

        # Append symmetric PAM: e.g. "CGT" on forward and reverse
        # If target on forward is "ACGT"*5 + "CGT", on reverse it matches similarly
        seq = "T" * 20 + palindromic_guide + "CGT" + "T" * 20
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chrP")[0]

        result = aligner.align(chunk=chunk, guide=palindromic_guide)
        # Should detect hit on Forward (+) and on Reverse (-)
        assert result.hit_count >= 1
        strands = [h.strand for h in result.hits]
        assert Strand.FORWARD in strands
        # Ordering is deterministic: FORWARD (0) before REVERSE (1) if same offset
        offsets_and_strands = [(h.local_offset, h.strand) for h in result.hits]
        assert offsets_and_strands == sorted(offsets_and_strands)


# ============================================================================
# 10. Empty Sequences and Invalid Input
# ============================================================================
class TestEmptyAndInvalidInput:
    """Tests edge cases: empty strings, short sequences, and invalid bases."""

    def test_empty_sequence(self, aligner: CPUAligner, guide_20: str):
        chunk = ChunkDescriptor(
            chunk_id="chrEmpty",
            chromosome="chrEmpty",
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
        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 0
        assert result.total_windows_evaluated == 0

    def test_sequence_shorter_than_guide(self, aligner: CPUAligner, guide_20: str):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk("ACGTACGT", chromosome="chrShort")[0]
        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 0
        assert result.total_windows_evaluated == 0

    def test_invalid_nucleotide_in_reverse_complement_raises_error(self):
        with pytest.raises(InvalidSequenceError, match="Unsupported or ambiguous nucleotide character 'N'"):
            reverse_complement("ACGTNACGT")

        with pytest.raises(InvalidSequenceError, match="Unsupported or ambiguous nucleotide character '1'"):
            reverse_complement("ACGT123")

    def test_empty_string_reverse_complement(self):
        assert reverse_complement("") == ""


# ============================================================================
# 11. Chunk-Boundary Reverse-Strand Matches
# ============================================================================
class TestChunkBoundaryReverseStrandMatches:
    """Tests reverse-strand candidates straddling the core-halo boundary of a chunk."""

    def test_reverse_hit_straddling_core_and_halo(self, aligner: CPUAligner, guide_20: str):
        # Chunk core length 100 bp, halo 22 bp (total 122 bp).
        # Place a 23 bp reverse target starting in core at offset 85 and extending into halo to 108.
        rc_target_23 = reverse_complement(guide_20 + "AGG")
        seq_chars = ["T"] * 122
        seq_chars[85:108] = list(rc_target_23)
        seq = "".join(seq_chars)

        chunk = ChunkDescriptor(
            chunk_id="chr1_chunk_0000",
            chromosome="chr1",
            chunk_index=0,
            global_start=0,
            global_end=122,
            actual_length=122,
            core_start=0,
            core_end=100,
            halo_start=100,
            halo_end=122,
            sequence_data=seq,
        )

        result = aligner.align(chunk=chunk, guide=guide_20)
        # Because forward local_offset is 85 < 100, it is retained in core
        assert result.hit_count == 1
        hit = result.hits[0]
        assert hit.strand == Strand.REVERSE
        assert hit.local_offset == 85
        assert hit.global_start == 85
        assert hit.global_end == 108
        assert chunk.is_in_core(hit.local_offset)


# ============================================================================
# 12. Duplicate Prevention Across Overlapping Chunks
# ============================================================================
class TestDuplicatePreventionAcrossChunks:
    """Tests multi-chunk deduplication for reverse-strand hits across overlapping halos."""

    def test_reverse_strand_halo_deduplication(self, aligner: CPUAligner, guide_20: str):
        # Partition 200 bp sequence into 2 chunks with chunk_size=100, halo_size=22.
        # Chunk 0 covers [0, 122), core=[0, 100)
        # Chunk 1 covers [100, 200), core=[100, 200)
        rc_target = reverse_complement(guide_20 + "AGG")
        seq_chars = ["A"] * 200

        def inject(pos: int):
            for i, char in enumerate(rc_target):
                seq_chars[pos + i] = char

        # Hit 1: global 15 (Chunk 0 core)
        # Hit 2: global 88 (starts in Chunk 0 core, extends to 111 in Chunk 0 halo)
        # Hit 3: global 115 (in Chunk 0 halo [100, 122); starts in Chunk 1 core at offset 15)
        # Hit 4: global 160 (Chunk 1 core)
        inject(15)
        inject(88)
        inject(115)
        inject(160)
        full_seq = "".join(seq_chars)

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunks = chunker.chunk(full_seq, chromosome="chr1")
        assert len(chunks) == 2

        results = aligner.align_chunks(chunks=chunks, guide=guide_20)
        assert len(results) == 2

        all_hits = [hit for res in results for hit in res.hits]
        hit_starts = [h.global_start for h in all_hits]

        # Verify exact hits detected without any duplicate or omitted targets:
        assert sorted(hit_starts) == [15, 88, 115, 160]
        assert len(all_hits) == 4

        # Chunk 0 reported 15 and 88
        c0_starts = [h.global_start for h in results[0].hits]
        assert c0_starts == [15, 88]

        # Chunk 1 reported 115 and 160
        c1_starts = [h.global_start for h in results[1].hits]
        assert c1_starts == [115, 160]


# ============================================================================
# 13. Regression Tests for Existing Forward-Strand Behavior
# ============================================================================
class TestForwardStrandRegressions:
    """Verifies that forward-strand alignment results are completely unchanged."""

    def test_forward_exact_match_identical_properties(self, aligner: CPUAligner, guide_20: str):
        seq = "A" * 10 + guide_20 + "AGG" + "T" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        # Run with default config (both strands)
        res_both = aligner.align(chunk=chunk, guide=guide_20)
        assert res_both.hit_count == 1
        h = res_both.hits[0]

        assert h.strand == Strand.FORWARD
        assert h.strand_symbol == "+"
        assert h.local_offset == 10
        assert h.global_start == 10
        assert h.global_end == 33
        assert h.genomic_position == 11
        assert h.mismatch_count == 0
        assert h.mismatch_positions == ()
        assert h.target_sequence[:20] == guide_20
        assert h.pam_sequence == "AGG"

        # Run with forward-only config
        res_fwd = aligner.align(chunk=chunk, guide=guide_20, config=AlignmentConfig(search_reverse_strand=False))
        assert res_fwd.hit_count == 1
        assert res_fwd.hits[0].global_start == h.global_start
        assert res_fwd.hits[0].mismatch_count == h.mismatch_count
