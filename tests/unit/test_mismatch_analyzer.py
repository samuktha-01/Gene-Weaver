"""Unit tests for GeneWeaver MismatchAnalyzer.

Tests mismatch extraction, guide-relative coordinate conventions, seed/non-seed
partitioning, bitmask representation, strand handling, and candidate hit integration.
"""

import pytest

from geneweaver.alignment.models import (
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    Strand,
)
from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.scoring.mismatch_analyzer import MismatchAnalysis, MismatchAnalyzer
from geneweaver.scoring.pam_evaluator import (
    PAMValidationResult,
    ValidatedCandidateHit,
)


@pytest.fixture
def default_analyzer() -> MismatchAnalyzer:
    """Returns a MismatchAnalyzer with standard SpCas9 parameters (seed: 11-20, length: 20)."""
    return MismatchAnalyzer(seed_start=11, seed_end=20, spacer_length=20)


@pytest.fixture
def guide_spacer() -> str:
    """Standard 20 nt guide spacer."""
    return "GAGTCCGAGCAGAAGAAGAA"


class TestMismatchAnalysisBasics:
    """Tests basic sequence comparisons and mismatch characterization."""

    def test_exact_sequence_match(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Exact match should produce 0 mismatches, empty positions, and bitmask 0."""
        analysis = default_analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=guide_spacer)
        assert analysis.total_mismatches == 0
        assert analysis.seed_mismatches == 0
        assert analysis.non_seed_mismatches == 0
        assert analysis.mismatch_positions == ()
        assert analysis.seed_positions == ()
        assert analysis.non_seed_positions == ()
        assert analysis.bitmask == 0
        assert analysis.mismatch_details == ()
        assert analysis.is_exact_match is True
        assert analysis.has_seed_mismatches is False

    def test_one_mismatch_distal(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Single mismatch at distal position 1 (5' end)."""
        # Mutate position 1: G -> A
        target = "A" + guide_spacer[1:]
        analysis = default_analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=target)
        assert analysis.total_mismatches == 1
        assert analysis.mismatch_positions == (1,)
        assert analysis.non_seed_mismatches == 1
        assert analysis.seed_mismatches == 0
        assert analysis.non_seed_positions == (1,)
        assert analysis.seed_positions == ()
        assert analysis.bitmask == 1  # 1 << 0
        assert analysis.mismatch_details == ((1, "G", "A"),)
        assert analysis.is_exact_match is False
        assert analysis.has_seed_mismatches is False

    def test_one_mismatch_pam_proximal_pos20(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Single mismatch at position 20 (3' PAM-proximal seed position)."""
        # Mutate position 20: A -> C
        target = guide_spacer[:19] + "C"
        analysis = default_analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=target)
        assert analysis.total_mismatches == 1
        assert analysis.mismatch_positions == (20,)
        assert analysis.seed_mismatches == 1
        assert analysis.non_seed_mismatches == 0
        assert analysis.seed_positions == (20,)
        assert analysis.non_seed_positions == ()
        assert analysis.bitmask == (1 << 19)
        assert analysis.mismatch_details == ((20, "A", "C"),)
        assert analysis.has_seed_mismatches is True

    def test_multiple_mismatches(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Multiple mismatches across the protospacer."""
        # Mutate position 3 (G -> T) and position 15 (G -> C)
        target_list = list(guide_spacer)
        target_list[2] = "T"  # pos 3 (non-seed)
        target_list[14] = "C"  # pos 15 (seed)
        target = "".join(target_list)

        analysis = default_analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=target)
        assert analysis.total_mismatches == 2
        assert analysis.mismatch_positions == (3, 15)
        assert analysis.non_seed_mismatches == 1
        assert analysis.seed_mismatches == 1
        assert analysis.non_seed_positions == (3,)
        assert analysis.seed_positions == (15,)
        assert analysis.bitmask == (1 << 2) | (1 << 14)
        assert analysis.mismatch_details == ((3, "G", "T"), (15, "G", "C"))

    def test_multiple_seed_mismatches(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Multiple mismatches isolated within the seed region (positions 11-20)."""
        target_list = list(guide_spacer)
        target_list[10] = "C"  # pos 11 (A -> C)
        target_list[13] = "T"  # pos 14 (A -> T)
        target_list[19] = "G"  # pos 20 (A -> G)
        target = "".join(target_list)

        analysis = default_analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=target)
        assert analysis.total_mismatches == 3
        assert analysis.seed_mismatches == 3
        assert analysis.non_seed_mismatches == 0
        assert analysis.seed_positions == (11, 14, 20)
        assert analysis.non_seed_positions == ()

    def test_non_seed_region_mismatches(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Multiple mismatches isolated within the non-seed region (positions 1-10)."""
        target_list = list(guide_spacer)
        target_list[0] = "T"  # pos 1 (G -> T)
        target_list[4] = "A"  # pos 5 (C -> A)
        target_list[9] = "A"  # pos 10 (C -> A)
        target = "".join(target_list)

        analysis = default_analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=target)
        assert analysis.total_mismatches == 3
        assert analysis.non_seed_mismatches == 3
        assert analysis.seed_mismatches == 0
        assert analysis.non_seed_positions == (1, 5, 10)
        assert analysis.seed_positions == ()

    def test_mixed_seed_and_non_seed_mismatches(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Mixed mismatches across both seed and non-seed regions."""
        target_list = list(guide_spacer)
        target_list[1] = "C"  # pos 2 (A -> C)
        target_list[6] = "T"  # pos 7 (G -> T)
        target_list[11] = "T"  # pos 12 (G -> T)
        target_list[17] = "C"  # pos 18 (G -> C)
        target = "".join(target_list)

        analysis = default_analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=target)
        assert analysis.total_mismatches == 4
        assert analysis.non_seed_mismatches == 2
        assert analysis.seed_mismatches == 2
        assert analysis.non_seed_positions == (2, 7)
        assert analysis.seed_positions == (12, 18)
        assert analysis.mismatch_positions == (2, 7, 12, 18)


class TestConfigurableSeedRegion:
    """Tests custom seed region boundary configurations."""

    def test_custom_seed_boundaries(self, guide_spacer: str):
        """Custom seed region defined as positions 15-20 (length 6)."""
        analyzer = MismatchAnalyzer(seed_start=15, seed_end=20, spacer_length=20)
        target_list = list(guide_spacer)
        target_list[11] = "T"  # pos 12 (formerly seed, now non-seed)
        target_list[16] = "C"  # pos 17 (seed)
        target = "".join(target_list)

        analysis = analyzer.analyze_sequences(guide_seq=guide_spacer, target_seq=target)
        assert analysis.total_mismatches == 2
        assert analysis.seed_mismatches == 1
        assert analysis.seed_positions == (17,)
        assert analysis.non_seed_mismatches == 1
        assert analysis.non_seed_positions == (12,)

    def test_invalid_seed_boundaries_raise_value_error(self):
        """Invalid seed coordinates must raise ValueError."""
        with pytest.raises(ValueError, match="seed_start"):
            MismatchAnalyzer(seed_start=0, seed_end=20, spacer_length=20)
        with pytest.raises(ValueError, match="seed_end"):
            MismatchAnalyzer(seed_start=15, seed_end=10, spacer_length=20)
        with pytest.raises(ValueError, match="spacer_length"):
            MismatchAnalyzer(seed_start=11, seed_end=25, spacer_length=20)
        with pytest.raises(ValueError, match="spacer_length must be positive"):
            MismatchAnalyzer(seed_start=1, seed_end=5, spacer_length=0)


class TestStrandAndCandidateIntegration:
    """Tests integration with Forward and Reverse strand candidate models."""

    def test_forward_strand_candidate(self, default_analyzer: MismatchAnalyzer, guide_spacer: str):
        """Analyzes a forward strand ValidatedCandidateHit."""
        target_seq = guide_spacer[:18] + "TT" + "CGG"  # pos 19, 20 mutated + PAM
        raw_hit = RawCandidateHit(
            chunk_id="chunk_0",
            chromosome="NC_000001.11",
            local_offset=100,
            global_start=5000,
            global_end=5023,
            genomic_position=5001,
            strand=Strand.FORWARD,
            mismatch_info=MismatchInfo.from_bitmask((1 << 18) | (1 << 19)),
            pam_sequence="CGG",
            target_sequence=target_seq,
        )
        val_hit = ValidatedCandidateHit(
            candidate=raw_hit,
            pam_result=PAMValidationResult(is_valid=True, is_canonical=True, matched_motif="NGG"),
            chromosome="NC_000001.11",
            strand=Strand.FORWARD,
            protospacer_start=5000,
            protospacer_end=5020,
            pam_start=5020,
            pam_end=5023,
            pam_sequence="CGG",
            target_sequence=target_seq,
        )

        analysis = default_analyzer.analyze_candidate(candidate=val_hit, guide=guide_spacer)
        assert analysis.total_mismatches == 2
        assert analysis.mismatch_positions == (19, 20)
        assert analysis.seed_mismatches == 2
        assert analysis.non_seed_mismatches == 0
        assert analysis.bitmask == (1 << 18) | (1 << 19)

    def test_reverse_strand_candidate_uses_guide_relative_positions(
        self, default_analyzer: MismatchAnalyzer, guide_spacer: str
    ):
        """Reverse strand candidates must use guide-relative positions (5' to 3' of guide)."""
        # Target sequence is stored 5' to 3' in matched orientation
        # Mutate position 1 (distal): G -> A
        target_seq = "A" + guide_spacer[1:] + "CGG"
        raw_hit = RawCandidateHit(
            chunk_id="chunk_0",
            chromosome="NC_000001.11",
            local_offset=200,
            global_start=8000,
            global_end=8023,
            genomic_position=8001,
            strand=Strand.REVERSE,
            mismatch_info=MismatchInfo.from_bitmask(1 << 0),
            pam_sequence="CGG",
            target_sequence=target_seq,
        )
        val_hit = ValidatedCandidateHit(
            candidate=raw_hit,
            pam_result=PAMValidationResult(is_valid=True, is_canonical=True, matched_motif="NGG"),
            chromosome="NC_000001.11",
            strand=Strand.REVERSE,
            protospacer_start=8003,
            protospacer_end=8023,
            pam_start=8000,
            pam_end=8003,
            pam_sequence="CGG",
            target_sequence=target_seq,
        )

        analysis = default_analyzer.analyze_candidate(candidate=val_hit, guide=guide_spacer)
        assert analysis.total_mismatches == 1
        assert analysis.mismatch_positions == (1,)
        assert analysis.non_seed_mismatches == 1
        assert analysis.seed_mismatches == 0
        assert analysis.mismatch_details == ((1, "G", "A"),)

    def test_metadata_consistency_when_target_sequence_omitted(
        self, default_analyzer: MismatchAnalyzer, guide_spacer: str
    ):
        """When target sequence string is empty, reuses existing MismatchInfo metadata safely."""
        bitmask = (1 << 2) | (1 << 15)  # pos 3 (non-seed) and pos 16 (seed)
        raw_hit = RawCandidateHit(
            chunk_id="chunk_0",
            chromosome="NC_000001.11",
            local_offset=50,
            global_start=1000,
            global_end=1023,
            genomic_position=1001,
            strand=Strand.FORWARD,
            mismatch_info=MismatchInfo.from_bitmask(bitmask),
            pam_sequence="AGG",
            target_sequence="",  # Omitted target sequence string
        )

        analysis = default_analyzer.analyze_candidate(candidate=raw_hit, guide=guide_spacer)
        assert analysis.total_mismatches == 2
        assert analysis.mismatch_positions == (3, 16)
        assert analysis.non_seed_mismatches == 1
        assert analysis.seed_mismatches == 1
        assert analysis.non_seed_positions == (3,)
        assert analysis.seed_positions == (16,)
        assert analysis.bitmask == bitmask
        assert analysis.mismatch_details == ()  # Base details empty

    def test_analyze_batch_and_empty_collection(
        self, default_analyzer: MismatchAnalyzer, guide_spacer: str
    ):
        """Batch analysis handles empty collection and multiple candidates."""
        assert default_analyzer.analyze_batch([], guide=guide_spacer) == []

        cand1 = RawCandidateHit(
            chunk_id="c1", chromosome="chr1", local_offset=0, global_start=0, global_end=23,
            genomic_position=1, strand=Strand.FORWARD, mismatch_info=MismatchInfo.from_bitmask(0),
            pam_sequence="AGG", target_sequence=guide_spacer + "AGG"
        )
        cand2 = RawCandidateHit(
            chunk_id="c1", chromosome="chr1", local_offset=50, global_start=50, global_end=73,
            genomic_position=51, strand=Strand.FORWARD, mismatch_info=MismatchInfo.from_bitmask(1),
            pam_sequence="AGG", target_sequence="A" + guide_spacer[1:] + "AGG"
        )

        batch_result = default_analyzer.analyze_batch([cand1, cand2], guide=guide_spacer)
        assert len(batch_result) == 2
        assert batch_result[0].total_mismatches == 0
        assert batch_result[1].total_mismatches == 1


class TestValidationAndErrorHandling:
    """Tests sequence validation and length constraints."""

    def test_invalid_characters_raise_invalid_sequence_error(self, default_analyzer: MismatchAnalyzer):
        """Non-canonical nucleotide characters must raise InvalidSequenceError."""
        with pytest.raises(InvalidSequenceError):
            default_analyzer.analyze_sequences("GAGTCCGAGCAGAAGAAGAN", "GAGTCCGAGCAGAAGAAGAA")
        with pytest.raises(InvalidSequenceError):
            default_analyzer.analyze_sequences("GAGTCCGAGCAGAAGAAGAA", "GAGTCCGAGCAGAA1AAGAA")

    def test_sequence_shorter_than_spacer_length_raises_value_error(
        self, default_analyzer: MismatchAnalyzer
    ):
        """Sequences shorter than spacer_length (20 nt) must raise ValueError."""
        with pytest.raises(ValueError, match="shorter than spacer_length"):
            default_analyzer.analyze_sequences("GAGTCC", "GAGTCCGAGCAGAAGAAGAA")
        with pytest.raises(ValueError, match="shorter than spacer_length"):
            default_analyzer.analyze_sequences("GAGTCCGAGCAGAAGAAGAA", "GAGTCC")

    def test_empty_sequence_raises_value_error(self, default_analyzer: MismatchAnalyzer):
        """Empty sequence string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            default_analyzer.analyze_sequences("", "GAGTCCGAGCAGAAGAAGAA")

    def test_rna_and_lowercase_normalization(self, default_analyzer: MismatchAnalyzer):
        """RNA 'U' characters and lowercase strings are normalized without error."""
        guide_rna = "gaguccgagcagaagaagaa"
        target_dna = "GAGTCCGAGCAGAAGAAGAA"
        analysis = default_analyzer.analyze_sequences(guide_seq=guide_rna, target_seq=target_dna)
        assert analysis.total_mismatches == 0
        assert analysis.is_exact_match is True
