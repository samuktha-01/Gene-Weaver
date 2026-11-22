"""Comprehensive unit tests for GeneWeaver PAM Recognition and Validation Engine.

Covers:
1. PAM Pattern Tests:
   - Canonical SpCas9 motifs (AGG, TGG, CGG, GGG).
   - Incompatible PAM sequences (AGA, ATG, CAG, TGC, etc.).
   - Invalid pattern definitions and error handling.
   - Case-insensitive matching and lowercase input.
   - IUPAC pattern matching (canonical NGG, alternate NAG, NGA, custom SaCas9/Cas12a).
2. Strand-Aware Tests:
   - Forward-strand PAM detection and downstream adjacency.
   - Reverse-strand PAM detection and 5'->3' reverse-complement relationship.
   - Rejection of naive/incorrect PAM orientation on reverse strand.
   - Strict adjacency verification.
3. PAM Coordinate Tests:
   - Forward-strand PAM and protospacer coordinates (0-based and 1-based).
   - Reverse-strand PAM and protospacer coordinates (0-based and 1-based).
   - Boundary checks and interval lengths.
   - Arbitrary chromosome coordinates.
   - Chunk-boundary PAM detection.
   - Deduplication across overlapping chunk halos.
4. Integration Tests:
   - Exact alignment with compatible PAM.
   - Exact alignment with incompatible PAM.
   - Mismatched alignment with compatible PAM.
   - Simultaneous forward and reverse candidate sites.
   - Multiple candidate sites with filtering (filter_incompatible).
   - Missing or out-of-bounds PAM regions.
"""

import numpy as np
import pytest

from geneweaver.alignment import (
    AlignmentConfig,
    CPUAligner,
    GuideTarget,
    RawCandidateHit,
    Strand,
    reverse_complement,
)
from geneweaver.scoring.pam_evaluator import (
    IUPACMatcher,
    NucleaseConfig,
    PAMLocation,
    PAMPattern,
    PAMValidationResult,
    PAMValidator,
    SPCAS9,
    ValidatedCandidateHit,
)
from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker


@pytest.fixture
def validator() -> PAMValidator:
    """Provides a default SpCas9 PAMValidator instance."""
    return PAMValidator()


@pytest.fixture
def aligner() -> CPUAligner:
    """Provides a default CPUAligner instance."""
    return CPUAligner()


@pytest.fixture
def guide_20() -> str:
    """Standard 20 nt guide sequence for SpCas9 EMX1."""
    return "GAGTCCGAGCAGAAGAAGAA"


# ============================================================================
# 1. PAM Pattern Tests
# ============================================================================
class TestPAMPatternRecognition:
    """Tests IUPAC pattern compilation, canonical SpCas9 matching, and validation."""

    @pytest.mark.parametrize("pam", ["AGG", "TGG", "CGG", "GGG"])
    def test_canonical_spcas9_pams_are_valid(self, validator: PAMValidator, pam: str):
        result = validator.validate_pam_sequence(pam)
        assert isinstance(result, PAMValidationResult)
        assert result.is_valid is True
        assert result.is_canonical is True
        assert result.matched_motif == "NGG"
        assert result.cleavage_multiplier == 1.0
        assert result.pam_sequence == pam
        assert bool(result) is True

    @pytest.mark.parametrize("pam", ["AGA", "ATG", "CAG", "TGC", "TTT", "AAA", "CCC", "GAC"])
    def test_incompatible_pams_are_rejected(self, validator: PAMValidator, pam: str):
        # Non-canonical, non-alternate PAMs must be invalid
        result = validator.validate_pam_sequence(pam)
        assert result.is_valid is False
        assert result.is_canonical is False
        assert result.matched_motif is None
        assert result.cleavage_multiplier == 0.0
        assert bool(result) is False

    @pytest.mark.parametrize("motif", ["", "   ", "NG1", "123", "NG-", "NG!", "XYZ"])
    def test_invalid_pattern_definitions_raise_value_error(self, motif: str):
        with pytest.raises(ValueError, match="PAM pattern|Invalid IUPAC character"):
            IUPACMatcher.validate_pattern(motif)

        with pytest.raises(ValueError):
            PAMPattern(motif=motif)

    def test_negative_cleavage_multiplier_raises_value_error(self):
        with pytest.raises(ValueError, match="cleavage_multiplier cannot be negative"):
            PAMPattern(motif="NGG", cleavage_multiplier=-0.1)

    @pytest.mark.parametrize(
        "lower_pam,expected_upper",
        [("agg", "AGG"), ("tgg", "TGG"), ("cgg", "CGG"), ("ggg", "GGG")],
    )
    def test_lowercase_input_normalized_and_recognized(
        self, validator: PAMValidator, lower_pam: str, expected_upper: str
    ):
        result = validator.validate_pam_sequence(lower_pam)
        assert result.is_valid is True
        assert result.is_canonical is True
        assert result.pam_sequence == expected_upper

    def test_lowercase_registered_pattern(self):
        v = PAMValidator(NucleaseConfig(canonical_motif="ngg"))
        res = v.validate_pam_sequence("agg")
        assert res.is_valid is True
        assert res.is_canonical is True

    def test_iupac_alternate_motifs_spcas9(self):
        # Configure SpCas9 with alternate motifs: NAG (0.25) and NGA (0.10)
        cfg = NucleaseConfig(canonical_motif="NGG", alternate_motifs=("NAG", "NGA"))
        validator = PAMValidator(cfg)
        res_nag = validator.validate_pam_sequence("AAG")
        assert res_nag.is_valid is True
        assert res_nag.is_canonical is False
        assert res_nag.matched_motif == "NAG"
        assert res_nag.cleavage_multiplier == 0.25

        res_nga = validator.validate_pam_sequence("TGA")
        assert res_nga.is_valid is True
        assert res_nga.is_canonical is False
        assert res_nga.matched_motif == "NGA"
        assert res_nga.cleavage_multiplier == 0.10

        # When canonical_only is requested, alternate motifs are rejected
        res_nag_canonical_only = validator.validate_pam_sequence("AAG", canonical_only=True)
        assert res_nag_canonical_only.is_valid is False

    def test_custom_nuclease_configuration_sacas9(self):
        # SaCas9 has canonical PAM: NNGRRT
        sacas9_config = NucleaseConfig(
            name="SaCas9",
            canonical_motif="NNGRRT",
            alternate_motifs=(),
            protospacer_length=21,
        )
        validator_sa = PAMValidator(sacas9_config)

        # NNGRRT: N=any, N=any, G, R=A/G, R=A/G, T
        assert validator_sa.validate_pam_sequence("CCGAAT").is_valid is True
        assert validator_sa.validate_pam_sequence("TTGGGT").is_valid is True
        assert validator_sa.validate_pam_sequence("AAGAGT").is_valid is True
        # Fails last T
        assert validator_sa.validate_pam_sequence("CCGAAC").is_valid is False
        # Fails R (C is not A or G)
        assert validator_sa.validate_pam_sequence("CCGACT").is_valid is False

    def test_custom_nuclease_configuration_cas12a(self):
        # Cas12a (Cpf1): 5' TTTV PAM
        cas12a_config = NucleaseConfig(
            name="Cas12a",
            canonical_motif="TTTV",
            alternate_motifs=(),
            pam_location=PAMLocation.FIVE_PRIME,
            protospacer_length=23,
        )
        validator_cas12 = PAMValidator(cas12a_config)

        # V = A, C, G (not T)
        assert validator_cas12.validate_pam_sequence("TTTA").is_valid is True
        assert validator_cas12.validate_pam_sequence("TTTC").is_valid is True
        assert validator_cas12.validate_pam_sequence("TTTG").is_valid is True
        assert validator_cas12.validate_pam_sequence("TTTT").is_valid is False


# ============================================================================
# 2. Strand-Aware PAM Detection Tests
# ============================================================================
class TestStrandAwarePAMDetection:
    """Tests strand orientation, reverse-complement PAM interpretation, and adjacency."""

    def test_forward_strand_pam_detection(self, aligner: CPUAligner, validator: PAMValidator, guide_20: str):
        # Forward candidate with canonical PAM 'AGG'
        seq = "T" * 10 + guide_20 + "AGG" + "T" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        candidate = result.hits[0]

        val_hit = validator.validate_candidate(candidate)
        assert val_hit.strand == Strand.FORWARD
        assert val_hit.pam_sequence == "AGG"
        assert val_hit.is_valid_pam is True
        assert val_hit.is_canonical_pam is True
        assert val_hit.matched_motif == "NGG"
        # Adjacency: PAM immediately downstream of protospacer
        assert val_hit.pam_start == val_hit.protospacer_end

    def test_reverse_strand_pam_detection(self, aligner: CPUAligner, validator: PAMValidator, guide_20: str):
        # Target on reverse strand: guide_20 + "TGG" (canonical PAM)
        # On the forward reference strand, this sequence appears as:
        # RC(guide_20 + "TGG") = RC("TGG") + RC(guide_20) = "CCA" + RC(guide_20)
        rc_pam = reverse_complement("TGG")  # "CCA"
        rc_spacer = reverse_complement(guide_20)
        fwd_target = rc_pam + rc_spacer
        assert len(fwd_target) == 23

        seq = "A" * 15 + fwd_target + "C" * 15
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr2")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        candidate = result.hits[0]

        val_hit = validator.validate_candidate(candidate)
        assert val_hit.strand == Strand.REVERSE
        # Biological PAM in 5' to 3' orientation of target strand is TGG
        assert val_hit.pam_sequence == "TGG"
        assert val_hit.is_valid_pam is True
        assert val_hit.is_canonical_pam is True
        # Forward reference sequence at PAM locus is 'CCA'
        assert val_hit.forward_sequence[:3] == "CCA"
        # Adjacency: On reverse strand, PAM is adjacent to 3' end of protospacer,
        # which corresponds to the 5' side (lower coordinate) on forward reference
        assert val_hit.pam_end == val_hit.protospacer_start

    def test_reverse_strand_pam_rejection_of_incompatible_motif(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        # Target on reverse strand with incompatible PAM 'AGA'
        # Forward strand sequence: RC("AGA") + RC(guide_20) = "TCT" + RC(guide_20)
        rc_target = reverse_complement(guide_20 + "AGA")
        seq = "G" * 12 + rc_target + "A" * 12
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr3")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        candidate = result.hits[0]

        val_hit = validator.validate_candidate(candidate)
        assert val_hit.strand == Strand.REVERSE
        assert val_hit.pam_sequence == "AGA"
        assert val_hit.is_valid_pam is False
        assert val_hit.is_canonical_pam is False

    def test_reverse_strand_does_not_naively_search_ngg_on_forward_strand(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        # If someone naively looked for 'TGG' downstream on the forward strand instead of
        # the reverse-complement PAM on the correct side:
        # Forward strand: "TTT" + RC(guide_20) + "TGG"
        # True reverse strand target: RC("TTT" + RC(guide_20)) = guide_20 + "AAA" (PAM is AAA, invalid!)
        # Even though "TGG" is present downstream on the forward strand at the 3' side of the forward window,
        # for a reverse-strand target, that is the WRONG SIDE (5' end of reverse target)!
        rc_spacer = reverse_complement(guide_20)
        seq = "C" * 10 + "TTT" + rc_spacer + "TGG" + "C" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr4")[0]

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        candidate = result.hits[0]

        val_hit = validator.validate_candidate(candidate)
        assert val_hit.strand == Strand.REVERSE
        # Candidate's true PAM is RC("TTT") = "AAA"
        assert val_hit.pam_sequence == "AAA"
        # Must be rejected because 'AAA' is not a valid PAM
        assert val_hit.is_valid_pam is False

    def test_extract_pam_from_chunk_forward_and_reverse(self, validator: PAMValidator, guide_20: str):
        # Build chunk with known forward and reverse targets
        fwd_target = guide_20 + "AGG"
        rev_target = reverse_complement(guide_20 + "CGG")  # "CCG" + RC(guide_20)

        seq = "A" * 10 + fwd_target + "T" * 20 + rev_target + "G" * 10
        chunker = GenomicChunker(chunk_size=150, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chrExtract")[0]

        # 1. Forward hit: protospacer at [10, 30)
        pam_fwd, bounds_fwd = validator.extract_pam_from_chunk(
            chunk=chunk,
            protospacer_start=10,
            protospacer_end=30,
            strand=Strand.FORWARD,
            pam_length=3,
        )
        assert pam_fwd == "AGG"
        assert bounds_fwd == (30, 33)

        # 2. Reverse hit: rev_target starts at 53 (len 23): "CCG" at [53, 56), RC(guide_20) at [56, 76)
        pam_rev, bounds_rev = validator.extract_pam_from_chunk(
            chunk=chunk,
            protospacer_start=56,
            protospacer_end=76,
            strand=Strand.REVERSE,
            pam_length=3,
        )
        # Reverse complement of 'CCG' is 'CGG'
        assert pam_rev == "CGG"
        assert bounds_rev == (53, 56)


# ============================================================================
# 3. PAM Coordinate Mapping Tests
# ============================================================================
class TestPAMCoordinateMapping:
    """Tests exact coordinate conventions for protospacer and PAM boundaries."""

    def test_forward_pam_coordinates_with_arbitrary_global_start(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        global_base = 25_000_000
        fwd_target = guide_20 + "AGG"
        seq = "T" * 50 + fwd_target + "A" * 27

        chunk = ChunkDescriptor(
            chunk_id="chr7_chunk_0025",
            chromosome="chr7",
            chunk_index=25,
            global_start=global_base,
            global_end=global_base + len(seq),
            actual_length=len(seq),
            core_start=global_base,
            core_end=global_base + len(seq),
            halo_start=global_base + len(seq),
            halo_end=global_base + len(seq),
            sequence_data=seq,
        )

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        candidate = result.hits[0]

        val_hit = validator.validate_candidate(candidate)
        assert val_hit.chromosome == "chr7"
        assert val_hit.strand == Strand.FORWARD

        # 0-based half-open coordinates:
        # Protospacer: [global_base + 50, global_base + 70)
        # PAM:         [global_base + 70, global_base + 73)
        assert val_hit.protospacer_start == global_base + 50
        assert val_hit.protospacer_end == global_base + 70
        assert val_hit.pam_start == global_base + 70
        assert val_hit.pam_end == global_base + 73

        # 1-based closed coordinates:
        # Protospacer: [global_base + 51, global_base + 70]
        # PAM:         [global_base + 71, global_base + 73]
        assert val_hit.protospacer_start_1based == global_base + 51
        assert val_hit.protospacer_end_1based == global_base + 70
        assert val_hit.pam_start_1based == global_base + 71
        assert val_hit.pam_end_1based == global_base + 73

        # Length invariants
        assert val_hit.protospacer_end - val_hit.protospacer_start == 20
        assert val_hit.pam_end - val_hit.pam_start == 3
        assert val_hit.pam_start == val_hit.protospacer_end

    def test_reverse_pam_coordinates_with_arbitrary_global_start(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        global_base = 40_000_000
        rc_target = reverse_complement(guide_20 + "GGG")  # "CCC" + RC(guide_20)
        seq = "C" * 35 + rc_target + "G" * 42

        chunk = ChunkDescriptor(
            chunk_id="chrX_chunk_0040",
            chromosome="chrX",
            chunk_index=40,
            global_start=global_base,
            global_end=global_base + len(seq),
            actual_length=len(seq),
            core_start=global_base,
            core_end=global_base + len(seq),
            halo_start=global_base + len(seq),
            halo_end=global_base + len(seq),
            sequence_data=seq,
        )

        result = aligner.align(chunk, guide=guide_20)
        assert result.hit_count == 1
        candidate = result.hits[0]

        val_hit = validator.validate_candidate(candidate)
        assert val_hit.chromosome == "chrX"
        assert val_hit.strand == Strand.REVERSE

        # On forward reference:
        # Target spans [global_base + 35, global_base + 58) (len 23)
        # PAM is at the lower coordinate: [global_base + 35, global_base + 38)
        # Protospacer is at:              [global_base + 38, global_base + 58)
        assert val_hit.pam_start == global_base + 35
        assert val_hit.pam_end == global_base + 38
        assert val_hit.protospacer_start == global_base + 38
        assert val_hit.protospacer_end == global_base + 58

        # 1-based closed coordinates:
        assert val_hit.pam_start_1based == global_base + 36
        assert val_hit.pam_end_1based == global_base + 38
        assert val_hit.protospacer_start_1based == global_base + 39
        assert val_hit.protospacer_end_1based == global_base + 58

        # Length invariants
        assert val_hit.pam_end - val_hit.pam_start == 3
        assert val_hit.protospacer_end - val_hit.protospacer_start == 20
        assert val_hit.pam_end == val_hit.protospacer_start
        assert val_hit.pam_sequence == "GGG"

    def test_chunk_boundary_and_halo_deduplication(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        # 200 bp sequence partitioned into 2 chunks with chunk_size=100, halo_size=22
        # Place forward hit at 85 (straddles core-halo boundary of Chunk 0)
        # Place reverse hit at 110 (Chunk 1 core, Chunk 0 halo)
        fwd_target = guide_20 + "AGG"
        rev_target = reverse_complement(guide_20 + "CGG")

        seq_chars = ["A"] * 200
        for i, c in enumerate(fwd_target):
            seq_chars[85 + i] = c
        for i, c in enumerate(rev_target):
            seq_chars[110 + i] = c
        seq = "".join(seq_chars)

        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunks = chunker.chunk(seq, chromosome="chrBound")
        assert len(chunks) == 2

        val_hits = aligner.align_chunks_and_validate(
            chunks=chunks,
            guide=guide_20,
            validator=validator,
            filter_incompatible=True,
        )

        assert len(val_hits) == 2
        starts = [h.global_start for h in val_hits]
        assert sorted(starts) == [85, 110]
        # Pos 85 is Forward hit
        h85 = [h for h in val_hits if h.global_start == 85][0]
        assert h85.strand == Strand.FORWARD
        assert h85.pam_sequence == "AGG"

        # Pos 110 is Reverse hit
        h110 = [h for h in val_hits if h.global_start == 110][0]
        assert h110.strand == Strand.REVERSE
        assert h110.pam_sequence == "CGG"


# ============================================================================
# 4. Integration Tests With CPU Alignment
# ============================================================================
class TestCPUAlignmentIntegration:
    """Tests integrated PAM validation workflow with CPUAligner."""

    def test_exact_alignment_with_compatible_pam(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        seq = "C" * 10 + guide_20 + "AGG" + "T" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        val_hits = aligner.align_and_validate(chunk=chunk, guide=guide_20, validator=validator)
        assert len(val_hits) == 1
        h = val_hits[0]
        assert h.is_valid_pam is True
        assert h.is_canonical_pam is True
        assert h.mismatch_count == 0
        assert h.pam_sequence == "AGG"

    def test_exact_alignment_with_incompatible_pam(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        # Exact spacer match followed by incompatible PAM 'ATC'
        seq = "C" * 10 + guide_20 + "ATC" + "T" * 10
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        # 1. Without filtering: candidate is retained but marked invalid
        val_hits_all = aligner.align_and_validate(
            chunk=chunk, guide=guide_20, validator=validator, filter_incompatible=False
        )
        assert len(val_hits_all) == 1
        assert val_hits_all[0].is_valid_pam is False
        assert val_hits_all[0].pam_sequence == "ATC"
        assert val_hits_all[0].mismatch_count == 0

        # 2. With filter_incompatible: candidate is excluded
        val_hits_filtered = aligner.align_and_validate(
            chunk=chunk, guide=guide_20, validator=validator, filter_incompatible=True
        )
        assert len(val_hits_filtered) == 0

    def test_mismatches_preserved_with_compatible_pam(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        # 2 mismatches at guide positions 4 and 14, followed by canonical PAM 'CGG'
        mut = list(guide_20)
        mut[3] = "C" if mut[3] != "C" else "T"
        mut[13] = "A" if mut[13] != "A" else "G"
        mut_spacer = "".join(mut)

        seq = "A" * 15 + mut_spacer + "CGG" + "T" * 15
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chr1")[0]

        val_hits = aligner.align_and_validate(
            chunk=chunk,
            guide=guide_20,
            config=AlignmentConfig(max_mismatches=3),
            validator=validator,
        )
        assert len(val_hits) == 1
        h = val_hits[0]
        assert h.is_valid_pam is True
        assert h.mismatch_count == 2
        assert h.mismatch_positions == (4, 14)
        assert h.pam_sequence == "CGG"

    def test_multiple_candidates_filtering(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        # 3 candidate sites:
        # Site 1: Forward with canonical PAM 'AGG' (valid)
        # Site 2: Forward with incompatible PAM 'ATA' (invalid)
        # Site 3: Reverse with canonical PAM 'TGG' (valid)
        site1 = guide_20 + "AGG"
        site2 = guide_20 + "ATA"
        site3 = reverse_complement(guide_20 + "TGG")

        seq = site1 + "TTTT" + site2 + "TTTT" + site3
        chunker = GenomicChunker(chunk_size=150, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chrMulti")[0]

        # All 3 detected
        all_hits = aligner.align_and_validate(chunk=chunk, guide=guide_20, validator=validator)
        assert len(all_hits) == 3
        validity = [h.is_valid_pam for h in all_hits]
        assert validity == [True, False, True]

        # Only 2 valid retained with filter_incompatible
        valid_hits = aligner.align_and_validate(
            chunk=chunk, guide=guide_20, validator=validator, filter_incompatible=True
        )
        assert len(valid_hits) == 2
        assert all(h.is_valid_pam for h in valid_hits)
        assert [h.pam_sequence for h in valid_hits] == ["AGG", "TGG"]

    def test_missing_or_out_of_bounds_pam_region(
        self, aligner: CPUAligner, validator: PAMValidator, guide_20: str
    ):
        # Sequence ends exactly after the 20 nt spacer (0 bases for PAM)
        seq = "A" * 20 + guide_20
        assert len(seq) == 40
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk(seq, chromosome="chrTerminal")[0]

        val_hits = aligner.align_and_validate(chunk=chunk, guide=guide_20, validator=validator)
        assert len(val_hits) == 1
        h = val_hits[0]
        # PAM is truncated / out-of-bounds
        assert h.is_valid_pam is False
        assert len(h.pam_sequence) == 0
        assert "Incomplete or out-of-bounds PAM" in h.pam_result.details
