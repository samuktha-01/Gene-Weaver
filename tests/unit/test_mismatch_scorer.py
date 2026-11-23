"""Unit tests for GeneWeaver Mismatch Scoring Engine.

Tests the published Hsu-Zhang (2013) empirical positional weight model,
the ExperimentalPositionalScorer, PAM compatibility weighting, config validation,
and boundary conditions.
"""

import pytest

from geneweaver.alignment.models import (
    MismatchInfo,
    RawCandidateHit,
    Strand,
)
from geneweaver.scoring.mismatch_analyzer import MismatchAnalysis, MismatchAnalyzer
from geneweaver.scoring.mismatch_scorer import (
    BaseMismatchScorer,
    ExperimentalPositionalScorer,
    HsuZhangScorer,
    MismatchScoringConfig,
    ScoredCandidateHit,
)
from geneweaver.scoring.pam_evaluator import (
    PAMValidationResult,
    ValidatedCandidateHit,
)


@pytest.fixture
def hsu_scorer() -> HsuZhangScorer:
    """Returns an instance of the empirical HsuZhangScorer."""
    return HsuZhangScorer()


@pytest.fixture
def exp_scorer() -> ExperimentalPositionalScorer:
    """Returns an instance of ExperimentalPositionalScorer with default parameters."""
    return ExperimentalPositionalScorer(seed_weight=0.20, non_seed_weight=0.05)


def make_analysis(
    total_mismatches: int,
    positions: tuple[int, ...],
    seed_start: int = 11,
    seed_end: int = 20,
    spacer_length: int = 20,
) -> MismatchAnalysis:
    """Helper to synthesize MismatchAnalysis objects for scoring tests."""
    seed_pos = tuple(p for p in positions if seed_start <= p <= seed_end)
    non_seed_pos = tuple(p for p in positions if p < seed_start or p > seed_end)
    bitmask = sum(1 << (p - 1) for p in positions)
    return MismatchAnalysis(
        total_mismatches=total_mismatches,
        mismatch_positions=positions,
        seed_mismatches=len(seed_pos),
        seed_positions=seed_pos,
        non_seed_mismatches=len(non_seed_pos),
        non_seed_positions=non_seed_pos,
        bitmask=bitmask,
        mismatch_details=(),
        seed_start=seed_start,
        seed_end=seed_end,
        spacer_length=spacer_length,
    )


class TestHsuZhangScoring:
    """Tests the published Hsu-Zhang (2013) empirical model."""

    def test_exact_match_score_is_100(self, hsu_scorer: HsuZhangScorer):
        """Exact match (0 mismatches) must produce a perfect score of 100.0."""
        analysis = make_analysis(total_mismatches=0, positions=())
        score = hsu_scorer.score_mismatches(analysis)
        assert score == 100.0

    def test_single_mismatch_distal_pos1(self, hsu_scorer: HsuZhangScorer):
        """Single mismatch at distal position 1 (weight W=0.0) produces score of 100.0."""
        # Hsu Table: Pos 1 has weight 0.0, n=1 -> dist_term=1.0, count_term=1.0 -> 100.0
        analysis = make_analysis(total_mismatches=1, positions=(1,))
        score = hsu_scorer.score_mismatches(analysis)
        assert pytest.approx(score, rel=1e-3) == 100.0

    def test_single_mismatch_distal_pos3(self, hsu_scorer: HsuZhangScorer):
        """Single mismatch at position 3 (weight W=0.014)."""
        # Pos 3 has weight 0.014 -> 100 * (1 - 0.014) = 98.6
        analysis = make_analysis(total_mismatches=1, positions=(3,))
        score = hsu_scorer.score_mismatches(analysis)
        assert pytest.approx(score, rel=1e-3) == 98.6

    def test_single_mismatch_seed_pos14(self, hsu_scorer: HsuZhangScorer):
        """Single mismatch at seed peak position 14 (weight W=0.851)."""
        # Pos 14 has weight 0.851 -> 100 * (1 - 0.851) = 14.9
        analysis = make_analysis(total_mismatches=1, positions=(14,))
        score = hsu_scorer.score_mismatches(analysis)
        assert pytest.approx(score, rel=1e-3) == 14.9

    def test_single_mismatch_pam_proximal_pos20(self, hsu_scorer: HsuZhangScorer):
        """Single mismatch at PAM-proximal position 20 (weight W=0.669)."""
        # Pos 20 has weight 0.669 -> 100 * (1 - 0.669) = 33.1
        analysis = make_analysis(total_mismatches=1, positions=(20,))
        score = hsu_scorer.score_mismatches(analysis)
        assert pytest.approx(score, rel=1e-3) == 33.1

    def test_seed_mismatch_penalized_substantially_more_than_non_seed(
        self, hsu_scorer: HsuZhangScorer
    ):
        """A seed mismatch (pos 14) is penalized substantially more than a non-seed mismatch (pos 3)."""
        score_seed = hsu_scorer.score_mismatches(make_analysis(1, (14,)))
        score_distal = hsu_scorer.score_mismatches(make_analysis(1, (3,)))
        assert score_seed < score_distal
        assert score_distal > 90.0
        assert score_seed < 20.0

    def test_multiple_mismatch_distance_penalty(self, hsu_scorer: HsuZhangScorer):
        """Clustered mismatches incur a stronger penalty than dispersed mismatches."""
        # Clustered mismatches: positions 11 and 12 (distance = 1)
        # Dispersed mismatches: positions 1 and 20 (distance = 19)
        analysis_clustered = make_analysis(2, (11, 12))
        analysis_dispersed = make_analysis(2, (1, 20))

        score_clustered = hsu_scorer.score_mismatches(analysis_clustered)
        score_dispersed = hsu_scorer.score_mismatches(analysis_dispersed)

        assert score_clustered < score_dispersed

    def test_quadratic_mismatch_count_penalty(self, hsu_scorer: HsuZhangScorer):
        """Adding mismatches drops score rapidly due to quadratic count penalty (1 / n^2)."""
        score_1 = hsu_scorer.score_mismatches(make_analysis(1, (3,)))
        score_2 = hsu_scorer.score_mismatches(make_analysis(2, (3, 5)))
        score_3 = hsu_scorer.score_mismatches(make_analysis(3, (3, 5, 8)))

        assert score_1 > score_2 > score_3
        # Quadratic factor 1/4 drops 2-mismatch score significantly
        assert score_2 < (score_1 / 3.0)

    def test_reproducibility(self, hsu_scorer: HsuZhangScorer):
        """Identical inputs produce identical scores across successive invocations."""
        analysis = make_analysis(2, (4, 15))
        score_1 = hsu_scorer.score_mismatches(analysis)
        score_2 = hsu_scorer.score_mismatches(analysis)
        assert score_1 == score_2


class TestExperimentalPositionalScoring:
    """Tests the heuristic linear ExperimentalPositionalScorer."""

    def test_exact_match_score(self, exp_scorer: ExperimentalPositionalScorer):
        """Exact match produces 100.0."""
        assert exp_scorer.score_mismatches(make_analysis(0, ())) == 100.0

    def test_seed_and_non_seed_linear_deductions(self, exp_scorer: ExperimentalPositionalScorer):
        """Applies configured linear penalties."""
        # 1 non-seed mismatch: deduction = 0.05 -> score = 95.0
        analysis_non_seed = make_analysis(1, (3,))
        assert pytest.approx(exp_scorer.score_mismatches(analysis_non_seed)) == 95.0

        # 1 seed mismatch: deduction = 0.20 -> score = 80.0
        analysis_seed = make_analysis(1, (15,))
        assert pytest.approx(exp_scorer.score_mismatches(analysis_seed)) == 80.0

        # 2 seed + 1 non-seed: deduction = 2*0.20 + 1*0.05 = 0.45 -> score = 55.0
        analysis_mixed = make_analysis(3, (2, 12, 16))
        assert pytest.approx(exp_scorer.score_mismatches(analysis_mixed)) == 55.0

    def test_floor_at_zero(self, exp_scorer: ExperimentalPositionalScorer):
        """Score cannot drop below 0.0 even with many mismatches."""
        # 6 seed mismatches * 0.20 = 1.2 deduction -> remaining = 0 -> score = 0.0
        analysis = make_analysis(6, (11, 12, 13, 14, 15, 16))
        assert exp_scorer.score_mismatches(analysis) == 0.0

    def test_invalid_weights_raise_value_error(self):
        """Negative weights raise ValueError."""
        with pytest.raises(ValueError, match="seed_weight"):
            ExperimentalPositionalScorer(seed_weight=-0.1)
        with pytest.raises(ValueError, match="non_seed_weight"):
            ExperimentalPositionalScorer(non_seed_weight=-0.05)


class TestPAMCompatibilityIntegration:
    """Tests integration of PAM validation results into candidate scoring."""

    def test_canonical_pam_hit_receives_full_score(self, hsu_scorer: HsuZhangScorer):
        """Canonical NGG PAM receives 1.0 multiplier."""
        raw_hit = RawCandidateHit(
            chunk_id="c0", chromosome="chr1", local_offset=0, global_start=100, global_end=123,
            genomic_position=101, strand=Strand.FORWARD, mismatch_info=MismatchInfo.from_bitmask(0),
            pam_sequence="TGG", target_sequence="A"*20 + "TGG"
        )
        val_hit = ValidatedCandidateHit(
            candidate=raw_hit,
            pam_result=PAMValidationResult(is_valid=True, is_canonical=True, matched_motif="NGG", cleavage_multiplier=1.0),
            chromosome="chr1", strand=Strand.FORWARD, protospacer_start=100, protospacer_end=120,
            pam_start=120, pam_end=123, pam_sequence="TGG", target_sequence="A"*20 + "TGG"
        )
        analysis = make_analysis(0, ())
        scored = hsu_scorer.score_candidate(candidate=val_hit, analysis=analysis)

        assert scored.raw_score == 100.0
        assert scored.score == 100.0
        assert scored.pam_multiplier_applied == 1.0
        assert scored.is_valid_pam is True
        assert scored.is_canonical_pam is True

    def test_alternate_pam_multiplier_scaled(self, hsu_scorer: HsuZhangScorer):
        """Alternate NAG PAM with cleavage_multiplier 0.25 scales final score."""
        raw_hit = RawCandidateHit(
            chunk_id="c0", chromosome="chr1", local_offset=0, global_start=100, global_end=123,
            genomic_position=101, strand=Strand.FORWARD, mismatch_info=MismatchInfo.from_bitmask(0),
            pam_sequence="AAG", target_sequence="A"*20 + "AAG"
        )
        val_hit = ValidatedCandidateHit(
            candidate=raw_hit,
            pam_result=PAMValidationResult(is_valid=True, is_canonical=False, matched_motif="NAG", cleavage_multiplier=0.25),
            chromosome="chr1", strand=Strand.FORWARD, protospacer_start=100, protospacer_end=120,
            pam_start=120, pam_end=123, pam_sequence="AAG", target_sequence="A"*20 + "AAG"
        )
        analysis = make_analysis(0, ())
        scored = hsu_scorer.score_candidate(candidate=val_hit, analysis=analysis)

        assert scored.raw_score == 100.0
        assert scored.score == 25.0
        assert scored.pam_multiplier_applied == 0.25

    def test_incompatible_pam_penalized_to_zero(self, hsu_scorer: HsuZhangScorer):
        """Incompatible PAM receives 0.0 final score when penalize_incompatible_pam=True."""
        raw_hit = RawCandidateHit(
            chunk_id="c0", chromosome="chr1", local_offset=0, global_start=100, global_end=123,
            genomic_position=101, strand=Strand.FORWARD, mismatch_info=MismatchInfo.from_bitmask(0),
            pam_sequence="AGA", target_sequence="A"*20 + "AGA"
        )
        val_hit = ValidatedCandidateHit(
            candidate=raw_hit,
            pam_result=PAMValidationResult(is_valid=False, is_canonical=False, matched_motif=None, cleavage_multiplier=0.0),
            chromosome="chr1", strand=Strand.FORWARD, protospacer_start=100, protospacer_end=120,
            pam_start=120, pam_end=123, pam_sequence="AGA", target_sequence="A"*20 + "AGA"
        )
        analysis = make_analysis(0, ())
        config = MismatchScoringConfig(penalize_incompatible_pam=True, incompatible_pam_score=0.0)
        scored = hsu_scorer.score_candidate(candidate=val_hit, analysis=analysis, config=config)

        assert scored.raw_score == 100.0
        assert scored.score == 0.0
        assert scored.is_valid_pam is False
        assert scored.pam_multiplier_applied == 0.0

    def test_incompatible_pam_without_penalty_retains_raw_score(self, hsu_scorer: HsuZhangScorer):
        """When penalize_incompatible_pam=False, incompatible PAM hit retains its raw score."""
        raw_hit = RawCandidateHit(
            chunk_id="c0", chromosome="chr1", local_offset=0, global_start=100, global_end=123,
            genomic_position=101, strand=Strand.FORWARD, mismatch_info=MismatchInfo.from_bitmask(0),
            pam_sequence="AGA", target_sequence="A"*20 + "AGA"
        )
        val_hit = ValidatedCandidateHit(
            candidate=raw_hit,
            pam_result=PAMValidationResult(is_valid=False, is_canonical=False, matched_motif=None, cleavage_multiplier=0.0),
            chromosome="chr1", strand=Strand.FORWARD, protospacer_start=100, protospacer_end=120,
            pam_start=120, pam_end=123, pam_sequence="AGA", target_sequence="A"*20 + "AGA"
        )
        analysis = make_analysis(0, ())
        config = MismatchScoringConfig(penalize_incompatible_pam=False)
        scored = hsu_scorer.score_candidate(candidate=val_hit, analysis=analysis, config=config)

        assert scored.raw_score == 100.0
        assert scored.score == 100.0


class TestScoringConfigValidation:
    """Tests validation of MismatchScoringConfig."""

    def test_invalid_model_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid model_name"):
            MismatchScoringConfig(model_name="unsupported_model")

    def test_invalid_incompatible_pam_score_raises_value_error(self):
        with pytest.raises(ValueError, match="incompatible_pam_score"):
            MismatchScoringConfig(incompatible_pam_score=-5.0)
        with pytest.raises(ValueError, match="incompatible_pam_score"):
            MismatchScoringConfig(incompatible_pam_score=150.0)
