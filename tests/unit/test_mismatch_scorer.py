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
        """Single mismatch at PAM-proximal position 20 (published weight W=0.583)."""
        # Hsu et al. 2013: Pos 20 has weight 0.583 -> 100 * (1 - 0.583) = 41.7
        analysis = make_analysis(total_mismatches=1, positions=(20,))
        score = hsu_scorer.score_mismatches(analysis)
        assert pytest.approx(score, rel=1e-3) == 41.7


class TestHsuZhangIndependentScientificVerification:
    """Rigorous independent scientific verification of the Hsu-Zhang (2013) MIT model.

    All expected values are independently hand-derived from the original mathematical formula:
        Score = 100.0 * T_pos * T_dist * T_count
    where:
        T_pos = prod_{p in M} (1 - W[p])
        T_dist = 1.0 / (((19.0 - d_bar) / 19.0) * 4.0 + 1.0) with d_bar = (p_n - p_1) / (n - 1)
        T_count = 1.0 / n^2
    """

    def test_zero_mismatches_independent(self, hsu_scorer: HsuZhangScorer):
        """Exact match produces maximum benchmark score of 100.0."""
        # n = 0 -> exact match definition: 100.0
        analysis = make_analysis(total_mismatches=0, positions=())
        assert hsu_scorer.score_mismatches(analysis) == 100.0

    def test_single_mismatch_pos1_independent(self, hsu_scorer: HsuZhangScorer):
        """Position 1 mismatch: W[1] = 0.0, T_dist = 1.0, T_count = 1.0 -> 100.0."""
        # T_pos = 1 - 0.0 = 1.0
        # T_dist = 1.0 (single mismatch)
        # T_count = 1 / 1^2 = 1.0
        # Score = 100.0 * 1.0 * 1.0 * 1.0 = 100.0
        analysis = make_analysis(total_mismatches=1, positions=(1,))
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == 100.0

    def test_single_mismatch_pos14_seed_peak_independent(self, hsu_scorer: HsuZhangScorer):
        """Position 14 seed peak mismatch: W[14] = 0.851 -> Score = 14.90."""
        # T_pos = 1 - 0.851 = 0.149
        # T_dist = 1.0
        # T_count = 1.0
        # Score = 100.0 * 0.149 * 1.0 * 1.0 = 14.90
        analysis = make_analysis(total_mismatches=1, positions=(14,))
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == 14.90

    def test_single_mismatch_pos20_pam_proximal_independent(self, hsu_scorer: HsuZhangScorer):
        """Position 20 PAM-proximal mismatch: published W[20] = 0.583 -> Score = 41.70."""
        # T_pos = 1 - 0.583 = 0.417
        # T_dist = 1.0
        # T_count = 1.0
        # Score = 100.0 * 0.417 * 1.0 * 1.0 = 41.70
        analysis = make_analysis(total_mismatches=1, positions=(20,))
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == 41.70

    def test_two_mismatches_extreme_dispersed_pos1_and_pos20(self, hsu_scorer: HsuZhangScorer):
        """Two mismatches at positions (1, 20): d_bar = 19.0, T_dist = 1.0 -> Score = 10.425."""
        # T_pos = (1 - 0.0) * (1 - 0.583) = 1.0 * 0.417 = 0.417
        # d_bar = (20 - 1) / (2 - 1) = 19.0
        # T_dist = 1.0 / (((19.0 - 19.0) / 19.0) * 4.0 + 1.0) = 1.0 / 1.0 = 1.0
        # T_count = 1.0 / 2^2 = 0.25
        # Expected Score = 100.0 * 0.417 * 1.0 * 0.25 = 10.425
        analysis = make_analysis(total_mismatches=2, positions=(1, 20))
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == 10.425

    def test_two_mismatches_adjacent_in_seed_pos13_and_pos14(self, hsu_scorer: HsuZhangScorer):
        """Two adjacent mismatches in seed (13, 14): d_bar = 1.0, T_dist = 19/91 -> Score = 0.30101."""
        # W[13] = 0.613 -> 1 - 0.613 = 0.387
        # W[14] = 0.851 -> 1 - 0.851 = 0.149
        # T_pos = 0.387 * 0.149 = 0.057663
        # d_bar = (14 - 13) / (2 - 1) = 1.0
        # T_dist = 1.0 / (((19.0 - 1.0) / 19.0) * 4.0 + 1.0) = 1.0 / (72/19 + 1) = 19/91 ~= 0.2087912
        # T_count = 1.0 / 4 = 0.25
        # Expected Score = 100.0 * 0.057663 * (19 / 91) * 0.25 ~= 0.301010
        analysis = make_analysis(total_mismatches=2, positions=(13, 14))
        expected_score = 100.0 * 0.057663 * (19.0 / 91.0) * 0.25
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == expected_score

    def test_three_clustered_mismatches_pos11_12_13(self, hsu_scorer: HsuZhangScorer):
        """Three tightly clustered seed mismatches (11, 12, 13): d_bar = 1.0 -> Score = 0.245155."""
        # W[11]=0.445 (0.555), W[12]=0.508 (0.492), W[13]=0.613 (0.387)
        # T_pos = 0.555 * 0.492 * 0.387 = 0.10567422
        # d_bar = (13 - 11) / (3 - 1) = 2.0 / 2 = 1.0
        # T_dist = 19 / 91 ~= 0.2087912
        # T_count = 1 / 3^2 = 1 / 9
        # Expected Score = 100.0 * 0.10567422 * (19.0 / 91.0) * (1.0 / 9.0) ~= 0.245155
        analysis = make_analysis(total_mismatches=3, positions=(11, 12, 13))
        expected_score = 100.0 * (0.555 * 0.492 * 0.387) * (19.0 / 91.0) * (1.0 / 9.0)
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == expected_score

    def test_three_dispersed_mismatches_pos1_10_20(self, hsu_scorer: HsuZhangScorer):
        """Three dispersed mismatches (1, 10, 20): d_bar = 9.5 -> Score = 1.422433."""
        # W[1]=0.0 (1.0), W[10]=0.079 (0.921), W[20]=0.583 (0.417)
        # T_pos = 1.0 * 0.921 * 0.417 = 0.384057
        # d_bar = (20 - 1) / (3 - 1) = 19.0 / 2 = 9.5
        # T_dist = 1.0 / (((19.0 - 9.5) / 19.0) * 4.0 + 1.0) = 1.0 / (0.5 * 4.0 + 1.0) = 1.0 / 3.0
        # T_count = 1.0 / 3^2 = 1.0 / 9.0
        # Expected Score = 100.0 * 0.384057 * (1.0 / 3.0) * (1.0 / 9.0) = 100.0 * 0.384057 / 27.0 ~= 1.422433
        analysis = make_analysis(total_mismatches=3, positions=(1, 10, 20))
        expected_score = 100.0 * 0.384057 / 27.0
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == expected_score

    def test_four_mismatches_independent(self, hsu_scorer: HsuZhangScorer):
        """Four mismatches (1, 6, 12, 19): d_bar = 6.0, T_dist = 19/71, T_count = 1/16 -> Score = 0.156822."""
        # W[1]=0.0 (1.0), W[6]=0.395 (0.605), W[12]=0.508 (0.492), W[19]=0.685 (0.315)
        # T_pos = 1.0 * 0.605 * 0.492 * 0.315 = 0.09376338
        # d_bar = (19 - 1) / (4 - 1) = 18.0 / 3 = 6.0
        # T_dist = 1.0 / (((19.0 - 6.0) / 19.0) * 4.0 + 1.0) = 1.0 / (52/19 + 1) = 19 / 71 ~= 0.2676056
        # T_count = 1.0 / 4^2 = 1.0 / 16.0 = 0.0625
        # Expected Score = 100.0 * 0.09376338 * (19.0 / 71.0) * (1.0 / 16.0) ~= 0.156822
        analysis = make_analysis(total_mismatches=4, positions=(1, 6, 12, 19))
        expected_score = 100.0 * 0.09376338 * (19.0 / 71.0) / 16.0
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-4) == expected_score

    def test_position_ordering_invariance(self, hsu_scorer: HsuZhangScorer):
        """Mismatch positions passed in reversed or arbitrary order produce identical scores."""
        analysis_sorted = make_analysis(3, (1, 10, 20))
        analysis_reversed = make_analysis(3, (20, 10, 1))
        analysis_permuted = make_analysis(3, (10, 1, 20))

        score_sorted = hsu_scorer.score_mismatches(analysis_sorted)
        score_reversed = hsu_scorer.score_mismatches(analysis_reversed)
        score_permuted = hsu_scorer.score_mismatches(analysis_permuted)

        assert score_sorted == score_reversed == score_permuted
        assert pytest.approx(score_sorted, rel=1e-4) == 100.0 * 0.384057 / 27.0

    def test_independent_component_decomposition(self, hsu_scorer: HsuZhangScorer):
        """Independently verifies T_pos, T_dist, and T_count components."""
        # Positions: (6, 14, 20)
        # T_pos: (1 - 0.395) * (1 - 0.851) * (1 - 0.583) = 0.605 * 0.149 * 0.417 = 0.037590465
        t_pos = (1.0 - 0.395) * (1.0 - 0.851) * (1.0 - 0.583)
        # T_dist: d_bar = (20 - 6) / 2 = 14.0 / 2 = 7.0
        # denominator = ((19 - 7) / 19) * 4 + 1 = (12/19)*4 + 1 = 48/19 + 1 = 67/19
        t_dist = 19.0 / 67.0
        # T_count: 1 / 3^2 = 1 / 9
        t_count = 1.0 / 9.0
        expected_score = 100.0 * t_pos * t_dist * t_count

        analysis = make_analysis(3, (6, 14, 20))
        assert pytest.approx(hsu_scorer.score_mismatches(analysis), rel=1e-5) == expected_score

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
