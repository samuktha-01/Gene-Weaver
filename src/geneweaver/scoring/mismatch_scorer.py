"""CRISPR off-target mismatch scoring engine.

Provides modular mismatch scoring algorithms for evaluating off-target cleavage probability
and sequence similarity between CRISPR guide RNAs and aligned genomic candidate sites.

Includes:
1. HsuZhangScorer: Published empirical positional weight model for SpCas9 (Hsu et al., Nature
   Biotechnology 2013).
2. ExperimentalPositionalScorer: Configurable heuristic linear positional penalty model for
   testing, research, or baseline comparisons (explicitly unvalidated).
3. BaseMismatchScorer: Abstract base protocol for extensible scoring model development.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from geneweaver.alignment.models import RawCandidateHit, Strand
from geneweaver.scoring.mismatch_analyzer import MismatchAnalysis, MismatchAnalyzer
from geneweaver.scoring.pam_evaluator import ValidatedCandidateHit


@dataclass(frozen=True)
class MismatchScoringConfig:
    """Configuration options for mismatch scoring models and PAM compatibility weighting.

    Attributes:
        model_name: Identifier for the scoring model to utilize ('hsu_zhang' or 'experimental_positional').
        apply_pam_multiplier: If True, multiplies the sequence mismatch score by the PAM cleavage
            multiplier (e.g. 1.0 for NGG, 0.25 for NAG, 0.0 for invalid).
        penalize_incompatible_pam: If True, assigns incompatible_pam_score when the PAM is incompatible.
        incompatible_pam_score: Score assigned to candidate sites lacking a compatible PAM when
            penalize_incompatible_pam is True (default: 0.0).
        experimental_seed_weight: Penalty per seed mismatch for ExperimentalPositionalScorer (default: 0.20).
        experimental_non_seed_weight: Penalty per non-seed mismatch for ExperimentalPositionalScorer (default: 0.05).
    """

    model_name: str = "hsu_zhang"
    apply_pam_multiplier: bool = True
    penalize_incompatible_pam: bool = True
    incompatible_pam_score: float = 0.0
    experimental_seed_weight: float = 0.20
    experimental_non_seed_weight: float = 0.05

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        valid_models = ("hsu_zhang", "experimental_positional")
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name '{self.model_name}'. Permitted models: {valid_models}"
            )
        if self.incompatible_pam_score < 0.0 or self.incompatible_pam_score > 100.0:
            raise ValueError(
                f"incompatible_pam_score ({self.incompatible_pam_score}) must be between 0.0 and 100.0"
            )
        if self.experimental_seed_weight < 0.0:
            raise ValueError(
                f"experimental_seed_weight ({self.experimental_seed_weight}) cannot be negative"
            )
        if self.experimental_non_seed_weight < 0.0:
            raise ValueError(
                f"experimental_non_seed_weight ({self.experimental_non_seed_weight}) cannot be negative"
            )


@dataclass(frozen=True)
class ScoredCandidateHit:
    """Enriched candidate hit with mismatch analysis, numerical score, and PAM status.

    All scores use GeneWeaver's unified risk convention:
    Score ranges from 0.0 to 100.0. Higher score indicates HIGHER predicted off-target
    cutting risk / activity (100.0 represents maximum risk / exact match, 0.0 represents
    negligible predicted activity).

    Attributes:
        candidate: Underlying ValidatedCandidateHit or RawCandidateHit.
        score: Final calculated off-target score after PAM weighting and policy application.
        raw_score: Intrinsic sequence mismatch score (0.0 - 100.0) before PAM weighting.
        mismatch_analysis: Detailed MismatchAnalysis record.
        scoring_model: Unique identifier of the scoring model used.
        pam_multiplier_applied: Value of the PAM multiplier applied to calculate the final score.
    """

    candidate: Union[ValidatedCandidateHit, RawCandidateHit]
    score: float
    raw_score: float
    mismatch_analysis: MismatchAnalysis
    scoring_model: str
    pam_multiplier_applied: float = 1.0

    def __post_init__(self) -> None:
        """Validates score ranges."""
        if self.score < 0.0 or self.score > 100.0001:
            raise ValueError(f"Score ({self.score}) out of valid range [0.0, 100.0]")
        if self.raw_score < 0.0 or self.raw_score > 100.0001:
            raise ValueError(f"Raw score ({self.raw_score}) out of valid range [0.0, 100.0]")

    # --- Convenience properties delegating to candidate hit ---

    @property
    def chromosome(self) -> str:
        """Chromosome or contig accession identifier."""
        return self.candidate.chromosome

    @property
    def strand(self) -> Strand:
        """Genomic strand orientation (Strand.FORWARD or Strand.REVERSE)."""
        return self.candidate.strand

    @property
    def protospacer_start(self) -> int:
        """0-based inclusive global genomic start coordinate of protospacer."""
        if isinstance(self.candidate, ValidatedCandidateHit):
            return self.candidate.protospacer_start
        return self.candidate.global_start

    @property
    def protospacer_end(self) -> int:
        """0-based exclusive global genomic end coordinate of protospacer."""
        if isinstance(self.candidate, ValidatedCandidateHit):
            return self.candidate.protospacer_end
        return self.candidate.global_end

    @property
    def pam_start(self) -> Optional[int]:
        """0-based inclusive global genomic start coordinate of PAM (None if unvalidated)."""
        if isinstance(self.candidate, ValidatedCandidateHit):
            return self.candidate.pam_start
        return getattr(self.candidate, "pam_start", None)

    @property
    def pam_end(self) -> Optional[int]:
        """0-based exclusive global genomic end coordinate of PAM (None if unvalidated)."""
        if isinstance(self.candidate, ValidatedCandidateHit):
            return self.candidate.pam_end
        return getattr(self.candidate, "pam_end", None)

    @property
    def pam_sequence(self) -> str:
        """5' to 3' PAM sequence along the target strand orientation."""
        return self.candidate.pam_sequence

    @property
    def target_sequence(self) -> str:
        """Observed target sequence on the aligned strand."""
        return self.candidate.target_sequence

    @property
    def is_valid_pam(self) -> bool:
        """True if the candidate site possesses a validated compatible PAM motif."""
        if isinstance(self.candidate, ValidatedCandidateHit):
            return self.candidate.is_valid_pam
        return True  # If unvalidated raw hit, default True

    @property
    def is_canonical_pam(self) -> bool:
        """True if the candidate site possesses the primary canonical PAM motif (e.g. NGG)."""
        if isinstance(self.candidate, ValidatedCandidateHit):
            return self.candidate.is_canonical_pam
        return False

    @property
    def matched_motif(self) -> Optional[str]:
        """Matched IUPAC pattern motif (e.g. 'NGG', 'NAG') or None."""
        if isinstance(self.candidate, ValidatedCandidateHit):
            return self.candidate.matched_motif
        return None

    @property
    def mismatch_count(self) -> int:
        """Total number of mismatches along the protospacer."""
        return self.mismatch_analysis.total_mismatches

    @property
    def mismatch_positions(self) -> Tuple[int, ...]:
        """1-based guide-relative mismatch positions (1 to 20, 5' to 3')."""
        return self.mismatch_analysis.mismatch_positions

    @property
    def seed_mismatches(self) -> int:
        """Count of mismatches in the seed region."""
        return self.mismatch_analysis.seed_mismatches

    @property
    def non_seed_mismatches(self) -> int:
        """Count of mismatches outside the seed region."""
        return self.mismatch_analysis.non_seed_mismatches

    @property
    def is_exact_match(self) -> bool:
        """True if protospacer sequence exactly matches the guide RNA."""
        return self.mismatch_analysis.is_exact_match


class BaseMismatchScorer(ABC):
    """Abstract base class for CRISPR off-target mismatch scoring models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier of the scoring model."""
        pass

    @abstractmethod
    def score_mismatches(self, analysis: MismatchAnalysis) -> float:
        """Calculates intrinsic sequence mismatch score (0.0 to 100.0) based on mismatch analysis.

        Args:
            analysis: MismatchAnalysis record containing mismatch count and positions.

        Returns:
            Float score between 0.0 and 100.0.
        """
        pass

    def score_candidate(
        self,
        candidate: Union[ValidatedCandidateHit, RawCandidateHit],
        analysis: MismatchAnalysis,
        config: Optional[MismatchScoringConfig] = None,
    ) -> ScoredCandidateHit:
        """Scores a candidate hit and integrates PAM weighting and compatibility policies.

        Args:
            candidate: ValidatedCandidateHit or RawCandidateHit instance.
            analysis: MismatchAnalysis for this candidate.
            config: Optional MismatchScoringConfig parameters.

        Returns:
            ScoredCandidateHit instance containing raw and final scores.
        """
        cfg = config if config is not None else MismatchScoringConfig()
        raw_score = self.score_mismatches(analysis)

        # Determine PAM compatibility and multiplier
        if isinstance(candidate, ValidatedCandidateHit):
            if candidate.is_valid_pam:
                pam_multiplier = candidate.cleavage_multiplier if cfg.apply_pam_multiplier else 1.0
                final_score = raw_score * pam_multiplier
            else:
                if cfg.penalize_incompatible_pam:
                    final_score = cfg.incompatible_pam_score
                    pam_multiplier = 0.0
                else:
                    final_score = raw_score
                    pam_multiplier = 1.0
        else:
            # Unvalidated raw hit: no PAM penalty applied
            final_score = raw_score
            pam_multiplier = 1.0

        # Clamp final score to [0.0, 100.0]
        final_score = max(0.0, min(100.0, final_score))

        return ScoredCandidateHit(
            candidate=candidate,
            score=final_score,
            raw_score=raw_score,
            mismatch_analysis=analysis,
            scoring_model=self.model_name,
            pam_multiplier_applied=pam_multiplier,
        )

    def score_candidates(
        self,
        pairs: Sequence[Tuple[Union[ValidatedCandidateHit, RawCandidateHit], MismatchAnalysis]],
        config: Optional[MismatchScoringConfig] = None,
    ) -> List[ScoredCandidateHit]:
        """Scores a batch of (candidate, analysis) pairs.

        Args:
            pairs: Sequence of (candidate, analysis) tuples.
            config: Optional scoring configuration.

        Returns:
            List of ScoredCandidateHit instances in input order.
        """
        cfg = config if config is not None else MismatchScoringConfig()
        return [self.score_candidate(c, a, config=cfg) for c, a in pairs]


class HsuZhangScorer(BaseMismatchScorer):
    """Hsu-Zhang (2013) published empirical off-target scoring model for SpCas9.

    Scientific Source:
    Hsu, P.D., Scott, D.A., Weinstein, J.A., Ran, F.A., Konermann, S., Agarwala, V.,
    Li, Y., Fine, E.J., Wu, X., Shalem, O., Cradick, T.J., Marraffini, L.A., Bao, G.,
    & Zhang, F. (2013). "DNA targeting specificity of RNA-guided Cas9 nucleases."
    Nature Biotechnology, 31(9), 827-832. doi:10.1038/nbt.2647.

    Methodology:
    For an off-target site with n mismatches at 1-based guide-relative positions
    p in M (where 1 is 5' distal, and 20 is 3' PAM-proximal):

    1. If n == 0 (exact match):
       Score = 100.0

    2. If n >= 1:
       Score = 100.0 * T_pos * T_dist * T_count

       where:
       T_pos = prod_{p in M} (1 - W[p])
       - W is the published empirical positional weight table (20 nt array).
       - Positions in the seed region (positions 11-20) carry large penalty weights
         (e.g., position 14 weight = 0.851), whereas distal positions (1-2) have weight 0.0.

       T_dist: Pairwise mismatch distance penalty:
       - If n == 1: T_dist = 1.0 (mean pairwise distance d_bar is defined as 19.0).
       - If n >= 2:
         d_bar = (sum_{i < j} |p_j - p_i|) / (n * (n - 1) / 2)
         T_dist = 1.0 / (((19.0 - d_bar) / 19.0) * 4.0 + 1.0)
         Clustered / consecutive mismatches result in a small d_bar and strong penalty.

       T_count = 1.0 / (n^2)
       - Quadratic penalty on total mismatch count.

    Documented Model Limitations:
    1. Specifically calibrated for Streptococcus pyogenes Cas9 (SpCas9) with 20 nt spacer
       and 5'-NGG-3' PAM motif.
    2. Calibrated from empirical human cell (HEK293FT) cleavage data primarily for n <= 4
       mismatches; extrapolation to high mismatch counts exhibits reduced empirical precision.
    3. Positional penalty does not differentiate specific nucleotide substitution types
       (e.g., rG:dT wobble vs. purine-purine clash).
    4. Does not incorporate chromatin accessibility, DNA methylation, or secondary RNA structures.
    """

    # Published empirical positional mismatch weight table (1-based positions 1 to 20)
    # Positions 1-2 carry 0 penalty; seed positions (11-20) carry heavy penalties up to 0.851
    POSITIONAL_WEIGHTS: Tuple[float, ...] = (
        0.0,    # Position 1  (5' distal)
        0.0,    # Position 2
        0.014,  # Position 3
        0.0,    # Position 4
        0.0,    # Position 5
        0.395,  # Position 6
        0.317,  # Position 7
        0.0,    # Position 8
        0.389,  # Position 9
        0.079,  # Position 10
        0.445,  # Position 11 (Seed start)
        0.508,  # Position 12
        0.613,  # Position 13
        0.851,  # Position 14 (Seed peak)
        0.732,  # Position 15
        0.828,  # Position 16
        0.615,  # Position 17
        0.804,  # Position 18
        0.685,  # Position 19
        0.669,  # Position 20 (PAM-proximal)
    )

    @property
    def model_name(self) -> str:
        return "hsu_zhang"

    def score_mismatches(self, analysis: MismatchAnalysis) -> float:
        """Calculates Hsu-Zhang (2013) off-target score from MismatchAnalysis.

        Args:
            analysis: MismatchAnalysis record.

        Returns:
            Score between 0.0 and 100.0.
        """
        n = analysis.total_mismatches
        if n == 0:
            return 100.0

        positions = sorted(analysis.mismatch_positions)

        # 1. Positional weight product: prod_{p in M} (1 - W[p])
        pos_term = 1.0
        weights_len = len(self.POSITIONAL_WEIGHTS)
        for p in positions:
            idx = p - 1  # convert 1-based position to 0-based index
            if 0 <= idx < weights_len:
                w = self.POSITIONAL_WEIGHTS[idx]
            else:
                # If position beyond 20 nt, use PAM-proximal weight as upper boundary
                w = self.POSITIONAL_WEIGHTS[-1]
            pos_term *= (1.0 - w)

        # 2. Pairwise distance factor: T_dist
        if n == 1:
            dist_term = 1.0
        else:
            num_pairs = (n * (n - 1)) // 2
            sum_distances = sum(
                abs(p2 - p1) for p1, p2 in itertools.combinations(positions, 2)
            )
            d_bar = sum_distances / num_pairs
            # Standard MIT formula: 1 / (((19 - d_bar) / 19) * 4 + 1)
            dist_term = 1.0 / (((19.0 - d_bar) / 19.0) * 4.0 + 1.0)

        # 3. Mismatch count factor: 1 / n^2
        count_term = 1.0 / (n * n)

        # Overall score scaled to 100.0
        score = 100.0 * pos_term * dist_term * count_term
        return max(0.0, min(100.0, score))


class ExperimentalPositionalScorer(BaseMismatchScorer):
    """Heuristic linear positional penalty mismatch scoring model.

    WARNING:
    This model is EXPERIMENTAL and NOT biologically validated for clinical or cleavage
    predictions. It serves solely as a configurable baseline for comparative analysis,
    synthetic benchmarking, and parameter sensitivity experimentation.

    Methodology:
    Calculates score by applying a configurable linear penalty for each seed-region mismatch
    and a smaller penalty for non-seed mismatches:
    Score = 100.0 * max(0.0, 1.0 - (seed_mismatches * seed_weight + non_seed_mismatches * non_seed_weight))
    """

    def __init__(
        self,
        seed_weight: float = 0.20,
        non_seed_weight: float = 0.05,
    ) -> None:
        """Initializes the ExperimentalPositionalScorer with custom penalty weights.

        Args:
            seed_weight: Penalty deduction per seed mismatch (default: 0.20).
            non_seed_weight: Penalty deduction per non-seed mismatch (default: 0.05).
        """
        if seed_weight < 0.0:
            raise ValueError(f"seed_weight ({seed_weight}) cannot be negative")
        if non_seed_weight < 0.0:
            raise ValueError(f"non_seed_weight ({non_seed_weight}) cannot be negative")

        self._seed_weight = seed_weight
        self._non_seed_weight = non_seed_weight

    @property
    def model_name(self) -> str:
        return "experimental_positional"

    @property
    def seed_weight(self) -> float:
        return self._seed_weight

    @property
    def non_seed_weight(self) -> float:
        return self._non_seed_weight

    def score_mismatches(self, analysis: MismatchAnalysis) -> float:
        """Calculates experimental heuristic score from MismatchAnalysis."""
        if analysis.is_exact_match:
            return 100.0

        penalty = (
            analysis.seed_mismatches * self._seed_weight
            + analysis.non_seed_mismatches * self._non_seed_weight
        )
        remaining = max(0.0, 1.0 - penalty)
        return remaining * 100.0
