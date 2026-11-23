"""Candidate ranking engine for CRISPR off-target alignment hits.

Provides deterministic sorting, filtering, and summarization of scored candidate off-target
sites according to predicted cutting risk and sequence similarity.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from geneweaver.alignment.models import GuideTarget
from geneweaver.scoring.mismatch_analyzer import MismatchAnalysis, MismatchAnalyzer
from geneweaver.scoring.mismatch_scorer import (
    BaseMismatchScorer,
    HsuZhangScorer,
    MismatchScoringConfig,
    ScoredCandidateHit,
)
from geneweaver.scoring.pam_evaluator import ValidatedCandidateHit


@dataclass(frozen=True)
class RankingConfig:
    """Configuration options and threshold filters for off-target candidate ranking.

    Attributes:
        min_score: Minimum allowable score (0.0 - 100.0). Candidates with scores below this
            cutoff are filtered out. If None, no score threshold is applied.
        max_mismatches: Maximum allowable total mismatches. Candidates exceeding this count
            are filtered out. If None, no limit is applied.
        max_seed_mismatches: Maximum allowable seed-region mismatches. If None, no limit is applied.
        require_valid_pam: If True, only candidates with compatible PAM motifs are retained.
        canonical_pam_only: If True, only candidates with the primary canonical PAM motif (e.g. NGG)
            are retained.
        top_k: Maximum number of ranked candidates to return. If None, all passing candidates are returned.
    """

    min_score: Optional[float] = None
    max_mismatches: Optional[int] = None
    max_seed_mismatches: Optional[int] = None
    require_valid_pam: bool = False
    canonical_pam_only: bool = False
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        """Validates ranking configuration thresholds."""
        if self.min_score is not None:
            if self.min_score < 0.0 or self.min_score > 100.0:
                raise ValueError(f"min_score ({self.min_score}) must be between 0.0 and 100.0")
        if self.max_mismatches is not None and self.max_mismatches < 0:
            raise ValueError(f"max_mismatches ({self.max_mismatches}) cannot be negative")
        if self.max_seed_mismatches is not None and self.max_seed_mismatches < 0:
            raise ValueError(f"max_seed_mismatches ({self.max_seed_mismatches}) cannot be negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k ({self.top_k}) must be a positive integer")


@dataclass(frozen=True)
class CandidateRankingResult:
    """Immutable result container representing a deterministically ranked collection of candidate hits.

    Candidates are ordered in descending order of predicted off-target cutting risk (score).

    Attributes:
        hits: Immutable tuple of ScoredCandidateHit instances in ranked order.
        total_candidates: Total count of candidate hits evaluated prior to filtering.
        retained_count: Total count of candidates passing all filter criteria.
        scoring_model: Unique identifier of the scoring model employed.
        ranking_order: Description of ranking direction (default: 'descending_score').
    """

    hits: Tuple[ScoredCandidateHit, ...]
    total_candidates: int
    retained_count: int
    scoring_model: str
    ranking_order: str = "descending_score"

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def __getitem__(self, index: Union[int, slice]):
        return self.hits[index]

    def top(self, k: int) -> Tuple[ScoredCandidateHit, ...]:
        """Returns the top k highest-risk off-target candidates.

        Args:
            k: Maximum number of top hits to return.

        Returns:
            Tuple of up to k ScoredCandidateHit instances.
        """
        if k <= 0:
            return ()
        return self.hits[:k]

    def by_chromosome(self) -> Dict[str, List[ScoredCandidateHit]]:
        """Groups ranked candidates by chromosome/accession identifier.

        Returns:
            Dictionary mapping chromosome name to list of ScoredCandidateHits in ranked order.
        """
        grouped: Dict[str, List[ScoredCandidateHit]] = defaultdict(list)
        for hit in self.hits:
            grouped[hit.chromosome].append(hit)
        return dict(grouped)

    def summary(self) -> Dict[str, Any]:
        """Generates statistical and diagnostic summary metrics for the ranked candidates."""
        exact_matches = sum(1 for h in self.hits if h.is_exact_match)
        canonical_pams = sum(1 for h in self.hits if h.is_canonical_pam)
        valid_pams = sum(1 for h in self.hits if h.is_valid_pam)
        top_score = self.hits[0].score if self.hits else 0.0
        lowest_score = self.hits[-1].score if self.hits else 0.0

        return {
            "total_evaluated": self.total_candidates,
            "retained_count": self.retained_count,
            "scoring_model": self.scoring_model,
            "top_score": top_score,
            "lowest_score": lowest_score,
            "exact_matches": exact_matches,
            "valid_pams": valid_pams,
            "canonical_pams": canonical_pams,
        }


class CandidateRanker:
    """Candidate ranking engine implementing deterministic sorting and multi-criteria filtering.

    Ranking Order Convention:
    Higher score indicates HIGHER predicted off-target cutting risk / activity.
    Therefore, candidates are sorted in DESCENDING order of score (100.0 is top rank).

    Deterministic Tie-Breaking Order:
    When candidates have identical numerical scores, ties are deterministically resolved by:
    1. Score: Descending (highest risk first)
    2. Mismatch count: Ascending (fewer total mismatches first)
    3. Seed mismatches: Ascending (fewer seed mismatches first)
    4. Chromosome accession: Ascending lexicographical order (e.g. 'NC_000001.11' before 'NC_000002.12')
    5. Genomic start coordinate: Ascending numerical order (earlier locus in contig first)
    6. Strand sign: Ascending ('+' before '-')
    """

    def __init__(self, config: Optional[RankingConfig] = None) -> None:
        """Initializes CandidateRanker with optional default RankingConfig.

        Args:
            config: Default RankingConfig to apply if none is passed to rank().
        """
        self._config = config if config is not None else RankingConfig()

    @property
    def config(self) -> RankingConfig:
        """Default ranking configuration."""
        return self._config

    def rank(
        self,
        candidates: Iterable[ScoredCandidateHit],
        config: Optional[RankingConfig] = None,
    ) -> CandidateRankingResult:
        """Ranks scored candidate hits deterministically with optional threshold filtering.

        Args:
            candidates: Collection of ScoredCandidateHit instances.
            config: Optional override ranking configuration.

        Returns:
            CandidateRankingResult containing ranked, filtered hits.
        """
        cfg = config if config is not None else self._config
        cand_list = list(candidates)
        total_count = len(cand_list)

        if not cand_list:
            return CandidateRankingResult(
                hits=(),
                total_candidates=0,
                retained_count=0,
                scoring_model="none",
            )

        scoring_model = cand_list[0].scoring_model

        # 1. Apply explicit threshold filters
        filtered: List[ScoredCandidateHit] = []
        for hit in cand_list:
            if cfg.min_score is not None and hit.score < cfg.min_score:
                continue
            if cfg.max_mismatches is not None and hit.mismatch_count > cfg.max_mismatches:
                continue
            if cfg.max_seed_mismatches is not None and hit.seed_mismatches > cfg.max_seed_mismatches:
                continue
            if cfg.require_valid_pam and not hit.is_valid_pam:
                continue
            if cfg.canonical_pam_only and not hit.is_canonical_pam:
                continue
            filtered.append(hit)

        # 2. Deterministic multi-tier sort
        # Key: (-score, mismatches, seed_mismatches, chromosome, protospacer_start, strand_sign)
        sorted_hits = sorted(
            filtered,
            key=lambda h: (
                -h.score,
                h.mismatch_count,
                h.seed_mismatches,
                h.chromosome,
                h.protospacer_start,
                h.strand.sign,
            ),
        )

        # 3. Limit to top_k if requested
        if cfg.top_k is not None:
            sorted_hits = sorted_hits[: cfg.top_k]

        return CandidateRankingResult(
            hits=tuple(sorted_hits),
            total_candidates=total_count,
            retained_count=len(sorted_hits),
            scoring_model=scoring_model,
        )

    def rank_validated_hits(
        self,
        validated_hits: Iterable[ValidatedCandidateHit],
        guide: Union[GuideTarget, str],
        scorer: Optional[BaseMismatchScorer] = None,
        analyzer: Optional[MismatchAnalyzer] = None,
        scoring_config: Optional[MismatchScoringConfig] = None,
        ranking_config: Optional[RankingConfig] = None,
    ) -> CandidateRankingResult:
        """End-to-end pipeline method: analyzes, scores, and ranks ValidatedCandidateHit objects.

        Args:
            validated_hits: Collection of ValidatedCandidateHit objects from alignment & PAM validation.
            guide: Target CRISPR guide sequence or GuideTarget.
            scorer: Mismatch scoring model (defaults to HsuZhangScorer).
            analyzer: Mismatch analyzer (defaults to MismatchAnalyzer with standard 11-20 seed).
            scoring_config: Optional scoring configuration.
            ranking_config: Optional ranking configuration.

        Returns:
            CandidateRankingResult containing fully evaluated and ranked candidates.
        """
        active_scorer = scorer if scorer is not None else HsuZhangScorer()
        active_analyzer = analyzer if analyzer is not None else MismatchAnalyzer()
        active_scoring_cfg = scoring_config if scoring_config is not None else MismatchScoringConfig()

        scored_hits: List[ScoredCandidateHit] = []
        for v_hit in validated_hits:
            analysis = active_analyzer.analyze_candidate(candidate=v_hit, guide=guide)
            scored = active_scorer.score_candidate(
                candidate=v_hit,
                analysis=analysis,
                config=active_scoring_cfg,
            )
            scored_hits.append(scored)

        return self.rank(scored_hits, config=ranking_config)
