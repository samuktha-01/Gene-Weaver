"""Unit tests for GeneWeaver CandidateRanker.

Tests deterministic sorting, multi-criteria tie-breaking, filtering,
metadata preservation, edge cases, and end-to-end CPU alignment integration.
"""

import pytest

from geneweaver.alignment.cpu_aligner import CPUAligner
from geneweaver.alignment.models import (
    AlignmentConfig,
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    Strand,
)
from geneweaver.scoring.candidate_ranker import (
    CandidateRanker,
    CandidateRankingResult,
    RankingConfig,
)
from geneweaver.scoring.mismatch_analyzer import MismatchAnalysis, MismatchAnalyzer
from geneweaver.scoring.mismatch_scorer import (
    HsuZhangScorer,
    ScoredCandidateHit,
)
from geneweaver.scoring.pam_evaluator import (
    PAMValidationResult,
    ValidatedCandidateHit,
)
from geneweaver.sequence_processor.chunker import ChunkDescriptor


def make_scored_hit(
    score: float,
    raw_score: float,
    chromosome: str = "chr1",
    global_start: int = 1000,
    strand: Strand = Strand.FORWARD,
    mismatch_count: int = 0,
    seed_mismatches: int = 0,
    is_valid_pam: bool = True,
    is_canonical_pam: bool = True,
    pam_sequence: str = "CGG",
) -> ScoredCandidateHit:
    """Helper to synthesize ScoredCandidateHit instances with exact properties."""
    bitmask = sum(1 << i for i in range(mismatch_count))
    m_info = MismatchInfo.from_bitmask(bitmask)

    raw_hit = RawCandidateHit(
        chunk_id="chunk_0",
        chromosome=chromosome,
        local_offset=global_start,
        global_start=global_start,
        global_end=global_start + 23,
        genomic_position=global_start + 1,
        strand=strand,
        mismatch_info=m_info,
        pam_sequence=pam_sequence,
        target_sequence="A" * 20 + pam_sequence,
    )

    val_hit = ValidatedCandidateHit(
        candidate=raw_hit,
        pam_result=PAMValidationResult(
            is_valid=is_valid_pam,
            is_canonical=is_canonical_pam,
            matched_motif="NGG" if is_canonical_pam else ("NAG" if is_valid_pam else None),
            cleavage_multiplier=1.0 if is_canonical_pam else (0.25 if is_valid_pam else 0.0),
        ),
        chromosome=chromosome,
        strand=strand,
        protospacer_start=global_start if strand == Strand.FORWARD else global_start + 3,
        protospacer_end=global_start + 20 if strand == Strand.FORWARD else global_start + 23,
        pam_start=global_start + 20 if strand == Strand.FORWARD else global_start,
        pam_end=global_start + 23 if strand == Strand.FORWARD else global_start + 3,
        pam_sequence=pam_sequence,
        target_sequence="A" * 20 + pam_sequence,
    )

    analysis = MismatchAnalysis(
        total_mismatches=mismatch_count,
        mismatch_positions=tuple(range(1, mismatch_count + 1)),
        seed_mismatches=seed_mismatches,
        seed_positions=tuple(range(11, 11 + seed_mismatches)),
        non_seed_mismatches=mismatch_count - seed_mismatches,
        non_seed_positions=tuple(range(1, 1 + mismatch_count - seed_mismatches)),
        bitmask=bitmask,
        mismatch_details=(),
        seed_start=11,
        seed_end=20,
        spacer_length=20,
    )

    return ScoredCandidateHit(
        candidate=val_hit,
        score=score,
        raw_score=raw_score,
        mismatch_analysis=analysis,
        scoring_model="hsu_zhang",
        pam_multiplier_applied=1.0 if is_canonical_pam else (0.25 if is_valid_pam else 0.0),
    )


class TestCandidateRankerBasicOrdering:
    """Tests basic ranking and score ordering."""

    def test_rank_multiple_candidates_descending_score(self):
        """Candidates are sorted descending by score (highest risk first)."""
        hit_low = make_scored_hit(score=10.0, raw_score=10.0, global_start=100)
        hit_high = make_scored_hit(score=95.0, raw_score=95.0, global_start=200)
        hit_mid = make_scored_hit(score=50.0, raw_score=50.0, global_start=300)

        ranker = CandidateRanker()
        result = ranker.rank([hit_low, hit_high, hit_mid])

        assert len(result) == 3
        assert result.total_candidates == 3
        assert result.retained_count == 3
        assert [h.score for h in result.hits] == [95.0, 50.0, 10.0]

    def test_empty_candidate_collection(self):
        """Empty input collection returns empty result."""
        ranker = CandidateRanker()
        result = ranker.rank([])
        assert len(result) == 0
        assert result.total_candidates == 0
        assert result.retained_count == 0
        assert result.hits == ()
        assert result.summary()["total_evaluated"] == 0

    def test_single_candidate(self):
        """Single candidate returns result of length 1."""
        hit = make_scored_hit(score=80.0, raw_score=80.0)
        ranker = CandidateRanker()
        result = ranker.rank([hit])
        assert len(result) == 1
        assert result.hits[0].score == 80.0
        assert result.top(1)[0] == hit
        assert result.top(5)[0] == hit


class TestDeterministicTieBreaking:
    """Tests deterministic tie-breaking rules when scores are identical."""

    def test_tie_breaking_by_mismatch_count(self):
        """When scores are equal, candidate with fewer mismatches ranks higher."""
        hit_more_mm = make_scored_hit(score=50.0, raw_score=50.0, mismatch_count=3, seed_mismatches=1, global_start=100)
        hit_fewer_mm = make_scored_hit(score=50.0, raw_score=50.0, mismatch_count=1, seed_mismatches=1, global_start=200)

        ranker = CandidateRanker()
        result = ranker.rank([hit_more_mm, hit_fewer_mm])
        assert result.hits[0] == hit_fewer_mm
        assert result.hits[1] == hit_more_mm

    def test_tie_breaking_by_seed_mismatches(self):
        """When scores and total mismatches are equal, candidate with fewer seed mismatches ranks higher."""
        hit_more_seed = make_scored_hit(score=50.0, raw_score=50.0, mismatch_count=2, seed_mismatches=2, global_start=100)
        hit_fewer_seed = make_scored_hit(score=50.0, raw_score=50.0, mismatch_count=2, seed_mismatches=1, global_start=200)

        ranker = CandidateRanker()
        result = ranker.rank([hit_more_seed, hit_fewer_seed])
        assert result.hits[0] == hit_fewer_seed
        assert result.hits[1] == hit_more_seed

    def test_tie_breaking_by_chromosome_and_position(self):
        """When scores and mismatches are equal, chromosome and coordinate break ties."""
        hit_chr2 = make_scored_hit(score=50.0, raw_score=50.0, chromosome="NC_000002.12", global_start=100)
        hit_chr1_later = make_scored_hit(score=50.0, raw_score=50.0, chromosome="NC_000001.11", global_start=500)
        hit_chr1_earlier = make_scored_hit(score=50.0, raw_score=50.0, chromosome="NC_000001.11", global_start=200)

        ranker = CandidateRanker()
        result = ranker.rank([hit_chr2, hit_chr1_later, hit_chr1_earlier])
        assert result.hits[0] == hit_chr1_earlier
        assert result.hits[1] == hit_chr1_later
        assert result.hits[2] == hit_chr2

    def test_tie_breaking_by_strand(self):
        """When locus and scores are identical, Forward (+) strand ranks before Reverse (-) strand."""
        hit_rev = make_scored_hit(score=50.0, raw_score=50.0, chromosome="chr1", global_start=100, strand=Strand.REVERSE)
        hit_fwd = make_scored_hit(score=50.0, raw_score=50.0, chromosome="chr1", global_start=100, strand=Strand.FORWARD)

        ranker = CandidateRanker()
        result = ranker.rank([hit_rev, hit_fwd])
        assert result.hits[0] == hit_fwd
        assert result.hits[1] == hit_rev


class TestRankingFilters:
    """Tests multi-criteria filtering during ranking."""

    def test_filter_min_score(self):
        """Filters out candidates with score below min_score cutoff."""
        h1 = make_scored_hit(score=90.0, raw_score=90.0)
        h2 = make_scored_hit(score=40.0, raw_score=40.0)
        h3 = make_scored_hit(score=10.0, raw_score=10.0)

        ranker = CandidateRanker()
        result = ranker.rank([h1, h2, h3], config=RankingConfig(min_score=50.0))
        assert len(result) == 1
        assert result.hits[0] == h1
        assert result.total_candidates == 3
        assert result.retained_count == 1

    def test_filter_max_mismatches_and_seed_mismatches(self):
        """Filters candidates exceeding maximum mismatch thresholds."""
        h1 = make_scored_hit(score=80.0, raw_score=80.0, mismatch_count=1, seed_mismatches=0)
        h2 = make_scored_hit(score=60.0, raw_score=60.0, mismatch_count=2, seed_mismatches=1)
        h3 = make_scored_hit(score=40.0, raw_score=40.0, mismatch_count=4, seed_mismatches=3)

        ranker = CandidateRanker()
        res_mm = ranker.rank([h1, h2, h3], config=RankingConfig(max_mismatches=2))
        assert len(res_mm) == 2
        assert res_mm.hits == (h1, h2)

        res_seed = ranker.rank([h1, h2, h3], config=RankingConfig(max_seed_mismatches=0))
        assert len(res_seed) == 1
        assert res_seed.hits == (h1,)

    def test_filter_pam_compatibility(self):
        """Filters candidates based on PAM compatibility and canonical requirements."""
        h_canonical = make_scored_hit(score=100.0, raw_score=100.0, is_valid_pam=True, is_canonical_pam=True)
        h_alternate = make_scored_hit(score=25.0, raw_score=100.0, is_valid_pam=True, is_canonical_pam=False)
        h_invalid = make_scored_hit(score=0.0, raw_score=100.0, is_valid_pam=False, is_canonical_pam=False)

        ranker = CandidateRanker()
        # require_valid_pam should retain canonical and alternate, drop invalid
        res_valid = ranker.rank([h_canonical, h_alternate, h_invalid], config=RankingConfig(require_valid_pam=True))
        assert len(res_valid) == 2
        assert h_invalid not in res_valid.hits

        # canonical_pam_only should only retain canonical
        res_canonical = ranker.rank([h_canonical, h_alternate, h_invalid], config=RankingConfig(canonical_pam_only=True))
        assert len(res_canonical) == 1
        assert res_canonical.hits == (h_canonical,)

    def test_limit_top_k(self):
        """Limits results to top_k highest risk candidates."""
        hits = [make_scored_hit(score=float(i), raw_score=float(i), global_start=i*10) for i in range(10)]
        ranker = CandidateRanker()
        result = ranker.rank(hits, config=RankingConfig(top_k=3))
        assert len(result) == 3
        assert [h.score for h in result.hits] == [9.0, 8.0, 7.0]

    def test_invalid_ranking_config_raises_value_error(self):
        """Invalid threshold parameters raise ValueError."""
        with pytest.raises(ValueError, match="min_score"):
            RankingConfig(min_score=-1.0)
        with pytest.raises(ValueError, match="min_score"):
            RankingConfig(min_score=105.0)
        with pytest.raises(ValueError, match="max_mismatches"):
            RankingConfig(max_mismatches=-1)
        with pytest.raises(ValueError, match="top_k"):
            RankingConfig(top_k=0)


class TestLargeCollectionAndMetadataPreservation:
    """Tests scalability and metadata preservation on large synthetic batches."""

    def test_large_candidate_collection_sorting(self):
        """Ranks 1,000 synthetic candidates quickly and deterministically."""
        hits = [
            make_scored_hit(
                score=float(i % 100),
                raw_score=float(i % 100),
                chromosome=f"chr{(i % 5) + 1}",
                global_start=i * 50,
                mismatch_count=(i % 4),
            )
            for i in range(1000)
        ]
        ranker = CandidateRanker()
        result = ranker.rank(hits)
        assert len(result) == 1000

        # Verify strict descending order
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_and_properties_preserved(self):
        """Candidate properties, coordinates, strand, and PAM metadata remain intact."""
        hit = make_scored_hit(
            score=77.5,
            raw_score=77.5,
            chromosome="NC_000001.11",
            global_start=15000,
            strand=Strand.FORWARD,
            mismatch_count=1,
            seed_mismatches=0,
            is_valid_pam=True,
            is_canonical_pam=True,
            pam_sequence="GGG",
        )
        ranker = CandidateRanker()
        result = ranker.rank([hit])
        ranked = result.hits[0]

        assert ranked.chromosome == "NC_000001.11"
        assert ranked.strand == Strand.FORWARD
        assert ranked.protospacer_start == 15000
        assert ranked.protospacer_end == 15020
        assert ranked.pam_start == 15020
        assert ranked.pam_end == 15023
        assert ranked.pam_sequence == "GGG"
        assert ranked.is_valid_pam is True
        assert ranked.is_canonical_pam is True
        assert ranked.score == 77.5

        # Group by chromosome helper
        by_chr = result.by_chromosome()
        assert "NC_000001.11" in by_chr
        assert len(by_chr["NC_000001.11"]) == 1


class TestEndToEndAlignmentIntegration:
    """Tests end-to-end workflow from CPUAligner through PAM validation, scoring, and ranking."""

    def test_cpu_aligner_align_validate_and_rank(self):
        """Executes full pipeline on synthetic chunk containing compatible and incompatible hits."""
        aligner = CPUAligner()
        guide_seq = "GAGTCCGAGCAGAAGAAGAA" + "CGG"  # 23 nt GuideTarget string
        guide = GuideTarget.from_string(guide_seq)

        # Build chunk containing:
        # 1. Exact match with CGG (score 100.0) at start
        # 2. Match with 1 distal mismatch at pos 3 and CGG at index 40
        # 3. Match with AGA (incompatible PAM) at index 80
        flank = "A" * 20
        site1 = guide_seq  # 20 nt + CGG
        site2 = guide_seq[:2] + "T" + guide_seq[3:]  # pos 3 mismatch + CGG
        site3 = guide_seq[:20] + "AGA"  # 0 mm + AGA PAM
        chunk_seq = flank + site1 + flank + site2 + flank + site3 + flank

        chunk = ChunkDescriptor(
            chunk_id="chunk_test",
            chromosome="NC_000001.11",
            chunk_index=0,
            global_start=1000,
            global_end=1000 + len(chunk_seq),
            actual_length=len(chunk_seq),
            core_start=0,
            core_end=len(chunk_seq),
            halo_start=len(chunk_seq),
            halo_end=len(chunk_seq),
            sequence_data=chunk_seq,
        )

        ranking_result: CandidateRankingResult = aligner.align_validate_and_rank(
            chunk=chunk,
            guide=guide,
            filter_incompatible=False,
        )

        assert isinstance(ranking_result, CandidateRankingResult)
        assert len(ranking_result) >= 3

        # Exact match with canonical PAM should be rank 0 with score 100.0
        top_hit = ranking_result.hits[0]
        assert top_hit.is_exact_match is True
        assert top_hit.is_canonical_pam is True
        assert top_hit.score == 100.0

        # Incompatible PAM hit should have score 0.0 (penalized)
        incompatible_hits = [h for h in ranking_result.hits if not h.is_valid_pam]
        assert len(incompatible_hits) >= 1
        assert incompatible_hits[0].score == 0.0
