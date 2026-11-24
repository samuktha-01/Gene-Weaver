"""End-to-end CPU sequence processing and off-target alignment pipeline for GeneWeaver.

Coordinates streaming FASTA parsing, genomic chunk partitioning, dual-strand vectorized
sliding-window alignment, strand-aware PAM validation, guide-relative mismatch analysis,
Hsu-Zhang empirical scoring, deterministic candidate ranking, and result export.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from geneweaver.alignment.cpu_aligner import CPUAligner
from geneweaver.alignment.models import AlignmentConfig, GuideTarget, Strand
from geneweaver.genomic_parser.exceptions import FastaParsingError, InvalidSequenceError
from geneweaver.genomic_parser.fasta_reader import FastaReader
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.genomic_parser.validator import SequenceValidator
from geneweaver.result_manager.exporter import ResultExporter
from geneweaver.scoring.candidate_ranker import (
    CandidateRanker,
    CandidateRankingResult,
    RankingConfig,
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
    NucleaseConfig,
    PAMValidator,
    SPCAS9,
    ValidatedCandidateHit,
)
from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker

logger = logging.getLogger("geneweaver.pipeline.cpu")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration options for the end-to-end GeneWeaver CPU alignment pipeline.

    Attributes:
        fasta_path: Path to the reference FASTA file (.fa, .fasta, .fna, .gz).
        guide_sequence: 20 nt spacer or 23 nt guide (spacer + PAM) sequence.
        nuclease: Nuclease configuration object or name ('SpCas9', 'SaCas9', 'Cas12a').
        max_mismatches: Maximum allowable mismatches along the 20 nt protospacer (default: 4).
        max_seed_mismatches: Maximum allowable mismatches within the seed region (default: 2).
        min_score: Minimum off-target score cutoff (0.0 to 100.0). If None, no score cutoff is applied.
        canonical_pam_only: If True, only candidates matching the primary canonical PAM are retained.
        search_reverse_strand: If True, searches both Forward (+) and Reverse (-) strands (default: True).
        chunk_size: Nominal core length per genomic chunk in base pairs (default: 10,000,000 bp).
        halo_size: Trailing overlap halo in base pairs (default: 22 bp).
        batch_size: Processing batch size for vectorized sliding windows (default: 100,000).
        top_k: Limit ranking results to top K highest-risk candidates.
        scoring_model: Name of mismatch scoring model ('hsu_zhang' or 'experimental_positional').
        filter_incompatible_pam: If True, candidate hits with invalid PAMs are removed before scoring.
        output_csv: Optional file path to export ranked results as CSV.
        output_json: Optional file path to export ranked results and run metadata as JSON.
    """

    fasta_path: Union[Path, str]
    guide_sequence: str
    nuclease: Union[str, NucleaseConfig] = "SpCas9"
    max_mismatches: int = 4
    max_seed_mismatches: Optional[int] = 2
    min_score: Optional[float] = None
    canonical_pam_only: bool = False
    search_reverse_strand: bool = True
    chunk_size: int = 10_000_000
    halo_size: int = 22
    batch_size: int = 100_000
    top_k: Optional[int] = None
    scoring_model: str = "hsu_zhang"
    filter_incompatible_pam: bool = False
    target_chromosome: Optional[str] = None
    target_start: Optional[int] = None
    target_strand: Optional[Union[str, Strand]] = None
    output_csv: Optional[Union[Path, str]] = None
    output_json: Optional[Union[Path, str]] = None

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if self.max_mismatches < 0 or self.max_mismatches > 20:
            raise ValueError(
                f"max_mismatches ({self.max_mismatches}) must be between 0 and 20"
            )
        if self.max_seed_mismatches is not None:
            if self.max_seed_mismatches < 0 or self.max_seed_mismatches > self.max_mismatches:
                raise ValueError(
                    f"max_seed_mismatches ({self.max_seed_mismatches}) must be between 0 and max_mismatches ({self.max_mismatches})"
                )
        if self.min_score is not None and (self.min_score < 0.0 or self.min_score > 100.0):
            raise ValueError(f"min_score ({self.min_score}) must be between 0.0 and 100.0")
        if self.chunk_size <= self.halo_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be greater than halo_size ({self.halo_size})"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k ({self.top_k}) must be positive")


@dataclass
class PipelineStatistics:
    """Execution telemetry and diagnostic performance metrics for a pipeline run."""

    total_records: int = 0
    total_chunks: int = 0
    total_base_pairs: int = 0
    total_windows_evaluated: int = 0
    raw_candidates_detected: int = 0
    pam_compatible_candidates: int = 0
    pam_incompatible_candidates: int = 0
    total_scored_candidates: int = 0
    retained_candidates: int = 0
    on_target_candidates: int = 0
    off_target_candidates: int = 0
    mismatch_distribution: Dict[int, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts statistics into a serializable dictionary."""
        return {
            "total_records": self.total_records,
            "total_chunks": self.total_chunks,
            "total_base_pairs": self.total_base_pairs,
            "total_windows_evaluated": self.total_windows_evaluated,
            "raw_candidates_detected": self.raw_candidates_detected,
            "pam_compatible_candidates": self.pam_compatible_candidates,
            "pam_incompatible_candidates": self.pam_incompatible_candidates,
            "total_scored_candidates": self.total_scored_candidates,
            "retained_candidates": self.retained_candidates,
            "on_target_candidates": self.on_target_candidates,
            "off_target_candidates": self.off_target_candidates,
            "mismatch_distribution": {str(k): v for k, v in sorted(self.mismatch_distribution.items())},
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "throughput_bp_per_sec": (
                round(self.total_base_pairs / self.elapsed_seconds, 2)
                if self.elapsed_seconds > 0
                else 0.0
            ),
        }


@dataclass(frozen=True)
class PipelineResult:
    """Complete output container for the GeneWeaver CPU alignment pipeline.

    Attributes:
        ranking_result: CandidateRankingResult containing ranked ScoredCandidateHit instances.
        statistics: PipelineStatistics telemetric metrics.
        config: PipelineConfig configuration used for this execution.
        csv_path: Resolved path to the exported CSV file if requested.
        json_path: Resolved path to the exported JSON file if requested.
        guide_target: Validated GuideTarget instance.
    """

    ranking_result: CandidateRankingResult
    statistics: PipelineStatistics
    config: PipelineConfig
    csv_path: Optional[Path] = None
    json_path: Optional[Path] = None
    guide_target: Optional[GuideTarget] = None

    @property
    def hits(self) -> Tuple[ScoredCandidateHit, ...]:
        """Convenience accessor for ranked hits."""
        return self.ranking_result.hits

    def __len__(self) -> int:
        return len(self.ranking_result)

    def summary(self) -> Dict[str, Any]:
        """Returns run summary combining ranking results and telemetry statistics."""
        summary_data = self.statistics.to_dict()
        summary_data.update(self.ranking_result.summary())
        return summary_data


class CPUPipeline:
    """Executes the complete end-to-end CPU sequence processing and off-target search pipeline.

    Orchestrates streaming FASTA reading, chunk partitioning with halo deduplication,
    dual-strand vectorized Hamming-distance alignment, strand-aware PAM validation,
    guide-relative mismatch analysis, Hsu-Zhang (2013) scoring, deterministic ranking,
    and output serialization.
    """

    def __init__(self, aligner: Optional[CPUAligner] = None) -> None:
        """Initializes the CPUPipeline.

        Args:
            aligner: Optional custom CPUAligner instance (defaults to standard CPUAligner).
        """
        self._aligner = aligner or CPUAligner()

    def run(
        self,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """Executes the pipeline according to the provided configuration.

        Args:
            config: PipelineConfig defining input files, thresholds, and outputs.
            progress_callback: Optional callback accepting (status_message, percentage_complete_float).

        Returns:
            PipelineResult containing ranked candidate hits, performance statistics, and file paths.

        Raises:
            FileNotFoundError: If the input FASTA file does not exist.
            InvalidSequenceError: If the guide or sequences contain invalid characters.
            FastaParsingError: If FASTA parsing fails.
        """
        tracemalloc.start()
        start_time = time.perf_counter()
        stats = PipelineStatistics()

        def _report(msg: str, pct: float = 0.0) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg, pct)

        _report("Validating input configuration and sequences...", 0.0)

        # 1. Validate inputs and resolve guide target
        fasta_path = SequenceValidator.validate_fasta_path(config.fasta_path)
        guide_target = self._resolve_guide_target(config.guide_sequence)
        nuclease_cfg = self._resolve_nuclease_config(config.nuclease)

        align_config = AlignmentConfig(
            max_mismatches=config.max_mismatches,
            seed_max_mismatches=config.max_seed_mismatches or config.max_mismatches,
            seed_length=10,
            canonical_pam_only=config.canonical_pam_only,
            search_reverse_strand=config.search_reverse_strand,
            batch_size=config.batch_size,
        )

        pam_validator = PAMValidator(config=nuclease_cfg)
        analyzer = MismatchAnalyzer(seed_start=11, seed_end=20, spacer_length=20)
        scorer = self._resolve_scorer(config.scoring_model)
        scoring_cfg = MismatchScoringConfig(
            model_name=config.scoring_model,
            apply_pam_multiplier=True,
            penalize_incompatible_pam=True,
            incompatible_pam_score=0.0,
        )

        ranker = CandidateRanker()
        ranking_cfg = RankingConfig(
            min_score=config.min_score,
            max_mismatches=config.max_mismatches,
            max_seed_mismatches=config.max_seed_mismatches,
            require_valid_pam=config.filter_incompatible_pam,
            canonical_pam_only=config.canonical_pam_only,
            top_k=config.top_k,
        )

        reader = FastaReader(file_path=fasta_path)
        chunker = GenomicChunker(chunk_size=config.chunk_size, halo_size=config.halo_size)

        all_scored_hits: List[ScoredCandidateHit] = []
        fasta_records_map: Dict[str, FastaRecord] = {}

        _report("Streaming genomic records...", 5.0)

        # 2. Stream records and process chunk by chunk
        for record in reader.stream_records():
            stats.total_records += 1
            stats.total_base_pairs += record.length
            fasta_records_map[record.identifier] = record

            _report(
                f"Processing record '{record.identifier}' (length: {record.length:,} bp)...",
                10.0,
            )

            for chunk in chunker.stream_chunks(record):
                stats.total_chunks += 1

                # Align candidates across chunk (both strands)
                align_result = self._aligner.align(
                    chunk=chunk,
                    guide=guide_target,
                    config=align_config,
                )
                stats.total_windows_evaluated += align_result.total_windows_evaluated
                raw_hits = align_result.hits
                stats.raw_candidates_detected += len(raw_hits)

                for r in raw_hits:
                    stats.mismatch_distribution[r.mismatch_count] = (
                        stats.mismatch_distribution.get(r.mismatch_count, 0) + 1
                    )

                if not raw_hits:
                    continue

                # Validate PAM sequences
                val_hits = pam_validator.validate_candidates(
                    raw_hits,
                    canonical_only=config.canonical_pam_only,
                    filter_incompatible=config.filter_incompatible_pam,
                )

                for v in val_hits:
                    if v.is_valid_pam:
                        stats.pam_compatible_candidates += 1
                    else:
                        stats.pam_incompatible_candidates += 1

                if not val_hits:
                    continue

                # Analyze mismatches
                analyses = analyzer.analyze_batch(val_hits, guide=guide_target)

                # Score mismatches
                pairs = list(zip(val_hits, analyses))
                chunk_scored = scorer.score_candidates(pairs, config=scoring_cfg)
                stats.total_scored_candidates += len(chunk_scored)

                all_scored_hits.extend(chunk_scored)

        # 3. Classify on-target vs off-target candidates
        classified_hits = self._classify_candidates(
            all_scored_hits,
            config=config,
            fasta_records=fasta_records_map,
        )

        _report(
            f"Ranking {len(classified_hits):,} accumulated candidate hits...",
            85.0,
        )

        # 4. Deterministic candidate ranking
        ranking_result = ranker.rank(classified_hits, config=ranking_cfg)
        stats.retained_candidates = ranking_result.retained_count
        stats.on_target_candidates = sum(1 for h in ranking_result.hits if h.is_on_target)
        stats.off_target_candidates = sum(1 for h in ranking_result.hits if not h.is_on_target)

        # 5. Measure elapsed time and peak memory
        stats.elapsed_seconds = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats.peak_memory_mb = peak_mem / (1024 * 1024)

        # 6. Export results
        csv_path = None
        if config.output_csv:
            _report(f"Exporting results to CSV: {config.output_csv}", 90.0)
            csv_path = ResultExporter.export_csv(
                results=ranking_result,
                output_path=config.output_csv,
                fasta_records=fasta_records_map,
            )

        json_path = None
        if config.output_json:
            _report(f"Exporting results to JSON: {config.output_json}", 95.0)
            run_metadata = {
                "pipeline": "GeneWeaver CPU Alignment Pipeline",
                "version": "0.1.0",
                "fasta_path": str(fasta_path),
                "guide_sequence": config.guide_sequence,
                "spacer_sequence": guide_target.spacer,
                "nuclease": nuclease_cfg.name,
                "scoring_model": config.scoring_model,
                "search_reverse_strand": config.search_reverse_strand,
                "target_chromosome": config.target_chromosome,
                "target_start": config.target_start,
                "target_strand": str(config.target_strand) if config.target_strand else None,
            }
            json_path = ResultExporter.export_json(
                results=ranking_result,
                output_path=config.output_json,
                metadata=run_metadata,
                statistics=stats.to_dict(),
                fasta_records=fasta_records_map,
            )

        _report("Pipeline execution complete.", 100.0)

        return PipelineResult(
            ranking_result=ranking_result,
            statistics=stats,
            config=config,
            csv_path=csv_path,
            json_path=json_path,
            guide_target=guide_target,
        )

    def _resolve_guide_target(self, guide_input: str) -> GuideTarget:
        """Validates and normalizes guide input into a 23 nt GuideTarget.

        Accepts:
        - 20 nt spacer string: appends placeholder canonical PAM 'AGG' to enable 23 nt alignment.
        - 23 nt guide string (spacer + PAM): validates standard 23 nt guide.
        """
        clean = guide_input.strip().upper().replace("U", "T")
        if len(clean) == 20:
            # 20 nt spacer provided: validate spacer and append default canonical SpCas9 PAM
            validated_spacer = SequenceValidator.validate_guide(clean, expected_len=20)
            return GuideTarget.from_string(validated_spacer + "AGG")
        elif len(clean) == 23:
            # Full 23 nt guide (20 nt spacer + 3 nt PAM) provided
            return GuideTarget.from_string(clean)
        else:
            raise ValueError(
                f"Guide sequence length must be 20 nt (spacer only) or 23 nt (spacer + PAM), got {len(clean)} nt"
            )

    @staticmethod
    def _resolve_nuclease_config(nuclease: Union[str, NucleaseConfig]) -> NucleaseConfig:
        """Resolves nuclease configuration from string identifier or object."""
        if isinstance(nuclease, NucleaseConfig):
            return nuclease
        if isinstance(nuclease, str):
            clean = nuclease.strip().lower()
            if clean in ("spcas9", "cas9"):
                return SPCAS9
            raise ValueError(
                f"Unsupported nuclease system '{nuclease}'. Supported: 'SpCas9'"
            )
        raise TypeError(f"Invalid nuclease configuration type: {type(nuclease)}")

    @staticmethod
    def _resolve_scorer(model_name: str) -> BaseMismatchScorer:
        """Resolves mismatch scoring engine instance by model identifier."""
        clean = model_name.strip().lower()
        if clean in ("hsu_zhang", "hsu", "mit"):
            return HsuZhangScorer()
        elif clean in ("experimental_positional", "experimental", "linear"):
            return ExperimentalPositionalScorer()
        else:
            raise ValueError(
                f"Unsupported scoring model '{model_name}'. Supported: 'hsu_zhang', 'experimental_positional'"
            )

    @staticmethod
    def _classify_candidates(
        hits: Sequence[ScoredCandidateHit],
        config: PipelineConfig,
        fasta_records: Dict[str, FastaRecord],
    ) -> List[ScoredCandidateHit]:
        """Classifies candidate sites as ON_TARGET or OFF_TARGET.

        Classification logic:
        1. If user provides explicit target locus coordinates (target_start, and optionally
           target_chromosome and target_strand):
           - Matches on chromosome (case-insensitive substring or accession),
             start coordinate (genomic 1-based or region-relative), and strand (if given)
             are labeled 'ON_TARGET' with basis 'user_specified_coordinates'.
           - All other candidates are labeled 'OFF_TARGET' with basis 'non_target_locus'.

        2. If target locus coordinates are NOT provided:
           - Candidates with 0 mismatches, valid PAM, and score == 100.0:
             - Exactly 1 unique match: labeled 'ON_TARGET' with basis 'inferred_unique_exact_match'.
             - Multiple matches: labeled 'ON_TARGET' with basis 'inferred_multiple_exact_matches'
               (with a logged warning).
           - All candidates with mismatches > 0 or non-canonical/imperfect PAM are labeled
             'OFF_TARGET' with basis 'off_target_mismatches'.
        """
        if not hits:
            return []

        user_specified = config.target_start is not None
        classified: List[ScoredCandidateHit] = []

        if user_specified:
            target_chr = (config.target_chromosome or "").strip().lower()
            target_start = config.target_start
            target_strand: Optional[Strand] = None
            if config.target_strand is not None:
                if isinstance(config.target_strand, Strand):
                    target_strand = config.target_strand
                else:
                    target_strand = Strand.from_symbol(str(config.target_strand))

            for hit in hits:
                record = fasta_records.get(hit.chromosome)
                coords = ResultExporter.calculate_genomic_coordinates(hit, record)
                gen_start = coords["genomic_start"]
                proto_start = hit.protospacer_start

                chr_match = True
                if target_chr:
                    hit_chr = hit.chromosome.lower()
                    chr_match = (
                        target_chr == hit_chr
                        or target_chr in hit_chr
                        or hit_chr in target_chr
                    )

                coord_match = (
                    target_start == gen_start
                    or target_start == proto_start
                    or target_start == proto_start + 1
                )

                strand_match = True
                if target_strand is not None:
                    strand_match = (hit.strand == target_strand)

                if chr_match and coord_match and strand_match:
                    classified.append(
                        hit.with_classification(
                            site_type="ON_TARGET",
                            is_on_target=True,
                            classification_basis="user_specified_coordinates",
                        )
                    )
                else:
                    classified.append(
                        hit.with_classification(
                            site_type="OFF_TARGET",
                            is_on_target=False,
                            classification_basis="non_target_locus",
                        )
                    )
        else:
            # Inferred classification
            exact_hits = [
                h for h in hits
                if h.mismatch_count == 0 and h.is_valid_pam and h.score >= 99.99
            ]

            if len(exact_hits) == 1:
                exact_hit_id = id(exact_hits[0])
                for hit in hits:
                    if id(hit) == exact_hit_id:
                        classified.append(
                            hit.with_classification(
                                site_type="ON_TARGET",
                                is_on_target=True,
                                classification_basis="inferred_unique_exact_match",
                            )
                        )
                    else:
                        classified.append(
                            hit.with_classification(
                                site_type="OFF_TARGET",
                                is_on_target=False,
                                classification_basis="off_target_mismatches",
                            )
                        )
            elif len(exact_hits) > 1:
                logger.warning(
                    f"Found {len(exact_hits)} exact 0-mismatch candidate matches without user-specified target coordinates. "
                    "All exact matches are tagged as ON_TARGET ('inferred_multiple_exact_matches'). "
                    "Supply --target-chr and --target-start to resolve the intended on-target locus unambiguously."
                )
                exact_hit_ids = {id(h) for h in exact_hits}
                for hit in hits:
                    if id(hit) in exact_hit_ids:
                        classified.append(
                            hit.with_classification(
                                site_type="ON_TARGET",
                                is_on_target=True,
                                classification_basis="inferred_multiple_exact_matches",
                            )
                        )
                    else:
                        classified.append(
                            hit.with_classification(
                                site_type="OFF_TARGET",
                                is_on_target=False,
                                classification_basis="off_target_mismatches",
                            )
                        )
            else:
                for hit in hits:
                    classified.append(
                        hit.with_classification(
                            site_type="OFF_TARGET",
                            is_on_target=False,
                            classification_basis="off_target_mismatches",
                        )
                    )

        return classified
