"""Instrumented pipeline execution engine for granular stage profiling.

Executes the GeneWeaver CRISPR CPU processing workflow while measuring high-resolution
wall-clock durations across FASTA reading, chunk partitioning, forward alignment,
reverse-strand alignment, PAM validation, mismatch scoring, candidate ranking, and export.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from benchmarks.models import (
    BenchmarkRunResult,
    MemoryMetrics,
    PipelineCounts,
    StageTimings,
    ThroughputMetrics,
)
from benchmarks.system_profiler import MemoryTracker, SystemProfiler
from geneweaver.alignment.cpu_aligner import CPUAligner
from geneweaver.alignment.models import AlignmentConfig, GuideTarget, Strand
from geneweaver.alignment.reverse_complement import reverse_complement
from geneweaver.genomic_parser.fasta_reader import FastaReader
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.genomic_parser.validator import SequenceValidator
from geneweaver.pipeline.cpu_pipeline import CPUPipeline, PipelineConfig
from geneweaver.result_manager.exporter import ResultExporter
from geneweaver.scoring.candidate_ranker import CandidateRanker, RankingConfig
from geneweaver.scoring.mismatch_analyzer import MismatchAnalyzer
from geneweaver.scoring.mismatch_scorer import (
    HsuZhangScorer,
    MismatchScoringConfig,
    ScoredCandidateHit,
)
from geneweaver.scoring.pam_evaluator import PAMValidator, SPCAS9
from geneweaver.sequence_processor.chunker import GenomicChunker


class PipelineProfiler:
    """Profiles the GeneWeaver CPU alignment pipeline with granular stage instrumentation."""

    def __init__(self, aligner: Optional[CPUAligner] = None) -> None:
        self.aligner = aligner or CPUAligner()

    def profile_run(
        self,
        config: PipelineConfig,
        run_id: int = 1,
        is_warmup: bool = False,
        dataset_name: str = "custom",
    ) -> BenchmarkRunResult:
        """Executes a fully instrumented single run of the pipeline.

        Measures individual stage durations, throughput rates, memory usage,
        and element counts while maintaining full scientific correctness.

        Args:
            config: PipelineConfig with dataset path, guide, thresholds, and outputs.
            run_id: Numeric run identifier.
            is_warmup: Whether this run is a warm-up iteration.
            dataset_name: Informational dataset tag.

        Returns:
            BenchmarkRunResult containing complete performance telemetry.
        """
        # Timing accumulators (seconds)
        t_fasta_parsing = 0.0
        t_chunking = 0.0
        t_fwd_align = 0.0
        t_rev_align = 0.0
        t_pam_val = 0.0
        t_scoring = 0.0
        t_ranking = 0.0
        t_export = 0.0

        counts = {
            "total_base_pairs": 0,
            "total_chunks": 0,
            "total_windows": 0,
            "raw_candidates": 0,
            "pam_compatible_candidates": 0,
            "retained_candidates": 0,
            "on_target_candidates": 0,
            "off_target_candidates": 0,
            "mismatch_distribution": {},
        }

        # Initialize pipeline engines
        fasta_path = SequenceValidator.validate_fasta_path(config.fasta_path)
        guide_target = CPUPipeline()._resolve_guide_target(config.guide_sequence)
        nuclease_cfg = CPUPipeline._resolve_nuclease_config(config.nuclease)

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
        scorer = HsuZhangScorer()
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

        chunker = GenomicChunker(chunk_size=config.chunk_size, halo_size=config.halo_size)
        reader = FastaReader(file_path=fasta_path)

        all_scored_hits: List[ScoredCandidateHit] = []
        fasta_records_map: Dict[str, FastaRecord] = {}

        # Guide parameters for alignment
        guide_spacer = guide_target.spacer if guide_target.spacer else guide_target.sequence
        guide_len = len(guide_spacer)
        guide_encoded = guide_target.encoded_2bit[:guide_len]
        powers = (1 << np.arange(guide_len, dtype=np.uint32))

        # Memory tracking
        with MemoryTracker() as mem_tracker:
            t_total_start = time.perf_counter()

            # 1. Stream FASTA records
            t0 = time.perf_counter()
            records_gen = reader.stream_records()
            t_fasta_parsing += time.perf_counter() - t0

            for record in records_gen:
                counts["total_base_pairs"] += record.length
                fasta_records_map[record.identifier] = record

                # 2. Partition into chunks
                t0 = time.perf_counter()
                chunks = list(chunker.stream_chunks(record))
                t_chunking += time.perf_counter() - t0
                counts["total_chunks"] += len(chunks)

                for chunk in chunks:
                    # Resolve sequence
                    seq_2bit, seq_str = self.aligner._resolve_chunk_sequence(chunk.sequence_data)
                    seq_len = seq_2bit.size
                    num_windows = max(0, seq_len - guide_len + 1)
                    counts["total_windows"] += num_windows

                    # 3. Forward Strand Alignment
                    t0 = time.perf_counter()
                    fwd_hits = self.aligner._scan_single_strand(
                        seq_2bit=seq_2bit,
                        seq_str=seq_str,
                        guide_encoded=guide_encoded,
                        guide_len=guide_len,
                        strand=Strand.FORWARD,
                        chunk=chunk,
                        effective_config=align_config,
                        powers=powers,
                    )
                    t_fwd_align += time.perf_counter() - t0

                    chunk_hits = list(fwd_hits)

                    # 4. Reverse Strand Alignment
                    if align_config.search_reverse_strand:
                        t0 = time.perf_counter()
                        rc_seq_2bit = np.ascontiguousarray((seq_2bit ^ 3)[::-1])
                        rc_seq_str = reverse_complement(seq_str) if seq_str is not None else None
                        rev_hits = self.aligner._scan_single_strand(
                            seq_2bit=rc_seq_2bit,
                            seq_str=rc_seq_str,
                            guide_encoded=guide_encoded,
                            guide_len=guide_len,
                            strand=Strand.REVERSE,
                            chunk=chunk,
                            effective_config=align_config,
                            powers=powers,
                        )
                        t_rev_align += time.perf_counter() - t0
                        chunk_hits.extend(rev_hits)

                    counts["raw_candidates"] += len(chunk_hits)
                    for h in chunk_hits:
                        mm_c = str(h.mismatch_count)
                        counts["mismatch_distribution"][mm_c] = (
                            counts["mismatch_distribution"].get(mm_c, 0) + 1
                        )

                    if not chunk_hits:
                        continue

                    # 5. PAM Validation
                    t0 = time.perf_counter()
                    val_hits = pam_validator.validate_candidates(
                        chunk_hits,
                        canonical_only=config.canonical_pam_only,
                        filter_incompatible=config.filter_incompatible_pam,
                    )
                    t_pam_val += time.perf_counter() - t0

                    for v in val_hits:
                        if v.is_valid_pam:
                            counts["pam_compatible_candidates"] += 1

                    if not val_hits:
                        continue

                    # 6. Mismatch Analysis and Scoring
                    t0 = time.perf_counter()
                    analyses = analyzer.analyze_batch(val_hits, guide=guide_target)
                    pairs = list(zip(val_hits, analyses))
                    chunk_scored = scorer.score_candidates(pairs, config=scoring_cfg)
                    t_scoring += time.perf_counter() - t0

                    all_scored_hits.extend(chunk_scored)

            # 7. Classification and Deterministic Ranking
            t0 = time.perf_counter()
            classified_hits = CPUPipeline._classify_candidates(
                all_scored_hits,
                config=config,
                fasta_records=fasta_records_map,
            )
            ranking_result = ranker.rank(classified_hits, config=ranking_cfg)
            t_ranking += time.perf_counter() - t0

            counts["retained_candidates"] = ranking_result.retained_count
            counts["on_target_candidates"] = sum(1 for h in ranking_result.hits if h.is_on_target)
            counts["off_target_candidates"] = sum(1 for h in ranking_result.hits if not h.is_on_target)

            # 8. Export Results if requested
            if config.output_csv or config.output_json:
                t0 = time.perf_counter()
                if config.output_csv:
                    ResultExporter.export_csv(
                        results=ranking_result,
                        output_path=config.output_csv,
                        fasta_records=fasta_records_map,
                    )
                if config.output_json:
                    ResultExporter.export_json(
                        results=ranking_result,
                        output_path=config.output_json,
                        fasta_records=fasta_records_map,
                    )
                t_export += time.perf_counter() - t0

            t_total_execution = time.perf_counter() - t_total_start

        # Collect candidate signatures (fingerprint for correctness comparison across chunk sizes)
        signatures = tuple(
            f"{h.chromosome}:{h.strand.sign}:{h.protospacer_start}:{h.pam_sequence}:{h.mismatch_count}:{h.score:.2f}"
            for h in ranking_result.hits[:25]
        )

        timings = StageTimings(
            fasta_parsing_s=t_fasta_parsing,
            chunking_s=t_chunking,
            forward_alignment_s=t_fwd_align,
            reverse_alignment_s=t_rev_align,
            pam_validation_s=t_pam_val,
            mismatch_analysis_and_scoring_s=t_scoring,
            ranking_and_classification_s=t_ranking,
            export_s=t_export,
            total_execution_s=t_total_execution,
        )

        throughput = ThroughputMetrics.calculate(
            total_bp=counts["total_base_pairs"],
            total_windows=counts["total_windows"],
            raw_candidates=counts["raw_candidates"],
            elapsed_seconds=t_total_execution,
        )

        mem_metrics = mem_tracker.get_metrics()

        pipe_counts = PipelineCounts(
            total_base_pairs=counts["total_base_pairs"],
            total_chunks=counts["total_chunks"],
            total_windows=counts["total_windows"],
            raw_candidates=counts["raw_candidates"],
            pam_compatible_candidates=counts["pam_compatible_candidates"],
            retained_candidates=counts["retained_candidates"],
            on_target_candidates=counts["on_target_candidates"],
            off_target_candidates=counts["off_target_candidates"],
            mismatch_distribution=counts["mismatch_distribution"],
        )

        return BenchmarkRunResult(
            run_id=run_id,
            is_warmup=is_warmup,
            dataset_name=dataset_name,
            chunk_size=config.chunk_size,
            max_mismatches=config.max_mismatches,
            batch_size=config.batch_size,
            search_reverse_strand=config.search_reverse_strand,
            timings=timings,
            throughput=throughput,
            memory=mem_metrics,
            counts=pipe_counts,
            candidate_signatures=signatures,
        )
