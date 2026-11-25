"""Reproducible CPU benchmarking orchestration engine for GeneWeaver.

Coordinates automated warm-up iterations, repeated benchmark runs, statistical aggregation,
chunk-size scaling experiments, mismatch-threshold comparisons, cross-configuration
correctness verification, and output serialization to CSV and JSON inside outputs/.
"""

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from benchmarks.models import (
    AggregatedBenchmarkResult,
    BenchmarkRunResult,
    BenchmarkSuiteReport,
    MetricStats,
    PipelineCounts,
)
from benchmarks.pipeline_profiler import PipelineProfiler
from benchmarks.synthetic_generator import SyntheticGenomicGenerator
from benchmarks.system_profiler import SystemProfiler
from geneweaver.pipeline.cpu_pipeline import PipelineConfig

logger = logging.getLogger("geneweaver.benchmarks")


class CPUBenchmarkRunner:
    """Orchestrates comprehensive CPU alignment and pipeline performance benchmarks."""

    def __init__(self, profiler: Optional[PipelineProfiler] = None) -> None:
        self.profiler = profiler or PipelineProfiler()

    def run_benchmark_matrix(
        self,
        fasta_path: Union[Path, str],
        guide_sequence: str = "GCTTATCCAGTCTGCTACGC",
        chunk_sizes: Sequence[int] = (100_000, 250_000, 500_000, 1_000_000),
        mismatch_thresholds: Sequence[int] = (3,),
        iterations: int = 3,
        warmup_runs: int = 1,
        batch_size: int = 100_000,
        search_reverse_strand: bool = True,
        dataset_name: str = "dataset",
        output_dir: Optional[Union[Path, str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> BenchmarkSuiteReport:
        """Executes a full matrix benchmark across chunk sizes and mismatch thresholds.

        Args:
            fasta_path: Path to reference FASTA file.
            guide_sequence: 20 nt spacer or 23 nt guide.
            chunk_sizes: Sequence of chunk core sizes in base pairs to evaluate.
            mismatch_thresholds: Sequence of max_mismatches thresholds to evaluate.
            iterations: Number of measured iterations per configuration.
            warmup_runs: Number of initial untimed/warm-up iterations.
            batch_size: Vectorized sliding window batch size.
            search_reverse_strand: Whether to search both strands.
            dataset_name: Identifier for the dataset being tested.
            output_dir: Directory to save CSV and JSON reports (default: 'outputs').
            progress_callback: Optional status message callback.

        Returns:
            BenchmarkSuiteReport containing all aggregated and individual run results.
        """
        def _log(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        env_info = SystemProfiler.get_environment_info()

        report = BenchmarkSuiteReport(
            benchmark_title=f"GeneWeaver CPU Benchmark - {dataset_name}",
            timestamp_utc=timestamp_str,
            environment=env_info,
        )

        all_raw_runs: List[BenchmarkRunResult] = []
        aggregated_results: List[AggregatedBenchmarkResult] = []

        total_configs = len(chunk_sizes) * len(mismatch_thresholds)
        config_idx = 0

        for max_mm in mismatch_thresholds:
            # Reference signatures for cross-chunk correctness verification scoped per mismatch threshold
            reference_signatures: Optional[Tuple[str, ...]] = None
            reference_counts: Optional[PipelineCounts] = None

            for chunk_sz in chunk_sizes:
                config_idx += 1
                _log(
                    f"[{config_idx}/{total_configs}] Benchmarking chunk_size={chunk_sz:,} bp, "
                    f"max_mismatches={max_mm} ({iterations} runs + {warmup_runs} warmup)..."
                )

                pipe_config = PipelineConfig(
                    fasta_path=fasta_path,
                    guide_sequence=guide_sequence,
                    max_mismatches=max_mm,
                    chunk_size=chunk_sz,
                    halo_size=22,
                    batch_size=batch_size,
                    search_reverse_strand=search_reverse_strand,
                )

                # 1. Warm-up runs
                for w_idx in range(warmup_runs):
                    _log(f"  Warm-up run {w_idx + 1}/{warmup_runs}...")
                    w_res = self.profiler.profile_run(
                        config=pipe_config,
                        run_id=w_idx,
                        is_warmup=True,
                        dataset_name=dataset_name,
                    )
                    all_raw_runs.append(w_res)

                # 2. Measured iterations
                measured_runs: List[BenchmarkRunResult] = []
                for it in range(1, iterations + 1):
                    _log(f"  Iteration {it}/{iterations}...")
                    m_res = self.profiler.profile_run(
                        config=pipe_config,
                        run_id=it,
                        is_warmup=False,
                        dataset_name=dataset_name,
                    )
                    measured_runs.append(m_res)
                    all_raw_runs.append(m_res)

                # 3. Correctness Verification
                # Check candidate counts and signatures against the reference run
                current_sigs = measured_runs[0].candidate_signatures
                current_counts = measured_runs[0].counts

                if reference_signatures is None:
                    # Establish reference from first evaluated configuration
                    reference_signatures = current_sigs
                    reference_counts = current_counts
                    correctness_verified = True
                    correctness_notes = "Established as baseline configuration"
                else:
                    counts_match = (
                        current_counts.retained_candidates == reference_counts.retained_candidates
                        and current_counts.on_target_candidates == reference_counts.on_target_candidates
                    )
                    sigs_match = (current_sigs == reference_signatures)

                    if counts_match and sigs_match:
                        correctness_verified = True
                        correctness_notes = (
                            f"Verified: identical {current_counts.retained_candidates} candidates "
                            "and coordinates across chunk sizes"
                        )
                    else:
                        correctness_verified = False
                        correctness_notes = (
                            f"Discrepancy: candidate count (got {current_counts.retained_candidates}, "
                            f"expected {reference_counts.retained_candidates}) or signature mismatch"
                        )
                        _log(f"  [WARNING] Correctness discrepancy detected: {correctness_notes}")

                # 4. Statistical aggregation across measured runs
                agg = self._aggregate_runs(
                    runs=measured_runs,
                    dataset_name=dataset_name,
                    chunk_size=chunk_sz,
                    max_mismatches=max_mm,
                    batch_size=batch_size,
                    search_reverse_strand=search_reverse_strand,
                    correctness_verified=correctness_verified,
                    correctness_notes=correctness_notes,
                )
                aggregated_results.append(agg)

        report.raw_runs = all_raw_runs
        report.aggregated_results = aggregated_results

        # Export benchmark artifacts if requested
        if output_dir:
            out_d = Path(output_dir).resolve()
            out_d.mkdir(parents=True, exist_ok=True)
            self.export_reports(report, out_d, dataset_name=dataset_name)

        return report

    @staticmethod
    def _aggregate_runs(
        runs: List[BenchmarkRunResult],
        dataset_name: str,
        chunk_size: int,
        max_mismatches: int,
        batch_size: int,
        search_reverse_strand: bool,
        correctness_verified: bool,
        correctness_notes: str,
    ) -> AggregatedBenchmarkResult:
        """Computes statistical metrics across repeated runs of one configuration."""
        total_times = [r.timings.total_execution_s for r in runs]
        fwd_times = [r.timings.forward_alignment_s for r in runs]
        rev_times = [r.timings.reverse_alignment_s for r in runs]
        align_times = [r.timings.total_alignment_s for r in runs]
        pam_times = [r.timings.pam_validation_s for r in runs]
        score_times = [r.timings.mismatch_analysis_and_scoring_s for r in runs]
        mbp_rates = [r.throughput.mbp_per_second for r in runs]
        win_rates = [r.throughput.windows_per_second for r in runs]

        peak_mems = [r.memory.python_peak_memory_mb for r in runs]
        rss_mems = [r.memory.process_memory_rss_mb for r in runs]

        return AggregatedBenchmarkResult(
            dataset_name=dataset_name,
            chunk_size=chunk_size,
            max_mismatches=max_mismatches,
            batch_size=batch_size,
            search_reverse_strand=search_reverse_strand,
            iterations_measured=len(runs),
            total_execution_stats=MetricStats.from_values(total_times),
            forward_alignment_stats=MetricStats.from_values(fwd_times),
            reverse_alignment_stats=MetricStats.from_values(rev_times),
            total_alignment_stats=MetricStats.from_values(align_times),
            pam_validation_stats=MetricStats.from_values(pam_times),
            scoring_stats=MetricStats.from_values(score_times),
            throughput_mbp_stats=MetricStats.from_values(mbp_rates),
            throughput_windows_stats=MetricStats.from_values(win_rates),
            python_peak_memory_mb=float(np.median(peak_mems)) if peak_mems else 0.0,
            process_memory_rss_mb=float(np.median(rss_mems)) if rss_mems else 0.0,
            counts=runs[0].counts,
            correctness_verified=correctness_verified,
            correctness_notes=correctness_notes,
        )

    @classmethod
    def export_reports(
        cls,
        report: BenchmarkSuiteReport,
        output_dir: Path,
        dataset_name: str,
    ) -> Tuple[Path, Path]:
        """Exports benchmark metrics to structured CSV and JSON files in outputs/."""
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"benchmark_{dataset_name}_{timestamp_slug}.csv"
        json_path = output_dir / f"benchmark_{dataset_name}_{timestamp_slug}.json"

        # 1. JSON Export
        with open(json_path, mode="wt", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)

        # 2. CSV Export (Tabular Aggregated Results)
        csv_rows = []
        for agg in report.aggregated_results:
            row = {
                "dataset": agg.dataset_name,
                "chunk_size_bp": agg.chunk_size,
                "max_mismatches": agg.max_mismatches,
                "iterations": agg.iterations_measured,
                "median_total_time_s": agg.total_execution_stats.median,
                "mean_total_time_s": agg.total_execution_stats.mean,
                "stddev_total_time_s": agg.total_execution_stats.stddev,
                "median_total_alignment_s": agg.total_alignment_stats.median,
                "median_forward_align_s": agg.forward_alignment_stats.median,
                "median_reverse_align_s": agg.reverse_alignment_stats.median,
                "median_pam_val_s": agg.pam_validation_stats.median,
                "median_scoring_s": agg.scoring_stats.median,
                "median_throughput_mbp_s": agg.throughput_mbp_stats.median,
                "median_windows_per_s": agg.throughput_windows_stats.median,
                "python_peak_mem_mb": agg.python_peak_memory_mb,
                "process_rss_mem_mb": agg.process_memory_rss_mb,
                "total_bp": agg.counts.total_base_pairs,
                "total_chunks": agg.counts.total_chunks,
                "raw_candidates": agg.counts.raw_candidates,
                "retained_candidates": agg.counts.retained_candidates,
                "on_target_candidates": agg.counts.on_target_candidates,
                "off_target_candidates": agg.counts.off_target_candidates,
                "correctness_verified": agg.correctness_verified,
            }
            csv_rows.append(row)

        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with open(csv_path, mode="wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        return csv_path, json_path

    @classmethod
    def format_summary_table(cls, report: BenchmarkSuiteReport) -> str:
        """Formats aggregated benchmark results into an ASCII table."""
        if not report.aggregated_results:
            return "No benchmark results to display."

        headers = [
            "Chunk Size",
            "MM",
            "Median Time",
            "StdDev",
            "Throughput",
            "Alignment",
            "PAM Val",
            "Scoring",
            "Peak PyRAM",
            "Process RSS",
            "Candidates",
            "Correct",
        ]

        rows = []
        for agg in report.aggregated_results:
            rows.append(
                [
                    f"{agg.chunk_size:,} bp",
                    str(agg.max_mismatches),
                    f"{agg.total_execution_stats.median:.3f} s",
                    f"±{agg.total_execution_stats.stddev:.3f} s",
                    f"{agg.throughput_mbp_stats.median:.2f} Mbp/s",
                    f"{agg.total_alignment_stats.median:.3f} s",
                    f"{agg.pam_validation_stats.median:.4f} s",
                    f"{agg.scoring_stats.median:.4f} s",
                    f"{agg.python_peak_memory_mb:.1f} MB",
                    f"{agg.process_memory_rss_mb:.1f} MB",
                    f"{agg.counts.retained_candidates}",
                    "YES" if agg.correctness_verified else "FAIL",
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for idx, val in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(str(val)))

        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        row_lines = [
            " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
            for row in rows
        ]

        return "\n".join([header_line, sep_line] + row_lines)
