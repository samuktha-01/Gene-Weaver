"""Unit tests for the GeneWeaver CPU performance benchmarking suite.

Verifies:
1. Metric and throughput calculations.
2. Repeated-run statistical aggregation (mean, median, stddev, IQR).
3. Memory measurement behavior with psutil and fallback handling.
4. Deterministic synthetic sequence generation and reproducibility with fixed seeds.
5. Benchmark runner execution, cross-chunk-size correctness verification, and CSV/JSON export.
"""

import csv
import json
from pathlib import Path
import tempfile
import pytest

from benchmarks.cpu_benchmark import CPUBenchmarkRunner
from benchmarks.models import (
    AggregatedBenchmarkResult,
    BenchmarkRunResult,
    EnvironmentInfo,
    MemoryMetrics,
    MetricStats,
    PipelineCounts,
    StageTimings,
    ThroughputMetrics,
)
from benchmarks.synthetic_generator import SyntheticGenomicGenerator
from benchmarks.system_profiler import MemoryTracker, SystemProfiler
from geneweaver.genomic_parser.fasta_reader import FastaReader


class TestMetricCalculations:
    """Verifies timing, throughput, and rate calculations."""

    def test_throughput_calculation_positive(self):
        """Verifies throughput metrics for valid execution durations."""
        throughput = ThroughputMetrics.calculate(
            total_bp=10_000_000,
            total_windows=9_999_981,
            raw_candidates=50,
            elapsed_seconds=2.0,
        )
        assert throughput.bp_per_second == 5_000_000.0
        assert throughput.mbp_per_second == 5.0
        assert throughput.windows_per_second == 4_999_990.5
        assert throughput.candidates_per_second == 25.0

    def test_throughput_calculation_zero_duration(self):
        """Zero duration safely produces zeroed throughput rates without ZeroDivisionError."""
        throughput = ThroughputMetrics.calculate(
            total_bp=1_000,
            total_windows=980,
            raw_candidates=5,
            elapsed_seconds=0.0,
        )
        assert throughput.bp_per_second == 0.0
        assert throughput.mbp_per_second == 0.0
        assert throughput.windows_per_second == 0.0

    def test_stage_timings_total_alignment(self):
        """Verifies total alignment time combines forward and reverse strands."""
        timings = StageTimings(
            forward_alignment_s=1.25,
            reverse_alignment_s=1.75,
            pam_validation_s=0.10,
            total_execution_s=3.50,
        )
        assert timings.total_alignment_s == 3.0
        d = timings.to_dict()
        assert d["total_alignment_s"] == 3.0
        assert d["forward_alignment_s"] == 1.25


class TestStatisticalAggregation:
    """Verifies statistical metric summaries for repeated benchmark runs."""

    def test_metric_stats_odd_elements(self):
        """Calculates median, mean, min, max, stddev, and IQR on odd number of elements."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = MetricStats.from_values(values)

        assert stats.median == 30.0
        assert stats.mean == 30.0
        assert stats.min_val == 10.0
        assert stats.max_val == 50.0
        assert stats.iqr == 20.0  # 40 - 20
        assert stats.stddev > 0.0

    def test_metric_stats_even_elements(self):
        """Calculates median and IQR on even number of elements."""
        values = [1.0, 2.0, 3.0, 4.0]
        stats = MetricStats.from_values(values)

        assert stats.median == 2.5
        assert stats.mean == 2.5
        assert stats.min_val == 1.0
        assert stats.max_val == 4.0

    def test_metric_stats_single_element(self):
        """Single element returns zero stddev and IQR."""
        stats = MetricStats.from_values([42.5])
        assert stats.median == 42.5
        assert stats.mean == 42.5
        assert stats.stddev == 0.0
        assert stats.iqr == 0.0

    def test_metric_stats_empty_list(self):
        """Empty sequence returns safe zeroed metrics."""
        stats = MetricStats.from_values([])
        assert stats.median == 0.0
        assert stats.mean == 0.0
        assert stats.stddev == 0.0


class TestMemoryTracking:
    """Verifies memory tracking behavior and optional dependency fallbacks."""

    def test_memory_tracker_context(self):
        """MemoryTracker tracks non-zero Python memory allocations."""
        with MemoryTracker() as tracker:
            # Allocate a temporary buffer
            buf = [b"A" * 1024 for _ in range(500)]
            assert len(buf) == 500

        metrics = tracker.get_metrics()
        assert metrics.python_peak_memory_mb > 0.0
        assert isinstance(metrics.measurement_method, str)

    def test_system_profiler_environment(self):
        """SystemProfiler discovers host architecture, CPU cores, and Python details."""
        env = SystemProfiler.get_environment_info()
        assert env.python_version != ""
        assert env.cpu_count_logical >= 1
        assert env.os_system != ""
        d = env.to_dict()
        assert "python_version" in d
        assert "cpu_count_logical" in d

    def test_fallback_when_psutil_unavailable(self, monkeypatch):
        """Verifies graceful fallback behavior when psutil is simulated as unavailable."""
        monkeypatch.setattr(SystemProfiler, "_has_psutil", False)
        rss, vms, method = SystemProfiler.get_process_memory_mb()
        assert rss >= 0.0
        assert vms >= 0.0
        assert isinstance(method, str)


class TestSyntheticGenerator:
    """Verifies deterministic synthetic sequence generation and reproducibility."""

    def test_reproducibility_with_fixed_seed(self):
        """Identical random seeds generate 100% identical nucleotide sequences."""
        seq1 = SyntheticGenomicGenerator.generate_sequence(length_bp=10_000, seed=123)
        seq2 = SyntheticGenomicGenerator.generate_sequence(length_bp=10_000, seed=123)
        assert seq1 == seq2

    def test_divergence_with_different_seeds(self):
        """Different seeds generate distinct nucleotide sequences."""
        seq1 = SyntheticGenomicGenerator.generate_sequence(length_bp=1_000, seed=10)
        seq2 = SyntheticGenomicGenerator.generate_sequence(length_bp=1_000, seed=20)
        assert seq1 != seq2

    def test_synthetic_fasta_creation_and_parsing(self, tmp_path):
        """Generates synthetic FASTA file and verifies parsing through FastaReader."""
        fasta_path = tmp_path / "test_synth.fa"
        guide_spacer = "GCTTATCCAGTCTGCTACGC"

        SyntheticGenomicGenerator.create_benchmark_fasta(
            output_path=fasta_path,
            length_bp=50_000,
            guide_spacer=guide_spacer,
            seed=42,
            header_identifier="synth_chr1",
        )

        assert fasta_path.exists()
        reader = FastaReader(file_path=fasta_path)
        records = list(reader.stream_records())
        assert len(records) == 1
        rec = records[0]
        assert rec.identifier == "synth_chr1"
        assert rec.length == 50_000

        # Verify embedded forward on-target sequence exists
        assert (guide_spacer + "TGG") in rec.sequence


class TestBenchmarkRunner:
    """Verifies automated benchmarking execution, chunk scaling, and report generation."""

    def test_benchmark_matrix_and_correctness_across_chunk_sizes(self, tmp_path):
        """Verifies benchmark execution across multiple chunk sizes yields identical candidate hits."""
        fasta_path = tmp_path / "benchmark_test.fa"
        guide = "GCTTATCCAGTCTGCTACGC"

        SyntheticGenomicGenerator.create_benchmark_fasta(
            output_path=fasta_path,
            length_bp=60_000,
            guide_spacer=guide,
            seed=42,
        )

        runner = CPUBenchmarkRunner()
        out_dir = tmp_path / "outputs"

        report = runner.run_benchmark_matrix(
            fasta_path=fasta_path,
            guide_sequence=guide,
            chunk_sizes=[25_000, 50_000],
            mismatch_thresholds=[3],
            iterations=2,
            warmup_runs=1,
            batch_size=20_000,
            dataset_name="unit_test_dataset",
            output_dir=out_dir,
        )

        # 2 configurations (25k, 50k chunk sizes)
        assert len(report.aggregated_results) == 2

        # Verify correctness: biological results must be identical across chunk sizes
        agg25 = report.aggregated_results[0]
        agg50 = report.aggregated_results[1]

        assert agg25.correctness_verified is True
        assert agg50.correctness_verified is True
        assert agg25.counts.retained_candidates == agg50.counts.retained_candidates
        assert agg25.counts.on_target_candidates == agg50.counts.on_target_candidates
        assert agg25.counts.raw_candidates == agg50.counts.raw_candidates

        # Verify exported files exist
        csv_files = list(out_dir.glob("*.csv"))
        json_files = list(out_dir.glob("*.json"))
        assert len(csv_files) == 1
        assert len(json_files) == 1

        # Check JSON report structure
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert "benchmark_title" in data
        assert "environment" in data
        assert "aggregated_results" in data
        assert "raw_runs" in data

        # Check CSV report structure
        lines = csv_files[0].read_text(encoding="utf-8").strip().splitlines()
        reader = csv.DictReader(lines)
        rows = list(reader)
        assert len(rows) == 2
        assert "chunk_size_bp" in rows[0]
        assert "median_total_time_s" in rows[0]
        assert "correctness_verified" in rows[0]
