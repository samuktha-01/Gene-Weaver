"""Data models and telemetry containers for GeneWeaver CPU performance benchmarking.

Provides typed dataclasses for stage execution timings, throughput rates,
memory consumption metrics, pipeline counts, repeated run aggregation,
environment metadata, and comprehensive benchmark suite reports.
"""

from dataclasses import asdict, dataclass, field
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class EnvironmentInfo:
    """System hardware and software environment specifications."""

    python_version: str = sys.version.split()[0]
    python_compiler: str = platform.python_compiler()
    os_system: str = platform.system()
    os_release: str = platform.release()
    os_machine: str = platform.machine()
    processor: str = platform.processor() or "Unknown"
    cpu_count_logical: int = 1
    cpu_count_physical: Optional[int] = None
    total_ram_gb: float = 0.0
    numpy_version: str = ""
    geneweaver_version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Converts environment details to a serializable dictionary."""
        return {
            "python_version": self.python_version,
            "python_compiler": self.python_compiler,
            "os_system": self.os_system,
            "os_release": self.os_release,
            "os_machine": self.os_machine,
            "processor": self.processor,
            "cpu_count_logical": self.cpu_count_logical,
            "cpu_count_physical": self.cpu_count_physical,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "numpy_version": self.numpy_version,
            "geneweaver_version": self.geneweaver_version,
        }


@dataclass(frozen=True)
class StageTimings:
    """Execution timings in seconds across granular pipeline processing phases."""

    fasta_parsing_s: float = 0.0
    chunking_s: float = 0.0
    forward_alignment_s: float = 0.0
    reverse_alignment_s: float = 0.0
    pam_validation_s: float = 0.0
    mismatch_analysis_and_scoring_s: float = 0.0
    ranking_and_classification_s: float = 0.0
    export_s: float = 0.0
    total_execution_s: float = 0.0

    @property
    def total_alignment_s(self) -> float:
        """Combined alignment time across both strands."""
        return self.forward_alignment_s + self.reverse_alignment_s

    def to_dict(self) -> Dict[str, float]:
        """Converts stage timings to a dictionary rounded to 6 decimal places."""
        return {
            "fasta_parsing_s": round(self.fasta_parsing_s, 6),
            "chunking_s": round(self.chunking_s, 6),
            "forward_alignment_s": round(self.forward_alignment_s, 6),
            "reverse_alignment_s": round(self.reverse_alignment_s, 6),
            "total_alignment_s": round(self.total_alignment_s, 6),
            "pam_validation_s": round(self.pam_validation_s, 6),
            "mismatch_analysis_and_scoring_s": round(self.mismatch_analysis_and_scoring_s, 6),
            "ranking_and_classification_s": round(self.ranking_and_classification_s, 6),
            "export_s": round(self.export_s, 6),
            "total_execution_s": round(self.total_execution_s, 6),
        }


@dataclass(frozen=True)
class ThroughputMetrics:
    """Throughput rate measurements based on total bases and windows evaluated."""

    bp_per_second: float = 0.0
    mbp_per_second: float = 0.0
    windows_per_second: float = 0.0
    candidates_per_second: float = 0.0

    @classmethod
    def calculate(
        cls,
        total_bp: int,
        total_windows: int,
        raw_candidates: int,
        elapsed_seconds: float,
    ) -> "ThroughputMetrics":
        """Calculates throughput metrics from raw counts and elapsed duration."""
        if elapsed_seconds <= 0.0:
            return cls()

        bp_sec = total_bp / elapsed_seconds
        mbp_sec = bp_sec / 1_000_000.0
        win_sec = total_windows / elapsed_seconds
        cand_sec = raw_candidates / elapsed_seconds

        return cls(
            bp_per_second=bp_sec,
            mbp_per_second=mbp_sec,
            windows_per_second=win_sec,
            candidates_per_second=cand_sec,
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts throughput metrics to a serializable dictionary."""
        return {
            "bp_per_second": round(self.bp_per_second, 2),
            "mbp_per_second": round(self.mbp_per_second, 4),
            "windows_per_second": round(self.windows_per_second, 2),
            "candidates_per_second": round(self.candidates_per_second, 2),
        }


@dataclass(frozen=True)
class MemoryMetrics:
    """Memory consumption measurements distinguishing Python from total OS process memory."""

    python_peak_memory_mb: float = 0.0
    process_memory_rss_mb: float = 0.0
    process_memory_vms_mb: float = 0.0
    measurement_method: str = "tracemalloc"

    def to_dict(self) -> Dict[str, Any]:
        """Converts memory metrics to a serializable dictionary."""
        return {
            "python_peak_memory_mb": round(self.python_peak_memory_mb, 2),
            "process_memory_rss_mb": round(self.process_memory_rss_mb, 2),
            "process_memory_vms_mb": round(self.process_memory_vms_mb, 2),
            "measurement_method": self.measurement_method,
        }


@dataclass(frozen=True)
class PipelineCounts:
    """Summary of genomic elements, windows, and candidate sites processed."""

    total_base_pairs: int = 0
    total_chunks: int = 0
    total_windows: int = 0
    raw_candidates: int = 0
    pam_compatible_candidates: int = 0
    retained_candidates: int = 0
    on_target_candidates: int = 0
    off_target_candidates: int = 0
    mismatch_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts pipeline element counts to a dictionary."""
        return {
            "total_base_pairs": self.total_base_pairs,
            "total_chunks": self.total_chunks,
            "total_windows": self.total_windows,
            "raw_candidates": self.raw_candidates,
            "pam_compatible_candidates": self.pam_compatible_candidates,
            "retained_candidates": self.retained_candidates,
            "on_target_candidates": self.on_target_candidates,
            "off_target_candidates": self.off_target_candidates,
            "mismatch_distribution": self.mismatch_distribution,
        }


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Complete diagnostic results container for an individual benchmark run iteration."""

    run_id: int
    is_warmup: bool
    dataset_name: str
    chunk_size: int
    max_mismatches: int
    batch_size: int
    search_reverse_strand: bool
    timings: StageTimings
    throughput: ThroughputMetrics
    memory: MemoryMetrics
    counts: PipelineCounts
    candidate_signatures: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Converts individual run metrics to a serializable dictionary."""
        return {
            "run_id": self.run_id,
            "is_warmup": self.is_warmup,
            "dataset_name": self.dataset_name,
            "chunk_size": self.chunk_size,
            "max_mismatches": self.max_mismatches,
            "batch_size": self.batch_size,
            "search_reverse_strand": self.search_reverse_strand,
            "timings": self.timings.to_dict(),
            "throughput": self.throughput.to_dict(),
            "memory": self.memory.to_dict(),
            "counts": self.counts.to_dict(),
            "candidate_signatures_count": len(self.candidate_signatures),
        }


@dataclass(frozen=True)
class MetricStats:
    """Summary statistics (median, mean, stddev, min, max, IQR) for repeated metrics."""

    median: float
    mean: float
    stddev: float
    min_val: float
    max_val: float
    iqr: float

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricStats":
        """Calculates statistical summary from a non-empty sequence of floats."""
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        import numpy as np

        arr = np.array(values, dtype=np.float64)
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        stddev = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = float(q75 - q25)

        return cls(
            median=round(median, 6),
            mean=round(mean, 6),
            stddev=round(stddev, 6),
            min_val=round(min_val, 6),
            max_val=round(max_val, 6),
            iqr=round(iqr, 6),
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts summary statistics to a dictionary."""
        return {
            "median": self.median,
            "mean": self.mean,
            "stddev": self.stddev,
            "min": self.min_val,
            "max": self.max_val,
            "iqr": self.iqr,
        }


@dataclass(frozen=True)
class AggregatedBenchmarkResult:
    """Statistical aggregation across multiple repeated runs for a specific configuration."""

    dataset_name: str
    chunk_size: int
    max_mismatches: int
    batch_size: int
    search_reverse_strand: bool
    iterations_measured: int
    total_execution_stats: MetricStats
    forward_alignment_stats: MetricStats
    reverse_alignment_stats: MetricStats
    total_alignment_stats: MetricStats
    pam_validation_stats: MetricStats
    scoring_stats: MetricStats
    throughput_mbp_stats: MetricStats
    throughput_windows_stats: MetricStats
    python_peak_memory_mb: float
    process_memory_rss_mb: float
    counts: PipelineCounts
    correctness_verified: bool = True
    correctness_notes: str = "Verified"

    def to_dict(self) -> Dict[str, Any]:
        """Converts aggregated results to a serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "chunk_size": self.chunk_size,
            "max_mismatches": self.max_mismatches,
            "batch_size": self.batch_size,
            "search_reverse_strand": self.search_reverse_strand,
            "iterations_measured": self.iterations_measured,
            "total_execution_time_s": self.total_execution_stats.to_dict(),
            "forward_alignment_s": self.forward_alignment_stats.to_dict(),
            "reverse_alignment_s": self.reverse_alignment_stats.to_dict(),
            "total_alignment_s": self.total_alignment_stats.to_dict(),
            "pam_validation_s": self.pam_validation_stats.to_dict(),
            "mismatch_scoring_s": self.scoring_stats.to_dict(),
            "throughput_mbp_sec": self.throughput_mbp_stats.to_dict(),
            "throughput_windows_sec": self.throughput_windows_stats.to_dict(),
            "python_peak_memory_mb": round(self.python_peak_memory_mb, 2),
            "process_memory_rss_mb": round(self.process_memory_rss_mb, 2),
            "counts": self.counts.to_dict(),
            "correctness_verified": self.correctness_verified,
            "correctness_notes": self.correctness_notes,
        }


@dataclass
class BenchmarkSuiteReport:
    """Comprehensive container for an entire benchmarking session output."""

    benchmark_title: str
    timestamp_utc: str
    environment: EnvironmentInfo
    aggregated_results: List[AggregatedBenchmarkResult] = field(default_factory=list)
    raw_runs: List[BenchmarkRunResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts complete benchmark suite report to serializable format."""
        return {
            "benchmark_title": self.benchmark_title,
            "timestamp_utc": self.timestamp_utc,
            "environment": self.environment.to_dict(),
            "aggregated_results": [r.to_dict() for r in self.aggregated_results],
            "raw_runs": [r.to_dict() for r in self.raw_runs],
        }
