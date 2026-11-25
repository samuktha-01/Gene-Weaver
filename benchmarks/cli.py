"""Command-line interface (CLI) for the GeneWeaver CPU performance benchmarking suite.

Provides terminal entry points for running automated benchmarks across synthetic datasets
and real GRCh38 genomic regions, measuring stage timings, memory, and throughput,
verifying candidate correctness across chunk sizes, and exporting CSV/JSON reports.
"""

import argparse
import logging
from pathlib import Path
import sys
import tempfile
from typing import List, Optional

from benchmarks.cpu_benchmark import CPUBenchmarkRunner
from benchmarks.synthetic_generator import SyntheticGenomicGenerator
from benchmarks.system_profiler import SystemProfiler


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line options for CPU benchmarking."""
    parser = argparse.ArgumentParser(
        prog="geneweaver-benchmark",
        description="GeneWeaver CPU Benchmarking Suite: Measure throughput, memory, and scalability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark on small synthetic sequence (100k bp)
  python -m benchmarks.cli --dataset synthetic_small --iterations 3

  # Benchmark on medium synthetic sequence (1M bp) across multiple chunk sizes
  python -m benchmarks.cli --dataset synthetic_medium --chunk-sizes 100000 250000 500000 1000000

  # Benchmark on real GRCh38 chromosome 1 region
  python -m benchmarks.cli --dataset real_chr1 --chunk-sizes 1000000 5000000 10000000 --iterations 2
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic_small",
        choices=["synthetic_small", "synthetic_medium", "real_chr1", "custom"],
        help="Dataset profile to benchmark (default: 'synthetic_small').",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="Path to custom FASTA file (required if --dataset custom).",
    )
    parser.add_argument(
        "--guide",
        "-g",
        type=str,
        default="GCTTATCCAGTCTGCTACGC",
        help="CRISPR guide RNA sequence (default: 'GCTTATCCAGTCTGCTACGC').",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[100_000, 250_000, 500_000, 1_000_000],
        help="Sequence of chunk core sizes in base pairs to evaluate.",
    )
    parser.add_argument(
        "--mismatch-thresholds",
        type=int,
        nargs="+",
        default=[3],
        help="Sequence of maximum mismatch thresholds to evaluate (e.g. 2 3 4).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of timed benchmark repetitions per configuration (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of untimed warm-up iterations (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Sliding-window matrix evaluation batch size (default: 100,000).",
    )
    parser.add_argument(
        "--single-strand",
        action="store_true",
        help="Search forward strand only (default searches both strands).",
    )
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=None,
        help="Custom base-pair length for synthetic dataset generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save exported benchmark CSV and JSON reports (default: 'outputs').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic sequence generation (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed diagnostic logging output.",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main execution routine for the benchmarking CLI."""
    parsed = parse_arguments(args)

    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    env = SystemProfiler.get_environment_info()

    print("=" * 80)
    print("  GeneWeaver CPU Pipeline Performance & Scalability Benchmarking Suite")
    print("=" * 80)
    print(f" Platform:       {env.os_system} {env.os_release} ({env.os_machine})")
    print(f" Python:         {env.python_version} ({env.python_compiler})")
    print(f" Processor:      {env.processor}")
    print(f" CPU Cores:      {env.cpu_count_logical} logical" + (f", {env.cpu_count_physical} physical" if env.cpu_count_physical else ""))
    print(f" Total RAM:      {env.total_ram_gb:.1f} GB" if env.total_ram_gb > 0 else " Total RAM:      Unknown")
    print(f" NumPy Version:  {env.numpy_version}")
    print("-" * 80)
    print(f" Dataset Target: {parsed.dataset}")
    print(f" Guide Sequence: {parsed.guide}")
    print(f" Chunk Sizes:    {', '.join(f'{sz:,} bp' for sz in parsed.chunk_sizes)}")
    print(f" Mismatch Thresh:{', '.join(str(m) for m in parsed.mismatch_thresholds)}")
    print(f" Iterations:     {parsed.iterations} measured (+ {parsed.warmup} warm-up)")
    print(f" Strand Search:  {'Forward strand only' if parsed.single_strand else 'Dual-strand (+ and -)'}")
    print("=" * 80)
    print()

    # Resolve dataset path
    temp_dir_obj = None
    target_fasta: Path
    dataset_tag = parsed.dataset

    if parsed.dataset == "synthetic_small":
        size = parsed.synthetic_size or 100_000
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gw_bench_")
        target_fasta = Path(temp_dir_obj.name) / "synthetic_small.fa"
        print(f"Generating deterministic synthetic dataset ({size:,} bp, seed={parsed.seed})...")
        SyntheticGenomicGenerator.create_benchmark_fasta(
            output_path=target_fasta,
            length_bp=size,
            guide_spacer=parsed.guide[:20],
            seed=parsed.seed,
            header_identifier="synthetic_small_chr",
        )
        dataset_tag = f"synthetic_{size//1000}k"

    elif parsed.dataset == "synthetic_medium":
        size = parsed.synthetic_size or 1_000_000
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gw_bench_")
        target_fasta = Path(temp_dir_obj.name) / "synthetic_medium.fa"
        print(f"Generating deterministic synthetic dataset ({size:,} bp, seed={parsed.seed})...")
        SyntheticGenomicGenerator.create_benchmark_fasta(
            output_path=target_fasta,
            length_bp=size,
            guide_spacer=parsed.guide[:20],
            seed=parsed.seed,
            header_identifier="synthetic_medium_chr",
        )
        dataset_tag = f"synthetic_{size//1000}k"

    elif parsed.dataset == "real_chr1":
        candidate_paths = [
            Path("data/raw/GRCh38/chr1_region.fasta.fa"),
            Path("Gene-Weaver/data/raw/GRCh38/chr1_region.fasta.fa"),
        ]
        target_fasta = None
        for p in candidate_paths:
            if p.exists():
                target_fasta = p.resolve()
                break
        if target_fasta is None:
            print("[ERROR] Real reference dataset 'data/raw/GRCh38/chr1_region.fasta.fa' not found.", file=sys.stderr)
            return 1
        dataset_tag = "chr1_region"

    elif parsed.dataset == "custom":
        if not parsed.fasta:
            print("[ERROR] --fasta path must be specified when using --dataset custom", file=sys.stderr)
            return 1
        target_fasta = Path(parsed.fasta).resolve()
        if not target_fasta.exists():
            print(f"[ERROR] Custom FASTA file '{target_fasta}' does not exist.", file=sys.stderr)
            return 1
        dataset_tag = target_fasta.stem

    runner = CPUBenchmarkRunner()

    def progress_callback(msg: str) -> None:
        print(f"  {msg}")

    print("Executing benchmark matrix...\n")
    try:
        report = runner.run_benchmark_matrix(
            fasta_path=target_fasta,
            guide_sequence=parsed.guide,
            chunk_sizes=parsed.chunk_sizes,
            mismatch_thresholds=parsed.mismatch_thresholds,
            iterations=parsed.iterations,
            warmup_runs=parsed.warmup,
            batch_size=parsed.batch_size,
            search_reverse_strand=not parsed.single_strand,
            dataset_name=dataset_tag,
            output_dir=parsed.output_dir,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        print(f"\n[ERROR] Benchmark execution failed: {exc}", file=sys.stderr)
        if parsed.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if temp_dir_obj:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                pass

    print()
    print("=" * 80)
    print(f"  Benchmark Summary Results: {report.benchmark_title}")
    print("=" * 80)
    print(CPUBenchmarkRunner.format_summary_table(report))
    print("=" * 80)
    print()

    # Detailed stage breakdown for each aggregated configuration
    print("Granular Stage Timing Breakdown (Median Values):")
    print("-" * 80)
    for agg in report.aggregated_results:
        print(
            f" Chunk Size: {agg.chunk_size:,} bp (MM: {agg.max_mismatches}) | "
            f"Total: {agg.total_execution_stats.median:.3f} s ({agg.throughput_mbp_stats.median:.2f} Mbp/s)\n"
            f"   - Forward Alignment: {agg.forward_alignment_stats.median:.4f} s\n"
            f"   - Reverse Alignment: {agg.reverse_alignment_stats.median:.4f} s\n"
            f"   - PAM Validation:    {agg.pam_validation_stats.median:.4f} s\n"
            f"   - Mismatch Scoring:  {agg.scoring_stats.median:.4f} s\n"
            f"   - Peak Python RAM:   {agg.python_peak_memory_mb:.2f} MB\n"
            f"   - Process RSS RAM:   {agg.process_memory_rss_mb:.2f} MB\n"
            f"   - Correctness Check: {'PASSED (' + agg.correctness_notes + ')' if agg.correctness_verified else 'FAILED: ' + agg.correctness_notes}"
        )
        print()

    if parsed.output_dir:
        print(f"[OK] Benchmark CSV & JSON reports saved in: {Path(parsed.output_dir).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
