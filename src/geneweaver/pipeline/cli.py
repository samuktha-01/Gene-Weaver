"""Command-line interface (CLI) for the GeneWeaver end-to-end CPU alignment pipeline.

Provides a unified terminal entry point for executing genomic off-target searches,
validating inputs, displaying real-time progress, formatting tabular results,
and exporting structured CSV/JSON reports.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

from geneweaver.pipeline.cpu_pipeline import CPUPipeline, PipelineConfig, PipelineResult


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses and validates command-line arguments for the CPU pipeline."""
    parser = argparse.ArgumentParser(
        prog="geneweaver-cpu",
        description="GeneWeaver CPU Pipeline: End-to-end CRISPR off-target alignment, PAM validation, and ranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search both strands of a FASTA reference with standard SpCas9 parameters
  geneweaver-cpu -f reference.fa -g GAGTCCGAGCAGAAGAAGAA -o results.csv

  # Run with 23 nt guide (spacer + PAM), score threshold, and JSON export
  geneweaver-cpu -f reference.fa.gz -g GAGTCCGAGCAGAAGAAGAAAGG --min-score 5.0 -j report.json

  # Restrict to canonical NGG PAM and top 50 highest-risk sites
  geneweaver-cpu -f chr1.fa -g GAGTCCGAGCAGAAGAAGAA --canonical-pam-only -k 50 -o top50.csv
        """,
    )

    # Required parameters
    parser.add_argument(
        "--fasta",
        "-f",
        required=True,
        type=str,
        help="Path to the reference FASTA file (.fa, .fasta, .fna, or .gz compressed).",
    )
    parser.add_argument(
        "--guide",
        "-g",
        required=True,
        type=str,
        help="CRISPR guide RNA sequence (20 nt spacer or 23 nt spacer+PAM). RNA 'U' is normalized to 'T'.",
    )

    # Output parameters
    parser.add_argument(
        "--output-csv",
        "-o",
        type=str,
        default=None,
        help="Output destination path for CSV results file.",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        type=str,
        default=None,
        help="Output destination path for structured JSON results file.",
    )

    # Alignment & Filter parameters
    parser.add_argument(
        "--nuclease",
        type=str,
        default="SpCas9",
        help="Nuclease configuration system (default: 'SpCas9').",
    )
    parser.add_argument(
        "--max-mismatches",
        "-m",
        type=int,
        default=4,
        help="Maximum allowable mismatches along the 20 nt protospacer (default: 4).",
    )
    parser.add_argument(
        "--max-seed-mismatches",
        type=int,
        default=2,
        help="Maximum allowable mismatches in the PAM-proximal seed region (default: 2).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum numerical score threshold (0.0 to 100.0) for retained candidates.",
    )
    parser.add_argument(
        "--canonical-pam-only",
        action="store_true",
        help="If set, retains only sites with the primary canonical PAM motif (e.g. NGG).",
    )
    parser.add_argument(
        "--single-strand",
        action="store_true",
        help="If set, searches only the forward (+) strand (default searches both strands).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Limit the final ranked output to the top K highest-risk candidate sites.",
    )
    parser.add_argument(
        "--scoring-model",
        type=str,
        default="hsu_zhang",
        choices=["hsu_zhang", "experimental_positional"],
        help="Mismatch scoring model (default: 'hsu_zhang', Hsu et al. Nature Biotech 2013).",
    )

    # Chunking & Performance parameters
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000_000,
        help="Genomic chunk core size in base pairs (default: 10,000,000 bp / 10 Mbp).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Processing batch size for vectorized sliding-window matrix operations (default: 100,000).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed diagnostic logging output.",
    )

    return parser.parse_args(args)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats columns and rows into a clean ASCII table."""
    if not rows:
        return "No candidate off-target sites found."

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


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI execution routine."""
    parsed = parse_arguments(args)

    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 76)
    print("  GeneWeaver CPU Alignment and Off-Target Scoring Engine")
    print("=" * 76)
    print(f" Reference FASTA:  {parsed.fasta}")
    print(f" Guide Sequence:   {parsed.guide}")
    print(f" Nuclease System:  {parsed.nuclease}")
    print(f" Max Mismatches:   {parsed.max_mismatches} (Seed Max: {parsed.max_seed_mismatches})")
    print(f" Strand Search:    {'Forward (+) only' if parsed.single_strand else 'Dual-strand (+ and -)'}")
    print(f" Scoring Model:    {parsed.scoring_model}")
    print("=" * 76)
    print()

    config = PipelineConfig(
        fasta_path=parsed.fasta,
        guide_sequence=parsed.guide,
        nuclease=parsed.nuclease,
        max_mismatches=parsed.max_mismatches,
        max_seed_mismatches=parsed.max_seed_mismatches,
        min_score=parsed.min_score,
        canonical_pam_only=parsed.canonical_pam_only,
        search_reverse_strand=not parsed.single_strand,
        chunk_size=parsed.chunk_size,
        batch_size=parsed.batch_size,
        top_k=parsed.top_k,
        scoring_model=parsed.scoring_model,
        output_csv=parsed.output_csv,
        output_json=parsed.output_json,
    )

    pipeline = CPUPipeline()

    def progress_callback(msg: str, pct: float):
        print(f" [{pct:5.1f}%] {msg}")

    try:
        result: PipelineResult = pipeline.run(config, progress_callback=progress_callback)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline execution failed: {exc}", file=sys.stderr)
        if parsed.verbose:
            import traceback
            traceback.print_exc()
        return 1

    stats = result.statistics

    print()
    print("-" * 76)
    print("  Execution Telemetry & Summary Statistics")
    print("-" * 76)
    print(f" Total Records Processed:    {stats.total_records:,}")
    print(f" Total Chunks Evaluated:     {stats.total_chunks:,}")
    print(f" Total Base Pairs Scanned:   {stats.total_base_pairs:,} bp")
    print(f" Sliding Windows Evaluated:  {stats.total_windows_evaluated:,}")
    print(f" Raw Candidates Detected:    {stats.raw_candidates_detected:,}")
    print(f" PAM-Compatible Candidates:  {stats.pam_compatible_candidates:,}")
    print(f" Retained Off-Target Sites:  {stats.retained_candidates:,}")
    print(f" Wall-Clock Execution Time:  {stats.elapsed_seconds:.2f} seconds")
    if stats.elapsed_seconds > 0:
        throughput = stats.total_base_pairs / stats.elapsed_seconds
        print(f" Overall Scan Throughput:    {throughput:,.0f} bp/sec ({throughput/1e6:.2f} Mbp/sec)")
    print(f" Peak Process Memory Usage:  {stats.peak_memory_mb:.2f} MB")
    print("-" * 76)
    print()

    # Display Top 10 Ranked Hits in Table Format
    display_k = min(10, len(result.hits))
    print(f"Top {display_k} Off-Target Candidates (Sorted by Predicted Risk):")
    print()

    headers = [
        "Rank",
        "Score",
        "Chr",
        "Strand",
        "Start",
        "PAM",
        "Valid",
        "MM",
        "Seed",
        "Positions",
        "Target Sequence",
    ]

    table_rows = []
    for rank_idx, hit in enumerate(result.hits[:display_k], start=1):
        pos_str = ";".join(str(p) for p in hit.mismatch_positions) or "-"
        table_rows.append(
            [
                str(rank_idx),
                f"{hit.score:6.2f}",
                hit.chromosome[:16],
                hit.strand.sign,
                f"{hit.protospacer_start:,}",
                hit.pam_sequence,
                "Yes" if hit.is_valid_pam else "No",
                str(hit.mismatch_count),
                str(hit.seed_mismatches),
                pos_str,
                hit.target_sequence[:23],
            ]
        )

    print(format_table(headers, table_rows))
    print()

    # Output files confirmation
    if result.csv_path:
        print(f" [OK] Exported CSV results to:  {result.csv_path}")
    if result.json_path:
        print(f" [OK] Exported JSON report to: {result.json_path}")

    print()
    print("=" * 76)
    print("  [NOTE] Computational off-target scores reflect sequence similarity and")
    print("  empirical positional penalties (Hsu et al., 2013). They do not constitute")
    print("  guaranteed experimental cleavage confirmation.")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
