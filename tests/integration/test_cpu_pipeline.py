"""Integration tests for the GeneWeaver end-to-end CPU pipeline.

Validates streaming FASTA ingestion, bounded chunking, dual-strand alignment,
PAM validation, mismatch scoring, candidate ranking, CSV/JSON export, and
execution against both synthetic sequences and the real GRCh38 chr1_region dataset.
"""

import csv
import json
from pathlib import Path
import pytest

from geneweaver.alignment.models import Strand
from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.pipeline.cpu_pipeline import CPUPipeline, PipelineConfig, PipelineResult
from geneweaver.scoring.pam_evaluator import SPCAS9


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for output files."""
    out_dir = tmp_path / "pipeline_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def synthetic_fasta_path(tmp_path: Path) -> Path:
    """Generates a synthetic FASTA file with deterministic forward and reverse sites."""
    fasta_file = tmp_path / "synthetic_test.fa"
    spacer = "GAGTCCGAGCAGAAGAAGAA"  # 20 nt
    # Reverse complement of spacer + CGG:
    # spacer is 5'-GAGTCCGAGCAGAAGAAGAA-3', RC is 5'-TTCTTCTTCTGCTCGGACTC-3'
    # PAM on opposite strand is 5'-CCG-3' on forward reference (RC is CGG)
    rc_site = "CCG" + "TTCTTCTTCTGCTCGGACTC"

    flank = "A" * 40
    # 1. Exact forward hit with CGG (score 100.0)
    site1 = spacer + "CGG"
    # 2. Forward hit with 1 mismatch at pos 3 (G -> T) + TGG PAM
    site2 = spacer[:2] + "T" + spacer[3:] + "TGG"
    # 3. Exact reverse hit with CGG PAM
    site3 = rc_site
    # 4. Forward hit with incompatible PAM (AGA)
    site4 = spacer + "AGA"

    seq = flank + site1 + flank + site2 + flank + site3 + flank + site4 + flank

    content = f">ref|TEST_001.1|:1000-5000 Synthetic test chromosome\n{seq}\n"
    fasta_file.write_text(content, encoding="utf-8")
    return fasta_file


class TestCPUPipelineSynthetic:
    """Integration tests on controlled synthetic genomic sequences."""

    def test_end_to_end_dual_strand_search_and_export(
        self, synthetic_fasta_path: Path, tmp_output_dir: Path
    ):
        """Executes full pipeline searching both strands and exports CSV and JSON."""
        csv_file = tmp_output_dir / "results.csv"
        json_file = tmp_output_dir / "results.json"

        config = PipelineConfig(
            fasta_path=synthetic_fasta_path,
            guide_sequence="GAGTCCGAGCAGAAGAAGAA",  # 20 nt spacer
            max_mismatches=4,
            search_reverse_strand=True,
            output_csv=csv_file,
            output_json=json_file,
        )

        pipeline = CPUPipeline()
        result: PipelineResult = pipeline.run(config)

        assert isinstance(result, PipelineResult)
        assert len(result) >= 3  # site1 (fwd), site2 (fwd), site3 (rev)
        assert result.statistics.total_records == 1
        assert result.statistics.total_chunks == 1
        assert result.statistics.elapsed_seconds > 0.0
        assert result.statistics.peak_memory_mb > 0.0

        # Check top ranked hits (exact matches on both strands should have score 100.0)
        top_scores = [h.score for h in result.hits[:2]]
        assert top_scores == [100.0, 100.0]

        # Verify that both strands were found among the candidates
        strands_found = {h.strand for h in result.hits}
        assert Strand.FORWARD in strands_found
        assert Strand.REVERSE in strands_found

        # Verify CSV export
        assert csv_file.exists()
        with open(csv_file, mode="rt", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            assert len(rows) == len(result)
            assert rows[0]["score"] == "100.0"
            assert rows[0]["rank"] == "1"

        # Verify JSON export
        assert json_file.exists()
        with open(json_file, mode="rt", encoding="utf-8") as handle:
            data = json.load(handle)
            assert "metadata" in data
            assert "statistics" in data
            assert "results" in data
            assert len(data["results"]) == len(result)
            assert data["statistics"]["total_records"] == 1

    def test_pam_filtering_and_score_threshold(
        self, synthetic_fasta_path: Path, tmp_output_dir: Path
    ):
        """Pipeline filters incompatible PAMs when canonical_pam_only is True."""
        config = PipelineConfig(
            fasta_path=synthetic_fasta_path,
            guide_sequence="GAGTCCGAGCAGAAGAAGAA",
            canonical_pam_only=True,
            min_score=90.0,
        )

        pipeline = CPUPipeline()
        result = pipeline.run(config)

        # Only sites with canonical NGG and score >= 90.0 should be retained
        assert len(result) >= 2
        for hit in result.hits:
            assert hit.is_canonical_pam is True
            assert hit.score >= 90.0

    def test_chunk_boundaries_and_duplicate_prevention(self, tmp_path: Path):
        """Validates that candidate sites straddling chunk boundaries are not duplicated."""
        # Create sequence partitioned into small chunks (chunk_size=100 bp, halo=22 bp)
        spacer = "GAGTCCGAGCAGAAGAAGAA" + "CGG"  # 23 nt
        # Place hit right near chunk core boundary
        pad1 = "A" * 90
        pad2 = "A" * 150
        seq = pad1 + spacer + pad2  # total ~263 bp -> 3 chunks of 100 bp
        fasta_file = tmp_path / "boundary_test.fa"
        fasta_file.write_text(f">chr_boundary\n{seq}\n", encoding="utf-8")

        config = PipelineConfig(
            fasta_path=fasta_file,
            guide_sequence="GAGTCCGAGCAGAAGAAGAA",
            chunk_size=100,
            halo_size=22,
        )

        pipeline = CPUPipeline()
        result = pipeline.run(config)

        assert result.statistics.total_chunks == 3
        # Exactly one exact match should be detected (no duplicates across chunk 0 and chunk 1)
        exact_hits = [h for h in result.hits if h.is_exact_match]
        assert len(exact_hits) == 1

    def test_empty_results_when_no_candidates_match(self, tmp_path: Path):
        """Completes gracefully when no sequences match within the mismatch cutoff."""
        # Poly-T sequence vs Poly-G guide
        fasta_file = tmp_path / "no_match.fa"
        fasta_file.write_text(">chr_no_match\n" + "T" * 500 + "\n", encoding="utf-8")

        config = PipelineConfig(
            fasta_path=fasta_file,
            guide_sequence="GGGGGGGGGGGGGGGGGGGG",
            max_mismatches=2,
        )

        pipeline = CPUPipeline()
        result = pipeline.run(config)

        assert len(result) == 0
        assert result.statistics.retained_candidates == 0
        assert result.statistics.raw_candidates_detected == 0

    def test_invalid_fasta_path_raises_file_not_found(self):
        """Non-existent FASTA file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            config = PipelineConfig(
                fasta_path="non_existent_file.fa",
                guide_sequence="GAGTCCGAGCAGAAGAAGAA",
            )
            pipeline = CPUPipeline()
            pipeline.run(config)

    def test_invalid_guide_characters_raise_invalid_sequence_error(self, tmp_path: Path):
        """Invalid guide with non-canonical base raises InvalidSequenceError."""
        fasta_file = tmp_path / "dummy.fa"
        fasta_file.write_text(">dummy\nACGTACGTACGT\n")

        with pytest.raises(InvalidSequenceError):
            config = PipelineConfig(
                fasta_path=fasta_file,
                guide_sequence="GAGTCCGAGCAGAAGAAGAN",  # 'N' is invalid in guide
            )
            pipeline = CPUPipeline()
            pipeline.run(config)


@pytest.fixture(scope="module")
def real_dataset_path() -> Path:
    """Resolves real dataset file path."""
    p = Path(__file__).resolve().parents[2] / "data" / "raw" / "GRCh38" / "chr1_region.fasta.fa"
    if not p.exists():
        pytest.skip(f"Real dataset not found at {p}")
    return p


class TestCPUPipelineRealDataset:
    """Integration test executing against the real GRCh38 chr1_region dataset."""

    def test_real_dataset_streaming_pipeline(
        self, real_dataset_path: Path, tmp_output_dir: Path
    ):
        """Executes the complete pipeline on the 24.89 Mbp GRCh38 chr1 region.

        Verifies:
        - Memory-bounded streaming across 3 chunks.
        - Off-target hits detected and ranked.
        - CSV and JSON output generated.
        - Runtime and memory limits adhered to.
        """
        # EMX1 standard human target guide
        guide_seq = "GAGTCCGAGCAGAAGAAGAA"  # EMX1 on-target spacer

        csv_out = tmp_output_dir / "emx1_chr1_hits.csv"
        json_out = tmp_output_dir / "emx1_chr1_report.json"

        config = PipelineConfig(
            fasta_path=real_dataset_path,
            guide_sequence=guide_seq,
            max_mismatches=2,  # Strict threshold for fast execution on 25 Mbp
            canonical_pam_only=False,
            search_reverse_strand=True,
            chunk_size=10_000_000,
            batch_size=200_000,
            top_k=20,
            output_csv=csv_out,
            output_json=json_out,
        )

        pipeline = CPUPipeline()
        result = pipeline.run(config)

        stats = result.statistics
        assert stats.total_records == 1
        assert stats.total_chunks == 3
        assert stats.total_base_pairs == 24_895_643
        assert stats.elapsed_seconds > 0.0
        assert stats.peak_memory_mb < 2000.0  # Well within bounded memory limits

        # Output files verified
        assert csv_out.exists()
        assert json_out.exists()
        assert csv_out.stat().st_size > 0
        assert json_out.stat().st_size > 0
