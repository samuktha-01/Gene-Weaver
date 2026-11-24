"""Unit tests for GeneWeaver CPU Pipeline Correctness Audit.

Verifies:
1. Genomic coordinate transformations on both forward and reverse strands,
   including non-zero reference sub-region slice offsets and pam adjacency invariants.
2. On-target vs off-target candidate site classification (user-specified locus vs inferred).
3. Mismatch detection and distribution tracking across thresholds (0, 1, 2, 3, 4 mismatches).
4. CSV and JSON export schema completeness and coordinate field consistency.
"""

import json
from pathlib import Path
import pytest

from geneweaver.alignment.models import (
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    Strand,
)
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.pipeline.cpu_pipeline import CPUPipeline, PipelineConfig
from geneweaver.result_manager.exporter import ResultExporter
from geneweaver.scoring.candidate_ranker import CandidateRanker, RankingConfig
from geneweaver.scoring.mismatch_analyzer import MismatchAnalyzer
from geneweaver.scoring.mismatch_scorer import (
    HsuZhangScorer,
    MismatchScoringConfig,
    ScoredCandidateHit,
)
from geneweaver.scoring.pam_evaluator import (
    NucleaseConfig,
    PAMValidationResult,
    PAMValidator,
    SPCAS9,
    ValidatedCandidateHit,
)


def _make_mock_validated_hit(
    chromosome: str,
    strand: Strand,
    protospacer_start: int,
    protospacer_end: int,
    pam_start: int,
    pam_end: int,
    pam_sequence: str = "AGG",
    is_valid_pam: bool = True,
    is_canonical_pam: bool = True,
    mismatch_count: int = 0,
    protospacer_sequence: str = "GCTTATCCAGTCTGCTACGC",
) -> ValidatedCandidateHit:
    """Helper to build a realistic ValidatedCandidateHit."""
    pam_res = PAMValidationResult(
        is_valid=is_valid_pam,
        is_canonical=is_canonical_pam,
        matched_motif="NGG" if is_valid_pam else None,
        cleavage_multiplier=1.0 if is_canonical_pam else (0.25 if is_valid_pam else 0.0),
        pam_sequence=pam_sequence,
    )
    mismatch_info = MismatchInfo(
        count=mismatch_count,
        bitmask=0,
        positions=(),
        seed_count=0,
    )
    g_start = min(protospacer_start, pam_start)
    g_end = max(protospacer_end, pam_end)
    raw = RawCandidateHit(
        chunk_id="chunk_0",
        chromosome=chromosome,
        local_offset=protospacer_start,
        global_start=g_start,
        global_end=g_end,
        genomic_position=g_start + 1,
        strand=strand,
        mismatch_info=mismatch_info,
        pam_sequence=pam_sequence,
        target_sequence=protospacer_sequence + pam_sequence,
    )
    return ValidatedCandidateHit(
        candidate=raw,
        pam_result=pam_res,
        chromosome=chromosome,
        strand=strand,
        protospacer_start=protospacer_start,
        protospacer_end=protospacer_end,
        pam_start=pam_start,
        pam_end=pam_end,
        pam_sequence=pam_sequence,
        target_sequence=protospacer_sequence + pam_sequence,
    )


class TestGenomicCoordinateAudit:
    """Verifies coordinate conversion between region-relative and reference-assembly systems."""

    def test_forward_strand_with_header_offset(self):
        """Forward strand coordinate transformation with non-zero header offset (e.g. chr1 slice)."""
        # Header specifies: >ref|NC_000001.11|:85401937-110297579
        # 1-based start is 85,401,937 -> base_offset is 85,401,936
        record = FastaRecord(
            identifier="ref|NC_000001.11|:85401937-110297579",
            sequence="A" * 200_000,
        )

        val_hit = _make_mock_validated_hit(
            chromosome="ref|NC_000001.11|:85401937-110297579",
            strand=Strand.FORWARD,
            protospacer_start=100103,
            protospacer_end=100123,
            pam_start=100123,
            pam_end=100126,
            pam_sequence="TGG",
        )

        coords = ResultExporter.calculate_genomic_coordinates(val_hit, record)

        # Region coordinates (0-based half-open)
        assert coords["region_protospacer_start"] == 100103
        assert coords["region_protospacer_end"] == 100123
        assert coords["region_pam_start"] == 100123
        assert coords["region_pam_end"] == 100126

        # Reference-genome coordinates (1-based inclusive)
        # 85,401,936 + 100,103 + 1 = 85,502,040
        assert coords["genomic_start"] == 85502040
        # 85,401,936 + 100,123 = 85,502,059
        assert coords["genomic_end"] == 85502059
        # 85,401,936 + 100,123 + 1 = 85,502,060
        assert coords["genomic_pam_start"] == 85502060
        # 85,401,936 + 100,126 = 85,502,062
        assert coords["genomic_pam_end"] == 85502062

        # Adjacency invariant on forward strand: PAM immediately follows protospacer
        assert coords["genomic_pam_start"] == coords["genomic_end"] + 1

    def test_reverse_strand_with_header_offset(self):
        """Reverse strand coordinate transformation with non-zero header offset."""
        record = FastaRecord(
            identifier="ref|NC_000001.11|:85401937-110297579",
            sequence="A" * 300_000,
        )

        # On reverse strand, PAM occupies the lower reference coordinates [200000, 200003)
        # and protospacer occupies [200003, 200023)
        val_hit = _make_mock_validated_hit(
            chromosome="ref|NC_000001.11|:85401937-110297579",
            strand=Strand.REVERSE,
            protospacer_start=200003,
            protospacer_end=200023,
            pam_start=200000,
            pam_end=200003,
            pam_sequence="CGG",
        )

        coords = ResultExporter.calculate_genomic_coordinates(val_hit, record)

        # Region coordinates
        assert coords["region_protospacer_start"] == 200003
        assert coords["region_protospacer_end"] == 200023
        assert coords["region_pam_start"] == 200000
        assert coords["region_pam_end"] == 200003

        # Reference-genome coordinates (1-based inclusive)
        # base_offset = 85,401,936
        assert coords["genomic_pam_start"] == 85401936 + 200000 + 1  # 85601937
        assert coords["genomic_pam_end"] == 85401936 + 200003        # 85601939
        assert coords["genomic_start"] == 85401936 + 200003 + 1      # 85601940
        assert coords["genomic_end"] == 85401936 + 200023          # 85601959

        # Adjacency invariant on reverse strand: PAM precedes protospacer in genomic coordinates
        assert coords["genomic_pam_end"] + 1 == coords["genomic_start"]

    def test_whole_chromosome_zero_offset(self):
        """Whole-chromosome FASTA with no sub-region offset (base_offset = 0)."""
        record = FastaRecord(
            identifier="chr1",
            sequence="A" * 1000,
            description="Homo sapiens chromosome 1",
        )

        val_hit = _make_mock_validated_hit(
            chromosome="chr1",
            strand=Strand.FORWARD,
            protospacer_start=100,
            protospacer_end=120,
            pam_start=120,
            pam_end=123,
            pam_sequence="AGG",
        )

        coords = ResultExporter.calculate_genomic_coordinates(val_hit, record)

        # When coordinates is None, base_offset is 0
        assert coords["base_offset"] == 0
        assert coords["genomic_start"] == 101
        assert coords["genomic_end"] == 120
        assert coords["genomic_pam_start"] == 121
        assert coords["genomic_pam_end"] == 123
        assert coords["genomic_pam_start"] == coords["genomic_end"] + 1


class TestOnTargetClassificationAudit:
    """Verifies candidate site classification into ON_TARGET vs OFF_TARGET."""

    def _make_candidate(
        self,
        chrom: str,
        start: int,
        strand: Strand,
        mismatch_count: int,
        score: float,
        pam_valid: bool = True,
    ) -> ScoredCandidateHit:
        proto_start = start if strand == Strand.FORWARD else start + 3
        proto_end = start + 20 if strand == Strand.FORWARD else start + 23
        pam_start = start + 20 if strand == Strand.FORWARD else start
        pam_end = start + 23 if strand == Strand.FORWARD else start + 3

        val_hit = _make_mock_validated_hit(
            chromosome=chrom,
            strand=strand,
            protospacer_start=proto_start,
            protospacer_end=proto_end,
            pam_start=pam_start,
            pam_end=pam_end,
            pam_sequence="AGG" if pam_valid else "ACT",
            is_valid_pam=pam_valid,
            is_canonical_pam=pam_valid,
            mismatch_count=mismatch_count,
        )
        analyzer = MismatchAnalyzer()
        analysis = analyzer.analyze_candidate(val_hit, GuideTarget.from_string("GCTTATCCAGTCTGCTACGCAGG"))
        return ScoredCandidateHit(
            candidate=val_hit,
            score=score,
            raw_score=score,
            mismatch_analysis=analysis,
            scoring_model="hsu_zhang",
            pam_multiplier_applied=1.0,
        )

    def test_user_specified_target_coordinates(self):
        """User supplies explicit coordinates: matching site is ON_TARGET, others are OFF_TARGET."""
        record = FastaRecord(
            identifier="ref|NC_000001.11|:85401937-110297579",
            sequence="A" * 200_000,
        )
        records_map = {record.identifier: record}

        # Hit 1 is at 85502040 (genomic start)
        hit1 = self._make_candidate(
            record.identifier, start=100103, strand=Strand.FORWARD, mismatch_count=0, score=100.0
        )
        # Hit 2 is at another locus
        hit2 = self._make_candidate(
            record.identifier, start=150000, strand=Strand.FORWARD, mismatch_count=0, score=100.0
        )

        config = PipelineConfig(
            fasta_path="dummy.fa",
            guide_sequence="GCTTATCCAGTCTGCTACGCAGG",
            target_chromosome="NC_000001.11",
            target_start=85502040,  # Explicit 1-based genomic target
            target_strand=Strand.FORWARD,
        )

        classified = CPUPipeline._classify_candidates([hit1, hit2], config, records_map)

        assert classified[0].is_on_target is True
        assert classified[0].site_type == "ON_TARGET"
        assert classified[0].classification_basis == "user_specified_coordinates"

        assert classified[1].is_on_target is False
        assert classified[1].site_type == "OFF_TARGET"
        assert classified[1].classification_basis == "non_target_locus"

    def test_inferred_unique_exact_match(self):
        """Without user coordinates, a single 0-mismatch canonical match is inferred as ON_TARGET."""
        record = FastaRecord(
            identifier="chr1",
            sequence="A" * 1000,
        )
        records_map = {record.identifier: record}

        hit_on = self._make_candidate("chr1", start=100, strand=Strand.FORWARD, mismatch_count=0, score=100.0)
        hit_off = self._make_candidate("chr1", start=300, strand=Strand.REVERSE, mismatch_count=2, score=45.0)

        config = PipelineConfig(
            fasta_path="dummy.fa",
            guide_sequence="GCTTATCCAGTCTGCTACGCAGG",
        )

        classified = CPUPipeline._classify_candidates([hit_on, hit_off], config, records_map)

        assert classified[0].is_on_target is True
        assert classified[0].site_type == "ON_TARGET"
        assert classified[0].classification_basis == "inferred_unique_exact_match"

        assert classified[1].is_on_target is False
        assert classified[1].site_type == "OFF_TARGET"
        assert classified[1].classification_basis == "off_target_mismatches"

    def test_inferred_multiple_exact_matches(self):
        """Without user coordinates, multiple identical 0-mismatch matches are flagged as inferred_multiple."""
        record = FastaRecord(
            identifier="chr1",
            sequence="A" * 1000,
        )
        records_map = {record.identifier: record}

        hit1 = self._make_candidate("chr1", start=100, strand=Strand.FORWARD, mismatch_count=0, score=100.0)
        hit2 = self._make_candidate("chr1", start=500, strand=Strand.FORWARD, mismatch_count=0, score=100.0)

        config = PipelineConfig(
            fasta_path="dummy.fa",
            guide_sequence="GCTTATCCAGTCTGCTACGCAGG",
        )

        classified = CPUPipeline._classify_candidates([hit1, hit2], config, records_map)

        assert len(classified) == 2
        assert classified[0].is_on_target is True
        assert classified[0].classification_basis == "inferred_multiple_exact_matches"
        assert classified[1].is_on_target is True
        assert classified[1].classification_basis == "inferred_multiple_exact_matches"


class TestMismatchThresholdAndDistributionAudit:
    """Verifies mismatch detection across thresholds and telemetry breakdown."""

    def test_mismatch_distribution_synthetic(self, tmp_path):
        """Constructs synthetic records with exactly 0, 1, 2, 3, 4, 5 mismatches and checks detection."""
        guide_spacer = "GCTTATCCAGTCTGCTACGC"
        pam = "AGG"
        guide_target = guide_spacer + pam

        # Construct sequences with known mismatch counts (all canonical bases)
        s0 = guide_spacer + pam
        s1 = "ACTTATCCAGTCTGCTACGC" + pam
        s2 = "AATTATCCAGTCTGCTACGC" + pam
        s3 = "AAATATCCAGTCTGCTACGC" + pam
        s4 = "AAAAATCCAGTCTGCTACGC" + pam
        s5 = "AAAAAACCAGTCTGCTACGC" + pam

        spacer_pad = "A" * 50
        full_seq = spacer_pad.join(["", s0, s1, s2, s3, s4, s5, ""])

        fasta_file = tmp_path / "mismatch_test.fa"
        fasta_file.write_text(f">test_record\n{full_seq}\n")

        # Run pipeline with max_mismatches = 4
        config = PipelineConfig(
            fasta_path=fasta_file,
            guide_sequence=guide_target,
            max_mismatches=4,
            max_seed_mismatches=4,
            chunk_size=100_000,
            halo_size=25,
            search_reverse_strand=False,
        )

        pipeline = CPUPipeline()
        result = pipeline.run(config)

        # Sites with 0, 1, 2, 3, 4 mismatches should all be detected (1 each = 5 total)
        # Site with 5 mismatches should NOT be detected
        dist = result.statistics.mismatch_distribution
        assert dist.get(0) == 1
        assert dist.get(1) == 1
        assert dist.get(2) == 1
        assert dist.get(3) == 1
        assert dist.get(4) == 1
        assert dist.get(5, 0) == 0

        assert result.statistics.raw_candidates_detected == 5
        assert result.statistics.on_target_candidates == 1
        assert result.statistics.off_target_candidates == 4


class TestExportSchemaAudit:
    """Verifies exported CSV and JSON fields conform to the audited coordinate and classification rules."""

    def test_csv_and_json_schema(self, tmp_path):
        """Verifies CSV and JSON contain required coordinate columns and classification metadata."""
        fasta_file = tmp_path / "schema_test.fa"
        # Reference with offset :1000-2000
        guide_seq = "GCTTATCCAGTCTGCTACGCAGG"
        fasta_file.write_text(f">chr1:1001-2000\nAAAA{guide_seq}AAAA\n")

        csv_file = tmp_path / "out.csv"
        json_file = tmp_path / "out.json"

        config = PipelineConfig(
            fasta_path=fasta_file,
            guide_sequence=guide_seq,
            max_mismatches=2,
            chunk_size=50_000,
            halo_size=25,
            output_csv=csv_file,
            output_json=json_file,
        )

        pipeline = CPUPipeline()
        result = pipeline.run(config)

        assert csv_file.exists()
        assert json_file.exists()

        # Check CSV header
        lines = csv_file.read_text().strip().splitlines()
        header = lines[0].split(",")

        required_columns = [
            "rank",
            "site_type",
            "is_on_target",
            "classification_basis",
            "score",
            "chromosome",
            "strand",
            "genomic_start",
            "genomic_end",
            "genomic_pam_start",
            "genomic_pam_end",
            "region_protospacer_start",
            "region_protospacer_end",
            "region_pam_start",
            "region_pam_end",
            "pam_sequence",
            "is_valid_pam",
            "mismatch_count",
        ]
        for col in required_columns:
            assert col in header, f"Missing required column '{col}' in CSV export"

        # Check JSON structure
        data = json.loads(json_file.read_text())
        assert "metadata" in data
        assert "statistics" in data
        assert "results" in data

        stats = data["statistics"]
        assert "on_target_candidates" in stats
        assert "off_target_candidates" in stats
        assert "mismatch_distribution" in stats

        if data["results"]:
            first_hit = data["results"][0]
            assert "site_type" in first_hit
            assert "is_on_target" in first_hit
            assert "classification_basis" in first_hit
            assert "genomic_start" in first_hit
            assert "genomic_end" in first_hit
            assert "genomic_pam_start" in first_hit
            assert "genomic_pam_end" in first_hit
            assert "region_protospacer_start" in first_hit
            assert "region_protospacer_end" in first_hit
