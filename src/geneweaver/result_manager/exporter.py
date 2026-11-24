"""Result serialization and export module for GeneWeaver off-target search results.

Provides tabular CSV and structured JSON serializers for candidate off-target search results,
preserving exact genomic coordinates, chromosome/accession identifiers, strand orientations,
PAM motifs, mismatch positions, and calculated scores.

Coordinate System Conventions:
1. Reference Genome Coordinates (1-based inclusive):
   - genomic_start: 1-based start coordinate of the 20 nt protospacer on the reference chromosome.
   - genomic_end: 1-based end coordinate of the 20 nt protospacer on the reference chromosome.
   - genomic_pam_start: 1-based start coordinate of the 3 nt PAM on the reference chromosome.
   - genomic_pam_end: 1-based end coordinate of the 3 nt PAM on the reference chromosome.
   When FASTA headers contain regional offsets (e.g. '>ref|NC_000001.11|:85401937-110297579'),
   these fields are translated into absolute coordinates on the reference assembly.

2. Region-Relative Coordinates (0-based half-open):
   - region_protospacer_start, region_protospacer_end: 0-based slice indices within the FASTA record.
   - region_pam_start, region_pam_end: 0-based slice indices of the PAM within the FASTA record.
   - protospacer_start, protospacer_end, pam_start, pam_end: Aliases provided for backward compatibility.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from geneweaver.alignment.models import Strand
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.scoring.candidate_ranker import CandidateRankingResult
from geneweaver.scoring.mismatch_scorer import ScoredCandidateHit


class ResultExporter:
    """Serializes off-target alignment and scoring results to standard data formats."""

    CSV_FIELDNAMES = [
        "rank",
        "site_type",
        "is_on_target",
        "classification_basis",
        "score",
        "raw_score",
        "chromosome",
        "accession",
        "strand",
        "genomic_start",
        "genomic_end",
        "genomic_pam_start",
        "genomic_pam_end",
        "region_protospacer_start",
        "region_protospacer_end",
        "region_pam_start",
        "region_pam_end",
        "protospacer_start",
        "protospacer_end",
        "pam_start",
        "pam_end",
        "pam_sequence",
        "is_valid_pam",
        "is_canonical_pam",
        "matched_motif",
        "mismatch_count",
        "seed_mismatches",
        "non_seed_mismatches",
        "mismatch_positions",
        "target_sequence",
        "scoring_model",
    ]

    @classmethod
    def export_csv(
        cls,
        results: CandidateRankingResult,
        output_path: Union[Path, str],
        fasta_records: Optional[Dict[str, FastaRecord]] = None,
    ) -> Path:
        """Exports ranked off-target candidates to a comma-separated values (CSV) file.

        Args:
            results: CandidateRankingResult from CandidateRanker.
            output_path: Destination file path for the CSV output.
            fasta_records: Optional mapping of record identifiers to FastaRecord objects
                used to resolve chromosome header offsets and accession numbers.

        Returns:
            Resolved Path object of the generated CSV file.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        rows = cls._generate_rows(results, fasta_records=fasta_records)

        with open(out_p, mode="wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return out_p

    @classmethod
    def export_json(
        cls,
        results: CandidateRankingResult,
        output_path: Union[Path, str],
        metadata: Optional[Dict[str, Any]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        fasta_records: Optional[Dict[str, FastaRecord]] = None,
        indent: int = 2,
    ) -> Path:
        """Exports ranked off-target candidates and run metadata to a structured JSON file.

        Args:
            results: CandidateRankingResult from CandidateRanker.
            output_path: Destination file path for the JSON output.
            metadata: Optional dictionary of run configuration and environment metadata.
            statistics: Optional dictionary of processing and runtime performance metrics.
            fasta_records: Optional mapping of record identifiers to FastaRecord objects.
            indent: JSON indentation spaces (default: 2).

        Returns:
            Resolved Path object of the generated JSON file.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        payload = cls.to_dict(
            results=results,
            metadata=metadata,
            statistics=statistics,
            fasta_records=fasta_records,
        )

        with open(out_p, mode="wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, default=str)

        return out_p

    @classmethod
    def to_dict(
        cls,
        results: CandidateRankingResult,
        metadata: Optional[Dict[str, Any]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        fasta_records: Optional[Dict[str, FastaRecord]] = None,
    ) -> Dict[str, Any]:
        """Converts CandidateRankingResult into a structured Python dictionary."""
        rows = cls._generate_rows(results, fasta_records=fasta_records)

        return {
            "metadata": metadata or {},
            "statistics": statistics or results.summary(),
            "results": rows,
        }

    @staticmethod
    def calculate_genomic_coordinates(
        hit: Any,
        record: Optional[FastaRecord] = None,
    ) -> Dict[str, int]:
        """Calculates 1-based reference-genome coordinates and 0-based region coordinates.

        Coordinate conventions:
        - Reference Genome: 1-based inclusive coordinates [genomic_start, genomic_end].
          If the FASTA record was derived from a sub-region slice (e.g. :85401937-110297579),
          the slice's starting coordinate (85401937) is used as base offset (85401936) so that:
            genomic_start = base_offset + protospacer_start + 1
            genomic_end = base_offset + protospacer_end
            genomic_pam_start = base_offset + pam_start + 1
            genomic_pam_end = base_offset + pam_end
          Strand invariants:
            Forward strand (+): genomic_pam_start == genomic_end + 1
            Reverse strand (-): genomic_pam_end + 1 == genomic_start
        - Region-Relative: 0-based half-open coordinates [protospacer_start, protospacer_end).
        """
        if record and record.coordinates:
            base_offset = record.coordinates[0] - 1
        else:
            base_offset = 0

        proto_start = getattr(hit, "protospacer_start", getattr(hit, "global_start", 0))
        proto_end = getattr(hit, "protospacer_end", getattr(hit, "global_end", 0))
        pam_start = getattr(hit, "pam_start", None)
        pam_end = getattr(hit, "pam_end", None)

        genomic_start = base_offset + proto_start + 1
        genomic_end = base_offset + proto_end

        if pam_start is not None and pam_end is not None:
            genomic_pam_start = base_offset + pam_start + 1
            genomic_pam_end = base_offset + pam_end
        else:
            genomic_pam_start = -1
            genomic_pam_end = -1

        return {
            "base_offset": base_offset,
            "genomic_start": genomic_start,
            "genomic_end": genomic_end,
            "genomic_pam_start": genomic_pam_start,
            "genomic_pam_end": genomic_pam_end,
            "region_protospacer_start": proto_start,
            "region_protospacer_end": proto_end,
            "region_pam_start": pam_start if pam_start is not None else -1,
            "region_pam_end": pam_end if pam_end is not None else -1,
        }

    @classmethod
    def _generate_rows(
        cls,
        results: CandidateRankingResult,
        fasta_records: Optional[Dict[str, FastaRecord]] = None,
    ) -> List[Dict[str, Any]]:
        """Generates structured dictionary rows from CandidateRankingResult."""
        rows: List[Dict[str, Any]] = []

        for rank_idx, hit in enumerate(results.hits, start=1):
            chrom_id = hit.chromosome
            record_info = fasta_records.get(chrom_id) if fasta_records else None

            # Resolve accession
            if record_info:
                accession = record_info.accession
            else:
                accession = getattr(hit.candidate, "accession", chrom_id)

            # Calculate coordinates using helper
            coords = cls.calculate_genomic_coordinates(hit, record_info)

            mismatch_positions_str = ";".join(str(p) for p in hit.mismatch_positions)

            row = {
                "rank": rank_idx,
                "site_type": getattr(hit, "site_type", "OFF_TARGET"),
                "is_on_target": getattr(hit, "is_on_target", False),
                "classification_basis": getattr(hit, "classification_basis", "unassigned"),
                "score": round(hit.score, 4),
                "raw_score": round(hit.raw_score, 4),
                "chromosome": chrom_id,
                "accession": accession,
                "strand": hit.strand.sign,
                "genomic_start": coords["genomic_start"],
                "genomic_end": coords["genomic_end"],
                "genomic_pam_start": coords["genomic_pam_start"],
                "genomic_pam_end": coords["genomic_pam_end"],
                "region_protospacer_start": coords["region_protospacer_start"],
                "region_protospacer_end": coords["region_protospacer_end"],
                "region_pam_start": coords["region_pam_start"],
                "region_pam_end": coords["region_pam_end"],
                "protospacer_start": coords["region_protospacer_start"],
                "protospacer_end": coords["region_protospacer_end"],
                "pam_start": coords["region_pam_start"],
                "pam_end": coords["region_pam_end"],
                "pam_sequence": hit.pam_sequence,
                "is_valid_pam": hit.is_valid_pam,
                "is_canonical_pam": hit.is_canonical_pam,
                "matched_motif": hit.matched_motif or "",
                "mismatch_count": hit.mismatch_count,
                "seed_mismatches": hit.seed_mismatches,
                "non_seed_mismatches": hit.non_seed_mismatches,
                "mismatch_positions": mismatch_positions_str,
                "target_sequence": hit.target_sequence,
                "scoring_model": hit.scoring_model,
            }
            rows.append(row)

        return rows
