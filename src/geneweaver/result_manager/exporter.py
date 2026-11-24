"""Result serialization and export module for GeneWeaver off-target search results.

Provides tabular CSV and structured JSON serializers for candidate off-target search results,
preserving exact genomic coordinates, chromosome/accession identifiers, strand orientations,
PAM motifs, mismatch positions, and calculated scores.
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
        "score",
        "raw_score",
        "chromosome",
        "accession",
        "strand",
        "genomic_start",
        "genomic_end",
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
                header_coords = record_info.coordinates
            else:
                accession = getattr(hit.candidate, "accession", chrom_id)
                header_coords = None

            # Calculate 1-based genomic coordinates (incorporating chromosome sub-region offset if present)
            if header_coords:
                base_offset = header_coords[0] - 1  # 0-based start of sub-region
                genomic_start = base_offset + hit.protospacer_start + 1
                genomic_end = base_offset + hit.protospacer_end
            else:
                genomic_start = hit.protospacer_start + 1
                genomic_end = hit.protospacer_end

            mismatch_positions_str = ";".join(str(p) for p in hit.mismatch_positions)

            row = {
                "rank": rank_idx,
                "score": round(hit.score, 4),
                "raw_score": round(hit.raw_score, 4),
                "chromosome": chrom_id,
                "accession": accession,
                "strand": hit.strand.sign,
                "genomic_start": genomic_start,
                "genomic_end": genomic_end,
                "protospacer_start": hit.protospacer_start,
                "protospacer_end": hit.protospacer_end,
                "pam_start": hit.pam_start if hit.pam_start is not None else -1,
                "pam_end": hit.pam_end if hit.pam_end is not None else -1,
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
