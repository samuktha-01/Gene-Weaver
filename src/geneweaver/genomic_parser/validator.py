"""Validation utilities for FASTA files, genomic sequence headers, and CRISPR guides."""

import os
from pathlib import Path
from typing import Optional, Set, Tuple

from geneweaver.genomic_parser.exceptions import (
    EmptyFastaFileError,
    InvalidFileFormatError,
    InvalidSequenceError,
    MalformedHeaderError,
)
from geneweaver.sequence_processor.normalizer import SequenceNormalizer


class SequenceValidator:
    """Validator for genomic FASTA files, sequence records, and CRISPR guides."""

    SUPPORTED_EXTENSIONS: Set[str] = frozenset({
        ".fa",
        ".fasta",
        ".fna",
        ".fa.gz",
        ".fasta.gz",
        ".fna.gz",
    })

    def __init__(self, normalizer: Optional[SequenceNormalizer] = None):
        self.normalizer = normalizer or SequenceNormalizer()

    @classmethod
    def validate_fasta_path(cls, path: Path | str) -> Path:
        """Verifies file existence, read permissions, supported extension, and non-empty size.

        Args:
            path: Path to the FASTA file.

        Returns:
            Resolved Path instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the path references a directory.
            InvalidFileFormatError: If the file extension is not supported.
            EmptyFastaFileError: If the file size is 0 bytes.
        """
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {file_path}")

        if file_path.is_dir():
            raise IsADirectoryError(f"Provided path is a directory, not a file: {file_path}")

        # Check compound suffix for .gz files (e.g. .fasta.gz)
        suffixes = "".join(file_path.suffixes).lower()
        has_valid_ext = any(suffixes.endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS)
        if not has_valid_ext:
            valid_exts = ", ".join(sorted(cls.SUPPORTED_EXTENSIONS))
            raise InvalidFileFormatError(
                f"Unsupported file format '{suffixes}'. Supported extensions: {valid_exts}",
                file_path=str(file_path),
            )

        if file_path.stat().st_size == 0:
            raise EmptyFastaFileError(
                f"FASTA file is empty (0 bytes): {file_path}",
                file_path=str(file_path),
            )

        return file_path

    @classmethod
    def validate_header(
        cls,
        header_line: str,
        line_number: Optional[int] = None,
        file_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Validates a FASTA header line and extracts the sequence identifier and description.

        Args:
            header_line: Raw header text line.
            line_number: Source file line number for diagnostic reporting.
            file_path: Source file path for diagnostic reporting.

        Returns:
            Tuple of (identifier, description).

        Raises:
            MalformedHeaderError: If line does not start with '>' or lacks an identifier.
        """
        stripped = header_line.strip()
        if not stripped.startswith(">"):
            raise MalformedHeaderError(
                f"FASTA record header must start with '>': '{stripped[:40]}'",
                file_path=file_path,
                line_number=line_number,
            )

        # Remove leading '>'
        content = stripped[1:].strip()
        if not content:
            raise MalformedHeaderError(
                "FASTA record header contains no identifier after '>'",
                file_path=file_path,
                line_number=line_number,
            )

        # Split identifier from optional description on first whitespace
        parts = content.split(maxsplit=1)
        identifier = parts[0]
        description = parts[1] if len(parts) > 1 else ""

        return identifier, description

    @classmethod
    def validate_guide(cls, guide: str, expected_len: int = 23) -> str:
        """Validates a CRISPR guide RNA / target DNA sequence.

        Args:
            guide: 20nt protospacer + 3nt PAM sequence (e.g., 'GAGTCCGAGCAGAAGAAGAANGG').
            expected_len: Expected total length (default 23 for canonical SpCas9).

        Returns:
            Normalized uppercase DNA guide sequence.

        Raises:
            ValueError: If the guide sequence has an unexpected length.
            InvalidSequenceError: If the guide contains invalid nucleotide characters.
        """
        normalizer = SequenceNormalizer()
        normalized = normalizer.normalize(guide)

        if len(normalized) != expected_len:
            raise ValueError(
                f"Invalid CRISPR guide length: got {len(normalized)} nt, expected {expected_len} nt "
                f"(20 nt protospacer + 3 nt PAM)"
            )

        # Validate that all characters are canonical bases or valid IUPAC codes
        is_valid, err_msg = normalizer.validate(normalized, allow_ambiguity=True)
        if not is_valid:
            raise InvalidSequenceError(f"Invalid nucleotide in CRISPR guide: {err_msg}")

        return normalized
