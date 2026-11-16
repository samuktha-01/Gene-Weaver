"""Domain exception classes for genomic FASTA parsing and sequence validation."""

from typing import Optional


class FastaParsingError(Exception):
    """Base exception for all FASTA file parsing errors."""

    def __init__(self, message: str, file_path: Optional[str] = None, line_number: Optional[int] = None):
        self.file_path = file_path
        self.line_number = line_number
        context_parts = []
        if file_path:
            context_parts.append(f"file: {file_path}")
        if line_number is not None:
            context_parts.append(f"line: {line_number}")
        
        context_str = f" ({', '.join(context_parts)})" if context_parts else ""
        super().__init__(f"{message}{context_str}")


class EmptyFastaFileError(FastaParsingError):
    """Raised when a FASTA file is 0 bytes or contains no valid FASTA records."""


class MalformedHeaderError(FastaParsingError):
    """Raised when a FASTA record header is malformed, lacks '>', or has no identifier."""


class InvalidSequenceError(FastaParsingError):
    """Raised when a sequence contains invalid or unrecognized nucleotide characters."""


class EmptySequenceError(FastaParsingError):
    """Raised when a FASTA header record is followed by empty sequence data."""


class InvalidFileFormatError(FastaParsingError):
    """Raised when the file has an unsupported format or extension."""
