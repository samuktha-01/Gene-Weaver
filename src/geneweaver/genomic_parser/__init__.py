"""Genomic parser package for GeneWeaver.

Handles FASTA sequence streaming, file format validation, and genomic coordinate indexing
with memory-bounded execution.
"""

from geneweaver.genomic_parser.exceptions import (
    EmptyFastaFileError,
    EmptySequenceError,
    FastaParsingError,
    InvalidFileFormatError,
    InvalidSequenceError,
    MalformedHeaderError,
)
from geneweaver.genomic_parser.fasta_reader import FastaReader
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.genomic_parser.validator import SequenceValidator

__all__ = [
    "FastaReader",
    "FastaRecord",
    "SequenceValidator",
    "FastaParsingError",
    "EmptyFastaFileError",
    "MalformedHeaderError",
    "InvalidSequenceError",
    "EmptySequenceError",
    "InvalidFileFormatError",
]
