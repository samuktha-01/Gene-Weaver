"""Data structures representing parsed FASTA records."""

from dataclasses import dataclass
import re
from typing import Optional, Tuple


@dataclass(frozen=True)
class FastaRecord:
    """Represents a single parsed FASTA sequence record.

    Attributes:
        identifier: The unique sequence or chromosome identifier (first token after '>').
        sequence: The normalized nucleotide sequence (DNA uppercase).
        description: The full header description following the identifier, if any.
    """

    identifier: str
    sequence: str
    description: str = ""

    @property
    def length(self) -> int:
        """Returns the length of the nucleotide sequence in base pairs."""
        return len(self.sequence)

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def accession(self) -> str:
        """Extracts the reference accession if present in NCBI/RefSeq format (e.g. 'NC_000001.11')."""
        match = re.search(r"(?:ref\|)?([A-Z]{1,2}_\d+\.\d+)", self.identifier)
        if match:
            return match.group(1)
        return self.identifier

    @property
    def coordinates(self) -> Optional[Tuple[int, int]]:
        """Extracts 1-based genomic coordinates (start, end) if present in header (e.g. ':85401937-110297579')."""
        match = re.search(r":(\d+)-(\d+)", self.identifier)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def __repr__(self) -> str:
        desc_info = f", desc='{self.description[:30]}...'" if len(self.description) > 30 else (f", desc='{self.description}'" if self.description else "")
        return f"<FastaRecord id='{self.identifier}', length={self.length}{desc_info}>"
