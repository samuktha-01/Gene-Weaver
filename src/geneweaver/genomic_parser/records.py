"""Data structures representing parsed FASTA records."""

from dataclasses import dataclass


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

    def __repr__(self) -> str:
        desc_info = f", desc='{self.description[:30]}...'" if len(self.description) > 30 else (f", desc='{self.description}'" if self.description else "")
        return f"<FastaRecord id='{self.identifier}', length={self.length}{desc_info}>"
