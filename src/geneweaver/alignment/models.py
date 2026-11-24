"""Data models, configuration classes, and metadata structures for CRISPR sequence alignment.

These models define strongly typed, backend-agnostic contracts for sequence alignment inputs,
candidate hit coordinates, mismatch details, and alignment execution results.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from geneweaver.genomic_parser.validator import SequenceValidator
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder


class Strand(IntEnum):
    """Genomic DNA strand orientation.

    Inherits from IntEnum to allow zero-overhead scalar passing to CUDA kernels
    (0 = Forward, 1 = Reverse) while providing Pythonic symbol representations ('+', '-').
    """

    FORWARD = 0
    REVERSE = 1

    @property
    def sign(self) -> str:
        """Returns '+' for FORWARD and '-' for REVERSE."""
        return "+" if self == Strand.FORWARD else "-"

    def __str__(self) -> str:
        return self.sign

    def __repr__(self) -> str:
        return f"<Strand.{self.name}: '{self.sign}'>"

    @classmethod
    def from_symbol(cls, value: Union[str, int, "Strand"]) -> "Strand":
        """Converts symbol ('+', '-'), string ('FORWARD', 'REVERSE'), or int (0, 1) to Strand.

        Args:
            value: Input strand representation.

        Returns:
            Strand enum instance.

        Raises:
            ValueError: If the input cannot be resolved to a valid strand.
        """
        if isinstance(value, Strand):
            return value
        if isinstance(value, int):
            if value in (0, 1):
                return cls(value)
            raise ValueError(f"Invalid integer for Strand: {value}. Expected 0 (Forward) or 1 (Reverse).")
        if isinstance(value, str):
            clean = value.strip().upper()
            if clean in ("+", "FWD", "FORWARD", "0"):
                return cls.FORWARD
            if clean in ("-", "REV", "REVERSE", "1"):
                return cls.REVERSE
            raise ValueError(f"Invalid strand symbol: {repr(value)}. Expected '+' or '-'.")
        raise TypeError(f"Unsupported type for Strand conversion: {type(value)}")


@dataclass(frozen=True)
class MismatchInfo:
    """Detailed mismatch metadata for an aligned candidate sequence.

    Attributes:
        count: Total number of mismatched base pairs along the 20 nt protospacer.
        bitmask: Integer bitmask where bit i (0 <= i < 20) is 1 if position i is a mismatch.
        positions: Tuple of 1-based mismatch positions along the 20 nt spacer.
        seed_count: Number of mismatches occurring within the seed region (positions 11-20).
    """

    count: int
    bitmask: int
    positions: Tuple[int, ...]
    seed_count: int

    def __post_init__(self) -> None:
        if self.count < 0 or self.count > 20:
            raise ValueError(f"Mismatch count must be between 0 and 20, got {self.count}")
        if self.seed_count < 0 or self.seed_count > self.count:
            raise ValueError(f"Seed mismatch count ({self.seed_count}) cannot exceed total count ({self.count})")

    @classmethod
    def from_bitmask(
        cls,
        bitmask: int,
        spacer_length: int = 20,
        seed_length: int = 10,
    ) -> "MismatchInfo":
        """Constructs MismatchInfo from an integer bitmask.

        Args:
            bitmask: Bitmask where bit i is 1 if position i has a mismatch.
                     Bit 0 is the 5' end of the spacer; bit (spacer_length - 1) is PAM-proximal.
            spacer_length: Protospacer length (default: 20 nt).
            seed_length: Length of seed region adjacent to PAM (default: 10 nt, positions 11-20).

        Returns:
            MismatchInfo instance.
        """
        positions: List[int] = []
        seed_count = 0
        seed_start_idx = spacer_length - seed_length  # e.g., 20 - 10 = index 10 (1-based position 11)

        for i in range(spacer_length):
            if (bitmask >> i) & 1:
                pos_1based = i + 1
                positions.append(pos_1based)
                if i >= seed_start_idx:
                    seed_count += 1

        return cls(
            count=len(positions),
            bitmask=bitmask,
            positions=tuple(positions),
            seed_count=seed_count,
        )

    def is_within_tolerance(self, max_mismatches: int, max_seed_mismatches: Optional[int] = None) -> bool:
        """Checks if mismatches satisfy threshold constraints."""
        if self.count > max_mismatches:
            return False
        if max_seed_mismatches is not None and self.seed_count > max_seed_mismatches:
            return False
        return True

    def to_genomic_positions(
        self,
        global_start: int,
        strand: Strand,
        target_length: int = 23,
    ) -> Tuple[int, ...]:
        """Translates 1-based guide mismatch positions to 1-based global genomic coordinates.

        For Forward (+) strand hits:
            Guide position p (1-based, 5' to 3') maps directly to forward genomic position:
            genomic_pos = global_start + p.

        For Reverse (-) strand hits:
            The guide aligns to the reverse-complement strand (3' to 5' relative to forward).
            Guide position p (1-based, 5' to 3') corresponds to forward offset (target_length - p):
            genomic_pos = global_start + (target_length - p) + 1.

        Args:
            global_start: 0-based global genomic start position on the forward chromosome.
            strand: Strand orientation (Strand.FORWARD or Strand.REVERSE).
            target_length: Total length of the genomic target sequence window (default: 23 bp).

        Returns:
            Tuple of 1-based global genomic coordinates in ascending numerical order.
        """
        if strand == Strand.FORWARD:
            return tuple(global_start + p for p in self.positions)
        else:
            return tuple(sorted(global_start + (target_length - p) + 1 for p in self.positions))

    def to_forward_relative_positions(
        self,
        strand: Strand,
        target_length: int = 23,
    ) -> Tuple[int, ...]:
        """Translates 1-based guide mismatch positions to 1-based positions relative to the forward genomic window.

        Args:
            strand: Strand orientation (Strand.FORWARD or Strand.REVERSE).
            target_length: Total length of the genomic target window (default: 23 bp).

        Returns:
            Tuple of 1-based offsets (1..target_length) along the forward genomic reference.
        """
        if strand == Strand.FORWARD:
            return self.positions
        else:
            return tuple(sorted((target_length - p) + 1 for p in self.positions))


@dataclass(frozen=True)
class AlignmentConfig:
    """Runtime configuration and parameter thresholds for sequence alignment.

    Attributes:
        max_mismatches: Maximum allowable mismatches along the 20 nt protospacer (default: 4).
        seed_max_mismatches: Maximum allowable mismatches in the seed region (default: 2).
        seed_length: Length of PAM-proximal seed region in base pairs (default: 10).
        canonical_pam_only: If True, only matches canonical 'NGG' PAM. If False, also allows 'NAG' and 'NGA'.
        search_reverse_strand: If True, evaluates both Forward (+) and Reverse Complement (-) strands.
        batch_size: Processing window batch size for vectorized or GPU chunk operations.
    """

    max_mismatches: int = 4
    seed_max_mismatches: int = 2
    seed_length: int = 10
    canonical_pam_only: bool = False
    search_reverse_strand: bool = True
    batch_size: int = 65536

    def __post_init__(self) -> None:
        if self.max_mismatches < 0 or self.max_mismatches > 20:
            raise ValueError(f"max_mismatches must be between 0 and 20, got {self.max_mismatches}")
        if self.seed_max_mismatches < 0:
            raise ValueError(f"seed_max_mismatches cannot be negative, got {self.seed_max_mismatches}")
        if self.seed_max_mismatches > self.max_mismatches:
            raise ValueError(
                f"seed_max_mismatches ({self.seed_max_mismatches}) cannot exceed "
                f"max_mismatches ({self.max_mismatches})"
            )
        if self.seed_length <= 0 or self.seed_length > 20:
            raise ValueError(f"seed_length must be between 1 and 20, got {self.seed_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")


@dataclass(frozen=True)
class GuideTarget:
    """Validated CRISPR target guide RNA / target DNA sequence container.

    Attributes:
        sequence: Full 23 nt sequence (20 nt protospacer + 3 nt PAM).
        spacer: 20 nt guide spacer sequence.
        pam: 3 nt PAM motif (e.g. 'AGG', 'CGG', 'TGG', 'GGG', 'NAG').
        encoded_2bit: 1D NumPy array (uint8, values in {0, 1, 2, 3}) of length 23.
        encoded_packed: Optional 2-bit packed EncodedTwoBit representation.
    """

    sequence: str
    spacer: str
    pam: str
    encoded_2bit: np.ndarray = field(repr=False)
    encoded_packed: Optional[EncodedTwoBit] = field(default=None, repr=False)

    @classmethod
    def from_string(cls, guide_str: str, expected_length: int = 23) -> "GuideTarget":
        """Validates and constructs a GuideTarget from an ASCII string.

        Args:
            guide_str: 23 nt guide sequence (e.g. 'GAGTCCGAGCAGAAGAAGAAAGG').
            expected_length: Expected total length (default: 23 nt).

        Returns:
            Validated GuideTarget instance.
        """
        validated_seq = SequenceValidator.validate_guide(guide_str, expected_len=expected_length)
        spacer = validated_seq[:20]
        pam = validated_seq[20:23]

        encoded_2bit = TwoBitEncoder.encode(validated_seq)
        encoded_packed = TwoBitEncoder.pack_8bit(encoded_2bit, length=len(validated_seq))

        return cls(
            sequence=validated_seq,
            spacer=spacer,
            pam=pam,
            encoded_2bit=encoded_2bit,
            encoded_packed=encoded_packed,
        )

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class RawCandidateHit:
    """Represents a candidate off-target site detected during alignment.

    Attributes:
        chunk_id: Identifier of the genomic chunk containing the hit.
        chromosome: Chromosome or contig identifier.
        local_offset: 0-based start index of the 23-mer window within the chunk sequence.
        global_start: 0-based global genomic start position.
        global_end: 0-based global genomic end position (global_start + 23).
        genomic_position: 1-based global genomic start position (global_start + 1).
        strand: Strand orientation (Strand.FORWARD or Strand.REVERSE).
        mismatch_info: Detailed MismatchInfo record.
        pam_sequence: Observed 3 bp PAM sequence at this genomic locus.
        target_sequence: Observed 23 bp genomic sequence.
    """

    chunk_id: str
    chromosome: str
    local_offset: int
    global_start: int
    global_end: int
    genomic_position: int
    strand: Strand
    mismatch_info: MismatchInfo
    pam_sequence: str
    target_sequence: str = ""

    @property
    def mismatch_count(self) -> int:
        """Total mismatches in spacer (conforming to architectural contract)."""
        return self.mismatch_info.count

    @property
    def mismatch_bitmask(self) -> int:
        """Integer bitmask of mismatch locations (conforming to architectural contract)."""
        return self.mismatch_info.bitmask

    @property
    def strand_symbol(self) -> str:
        """String symbol '+' or '-'."""
        return self.strand.sign

    @property
    def mismatch_positions(self) -> Tuple[int, ...]:
        """Returns 1-based mismatch positions relative to the guide sequence (5' to 3', 1..20)."""
        return self.mismatch_info.positions

    @property
    def guide_mismatch_positions(self) -> Tuple[int, ...]:
        """Explicit alias for 1-based mismatch positions relative to the guide sequence (5' to 3')."""
        return self.mismatch_info.positions

    @property
    def genomic_mismatch_positions(self) -> Tuple[int, ...]:
        """Returns 1-based absolute genomic coordinates on the forward reference chromosome.

        For Forward (+) strand:
            Mismatches are at global_start + p.

        For Reverse (-) strand:
            Mismatches are at global_start + (target_length - p) + 1.
        """
        target_len = len(self.target_sequence) if self.target_sequence else 23
        return self.mismatch_info.to_genomic_positions(
            global_start=self.global_start,
            strand=self.strand,
            target_length=target_len,
        )

    @property
    def forward_relative_mismatch_positions(self) -> Tuple[int, ...]:
        """Returns 1-based mismatch positions relative to the forward genomic reference window."""
        target_len = len(self.target_sequence) if self.target_sequence else 23
        return self.mismatch_info.to_forward_relative_positions(
            strand=self.strand,
            target_length=target_len,
        )

    @property
    def forward_sequence(self) -> str:
        """Returns the nucleotide sequence as present on the forward (+) reference strand."""
        if self.strand == Strand.FORWARD or not self.target_sequence:
            return self.target_sequence
        from geneweaver.alignment.reverse_complement import reverse_complement
        return reverse_complement(self.target_sequence)

    @property
    def reference_sequence(self) -> str:
        """Alias for forward_sequence conforming to genomic coordinate conventions."""
        return self.forward_sequence


@dataclass(frozen=True)
class AlignmentResult:
    """Container for the output of an alignment execution over a single genomic chunk.

    Attributes:
        chunk_id: Identifier of the chunk processed.
        chromosome: Chromosome or contig identifier.
        hits: List of candidate hits detected whose window starts within the chunk core.
        total_windows_evaluated: Total count of 23-mer windows scanned.
        execution_time_ms: Elapsed execution time in milliseconds.
        backend: Name of the compute backend that generated the results (e.g. 'cpu-numpy', 'cuda-numba').
        config: Alignment configuration parameters used during execution.
    """

    chunk_id: str
    chromosome: str
    hits: List[RawCandidateHit]
    total_windows_evaluated: int
    execution_time_ms: float
    backend: str
    config: AlignmentConfig

    @property
    def hit_count(self) -> int:
        """Returns the number of off-target candidate hits detected."""
        return len(self.hits)

    @property
    def candidates(self) -> List[RawCandidateHit]:
        """Alias for hits conforming to candidate terminology."""
        return self.hits

    def __len__(self) -> int:
        return len(self.hits)

    def __repr__(self) -> str:
        return (
            f"<AlignmentResult chunk='{self.chunk_id}', hits={self.hit_count}, "
            f"windows={self.total_windows_evaluated}, time={self.execution_time_ms:.2f}ms, "
            f"backend='{self.backend}'>"
        )
