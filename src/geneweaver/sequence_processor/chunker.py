"""Genomic sequence chunking and overlapping halo boundary partitioner.

This module divides large genomic nucleotide sequences into manageable, fixed-size chunks
suitable for parallel GPU or CPU alignment, while preserving exact global coordinates and
enforcing overlapping halo margins across boundaries to prevent off-target target site loss.

Mathematical Formulation:
    For a sequence of total length N, nominal chunk size C, and halo margin H:
        Number of chunks M = ceil(N / C)
        For each chunk k in {0, ..., M - 1}:
            core_start_k  = k * C
            core_end_k    = min((k + 1) * C, N)
            chunk_start_k = core_start_k
            chunk_end_k   = min(core_end_k + H, N)

    Core Invariant:
        Every genomic position p in [0, N) belongs to exactly one chunk's core region:
            U_{k=0}^{M-1} [core_start_k, core_end_k) = [0, N)
        with [core_start_i, core_end_i) intersect [core_start_j, core_end_j) = empty for i != j.

    Halo Invariant:
        For a target search window of length L = 23 bp (20nt protospacer + 3nt PAM),
        a halo margin of H = L - 1 = 22 bp guarantees that any window beginning at the
        last base of Chunk k's core (core_end_k - 1) will have all 22 downstream bases
        present in Chunk k. Windows starting at or beyond core_end_k are deferred to
        Chunk k+1, eliminating duplicate reporting while preventing boundary dropouts.
"""

from dataclasses import dataclass
import math
from typing import Generator, Iterable, List, Optional, Tuple, Union

import numpy as np

from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder


@dataclass(frozen=True)
class ChunkDescriptor:
    """Descriptor and sequence container for a single partitioned genomic chunk.

    Attributes:
        chunk_id: Unique string identifier (e.g., 'chr1_chunk_0000').
        chromosome: Chromosome or contig identifier.
        chunk_index: 0-based sequential index of the chunk within the chromosome.
        global_start: 0-based inclusive global genomic start position.
        global_end: 0-based exclusive global genomic end position (including halo).
        actual_length: Total number of base pairs in this chunk (including halo).
        core_start: 0-based inclusive global start position of the core region.
        core_end: 0-based exclusive global end position of the core region.
        halo_start: 0-based inclusive global start position of the trailing halo.
        halo_end: 0-based exclusive global end position of the trailing halo.
        sequence_data: The sequence payload (string, 1D uint8 array, or EncodedTwoBit).
    """

    chunk_id: str
    chromosome: str
    chunk_index: int
    global_start: int
    global_end: int
    actual_length: int
    core_start: int
    core_end: int
    halo_start: int
    halo_end: int
    sequence_data: Union[str, np.ndarray, EncodedTwoBit]

    @property
    def core_length(self) -> int:
        """Returns the number of base pairs in the core region."""
        return self.core_end - self.core_start

    @property
    def halo_length(self) -> int:
        """Returns the number of base pairs in the trailing halo region."""
        return self.halo_end - self.halo_start

    @property
    def core_bounds(self) -> Tuple[int, int]:
        """Returns (core_start, core_end) in 0-based global coordinates."""
        return (self.core_start, self.core_end)

    @property
    def halo_bounds(self) -> Tuple[int, int]:
        """Returns (halo_start, halo_end) in 0-based global coordinates."""
        return (self.halo_start, self.halo_end)

    @property
    def global_bounds(self) -> Tuple[int, int]:
        """Returns (global_start, global_end) in 0-based global coordinates."""
        return (self.global_start, self.global_end)

    @property
    def local_core_bounds(self) -> Tuple[int, int]:
        """Returns (0, core_length) representing local chunk indices for the core region."""
        return (0, self.core_length)

    @property
    def local_halo_bounds(self) -> Tuple[int, int]:
        """Returns (core_length, actual_length) representing local indices for the halo region."""
        return (self.core_length, self.actual_length)

    @property
    def has_halo(self) -> bool:
        """Indicates whether this chunk has a non-empty trailing halo region."""
        return self.halo_length > 0

    @property
    def nominal_length(self) -> int:
        """Alias for core_length conforming to architectural specification."""
        return self.core_length

    @property
    def global_start_bp(self) -> int:
        """Alias for global_start conforming to architectural specification."""
        return self.global_start

    def to_1based_coordinate(self, local_offset: int) -> int:
        """Translates a 0-based local chunk offset into a 1-based global genomic coordinate.

        Args:
            local_offset: 0-based index within this chunk's sequence.

        Returns:
            1-based global genomic coordinate.
        """
        if local_offset < 0 or local_offset >= self.actual_length:
            raise IndexError(
                f"Local offset {local_offset} out of bounds for chunk of length {self.actual_length}"
            )
        return self.global_start + local_offset + 1

    def is_in_core(self, local_offset: int) -> bool:
        """Checks if a local chunk offset falls within the primary core region."""
        return 0 <= local_offset < self.core_length

    def is_in_halo(self, local_offset: int) -> bool:
        """Checks if a local chunk offset falls within the trailing halo region."""
        return self.core_length <= local_offset < self.actual_length

    def __len__(self) -> int:
        """Returns the actual sequence length of the chunk in base pairs."""
        return self.actual_length

    def __repr__(self) -> str:
        return (
            f"<ChunkDescriptor id='{self.chunk_id}', chrom='{self.chromosome}', "
            f"core=[{self.core_start}, {self.core_end}), halo_len={self.halo_length}, "
            f"actual_len={self.actual_length}>"
        )


class GenomicChunker:
    """Divides genomic DNA sequences into overlapping chunk partitions with coordinate tracking.

    Ensures that:
    - Every genomic coordinate is covered by exactly one core region.
    - Adjacent chunks share an overlapping halo margin (default 22 bp) to capture target
      sites straddling chunk boundaries.
    - Multiple chromosomes/contigs are partitioned independently without coordinate collision.
    """

    DEFAULT_CHUNK_SIZE: int = 10_000_000  # 10 Mbp
    DEFAULT_HALO_SIZE: int = 22           # 22 bp (L - 1 for 23 bp CRISPR window)

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, halo_size: int = DEFAULT_HALO_SIZE):
        """Initializes the GenomicChunker.

        Args:
            chunk_size: Nominal core chunk length in base pairs (default: 10,000,000 bp).
            halo_size: Overlapping trailing halo margin in base pairs (default: 22 bp).

        Raises:
            ValueError: If chunk_size <= 0, halo_size < 0, or chunk_size <= halo_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
        if halo_size < 0:
            raise ValueError(f"halo_size cannot be negative, got {halo_size}")
        if chunk_size <= halo_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be greater than halo_size ({halo_size})"
            )

        self.chunk_size = chunk_size
        self.halo_size = halo_size

    def chunk(
        self,
        sequence: Union[str, np.ndarray, EncodedTwoBit, FastaRecord],
        chromosome: str = "chr1",
    ) -> List[ChunkDescriptor]:
        """Partitions a single sequence into a list of ChunkDescriptor instances.

        Args:
            sequence: Sequence payload (string, 1D uint8 array, EncodedTwoBit, or FastaRecord).
            chromosome: Chromosome/contig identifier (used if sequence is not a FastaRecord).

        Returns:
            List of ChunkDescriptor instances in ascending genomic coordinate order.
        """
        return list(self.stream_chunks(sequence=sequence, chromosome=chromosome))

    def stream_chunks(
        self,
        sequence: Union[str, np.ndarray, EncodedTwoBit, FastaRecord],
        chromosome: str = "chr1",
    ) -> Generator[ChunkDescriptor, None, None]:
        """Lazily generates chunk descriptors for a single sequence.

        Args:
            sequence: Sequence payload (string, 1D uint8 array, EncodedTwoBit, or FastaRecord).
            chromosome: Chromosome/contig identifier.

        Yields:
            ChunkDescriptor instances in ascending genomic coordinate order.
        """
        chrom_id, seq_data, total_length = self._resolve_input(sequence, chromosome)

        if total_length == 0:
            return

        num_chunks = math.ceil(total_length / self.chunk_size)

        for chunk_idx in range(num_chunks):
            core_start = chunk_idx * self.chunk_size
            core_end = min(core_start + self.chunk_size, total_length)

            global_start = core_start
            global_end = min(core_end + self.halo_size, total_length)

            halo_start = core_end
            halo_end = global_end

            actual_len = global_end - global_start
            chunk_slice = self._slice_sequence(seq_data, global_start, global_end)

            chunk_id = f"{chrom_id}_chunk_{chunk_idx:04d}"

            yield ChunkDescriptor(
                chunk_id=chunk_id,
                chromosome=chrom_id,
                chunk_index=chunk_idx,
                global_start=global_start,
                global_end=global_end,
                actual_length=actual_len,
                core_start=core_start,
                core_end=core_end,
                halo_start=halo_start,
                halo_end=halo_end,
                sequence_data=chunk_slice,
            )

    def chunk_records(
        self,
        records: Iterable[FastaRecord],
    ) -> List[ChunkDescriptor]:
        """Partitions multiple FASTA records into chunk descriptors.

        Coordinates reset to 0 for each independent record.

        Args:
            records: Iterable of FastaRecord instances.

        Returns:
            List of ChunkDescriptor instances across all records.
        """
        return list(self.stream_records(records))

    def stream_records(
        self,
        records: Iterable[FastaRecord],
    ) -> Generator[ChunkDescriptor, None, None]:
        """Lazily streams chunk descriptors across multiple FASTA records.

        Args:
            records: Iterable of FastaRecord instances.

        Yields:
            ChunkDescriptor instances across all records.
        """
        for record in records:
            yield from self.stream_chunks(sequence=record)

    def partition(
        self,
        chromosome: str,
        sequence: Union[str, np.ndarray],
    ) -> List[ChunkDescriptor]:
        """Conforming method signature specified in architectural documentation."""
        return self.chunk(sequence=sequence, chromosome=chromosome)

    @staticmethod
    def _resolve_input(
        sequence: Union[str, np.ndarray, EncodedTwoBit, FastaRecord],
        default_chrom: str,
    ) -> Tuple[str, Union[str, np.ndarray, EncodedTwoBit], int]:
        """Resolves sequence length, data representation, and chromosome identifier."""
        if isinstance(sequence, FastaRecord):
            return sequence.identifier, sequence.sequence, len(sequence.sequence)

        if isinstance(sequence, str):
            return default_chrom, sequence, len(sequence)

        if isinstance(sequence, np.ndarray):
            return default_chrom, sequence, sequence.size

        if isinstance(sequence, EncodedTwoBit):
            return default_chrom, sequence, sequence.length

        raise TypeError(
            f"Unsupported sequence type {type(sequence)}. Expected str, np.ndarray, EncodedTwoBit, or FastaRecord."
        )

    @staticmethod
    def _slice_sequence(
        seq_data: Union[str, np.ndarray, EncodedTwoBit],
        start: int,
        end: int,
    ) -> Union[str, np.ndarray, EncodedTwoBit]:
        """Slices the sequence payload preserving the original data representation."""
        if isinstance(seq_data, (str, np.ndarray)):
            return seq_data[start:end]

        if isinstance(seq_data, EncodedTwoBit):
            # Unpack, slice, and re-pack to preserve EncodedTwoBit representation
            unpacked = TwoBitEncoder.unpack_8bit(seq_data)
            sliced = unpacked[start:end]
            return TwoBitEncoder.pack_8bit(sliced, length=sliced.size)

        raise TypeError(f"Cannot slice sequence data of type {type(seq_data)}")
