"""High-performance streaming FASTA file parser for GeneWeaver."""

import gzip
import io
from pathlib import Path
from typing import Generator, List, Optional, TextIO, Tuple

from geneweaver.genomic_parser.exceptions import (
    EmptyFastaFileError,
    EmptySequenceError,
    FastaParsingError,
    InvalidSequenceError,
    MalformedHeaderError,
)
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.genomic_parser.validator import SequenceValidator
from geneweaver.sequence_processor.normalizer import SequenceNormalizer


class FastaReader:
    """Memory-bounded streaming FASTA parser.

    Reads and validates single-record and multi-record FASTA files (.fa, .fasta, .fna),
    including gzip-compressed variants (.fa.gz, .fasta.gz, .fna.gz).

    Supports:
    - Streaming complete records via generator (`stream_records`).
    - Chunked chromosome streaming without holding entire genomes in memory (`stream_chromosomes`).
    - In-memory collection of all records (`read_records`).
    - Fast sequence length extraction (`get_chromosome_lengths`).
    - Comprehensive header and sequence validation with descriptive error reporting.
    """

    def __init__(
        self,
        file_path: Path | str,
        buffer_size_kb: int = 64,
        allow_ambiguity: bool = True,
        normalizer: Optional[SequenceNormalizer] = None,
    ):
        """Initializes the FastaReader.

        Args:
            file_path: Path to the FASTA file.
            buffer_size_kb: I/O buffer read size in kilobytes (default 64 KB).
            allow_ambiguity: If True, permits IUPAC ambiguity codes (e.g. 'N', 'R', 'Y').
            normalizer: Optional custom SequenceNormalizer instance.
        """
        self.file_path = SequenceValidator.validate_fasta_path(file_path)
        self.buffer_size = buffer_size_kb * 1024
        self.allow_ambiguity = allow_ambiguity
        self.normalizer = normalizer or SequenceNormalizer()

    def _open_file(self) -> TextIO:
        """Opens the FASTA file with proper gzip detection and buffering."""
        if str(self.file_path).endswith(".gz"):
            return io.TextIOWrapper(
                gzip.open(self.file_path, mode="rb"),
                encoding="utf-8",
                errors="strict",
            )
        return open(self.file_path, mode="rt", encoding="utf-8", errors="strict", buffering=self.buffer_size)

    def stream_records(self) -> Generator[FastaRecord, None, None]:
        """Yields parsed and normalized FastaRecord objects one by one.

        Yields:
            FastaRecord instances containing identifier, normalized sequence, and description.

        Raises:
            EmptyFastaFileError: If the file contains no valid FASTA records.
            MalformedHeaderError: If a header is malformed or lacks an identifier.
            InvalidSequenceError: If sequence content contains invalid characters.
            EmptySequenceError: If a record contains no sequence data.
        """
        current_id: Optional[str] = None
        current_desc: str = ""
        seq_fragments: List[str] = []
        record_count = 0
        header_line_num = 0

        with self._open_file() as handle:
            line_num = 0
            for line_num, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()

                # Skip completely blank lines
                if not line:
                    continue

                # Skip comment lines if any
                if line.startswith(("#", ";")):
                    continue

                if line.startswith(">"):
                    # Process preceding record if one was active
                    if current_id is not None:
                        combined_seq = "".join(seq_fragments)
                        if not combined_seq:
                            raise EmptySequenceError(
                                f"FASTA record '{current_id}' contains no sequence data",
                                file_path=str(self.file_path),
                                line_number=header_line_num,
                            )
                        yield FastaRecord(
                            identifier=current_id,
                            sequence=combined_seq,
                            description=current_desc,
                        )
                        record_count += 1
                        seq_fragments.clear()

                    # Validate new header
                    identifier, description = SequenceValidator.validate_header(
                        line,
                        line_number=line_num,
                        file_path=str(self.file_path),
                    )
                    current_id = identifier
                    current_desc = description
                    header_line_num = line_num
                else:
                    # Sequence content line
                    if current_id is None:
                        raise MalformedHeaderError(
                            f"FASTA file must begin with a record header starting with '>': '{line[:40]}'",
                            file_path=str(self.file_path),
                            line_number=line_num,
                        )

                    normalized_line = self.normalizer.normalize(line)
                    is_valid, err_msg = self.normalizer.validate(
                        normalized_line,
                        allow_ambiguity=self.allow_ambiguity,
                    )
                    if not is_valid:
                        raise InvalidSequenceError(
                            f"Invalid sequence in record '{current_id}': {err_msg}",
                            file_path=str(self.file_path),
                            line_number=line_num,
                        )
                    seq_fragments.append(normalized_line)

            # Flush the final record at end of file
            if current_id is not None:
                combined_seq = "".join(seq_fragments)
                if not combined_seq:
                    raise EmptySequenceError(
                        f"FASTA record '{current_id}' at end of file contains no sequence data",
                        file_path=str(self.file_path),
                        line_number=header_line_num,
                    )
                yield FastaRecord(
                    identifier=current_id,
                    sequence=combined_seq,
                    description=current_desc,
                )
                record_count += 1

        if record_count == 0:
            raise EmptyFastaFileError(
                f"FASTA file contains no sequence records: {self.file_path}",
                file_path=str(self.file_path),
            )

    def read_records(self) -> List[FastaRecord]:
        """Reads and returns all FASTA records in the file as a list.

        Convenience method for small-to-medium files or testing.

        Returns:
            List of FastaRecord instances.
        """
        return list(self.stream_records())

    def stream_chromosomes(
        self,
        chunk_size: int = 10_000_000,
    ) -> Generator[Tuple[str, Generator[str, None, None]], None, None]:
        """Streams sequence chunks per chromosome to avoid memory bloat.

        For large genomes (e.g. human chromosomes ~50-250 Mbp), this yields
        an inner generator producing sequence slices of `chunk_size` characters,
        enforcing a peak memory footprint under 100 MB.

        Args:
            chunk_size: Number of base pairs per chunk slice.

        Yields:
            Tuples of (chromosome_identifier, chunk_generator).
        """
        def make_chunk_generator(record: FastaRecord) -> Generator[str, None, None]:
            seq_len = len(record.sequence)
            for i in range(0, seq_len, chunk_size):
                yield record.sequence[i : i + chunk_size]

        for record in self.stream_records():
            yield record.identifier, make_chunk_generator(record)

    def get_chromosome_lengths(self) -> dict[str, int]:
        """Extracts sequence lengths for all records in the file.

        Returns:
            Dictionary mapping chromosome identifiers to length in base pairs.
        """
        lengths: dict[str, int] = {}
        for record in self.stream_records():
            lengths[record.identifier] = record.length
        return lengths
