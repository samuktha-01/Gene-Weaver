"""CPU sequence alignment engine using NumPy vectorized sliding-window search.

This module provides the CPU baseline alignment engine (CPUAligner) conforming to the
SequenceAligner protocol. It implements vectorized sliding-window sequence comparison
using NumPy strided views, zero-copy batch evaluation, and coordinate boundary tracking.
"""

from dataclasses import dataclass
import time
from typing import Iterable, List, Optional, Union

import numpy as np

from geneweaver.alignment.base import BaseSequenceAligner
from geneweaver.alignment.models import (
    AlignmentConfig,
    AlignmentResult,
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    Strand,
)
from geneweaver.alignment.reverse_complement import reverse_complement
from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.genomic_parser.validator import SequenceValidator
from geneweaver.sequence_processor.chunker import ChunkDescriptor
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder


class CPUAligner(BaseSequenceAligner):
    """CPU sequence alignment engine using NumPy vectorized sliding-window comparison.

    Implements off-target candidate detection by scanning genomic chunks against a
    CRISPR guide sequence. Uses strided 2D NumPy array views (`sliding_window_view`)
    for zero-copy candidate window generation and matrix multiplication for vectorized
    Hamming distance and mismatch bitmask calculations.

    Key Architectural Features:
    - Zero-copy windowing via `np.lib.stride_tricks.sliding_window_view`.
    - Bounded intermediate memory using chunk batching (`batch_size` in AlignmentConfig).
    - Halo deduplication: strictly retains candidates starting within the chunk core,
      preventing duplicate hits across adjacent partitioned chunks.
    - Full coordinate preservation: maintains 0-based global coordinates and 1-based
      genomic coordinates.
    - Deterministic output ordering sorted by genomic position.
    """

    @property
    def backend_name(self) -> str:
        """Returns the unique backend execution identifier."""
        return "cpu-numpy"

    @classmethod
    def _resolve_guide(cls, guide: Union[GuideTarget, str, np.ndarray]) -> GuideTarget:
        """Resolves guide input into a typed GuideTarget instance.

        Accepts:
        - Pre-constructed GuideTarget instances.
        - ASCII strings of length 20 (protospacer) or 23 (protospacer + PAM).
        - 1D NumPy uint8 arrays in {0, 1, 2, 3} of length 20 or 23.

        Args:
            guide: Target guide sequence.

        Returns:
            Validated GuideTarget instance.

        Raises:
            TypeError: If guide is neither GuideTarget, str, nor np.ndarray.
            ValueError: If guide length is not 20 or 23.
            InvalidSequenceError: If guide contains unsupported nucleotide characters.
        """
        if isinstance(guide, GuideTarget):
            return guide

        if isinstance(guide, str):
            clean = guide.strip()
            if len(clean) == 20:
                return GuideTarget.from_string(clean, expected_length=20)
            elif len(clean) == 23:
                return GuideTarget.from_string(clean, expected_length=23)
            else:
                raise ValueError(
                    f"Invalid CRISPR guide length: got {len(clean)} nt, expected 20 nt (or 23 nt with PAM)"
                )

        if isinstance(guide, np.ndarray):
            if guide.ndim != 1:
                raise ValueError(f"Guide array must be 1D, got shape {guide.shape}")
            if guide.size not in (20, 23):
                raise ValueError(
                    f"Invalid CRISPR guide length: got {guide.size} nt, expected 20 nt (or 23 nt with PAM)"
                )

            if guide.dtype == np.uint8 and np.all(guide <= 3):
                decoded = TwoBitEncoder.decode(guide)
            elif guide.dtype == np.uint8 and np.all((guide >= 65) & (guide <= 122)):
                decoded = guide.tobytes().decode("ascii")
            elif guide.dtype.kind in ("S", "U"):
                decoded = "".join(guide.astype(str))
            else:
                if np.any(guide > 3):
                    raise InvalidSequenceError(
                        "Invalid nucleotide values in guide array; values must be in 0..3 or valid ASCII"
                    )
                decoded = TwoBitEncoder.decode(guide.astype(np.uint8))

            clean = decoded.strip()
            if len(clean) == 20:
                return GuideTarget.from_string(clean, expected_length=20)
            elif len(clean) == 23:
                return GuideTarget.from_string(clean, expected_length=23)
            else:
                raise ValueError(
                    f"Invalid CRISPR guide length: got {len(clean)} nt, expected 20 nt (or 23 nt with PAM)"
                )

        raise TypeError(f"Expected guide to be GuideTarget, str, or np.ndarray, got {type(guide).__name__}")

    def _align_chunk(
        self,
        chunk: ChunkDescriptor,
        guide: GuideTarget,
        effective_config: AlignmentConfig,
    ) -> AlignmentResult:
        """Executes vectorized NumPy sliding-window alignment on a single genomic chunk.

        Evaluates Forward (+) and optionally Reverse (-) strands according to
        `effective_config.search_reverse_strand`. Preserves global genomic coordinates,
        core deduplication invariants, and bounded batch processing.

        Args:
            chunk: Validated ChunkDescriptor with sequence payload.
            guide: Resolved GuideTarget instance.
            effective_config: Active AlignmentConfig containing thresholds and batch size.

        Returns:
            AlignmentResult record containing candidate hits and execution telemetry.

        Raises:
            InvalidSequenceError: If genomic chunk contains non-canonical/invalid nucleotides.
        """
        start_time = time.perf_counter()

        # Resolve sequence payload to 1D unpacked 2-bit uint8 array (values in {0, 1, 2, 3})
        seq_2bit, seq_str = self._resolve_chunk_sequence(chunk.sequence_data)
        seq_len = seq_2bit.size

        # Resolve 20 nt guide target spacer to align
        guide_seq = guide.spacer if guide.spacer else guide.sequence
        guide_len = len(guide_seq)
        guide_encoded = guide.encoded_2bit[:guide_len]

        # Handle empty sequences or sequences shorter than guide
        if seq_len < guide_len:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AlignmentResult(
                chunk_id=chunk.chunk_id,
                chromosome=chunk.chromosome,
                hits=[],
                total_windows_evaluated=0,
                execution_time_ms=elapsed_ms,
                backend=self.backend_name,
                config=effective_config,
            )

        num_windows_per_strand = seq_len - guide_len + 1
        powers = (1 << np.arange(guide_len, dtype=np.uint32))
        hits: List[RawCandidateHit] = []

        # 1. Forward Strand (+) Alignment
        fwd_hits = self._scan_single_strand(
            seq_2bit=seq_2bit,
            seq_str=seq_str,
            guide_encoded=guide_encoded,
            guide_len=guide_len,
            strand=Strand.FORWARD,
            chunk=chunk,
            effective_config=effective_config,
            powers=powers,
        )
        hits.extend(fwd_hits)

        # 2. Reverse Strand (-) Alignment
        if effective_config.search_reverse_strand:
            # Vectorized reverse complement: in 2-bit encoding, comp(x) = x ^ 3; reverse via [::-1]
            rc_seq_2bit = np.ascontiguousarray((seq_2bit ^ 3)[::-1])
            rc_seq_str = reverse_complement(seq_str) if seq_str is not None else None

            rev_hits = self._scan_single_strand(
                seq_2bit=rc_seq_2bit,
                seq_str=rc_seq_str,
                guide_encoded=guide_encoded,
                guide_len=guide_len,
                strand=Strand.REVERSE,
                chunk=chunk,
                effective_config=effective_config,
                powers=powers,
            )
            hits.extend(rev_hits)

        # Sort candidate hits deterministically by local_offset, then strand (FORWARD 0 before REVERSE 1)
        if len(hits) > 1:
            hits.sort(key=lambda h: (h.local_offset, h.strand))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AlignmentResult(
            chunk_id=chunk.chunk_id,
            chromosome=chunk.chromosome,
            hits=hits,
            total_windows_evaluated=num_windows_per_strand,
            execution_time_ms=elapsed_ms,
            backend=self.backend_name,
            config=effective_config,
        )

    def _scan_single_strand(
        self,
        seq_2bit: np.ndarray,
        seq_str: Optional[str],
        guide_encoded: np.ndarray,
        guide_len: int,
        strand: Strand,
        chunk: ChunkDescriptor,
        effective_config: AlignmentConfig,
        powers: np.ndarray,
    ) -> List[RawCandidateHit]:
        """Scans a single strand sequence (forward or reverse-complement) using vectorized sliding windows.

        Args:
            seq_2bit: 1D unpacked 2-bit array of the strand sequence.
            seq_str: Optional ASCII string representation of the strand sequence.
            guide_encoded: 1D unpacked 2-bit array of the guide spacer.
            guide_len: Length of the guide spacer in nucleotides.
            strand: Strand orientation (Strand.FORWARD or Strand.REVERSE).
            chunk: ChunkDescriptor containing genomic coordinates and core boundaries.
            effective_config: Alignment thresholds and batch size.
            powers: Bitshift power array for dot-product bitmask construction.

        Returns:
            List of detected candidate hits strictly passing core filtering and mismatch cutoffs.
        """
        seq_len = seq_2bit.size
        num_windows = seq_len - guide_len + 1
        batch_size = effective_config.batch_size
        core_length = chunk.core_length
        hits: List[RawCandidateHit] = []

        # Process sliding windows in memory-bounded batches
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)

            # Slice sequence segment covering all windows in this batch
            chunk_slice = seq_2bit[batch_start : batch_end + guide_len - 1]
            batch_windows = np.lib.stride_tricks.sliding_window_view(
                chunk_slice, window_shape=guide_len
            )

            # Vectorized mismatch evaluation: (batch_count, guide_len) boolean array
            diffs = batch_windows != guide_encoded
            mismatch_counts = np.count_nonzero(diffs, axis=1)

            # Filter candidates within maximum mismatch threshold
            cand_mask = mismatch_counts <= effective_config.max_mismatches
            if not np.any(cand_mask):
                continue

            match_indices = np.flatnonzero(cand_mask)
            scan_offsets = batch_start + match_indices

            # Target sequence length (up to 23 bp, or remaining bases if near end)
            target_lens = np.minimum(23, seq_len - scan_offsets)

            # Coordinate translation: map scan offsets to 0-based local offsets on forward reference chunk
            if strand == Strand.FORWARD:
                fwd_offsets = scan_offsets
            else:
                # Reverse strand: index j in S_rev corresponds to forward local offset (N - j - target_len)
                fwd_offsets = seq_len - scan_offsets - target_lens

            # Halo deduplication: strictly retain candidates starting within the chunk core
            core_mask = (fwd_offsets < core_length) & (fwd_offsets >= 0)
            if not np.any(core_mask):
                continue

            filtered_match_indices = match_indices[core_mask]
            filtered_fwd_offsets = fwd_offsets[core_mask]
            filtered_scan_offsets = scan_offsets[core_mask]
            filtered_target_lens = target_lens[core_mask]
            filtered_diffs = diffs[filtered_match_indices]

            # Vectorized bitmask computation: diff matrix dot product with powers of 2
            bitmasks = filtered_diffs.astype(np.uint32) @ powers

            # Materialize RawCandidateHit instances for passing hits
            for idx in range(len(filtered_fwd_offsets)):
                fwd_offset = int(filtered_fwd_offsets[idx])
                scan_offset = int(filtered_scan_offsets[idx])
                target_len = int(filtered_target_lens[idx])
                bitmask = int(bitmasks[idx])

                mismatch_info = MismatchInfo.from_bitmask(
                    bitmask=bitmask,
                    spacer_length=guide_len,
                    seed_length=effective_config.seed_length,
                )

                global_start = chunk.global_start + fwd_offset
                global_end = global_start + target_len
                genomic_pos = chunk.to_1based_coordinate(fwd_offset)

                # Extract target sequence (5' to 3' of the matching strand) and PAM sequence
                if seq_str is not None:
                    target_seq = seq_str[scan_offset : scan_offset + target_len]
                else:
                    target_seq = TwoBitEncoder.decode(seq_2bit[scan_offset : scan_offset + target_len])

                pam_seq = target_seq[guide_len:] if target_len > guide_len else ""

                hit = RawCandidateHit(
                    chunk_id=chunk.chunk_id,
                    chromosome=chunk.chromosome,
                    local_offset=fwd_offset,
                    global_start=global_start,
                    global_end=global_end,
                    genomic_position=genomic_pos,
                    strand=strand,
                    mismatch_info=mismatch_info,
                    pam_sequence=pam_seq,
                    target_sequence=target_seq,
                )
                hits.append(hit)

        return hits

    def align_chunks(
        self,
        chunks: Iterable[ChunkDescriptor],
        guide: Union[GuideTarget, str, np.ndarray],
        config: Optional[AlignmentConfig] = None,
    ) -> List[AlignmentResult]:
        """Aligns a CRISPR guide sequence across multiple genomic chunks sequentially.

        Args:
            chunks: Iterable of ChunkDescriptor partitions.
            guide: GuideTarget, string, or 1D NumPy array.
            config: Optional override configuration.

        Returns:
            List of AlignmentResult records in chunk order.
        """
        return [self.align(chunk=c, guide=guide, config=config) for c in chunks]

    def align_and_validate(
        self,
        chunk: ChunkDescriptor,
        guide: Union[GuideTarget, str, np.ndarray],
        config: Optional[AlignmentConfig] = None,
        validator: Optional[object] = None,
        filter_incompatible: bool = False,
    ) -> List[object]:
        """Aligns a CRISPR guide against a genomic chunk and validates PAM motifs on candidates.

        Args:
            chunk: Genomic chunk descriptor.
            guide: Target guide sequence.
            config: Optional alignment configuration overrides.
            validator: Optional PAMValidator instance (defaults to standard SpCas9 validator).
            filter_incompatible: If True, filters out candidate hits lacking a valid PAM motif.

        Returns:
            List of ValidatedCandidateHit instances containing validation status and coordinates.
        """
        from geneweaver.scoring.pam_evaluator import PAMValidator
        pam_validator = validator if isinstance(validator, PAMValidator) else PAMValidator()
        result = self.align(chunk=chunk, guide=guide, config=config)
        canonical_only = (
            config.canonical_pam_only if config is not None else self.config.canonical_pam_only
        )
        return pam_validator.validate_alignment_result(
            result=result,
            canonical_only=canonical_only,
            filter_incompatible=filter_incompatible,
        )

    def align_chunks_and_validate(
        self,
        chunks: Iterable[ChunkDescriptor],
        guide: Union[GuideTarget, str, np.ndarray],
        config: Optional[AlignmentConfig] = None,
        validator: Optional[object] = None,
        filter_incompatible: bool = False,
    ) -> List[object]:
        """Aligns a CRISPR guide across multiple genomic chunks and validates PAM motifs.

        Args:
            chunks: Iterable of ChunkDescriptor partitions.
            guide: Target guide sequence.
            config: Optional alignment configuration overrides.
            validator: Optional PAMValidator instance.
            filter_incompatible: If True, returns only candidates with valid PAM motifs.

        Returns:
            List of ValidatedCandidateHit instances across all chunks.
        """
        from geneweaver.scoring.pam_evaluator import PAMValidator
        pam_validator = validator if isinstance(validator, PAMValidator) else PAMValidator()
        validated_hits = []
        for chunk in chunks:
            validated_hits.extend(
                self.align_and_validate(
                    chunk=chunk,
                    guide=guide,
                    config=config,
                    validator=pam_validator,
                    filter_incompatible=filter_incompatible,
                )
            )
        return validated_hits

    def align_validate_and_rank(
        self,
        chunk: ChunkDescriptor,
        guide: Union[GuideTarget, str, np.ndarray],
        config: Optional[AlignmentConfig] = None,
        validator: Optional[object] = None,
        filter_incompatible: bool = False,
        scorer: Optional[object] = None,
        ranking_config: Optional[object] = None,
    ) -> object:
        """Executes full alignment, PAM validation, mismatch scoring, and deterministic ranking.

        Args:
            chunk: Genomic chunk descriptor.
            guide: Target CRISPR guide sequence.
            config: Optional alignment configuration overrides.
            validator: Optional PAMValidator instance.
            filter_incompatible: If True, filters out PAM-incompatible hits before ranking.
            scorer: Optional BaseMismatchScorer (defaults to HsuZhangScorer).
            ranking_config: Optional RankingConfig thresholds.

        Returns:
            CandidateRankingResult containing ranked off-target candidates.
        """
        from geneweaver.scoring.candidate_ranker import CandidateRanker
        validated_hits = self.align_and_validate(
            chunk=chunk,
            guide=guide,
            config=config,
            validator=validator,
            filter_incompatible=filter_incompatible,
        )
        ranker = CandidateRanker()
        return ranker.rank_validated_hits(
            validated_hits=validated_hits,
            guide=guide,
            scorer=scorer,
            ranking_config=ranking_config,
        )

    @staticmethod
    def _resolve_chunk_sequence(
        seq_data: Union[str, np.ndarray, EncodedTwoBit],
    ) -> tuple[np.ndarray, Optional[str]]:
        """Validates and converts chunk sequence data to 1D unpacked 2-bit uint8 array.

        Args:
            seq_data: Sequence data payload from ChunkDescriptor.

        Returns:
            Tuple of (unpacked_2bit_array, original_string_if_available).

        Raises:
            InvalidSequenceError: If sequence contains invalid or ambiguous nucleotides.
            TypeError: If sequence data type is unsupported.
        """
        if isinstance(seq_data, str):
            if len(seq_data) == 0:
                return np.empty(0, dtype=np.uint8), seq_data
            seq_2bit = TwoBitEncoder.encode(seq_data)
            return seq_2bit, seq_data

        if isinstance(seq_data, np.ndarray):
            if seq_data.ndim != 1:
                raise ValueError(f"Chunk sequence array must be 1D, got shape {seq_data.shape}")
            if seq_data.size == 0:
                return np.empty(0, dtype=np.uint8), None

            if seq_data.dtype == np.uint8 and np.all(seq_data <= 3):
                return seq_data, None

            if seq_data.dtype == np.uint8:
                # ASCII byte array
                encoded = TwoBitEncoder._ASCII_TO_2BIT[seq_data]
                invalid_mask = encoded == 255
                if np.any(invalid_mask):
                    first_idx = int(np.argmax(invalid_mask))
                    inv_char = chr(seq_data[first_idx])
                    raise InvalidSequenceError(
                        f"Unsupported or ambiguous nucleotide character {repr(inv_char)} at index {first_idx}."
                    )
                return encoded, None

            if seq_data.dtype.kind in ("S", "U"):
                joined_str = "".join(seq_data.astype(str))
                seq_2bit = TwoBitEncoder.encode(joined_str)
                return seq_2bit, joined_str

            if np.any(seq_data > 3):
                raise InvalidSequenceError("Invalid 2-bit values > 3 in sequence array")
            return seq_data.astype(np.uint8), None

        if isinstance(seq_data, EncodedTwoBit):
            if len(seq_data) == 0:
                return np.empty(0, dtype=np.uint8), None
            seq_2bit = TwoBitEncoder.unpack_8bit(seq_data)
            return seq_2bit, None

        raise TypeError(
            f"Unsupported sequence data type: {type(seq_data).__name__}. "
            f"Expected str, np.ndarray, or EncodedTwoBit."
        )


__all__ = ["CPUAligner"]
