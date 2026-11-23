"""Mismatch analysis engine for GeneWeaver CRISPR alignment.

Provides detailed mismatch characterization between CRISPR guide RNA sequences and
aligned genomic candidate protospacers, including per-base transition identification,
guide-relative coordinate tracking, seed vs. non-seed partitioning, and bitmask representation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from geneweaver.alignment.models import GuideTarget, MismatchInfo, RawCandidateHit, Strand
from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.genomic_parser.validator import SequenceValidator
from geneweaver.scoring.pam_evaluator import ValidatedCandidateHit


@dataclass(frozen=True)
class MismatchAnalysis:
    """Detailed mismatch metadata and positional characterization for a candidate protospacer.

    All mismatch positions use the standard GeneWeaver guide-relative coordinate convention:
    1-based integer indices (1 to spacer_length, typically 1 to 20 for SpCas9) measured
    from 5' to 3' along the guide spacer. Position 1 is distal; position 20 is PAM-proximal.

    Attributes:
        total_mismatches: Total count of mismatched base pairs along the protospacer.
        mismatch_positions: Tuple of 1-based guide-relative mismatch positions in ascending order.
        seed_mismatches: Count of mismatches situated within the seed region.
        seed_positions: Tuple of 1-based guide-relative mismatch positions within the seed region.
        non_seed_mismatches: Count of mismatches situated outside the seed region.
        non_seed_positions: Tuple of 1-based guide-relative mismatch positions outside the seed region.
        bitmask: Integer bitmask where bit i (0 <= i < spacer_length) is 1 if position (i + 1) is mismatched.
        mismatch_details: Tuple of (position_1based, guide_base, target_base) tuples.
        seed_start: 1-based inclusive start index of the seed region (default: 11).
        seed_end: 1-based inclusive end index of the seed region (default: 20).
        spacer_length: Total length of the protospacer in nucleotides (default: 20).
    """

    total_mismatches: int
    mismatch_positions: Tuple[int, ...]
    seed_mismatches: int
    seed_positions: Tuple[int, ...]
    non_seed_mismatches: int
    non_seed_positions: Tuple[int, ...]
    bitmask: int
    mismatch_details: Tuple[Tuple[int, str, str], ...]
    seed_start: int = 11
    seed_end: int = 20
    spacer_length: int = 20

    def __post_init__(self) -> None:
        """Validates internal invariants of mismatch counts and coordinates."""
        if self.total_mismatches < 0 or self.total_mismatches > self.spacer_length:
            raise ValueError(
                f"total_mismatches ({self.total_mismatches}) must be between 0 and spacer_length ({self.spacer_length})"
            )
        if self.seed_mismatches < 0 or self.seed_mismatches > self.total_mismatches:
            raise ValueError(
                f"seed_mismatches ({self.seed_mismatches}) cannot exceed total_mismatches ({self.total_mismatches})"
            )
        if self.non_seed_mismatches < 0 or self.non_seed_mismatches > self.total_mismatches:
            raise ValueError(
                f"non_seed_mismatches ({self.non_seed_mismatches}) cannot exceed total_mismatches ({self.total_mismatches})"
            )
        if self.seed_mismatches + self.non_seed_mismatches != self.total_mismatches:
            raise ValueError(
                f"seed_mismatches ({self.seed_mismatches}) + non_seed_mismatches ({self.non_seed_mismatches}) "
                f"must equal total_mismatches ({self.total_mismatches})"
            )
        if len(self.mismatch_positions) != self.total_mismatches:
            raise ValueError(
                f"Length of mismatch_positions ({len(self.mismatch_positions)}) must equal total_mismatches ({self.total_mismatches})"
            )
        for pos in self.mismatch_positions:
            if pos < 1 or pos > self.spacer_length:
                raise ValueError(
                    f"Mismatch position {pos} out of range [1, {self.spacer_length}]"
                )

    @property
    def is_exact_match(self) -> bool:
        """True if the candidate protospacer has zero mismatches against the guide."""
        return self.total_mismatches == 0

    @property
    def has_seed_mismatches(self) -> bool:
        """True if any mismatches fall within the PAM-proximal seed region."""
        return self.seed_mismatches > 0

    @property
    def guide_relative_positions(self) -> Tuple[int, ...]:
        """Alias for 1-based mismatch positions relative to the guide sequence (5' to 3')."""
        return self.mismatch_positions


class MismatchAnalyzer:
    """Configurable mismatch analysis engine for CRISPR guide-target alignments.

    Extracts, validates, and partitions mismatches between guide RNAs and genomic targets.
    Supports configurable seed-region definitions (default: positions 11-20 for SpCas9)
    and integrates directly with GeneWeaver's existing RawCandidateHit and ValidatedCandidateHit
    models while maintaining guide-relative coordinate conventions across both forward and
    reverse strands.
    """

    def __init__(
        self,
        seed_start: int = 11,
        seed_end: int = 20,
        spacer_length: int = 20,
    ) -> None:
        """Initializes the MismatchAnalyzer with configurable seed region coordinates.

        Args:
            seed_start: 1-based inclusive start index of the seed region (default: 11).
            seed_end: 1-based inclusive end index of the seed region (default: 20).
            spacer_length: Expected length of the guide spacer in base pairs (default: 20).

        Raises:
            ValueError: If seed boundaries or spacer length are invalid.
        """
        if spacer_length <= 0:
            raise ValueError(f"spacer_length must be positive, got {spacer_length}")
        if seed_start < 1 or seed_start > spacer_length:
            raise ValueError(
                f"seed_start ({seed_start}) must be within [1, spacer_length={spacer_length}]"
            )
        if seed_end < seed_start or seed_end > spacer_length:
            raise ValueError(
                f"seed_end ({seed_end}) must be between seed_start ({seed_start}) and spacer_length ({spacer_length})"
            )

        self._seed_start = seed_start
        self._seed_end = seed_end
        self._spacer_length = spacer_length

    @property
    def seed_start(self) -> int:
        """1-based inclusive start index of the seed region."""
        return self._seed_start

    @property
    def seed_end(self) -> int:
        """1-based inclusive end index of the seed region."""
        return self._seed_end

    @property
    def spacer_length(self) -> int:
        """Expected protospacer length in nucleotides."""
        return self._spacer_length

    def analyze_sequences(
        self,
        guide_seq: str,
        target_seq: str,
    ) -> MismatchAnalysis:
        """Compares guide and target nucleotide sequences and produces a MismatchAnalysis.

        Args:
            guide_seq: Guide spacer sequence (at least spacer_length nt, e.g. 20 nt).
            target_seq: Observed genomic target sequence (at least spacer_length nt, e.g. 20 nt protospacer).

        Returns:
            MismatchAnalysis record with detailed mismatch breakdown.

        Raises:
            InvalidSequenceError: If sequences contain non-canonical or invalid bases.
            ValueError: If sequences are shorter than the configured spacer_length.
        """
        norm_guide = self._normalize_sequence(guide_seq, "guide_seq")
        norm_target = self._normalize_sequence(target_seq, "target_seq")

        if len(norm_guide) < self._spacer_length:
            raise ValueError(
                f"guide_seq length ({len(norm_guide)}) is shorter than spacer_length ({self._spacer_length})"
            )
        if len(norm_target) < self._spacer_length:
            raise ValueError(
                f"target_seq length ({len(norm_target)}) is shorter than spacer_length ({self._spacer_length})"
            )

        guide_spacer = norm_guide[: self._spacer_length]
        target_spacer = norm_target[: self._spacer_length]

        mismatch_positions: List[int] = []
        seed_positions: List[int] = []
        non_seed_positions: List[int] = []
        details: List[Tuple[int, str, str]] = []
        bitmask = 0

        for i in range(self._spacer_length):
            g_base = guide_spacer[i]
            t_base = target_spacer[i]
            pos_1based = i + 1

            if g_base != t_base:
                mismatch_positions.append(pos_1based)
                details.append((pos_1based, g_base, t_base))
                bitmask |= (1 << i)

                if self._seed_start <= pos_1based <= self._seed_end:
                    seed_positions.append(pos_1based)
                else:
                    non_seed_positions.append(pos_1based)

        return MismatchAnalysis(
            total_mismatches=len(mismatch_positions),
            mismatch_positions=tuple(mismatch_positions),
            seed_mismatches=len(seed_positions),
            seed_positions=tuple(seed_positions),
            non_seed_mismatches=len(non_seed_positions),
            non_seed_positions=tuple(non_seed_positions),
            bitmask=bitmask,
            mismatch_details=tuple(details),
            seed_start=self._seed_start,
            seed_end=self._seed_end,
            spacer_length=self._spacer_length,
        )

    def analyze_candidate(
        self,
        candidate: Union[ValidatedCandidateHit, RawCandidateHit],
        guide: Union[GuideTarget, str],
    ) -> MismatchAnalysis:
        """Analyzes mismatches for an existing alignment candidate hit.

        Leverages existing mismatch metadata on the candidate when available to avoid
        redundant sequence recomputation, while extracting nucleotide transitions and
        partitioning seed vs. non-seed regions using this analyzer's configuration.

        For both Forward (+) and Reverse (-) candidates:
        Candidate target sequences and mismatch positions are already stored in the
        guide-relative 5' to 3' orientation conforming to GeneWeaver's architectural contract.

        Args:
            candidate: ValidatedCandidateHit or RawCandidateHit instance.
            guide: GuideTarget or string guide sequence.

        Returns:
            MismatchAnalysis record.

        Raises:
            ValueError: If candidate sequence data is insufficient.
        """
        guide_spacer = self._resolve_guide_spacer(guide)
        norm_guide = self._normalize_sequence(guide_spacer, "guide")

        # Extract target sequence if available
        raw_hit = candidate.candidate if isinstance(candidate, ValidatedCandidateHit) else candidate
        target_seq = raw_hit.target_sequence

        # If target sequence is available, we perform complete sequence analysis
        # to capture specific base transitions (pos, guide_base, target_base).
        if target_seq and len(target_seq) >= self._spacer_length:
            return self.analyze_sequences(guide_seq=norm_guide, target_seq=target_seq)

        # Fallback to existing MismatchInfo metadata if full sequence string is missing
        mismatch_info: MismatchInfo = raw_hit.mismatch_info
        all_positions = mismatch_info.positions

        # Partition positions based on this analyzer's seed boundaries
        seed_pos = [p for p in all_positions if self._seed_start <= p <= self._seed_end]
        non_seed_pos = [p for p in all_positions if p < self._seed_start or p > self._seed_end]

        return MismatchAnalysis(
            total_mismatches=mismatch_info.count,
            mismatch_positions=all_positions,
            seed_mismatches=len(seed_pos),
            seed_positions=tuple(seed_pos),
            non_seed_mismatches=len(non_seed_pos),
            non_seed_positions=tuple(non_seed_pos),
            bitmask=mismatch_info.bitmask,
            mismatch_details=(),  # Base details unavailable without target string
            seed_start=self._seed_start,
            seed_end=self._seed_end,
            spacer_length=self._spacer_length,
        )

    def analyze_batch(
        self,
        candidates: Iterable[Union[ValidatedCandidateHit, RawCandidateHit]],
        guide: Union[GuideTarget, str],
    ) -> List[MismatchAnalysis]:
        """Analyzes a collection of candidate hits against a guide sequence.

        Args:
            candidates: Iterable of ValidatedCandidateHit or RawCandidateHit items.
            guide: Target CRISPR guide sequence.

        Returns:
            List of MismatchAnalysis records corresponding to each candidate.
        """
        guide_spacer = self._resolve_guide_spacer(guide)
        return [self.analyze_candidate(c, guide=guide_spacer) for c in candidates]

    def _resolve_guide_spacer(self, guide: Union[GuideTarget, str]) -> str:
        """Extracts the 20 nt spacer string from a GuideTarget or string."""
        if isinstance(guide, GuideTarget):
            return guide.spacer
        if isinstance(guide, str):
            clean = guide.strip()
            # If a 23 nt guide (spacer + PAM) was passed, take first spacer_length bases
            if len(clean) >= self._spacer_length:
                return clean[: self._spacer_length]
            return clean
        raise TypeError(f"Unsupported guide type: {type(guide)}")

    @staticmethod
    def _normalize_sequence(seq: str, name: str) -> str:
        """Normalizes and validates DNA/RNA nucleotide characters.

        Converts RNA 'U' to 'T' and standardizes to uppercase.

        Raises:
            InvalidSequenceError: If non-canonical characters are found.
        """
        if not seq:
            raise ValueError(f"{name} cannot be empty")
        upper = seq.strip().upper().replace("U", "T")
        invalid = [char for char in upper if char not in "ACGT"]
        if invalid:
            raise InvalidSequenceError(
                f"Invalid nucleotide character(s) in {name}: {set(invalid)}. Only A, C, G, T are permitted."
            )
        return upper
