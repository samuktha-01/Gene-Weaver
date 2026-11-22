"""Modular, strand-aware PAM (Protospacer Adjacent Motif) recognition and validation engine.

This module provides reusable, nuclease-agnostic PAM validation supporting standard IUPAC
degenerate nucleotide symbols, strand-aware PAM extraction, exact genomic coordinate mapping,
and seamless integration with GeneWeaver alignment results.

Coordinate Conventions:
    - 0-based half-open intervals: [start, end)
      - Used for internal slicing, BED format, and chunk-local indexing.
    - 1-based closed intervals: [start_1based, end_1based]
      - Used for genomic display, VCF/GFF reporting, and IGV visualization.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Dict, Iterable, List, Optional, Pattern, Set, Tuple, Union

from geneweaver.alignment.models import (
    AlignmentConfig,
    AlignmentResult,
    MismatchInfo,
    RawCandidateHit,
    Strand,
)
from geneweaver.alignment.reverse_complement import reverse_complement
from geneweaver.sequence_processor.chunker import ChunkDescriptor


class PAMLocation(Enum):
    """Location of the PAM motif relative to the target protospacer."""

    THREE_PRIME = "3prime"  # Downstream of protospacer in 5'->3' orientation (e.g. SpCas9, SaCas9)
    FIVE_PRIME = "5prime"    # Upstream of protospacer in 5'->3' orientation (e.g. Cas12a/Cpf1)


class IUPACMatcher:
    """IUPAC nucleotide pattern matching and regex compilation utility."""

    # IUPAC nucleotide ambiguity table
    IUPAC_MAP: Dict[str, Set[str]] = {
        "A": {"A"},
        "C": {"C"},
        "G": {"G"},
        "T": {"T"},
        "U": {"T"},  # RNA Uracil maps to DNA Thymine
        "R": {"A", "G"},
        "Y": {"C", "T"},
        "S": {"G", "C"},
        "W": {"A", "T"},
        "K": {"G", "T"},
        "M": {"A", "C"},
        "B": {"C", "G", "T"},
        "D": {"A", "G", "T"},
        "H": {"A", "C", "T"},
        "V": {"A", "C", "G"},
        "N": {"A", "C", "G", "T"},
    }

    # IUPAC character to regular expression group mapping
    IUPAC_REGEX_MAP: Dict[str, str] = {
        "A": "A",
        "C": "C",
        "G": "G",
        "T": "T",
        "U": "T",
        "R": "[AG]",
        "Y": "[CT]",
        "S": "[GC]",
        "W": "[AT]",
        "K": "[GT]",
        "M": "[AC]",
        "B": "[CGT]",
        "D": "[AGT]",
        "H": "[ACT]",
        "V": "[ACG]",
        "N": "[ACGT]",
    }

    @classmethod
    def validate_pattern(cls, pattern: str) -> str:
        """Validates that a PAM pattern consists solely of valid IUPAC nucleotide codes.

        Args:
            pattern: Raw PAM pattern string (e.g. 'NGG', 'NAG', 'TTTV').

        Returns:
            Normalized uppercase DNA pattern string.

        Raises:
            ValueError: If pattern is empty or contains non-IUPAC characters.
        """
        if not pattern or not isinstance(pattern, str):
            raise ValueError("PAM pattern must be a non-empty string.")

        clean = pattern.strip().upper().replace("U", "T")
        if not clean:
            raise ValueError("PAM pattern cannot be empty or purely whitespace.")

        invalid = set(clean) - set(cls.IUPAC_MAP.keys())
        if invalid:
            raise ValueError(
                f"Invalid IUPAC character(s) in PAM pattern '{pattern}': {sorted(invalid)}. "
                f"Allowed characters: {sorted(cls.IUPAC_MAP.keys())}"
            )

        return clean

    @classmethod
    def pattern_to_regex(cls, pattern: str) -> Pattern[str]:
        """Compiles an IUPAC motif string into a strict regex Pattern."""
        clean = cls.validate_pattern(pattern)
        regex_str = "^" + "".join(cls.IUPAC_REGEX_MAP[c] for c in clean) + "$"
        return re.compile(regex_str)

    @classmethod
    def matches(cls, sequence: str, pattern: str) -> bool:
        """Checks if a nucleotide sequence matches an IUPAC pattern.

        Args:
            sequence: Observed nucleotide sequence (case-insensitive).
            pattern: IUPAC PAM pattern.

        Returns:
            True if sequence strictly matches pattern length and nucleotide constraints.
        """
        if not sequence or not isinstance(sequence, str):
            return False

        clean_seq = sequence.strip().upper().replace("U", "T")
        try:
            compiled = cls.pattern_to_regex(pattern)
            return bool(compiled.match(clean_seq))
        except ValueError:
            return False


@dataclass(frozen=True)
class PAMPattern:
    """Represents a specific PAM motif pattern with biological weighting.

    Attributes:
        motif: IUPAC nucleotide pattern (e.g. 'NGG', 'NAG', 'NGA').
        is_canonical: Whether this is the primary canonical motif for the nuclease.
        cleavage_multiplier: Biological cleavage factor (e.g. 1.0 for NGG, 0.25 for NAG, 0.10 for NGA).
        name: Human-readable identifier.
    """

    motif: str
    is_canonical: bool = True
    cleavage_multiplier: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        clean = IUPACMatcher.validate_pattern(self.motif)
        object.__setattr__(self, "motif", clean)
        if self.cleavage_multiplier < 0.0:
            raise ValueError(f"cleavage_multiplier cannot be negative, got {self.cleavage_multiplier}")
        if not self.name:
            label = "canonical" if self.is_canonical else "non-canonical"
            object.__setattr__(self, "name", f"{label}-{clean}")

    @property
    def length(self) -> int:
        """Returns the length of the PAM motif in base pairs."""
        return len(self.motif)


@dataclass(frozen=True)
class NucleaseConfig:
    """CRISPR nuclease configuration defining PAM motif requirements and protospacer geometry.

    Attributes:
        name: Nuclease identifier (e.g. 'SpCas9', 'SaCas9', 'Cas12a').
        canonical_motif: Primary canonical PAM pattern (e.g. 'NGG').
        alternate_motifs: Tuple of alternate/non-canonical PAM patterns or PAMPattern instances.
        pam_location: Location of PAM relative to protospacer (THREE_PRIME or FIVE_PRIME).
        protospacer_length: Length of target spacer in base pairs (default: 20 nt).
    """

    name: str = "SpCas9"
    canonical_motif: str = "NGG"
    alternate_motifs: Tuple[Union[str, PAMPattern], ...] = ()
    pam_location: PAMLocation = PAMLocation.THREE_PRIME
    protospacer_length: int = 20

    def __post_init__(self) -> None:
        clean_canonical = IUPACMatcher.validate_pattern(self.canonical_motif)
        object.__setattr__(self, "canonical_motif", clean_canonical)
        if self.protospacer_length <= 0:
            raise ValueError(f"protospacer_length must be positive, got {self.protospacer_length}")

    @property
    def pam_length(self) -> int:
        """Returns the nominal canonical PAM length in base pairs."""
        return len(self.canonical_motif)


# Pre-configured nuclease profiles
SPCAS9 = NucleaseConfig(
    name="SpCas9",
    canonical_motif="NGG",
    alternate_motifs=(),
    pam_location=PAMLocation.THREE_PRIME,
    protospacer_length=20,
)


@dataclass(frozen=True)
class PAMValidationResult:
    """Detailed result of validating a candidate PAM sequence against registered patterns.

    Attributes:
        is_valid: True if candidate matches any registered canonical or alternate PAM pattern.
        is_canonical: True if candidate matches the primary canonical PAM pattern.
        matched_motif: The specific IUPAC motif pattern matched (e.g. 'NGG', 'NAG'), or None.
        cleavage_multiplier: Biological cleavage efficiency factor (1.0 for NGG, 0.25 for NAG, 0.0 if invalid).
        pam_sequence: Observed PAM nucleotide sequence in 5' to 3' orientation.
        details: Human-readable diagnostic description of validation outcome.
    """

    is_valid: bool
    is_canonical: bool
    matched_motif: Optional[str]
    cleavage_multiplier: float
    pam_sequence: str
    details: str

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass(frozen=True)
class ValidatedCandidateHit:
    """Enriched alignment candidate hit with strand-aware PAM validation and explicit coordinates.

    Attributes:
        candidate: Underlying RawCandidateHit from alignment engine.
        pam_result: PAMValidationResult detailing validation status and matched motif.
        chromosome: Chromosome or contig accession identifier.
        strand: Genomic strand orientation (Strand.FORWARD or Strand.REVERSE).
        protospacer_start: 0-based inclusive global genomic start coordinate of protospacer.
        protospacer_end: 0-based exclusive global genomic end coordinate of protospacer.
        pam_start: 0-based inclusive global genomic start coordinate of PAM.
        pam_end: 0-based exclusive global genomic end coordinate of PAM.
        pam_sequence: 5' to 3' PAM sequence in the target strand orientation.
        target_sequence: Observed target sequence on the aligned strand.
    """

    candidate: RawCandidateHit
    pam_result: PAMValidationResult
    chromosome: str
    strand: Strand
    protospacer_start: int
    protospacer_end: int
    pam_start: int
    pam_end: int
    pam_sequence: str
    target_sequence: str

    @property
    def is_valid_pam(self) -> bool:
        """True if the candidate site possesses a compatible PAM motif."""
        return self.pam_result.is_valid

    @property
    def is_canonical_pam(self) -> bool:
        """True if the candidate site possesses the primary canonical PAM motif."""
        return self.pam_result.is_canonical

    @property
    def cleavage_multiplier(self) -> float:
        """Biological PAM cleavage multiplier (e.g. 1.0 for NGG, 0.25 for NAG, 0.0 for invalid)."""
        return self.pam_result.cleavage_multiplier

    @property
    def matched_motif(self) -> Optional[str]:
        """Matched IUPAC pattern motif (e.g. 'NGG', 'NAG') or None."""
        return self.pam_result.matched_motif

    @property
    def mismatch_count(self) -> int:
        """Total number of mismatches along the protospacer."""
        return self.candidate.mismatch_count

    @property
    def mismatch_bitmask(self) -> int:
        """Integer bitmask of mismatch locations along the protospacer."""
        return self.candidate.mismatch_bitmask

    @property
    def mismatch_info(self) -> MismatchInfo:
        """MismatchInfo record with positional metadata."""
        return self.candidate.mismatch_info

    @property
    def mismatch_positions(self) -> Tuple[int, ...]:
        """1-based mismatch positions relative to the guide sequence (5' to 3', 1..20)."""
        return self.candidate.mismatch_positions

    @property
    def guide_mismatch_positions(self) -> Tuple[int, ...]:
        """Alias for 1-based mismatch positions relative to the guide sequence (5' to 3')."""
        return self.candidate.guide_mismatch_positions

    @property
    def genomic_mismatch_positions(self) -> Tuple[int, ...]:
        """1-based absolute genomic coordinates of mismatches on the forward reference chromosome."""
        return self.candidate.genomic_mismatch_positions

    @property
    def forward_relative_mismatch_positions(self) -> Tuple[int, ...]:
        """1-based mismatch positions relative to the forward genomic reference window."""
        return self.candidate.forward_relative_mismatch_positions

    @property
    def reference_accession(self) -> str:
        """Chromosome or reference sequence accession."""
        return self.chromosome

    @property
    def strand_symbol(self) -> str:
        """'+' for Forward strand, '-' for Reverse strand."""
        return self.strand.sign

    # 1-based closed coordinate conventions for genomic reporting (VCF/GFF)
    @property
    def protospacer_start_1based(self) -> int:
        """1-based inclusive genomic start coordinate of the protospacer."""
        return self.protospacer_start + 1

    @property
    def protospacer_end_1based(self) -> int:
        """1-based inclusive genomic end coordinate of the protospacer."""
        return self.protospacer_end

    @property
    def pam_start_1based(self) -> int:
        """1-based inclusive genomic start coordinate of the PAM motif."""
        return self.pam_start + 1

    @property
    def pam_end_1based(self) -> int:
        """1-based inclusive genomic end coordinate of the PAM motif."""
        return self.pam_end

    @property
    def global_start(self) -> int:
        """0-based inclusive global genomic start of the full candidate window."""
        return self.candidate.global_start

    @property
    def global_end(self) -> int:
        """0-based exclusive global genomic end of the full candidate window."""
        return self.candidate.global_end

    @property
    def genomic_position(self) -> int:
        """1-based inclusive genomic start of the full candidate window."""
        return self.candidate.genomic_position

    @property
    def forward_sequence(self) -> str:
        """Observed sequence on the forward (+) reference chromosome."""
        return self.candidate.forward_sequence

    @property
    def reference_sequence(self) -> str:
        """Alias for forward_sequence."""
        return self.candidate.reference_sequence


class PAMValidator:
    """Modular, strand-aware PAM recognition and validation engine for CRISPR target sites.

    Supports configurable nuclease profiles, multiple canonical and alternate IUPAC patterns,
    forward and reverse strand coordinate translation, and zero-loss integration with
    alignment engine outputs.
    """

    def __init__(self, config: Optional[NucleaseConfig] = None):
        """Initializes the PAM validator with a nuclease configuration.

        Args:
            config: NucleaseConfig instance (defaults to SpCas9).
        """
        self.config = config or SPCAS9
        self._patterns: List[PAMPattern] = []
        self._compiled_patterns: List[Tuple[PAMPattern, Pattern[str]]] = []
        self._init_registered_patterns()

    def _init_registered_patterns(self) -> None:
        """Registers canonical and alternate motifs defined by the active nuclease configuration."""
        # 1. Canonical pattern
        self.register_pattern(
            motif=self.config.canonical_motif,
            is_canonical=True,
            cleavage_multiplier=1.0,
            name=f"canonical-{self.config.canonical_motif}",
        )

        # 2. Alternate patterns
        for alt in self.config.alternate_motifs:
            if isinstance(alt, PAMPattern):
                self.register_pattern(
                    motif=alt.motif,
                    is_canonical=alt.is_canonical,
                    cleavage_multiplier=alt.cleavage_multiplier,
                    name=alt.name,
                )
            else:
                # Standard SpCas9 empirical weights (Doench 2016): NAG = 0.25, NGA = 0.10
                mult = 0.25 if alt == "NAG" else (0.10 if alt == "NGA" else 0.50)
                self.register_pattern(
                    motif=alt,
                    is_canonical=False,
                    cleavage_multiplier=mult,
                    name=f"alternate-{alt}",
                )

    def register_pattern(
        self,
        motif: str,
        is_canonical: bool = False,
        cleavage_multiplier: float = 1.0,
        name: str = "",
    ) -> PAMPattern:
        """Registers a new PAM motif pattern into the validation engine.

        Args:
            motif: IUPAC nucleotide pattern string (e.g. 'NGG', 'NAG', 'TTTV').
            is_canonical: If True, designates this motif as canonical.
            cleavage_multiplier: Cleavage efficiency factor (default: 1.0).
            name: Optional descriptive label.

        Returns:
            The registered PAMPattern instance.

        Raises:
            ValueError: If pattern is invalid or cleavage_multiplier is negative.
        """
        clean_motif = IUPACMatcher.validate_pattern(motif)
        compiled_re = IUPACMatcher.pattern_to_regex(clean_motif)
        pattern = PAMPattern(
            motif=clean_motif,
            is_canonical=is_canonical,
            cleavage_multiplier=cleavage_multiplier,
            name=name or clean_motif,
        )
        self._patterns.append(pattern)
        self._compiled_patterns.append((pattern, compiled_re))
        return pattern

    @property
    def registered_patterns(self) -> Tuple[PAMPattern, ...]:
        """Returns all registered PAM patterns."""
        return tuple(self._patterns)

    def matches_pattern(self, pam_sequence: str, pattern: str) -> bool:
        """Checks if a candidate PAM sequence matches a specific IUPAC pattern string.

        Args:
            pam_sequence: Candidate PAM nucleotide string (e.g. 'AGG', 'TGG').
            pattern: IUPAC pattern string (e.g. 'NGG', 'NAG').

        Returns:
            True if sequence matches the pattern.
        """
        return IUPACMatcher.matches(pam_sequence, pattern)

    def validate_pam_sequence(
        self,
        pam_sequence: str,
        canonical_only: bool = False,
    ) -> PAMValidationResult:
        """Validates a candidate PAM sequence string (5' to 3') against registered patterns.

        Args:
            pam_sequence: Observed PAM nucleotide string (e.g. 'AGG', 'AGA').
            canonical_only: If True, only matches canonical PAM motifs.

        Returns:
            PAMValidationResult containing match status, matched motif, and cleavage multiplier.
        """
        if not pam_sequence or not isinstance(pam_sequence, str):
            return PAMValidationResult(
                is_valid=False,
                is_canonical=False,
                matched_motif=None,
                cleavage_multiplier=0.0,
                pam_sequence=str(pam_sequence) if pam_sequence is not None else "",
                details="Empty or missing PAM sequence.",
            )

        clean = pam_sequence.strip().upper().replace("U", "T")
        if not clean:
            return PAMValidationResult(
                is_valid=False,
                is_canonical=False,
                matched_motif=None,
                cleavage_multiplier=0.0,
                pam_sequence=pam_sequence,
                details="Empty or whitespace-only PAM sequence.",
            )

        # Enforce canonical nucleotide characters for candidate sequence
        valid_dna = frozenset("ACGT")
        if not set(clean).issubset(valid_dna):
            invalid_chars = sorted(set(clean) - valid_dna)
            return PAMValidationResult(
                is_valid=False,
                is_canonical=False,
                matched_motif=None,
                cleavage_multiplier=0.0,
                pam_sequence=pam_sequence,
                details=f"Invalid nucleotide character(s) in PAM sequence: {invalid_chars}. Expected canonical DNA (A, C, G, T).",
            )

        # Check against registered patterns in order (canonical first)
        for pattern, compiled_re in self._compiled_patterns:
            if canonical_only and not pattern.is_canonical:
                continue

            if compiled_re.match(clean):
                return PAMValidationResult(
                    is_valid=True,
                    is_canonical=pattern.is_canonical,
                    matched_motif=pattern.motif,
                    cleavage_multiplier=pattern.cleavage_multiplier,
                    pam_sequence=clean,
                    details=f"Matches {pattern.name} ({pattern.motif}) with cleavage multiplier {pattern.cleavage_multiplier:.2f}.",
                )

        return PAMValidationResult(
            is_valid=False,
            is_canonical=False,
            matched_motif=None,
            cleavage_multiplier=0.0,
            pam_sequence=clean,
            details=f"Candidate PAM '{clean}' does not match any registered PAM motif for {self.config.name}.",
        )

    def validate_candidate(
        self,
        candidate: RawCandidateHit,
        canonical_only: bool = False,
    ) -> ValidatedCandidateHit:
        """Validates an alignment candidate hit and determines strand-aware PAM coordinates.

        Coordinate Logic:
            For Forward (+) Strand:
                Protospacer: [global_start, global_start + protospacer_length)
                PAM:         [global_start + protospacer_length, global_start + target_length)

            For Reverse (-) Strand:
                The target aligns 3' <- 5' relative to the forward reference.
                PAM is adjacent to the 3' end on the reverse strand (left side on forward reference):
                PAM:         [global_start, global_start + pam_length)
                Protospacer: [global_start + pam_length, global_start + target_length)

        Args:
            candidate: RawCandidateHit from CPU alignment engine.
            canonical_only: If True, only accepts canonical PAM patterns.

        Returns:
            ValidatedCandidateHit enriched with PAM validation metadata and exact coordinates.
        """
        strand = candidate.strand
        global_start = candidate.global_start
        global_end = candidate.global_end
        target_seq = candidate.target_sequence
        target_len = len(target_seq) if target_seq else (global_end - global_start)
        guide_len = self.config.protospacer_length
        expected_pam_len = self.config.pam_length

        pam_len = max(0, target_len - guide_len)
        pam_seq = candidate.pam_sequence

        # Detect truncated / out-of-bounds PAM region
        if len(pam_seq) < expected_pam_len:
            pam_result = PAMValidationResult(
                is_valid=False,
                is_canonical=False,
                matched_motif=None,
                cleavage_multiplier=0.0,
                pam_sequence=pam_seq,
                details=f"Incomplete or out-of-bounds PAM sequence (length {len(pam_seq)} < expected {expected_pam_len}).",
            )
        else:
            pam_result = self.validate_pam_sequence(pam_seq, canonical_only=canonical_only)

        # Coordinate translation according to strand orientation
        if strand == Strand.FORWARD:
            protospacer_start = global_start
            protospacer_end = global_start + guide_len
            pam_start = global_start + guide_len
            pam_end = global_start + target_len
        else:
            # Reverse strand: PAM is at the lower coordinate on forward reference
            pam_start = global_start
            pam_end = global_start + pam_len
            protospacer_start = global_start + pam_len
            protospacer_end = global_start + target_len

        return ValidatedCandidateHit(
            candidate=candidate,
            pam_result=pam_result,
            chromosome=candidate.chromosome,
            strand=strand,
            protospacer_start=protospacer_start,
            protospacer_end=protospacer_end,
            pam_start=pam_start,
            pam_end=pam_end,
            pam_sequence=pam_seq,
            target_sequence=target_seq,
        )

    def validate_alignment_result(
        self,
        result: AlignmentResult,
        canonical_only: bool = False,
        filter_incompatible: bool = False,
    ) -> List[ValidatedCandidateHit]:
        """Validates all candidate hits within an AlignmentResult record.

        Args:
            result: AlignmentResult produced by SequenceAligner.
            canonical_only: If True, only canonical PAM motifs are considered valid.
            filter_incompatible: If True, filters out candidate hits lacking a valid PAM.

        Returns:
            List of ValidatedCandidateHit instances.
        """
        return self.validate_candidates(
            candidates=result.hits,
            canonical_only=canonical_only,
            filter_incompatible=filter_incompatible,
        )

    def validate_candidates(
        self,
        candidates: Iterable[RawCandidateHit],
        canonical_only: bool = False,
        filter_incompatible: bool = False,
    ) -> List[ValidatedCandidateHit]:
        """Validates an iterable collection of RawCandidateHit instances.

        Args:
            candidates: Iterable of RawCandidateHit records.
            canonical_only: If True, only accepts canonical PAM patterns.
            filter_incompatible: If True, filters out candidates lacking a compatible PAM.

        Returns:
            List of ValidatedCandidateHit instances in candidate order.
        """
        validated_hits = [
            self.validate_candidate(c, canonical_only=canonical_only)
            for c in candidates
        ]
        if filter_incompatible:
            return [h for h in validated_hits if h.is_valid_pam]
        return validated_hits

    def extract_pam_from_chunk(
        self,
        chunk: ChunkDescriptor,
        protospacer_start: int,
        protospacer_end: int,
        strand: Strand,
        pam_length: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """Extracts the adjacent PAM sequence and coordinates from a genomic chunk descriptor.

        Args:
            chunk: ChunkDescriptor sequence payload.
            protospacer_start: 0-based global genomic start of the protospacer.
            protospacer_end: 0-based global genomic end of the protospacer.
            strand: Strand orientation (Strand.FORWARD or Strand.REVERSE).
            pam_length: Length of PAM in base pairs (defaults to active nuclease PAM length).

        Returns:
            Tuple of (pam_sequence_5to3, (pam_start_0based, pam_end_0based)) or (None, None) if out-of-bounds.
        """
        p_len = pam_length or self.config.pam_length

        if strand == Strand.FORWARD:
            pam_start = protospacer_end
            pam_end = protospacer_end + p_len
            local_pam_start = pam_start - chunk.global_start
            local_pam_end = pam_end - chunk.global_start

            if local_pam_start < 0 or local_pam_end > chunk.actual_length:
                return None, None

            if isinstance(chunk.sequence_data, str):
                pam_seq = chunk.sequence_data[local_pam_start:local_pam_end]
            else:
                from geneweaver.alignment.cpu_aligner import CPUAligner
                seq_2bit, _ = CPUAligner._resolve_chunk_sequence(chunk.sequence_data)
                from geneweaver.sequence_processor.encoder import TwoBitEncoder
                pam_seq = TwoBitEncoder.decode(seq_2bit[local_pam_start:local_pam_end])

            return pam_seq, (pam_start, pam_end)

        else:
            # Reverse strand: PAM is immediately to the left on the forward reference
            pam_start = protospacer_start - p_len
            pam_end = protospacer_start
            local_pam_start = pam_start - chunk.global_start
            local_pam_end = pam_end - chunk.global_start

            if local_pam_start < 0 or local_pam_end > chunk.actual_length:
                return None, None

            if isinstance(chunk.sequence_data, str):
                fwd_pam = chunk.sequence_data[local_pam_start:local_pam_end]
            else:
                from geneweaver.alignment.cpu_aligner import CPUAligner
                seq_2bit, _ = CPUAligner._resolve_chunk_sequence(chunk.sequence_data)
                from geneweaver.sequence_processor.encoder import TwoBitEncoder
                fwd_pam = TwoBitEncoder.decode(seq_2bit[local_pam_start:local_pam_end])

            # In 5' to 3' orientation on the reverse strand: reverse complement of forward sequence
            pam_seq = reverse_complement(fwd_pam)
            return pam_seq, (pam_start, pam_end)


__all__ = [
    "PAMLocation",
    "IUPACMatcher",
    "PAMPattern",
    "NucleaseConfig",
    "SPCAS9",
    "PAMValidationResult",
    "ValidatedCandidateHit",
    "PAMValidator",
]
