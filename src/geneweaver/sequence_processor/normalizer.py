"""Sequence normalization and IUPAC nucleotide sanitization utilities."""

import re
from typing import List, Optional, Set, Tuple


class SequenceNormalizer:
    """Sanitizes and normalizes genomic nucleotide sequences.

    Enforces standard uppercase DNA representation, converts RNA uracil ('U')
    to thymine ('T'), resolves IUPAC ambiguity codes, and detects unsequenced
    poly-N regions.
    """

    # Canonical DNA nucleotides
    CANONICAL_BASES: Set[str] = frozenset({"A", "C", "G", "T"})

    # Extended IUPAC nucleotide ambiguity codes:
    # R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C),
    # B (C/G/T), D (A/G/T), H (A/C/T), V (A/C/G), N (any)
    IUPAC_AMBIGUITY_CODES: Set[str] = frozenset({"R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N"})

    ALL_VALID_NUCLEOTIDES: Set[str] = CANONICAL_BASES | IUPAC_AMBIGUITY_CODES

    # Translation table: strips whitespace and maps U/u to T
    _TRANS_TABLE = str.maketrans({"U": "T", "u": "T"})

    # Precompiled regex patterns for fast nucleotide validation
    _CANONICAL_PATTERN = re.compile(r"^[ACGT]*$")
    _AMBIGUOUS_PATTERN = re.compile(r"^[ACGTNRYWSKMBDHV]*$")

    def normalize(self, sequence: str) -> str:
        """Normalizes a raw nucleotide sequence.

        Operations performed:
        - Strips leading, trailing, and internal whitespace (newlines, spaces, tabs).
        - Uppercases all nucleotide characters.
        - Translates RNA 'U' (Uracil) to DNA 'T' (Thymine).

        Args:
            sequence: Raw DNA/RNA sequence string.

        Returns:
            Normalized uppercase DNA sequence string.
        """
        # Fast path: check for whitespace before regex allocation
        if " " in sequence or "\t" in sequence or "\n" in sequence or "\r" in sequence:
            cleaned = re.sub(r"\s+", "", sequence)
        else:
            cleaned = sequence
        # Uppercase and translate RNA uracil to DNA thymine if needed
        uppercased = cleaned.upper()
        if "U" in uppercased:
            return uppercased.translate(self._TRANS_TABLE)
        return uppercased

    def validate(self, sequence: str, allow_ambiguity: bool = True) -> Tuple[bool, Optional[str]]:
        """Validates that all characters in the sequence are permissible nucleotides.

        Args:
            sequence: Sequence to check (must be normalized/uppercased).
            allow_ambiguity: If True, IUPAC codes (including 'N') are permitted.
                            If False, only canonical 'A', 'C', 'G', 'T' are permitted.

        Returns:
            Tuple of (is_valid, error_message_if_invalid).
        """
        pattern = self._AMBIGUOUS_PATTERN if allow_ambiguity else self._CANONICAL_PATTERN
        if pattern.match(sequence):
            return True, None

        allowed = self.ALL_VALID_NUCLEOTIDES if allow_ambiguity else self.CANONICAL_BASES
        for idx, char in enumerate(sequence):
            if char not in allowed:
                char_repr = repr(char)
                return False, f"Invalid nucleotide character {char_repr} at index {idx} (expected one of: {sorted(allowed)})"
        return True, None

    def find_poly_n_regions(self, sequence: str, min_length: int = 20) -> List[Tuple[int, int]]:
        """Identifies consecutive stretches of undetermined 'N' bases.

        Such regions (e.g. centromeres, telomeres) can be bypassed during GPU alignment
        to avoid redundant compute cycles.

        Args:
            sequence: Uppercase nucleotide sequence.
            min_length: Minimum length of consecutive 'N' bases to report.

        Returns:
            List of (start_index, end_index) 0-based intervals [start, end).
        """
        pattern = re.compile(rf"N{{{min_length},}}")
        return [(m.start(), m.end()) for m in pattern.finditer(sequence)]

    def reverse_complement(self, sequence: str) -> str:
        """Generates the reverse complement of a nucleotide sequence string.

        Args:
            sequence: Raw DNA/RNA sequence string.

        Returns:
            Normalized uppercase reverse-complemented DNA string.
        """
        from geneweaver.alignment.reverse_complement import reverse_complement
        return reverse_complement(sequence)
