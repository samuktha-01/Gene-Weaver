"""High-performance reverse complement generation for DNA sequences.

This module provides optimized, vectorized reverse complement operations for canonical
DNA sequences across string, 1D NumPy uint8 array (unpacked 2-bit and ASCII bytes),
and EncodedTwoBit representations.

Complement Rules:
    Base | Complement | 2-Bit Binary | 2-Bit Decimal
    ---- | ---------- | ------------ | -------------
    A    | T          | 00 -> 11     | 0 -> 3
    T    | A          | 11 -> 00     | 3 -> 0
    C    | G          | 01 -> 10     | 1 -> 2
    G    | C          | 10 -> 01     | 2 -> 1
    U    | A          | 11 -> 00     | 3 -> 0 (RNA Uracil)

Key Properties:
    - In 2-bit binary encoding, complement is bitwise XOR with 3 (`x ^ 3` or `3 - x`).
    - Reverse complement in NumPy: `np.ascontiguousarray((seq_2bit ^ 3)[::-1])`.
    - Strings: C-accelerated `str.translate` with strict validation.
    - Strict validation policy: Non-canonical or ambiguous bases ('N', 'R', 'Y', etc.)
      strictly raise `InvalidSequenceError`.
"""

import re
from typing import Optional, Union, overload

import numpy as np

from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder


# Translation table for canonical DNA nucleotides (supporting case-insensitivity and RNA Uracil)
_COMPLEMENT_MAP = {
    ord("A"): ord("T"),
    ord("a"): ord("T"),
    ord("T"): ord("A"),
    ord("t"): ord("A"),
    ord("C"): ord("G"),
    ord("c"): ord("G"),
    ord("G"): ord("C"),
    ord("g"): ord("C"),
    ord("U"): ord("A"),
    ord("u"): ord("A"),
}
_STR_TRANS_TABLE = str.maketrans({chr(k): chr(v) for k, v in _COMPLEMENT_MAP.items()})

_CANONICAL_BASES_SET = frozenset({"A", "C", "G", "T", "U", "a", "c", "g", "t", "u"})

# 256-byte ASCII lookup table for uint8 byte complementation (sentinel 255 for invalid characters)
_ASCII_COMPLEMENT_LUT = np.full(256, 255, dtype=np.uint8)
_ASCII_COMPLEMENT_LUT[ord("A")] = ord("T")
_ASCII_COMPLEMENT_LUT[ord("a")] = ord("T")
_ASCII_COMPLEMENT_LUT[ord("T")] = ord("A")
_ASCII_COMPLEMENT_LUT[ord("t")] = ord("A")
_ASCII_COMPLEMENT_LUT[ord("C")] = ord("G")
_ASCII_COMPLEMENT_LUT[ord("c")] = ord("G")
_ASCII_COMPLEMENT_LUT[ord("G")] = ord("C")
_ASCII_COMPLEMENT_LUT[ord("g")] = ord("C")
_ASCII_COMPLEMENT_LUT[ord("U")] = ord("A")
_ASCII_COMPLEMENT_LUT[ord("u")] = ord("A")


class ReverseComplement:
    """Namespace providing strongly-typed reverse-complement utility functions."""

    @classmethod
    def dna(cls, sequence: str) -> str:
        """Generates the reverse complement of a canonical DNA nucleotide string.

        Args:
            sequence: Raw DNA string (case-insensitive, canonical bases A, C, G, T, U).

        Returns:
            Uppercase reverse-complemented DNA string.

        Raises:
            InvalidSequenceError: If sequence contains unsupported or ambiguous characters.
        """
        if not sequence:
            return ""

        # Validate characters against canonical DNA alphabet
        seq_chars = set(sequence)
        invalid_chars = seq_chars - _CANONICAL_BASES_SET
        if invalid_chars:
            # Locate first invalid character for precise diagnostic error
            for idx, char in enumerate(sequence):
                if char not in _CANONICAL_BASES_SET:
                    raise InvalidSequenceError(
                        f"Unsupported or ambiguous nucleotide character {repr(char)} at index {idx}. "
                        f"Reverse complement strictly requires canonical DNA bases (A, C, G, T)."
                    )

        # Fast C-accelerated complement translation and slice reversal
        return sequence.translate(_STR_TRANS_TABLE)[::-1]

    @classmethod
    def array(cls, array: np.ndarray) -> np.ndarray:
        """Generates the reverse complement of a 1D NumPy array.

        Supports:
        - 1D uint8 unpacked 2-bit array with values in {0, 1, 2, 3}.
        - 1D uint8 ASCII byte array.
        - 1D string/unicode array.

        Args:
            array: 1D NumPy array representing sequence data.

        Returns:
            Contiguous 1D NumPy array containing the reverse-complemented sequence.

        Raises:
            ValueError: If array is not 1D.
            InvalidSequenceError: If array contains unsupported or ambiguous nucleotides.
        """
        if array.ndim != 1:
            raise ValueError(f"Input array must be 1D, got shape {array.shape}")
        if array.size == 0:
            return np.empty(0, dtype=array.dtype)

        # Unpacked 2-bit uint8 array (values 0..3)
        if array.dtype == np.uint8 and np.all(array <= 3):
            # In 2-bit: comp(x) = x ^ 3 (or 3 - x); reversed via [::-1]
            return np.ascontiguousarray((array ^ 3)[::-1])

        # ASCII byte array
        if array.dtype == np.uint8 and np.all((array >= 32) & (array <= 126)):
            complemented = _ASCII_COMPLEMENT_LUT[array]
            invalid_mask = complemented == 255
            if np.any(invalid_mask):
                first_invalid_idx = int(np.argmax(invalid_mask))
                inv_char = chr(array[first_invalid_idx])
                raise InvalidSequenceError(
                    f"Unsupported or ambiguous nucleotide character {repr(inv_char)} at index {first_invalid_idx}. "
                    f"Reverse complement strictly requires canonical DNA bases (A, C, G, T)."
                )
            return np.ascontiguousarray(complemented[::-1])

        # String / Unicode array
        if array.dtype.kind in ("S", "U"):
            joined_str = "".join(array.astype(str))
            rc_str = cls.dna(joined_str)
            return np.array(list(rc_str), dtype=array.dtype)

        # Unrecognized uint8 values > 3
        if array.dtype == np.uint8:
            invalid_mask = array > 3
            first_idx = int(np.argmax(invalid_mask))
            raise InvalidSequenceError(
                f"Invalid 2-bit nucleotide value {array[first_idx]} at index {first_idx}; "
                f"expected values in range 0..3."
            )

        raise TypeError(f"Unsupported NumPy array dtype for reverse complement: {array.dtype}")

    @classmethod
    def twobit(cls, encoded: EncodedTwoBit) -> EncodedTwoBit:
        """Generates the reverse complement of an EncodedTwoBit packed sequence.

        Args:
            encoded: EncodedTwoBit container.

        Returns:
            New EncodedTwoBit container containing the packed reverse-complement sequence.
        """
        if encoded.length == 0:
            return EncodedTwoBit(data=np.empty(0, dtype=np.uint8), length=0)

        unpacked = TwoBitEncoder.unpack_8bit(encoded)
        rc_unpacked = cls.array(unpacked)
        return TwoBitEncoder.pack_8bit(rc_unpacked, length=encoded.length)


@overload
def reverse_complement(sequence: str) -> str:
    ...


@overload
def reverse_complement(sequence: np.ndarray) -> np.ndarray:
    ...


@overload
def reverse_complement(sequence: EncodedTwoBit) -> EncodedTwoBit:
    ...


def reverse_complement(
    sequence: Union[str, np.ndarray, EncodedTwoBit],
) -> Union[str, np.ndarray, EncodedTwoBit]:
    """Generates the reverse complement of a DNA sequence.

    Dispatches to optimized handlers based on input representation:
    - `str`: returns uppercase reverse-complemented DNA string.
    - `np.ndarray`: returns contiguous reverse-complemented array.
    - `EncodedTwoBit`: returns new EncodedTwoBit container.

    Args:
        sequence: Input sequence representation.

    Returns:
        Reverse-complemented sequence in matching container type.

    Raises:
        InvalidSequenceError: If sequence contains invalid or ambiguous nucleotides.
        TypeError: If sequence type is unsupported.
    """
    if isinstance(sequence, str):
        return ReverseComplement.dna(sequence)
    if isinstance(sequence, np.ndarray):
        return ReverseComplement.array(sequence)
    if isinstance(sequence, EncodedTwoBit):
        return ReverseComplement.twobit(sequence)

    raise TypeError(
        f"Unsupported sequence type for reverse complement: {type(sequence).__name__}. "
        f"Expected str, np.ndarray, or EncodedTwoBit."
    )


__all__ = [
    "ReverseComplement",
    "reverse_complement",
]
