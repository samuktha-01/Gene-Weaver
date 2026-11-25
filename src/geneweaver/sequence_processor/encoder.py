"""Two-bit binary nucleotide encoder and packer for GeneWeaver.

This module provides high-throughput conversion between ASCII genomic sequences
(A, C, G, T) and compressed 2-bit representations in both unpacked (1 base/byte)
and packed (4 bases/byte in uint8, 32 bases/word in uint64) formats.

Encoding Specification:
    Nucleotide | Binary | Decimal | Bit Pattern
    ---------- | ------ | ------- | -----------
    A          | 00_2   | 0       | 0b00
    C          | 01_2   | 1       | 0b01
    G          | 10_2   | 2       | 0b10
    T / U      | 11_2   | 3       | 0b11

Packed Memory Layout (8-bit):
    A single uint8 byte stores 4 nucleotides using Big-Endian bit placement:
        Bit 7-6: Nucleotide i + 0 (most significant bits)
        Bit 5-4: Nucleotide i + 1
        Bit 3-2: Nucleotide i + 2
        Bit 1-0: Nucleotide i + 3 (least significant bits)

Limitations:
    - Ambiguous nucleotides (e.g. 'N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V')
      cannot be represented in a 2-bit alphabet without loss of information. Any sequence
      containing ambiguous or unsupported characters is strictly rejected with an
      `InvalidSequenceError`.
    - Sequences whose lengths are not multiples of 4 are right-padded with zero bits (00_2 = A)
      in the terminal byte. The exact original sequence length is preserved in the
      `EncodedTwoBit` descriptor to prevent artificial tail elongation during decoding.
"""

from dataclasses import dataclass
import math
import re
from typing import Optional, Tuple, Union

import numpy as np

from geneweaver.genomic_parser.exceptions import InvalidSequenceError


@dataclass(frozen=True)
class EncodedTwoBit:
    """Represents a 2-bit packed genomic nucleotide sequence.

    Attributes:
        data: 1D NumPy array of uint8 values (4 nucleotides packed per byte).
        length: Exact length of the original sequence in base pairs.
    """

    data: np.ndarray
    length: int

    @property
    def byte_size(self) -> int:
        """Returns the memory size of the packed byte array in bytes."""
        return self.data.nbytes

    def __len__(self) -> int:
        """Returns the length of the nucleotide sequence in base pairs."""
        return self.length

    def __repr__(self) -> str:
        return f"<EncodedTwoBit length={self.length} bp, bytes={self.byte_size} B>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EncodedTwoBit):
            return self.length == other.length and np.array_equal(self.data, other.data)
        return False


class TwoBitEncoder:
    """Converts DNA nucleotide sequences to and from 2-bit binary representations.

    Supports:
    - Unpacked 1-base-per-byte uint8 array (values in {0, 1, 2, 3}) for direct GPU
      thread-per-base indexing.
    - Packed 4-bases-per-byte uint8 array for 75% memory reduction in host storage
      and PCIe transfers.
    - Packed 32-bases-per-word uint64 array for bitwise Hamming distance operations.
    """

    # ASCII to 2-bit lookup table (256 entries, initialized to sentinel 255)
    _ASCII_TO_2BIT = np.full(256, 255, dtype=np.uint8)
    _ASCII_TO_2BIT[ord("A")] = 0
    _ASCII_TO_2BIT[ord("a")] = 0
    _ASCII_TO_2BIT[ord("C")] = 1
    _ASCII_TO_2BIT[ord("c")] = 1
    _ASCII_TO_2BIT[ord("G")] = 2
    _ASCII_TO_2BIT[ord("g")] = 2
    _ASCII_TO_2BIT[ord("T")] = 3
    _ASCII_TO_2BIT[ord("t")] = 3
    # RNA Uracil translates to Thymine
    _ASCII_TO_2BIT[ord("U")] = 3
    _ASCII_TO_2BIT[ord("u")] = 3

    # 2-bit integer to ASCII byte lookup
    _2BIT_TO_ASCII = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)

    @classmethod
    def encode(cls, sequence: str) -> np.ndarray:
        """Converts an ASCII DNA sequence to an unpacked 1D uint8 array with values in {0, 1, 2, 3}.

        Args:
            sequence: Raw or normalized DNA nucleotide string. Whitespace is stripped automatically.

        Returns:
            1D NumPy array with dtype=uint8, where each element is in {0, 1, 2, 3}.

        Raises:
            InvalidSequenceError: If any unsupported or ambiguous nucleotide character
                (including 'N', 'R', 'Y', or non-alphabetic symbols) is encountered.
        """
        # Strip internal and surrounding whitespace if present
        if " " in sequence or "\n" in sequence or "\t" in sequence or "\r" in sequence:
            cleaned = re.sub(r"\s+", "", sequence)
        else:
            cleaned = sequence
        if not cleaned:
            return np.empty(0, dtype=np.uint8)

        # Convert ASCII string to byte array without Python-level iteration
        ascii_bytes = np.frombuffer(cleaned.encode("latin-1"), dtype=np.uint8)
        encoded = cls._ASCII_TO_2BIT[ascii_bytes]

        # Validate that all characters mapped to valid 2-bit values
        invalid_mask = encoded == 255
        if np.any(invalid_mask):
            first_invalid_idx = int(np.argmax(invalid_mask))
            invalid_char = chr(ascii_bytes[first_invalid_idx])
            raise InvalidSequenceError(
                f"Unsupported or ambiguous nucleotide character {repr(invalid_char)} at index {first_invalid_idx}. "
                f"TwoBitEncoder strictly requires canonical DNA bases (A, C, G, T)."
            )

        return encoded

    @classmethod
    def decode(cls, encoded: Union[np.ndarray, EncodedTwoBit], length: Optional[int] = None) -> str:
        """Reconstructs the ASCII nucleotide sequence from an unpacked or packed 2-bit representation.

        Args:
            encoded: Either an unpacked 1D uint8 array (values in {0, 1, 2, 3})
                     or an EncodedTwoBit instance.
            length: Optional length cutoff if `encoded` is an unpacked array.

        Returns:
            Decoded uppercase ASCII DNA string.

        Raises:
            ValueError: If an unpacked array contains values outside {0, 1, 2, 3}.
        """
        if isinstance(encoded, EncodedTwoBit):
            return cls.decode_packed(encoded)

        if encoded.size == 0:
            return ""

        target_len = length if length is not None else encoded.size
        sliced = encoded[:target_len]

        # Validate range of unpacked values
        if np.any(sliced > 3):
            invalid_val = int(sliced[np.argmax(sliced > 3)])
            raise ValueError(
                f"Invalid 2-bit integer value {invalid_val}; values must be in range 0..3."
            )

        ascii_bytes = cls._2BIT_TO_ASCII[sliced]
        return ascii_bytes.tobytes().decode("ascii")

    @classmethod
    def pack_8bit(cls, encoded_uint8: np.ndarray, length: Optional[int] = None) -> EncodedTwoBit:
        """Packs an unpacked 1-base-per-byte uint8 array into a 4-bases-per-byte uint8 array.

        Handles sequences whose lengths are not multiples of 4 by zero-padding the final byte.
        The exact sequence length is stored in the returned EncodedTwoBit descriptor.

        Args:
            encoded_uint8: 1D NumPy array with values in {0, 1, 2, 3}.
            length: Sequence length (defaults to len(encoded_uint8)).

        Returns:
            EncodedTwoBit containing packed uint8 array and original sequence length.
        """
        seq_len = length if length is not None else encoded_uint8.size
        if seq_len == 0:
            return EncodedTwoBit(data=np.empty(0, dtype=np.uint8), length=0)

        # Slice to exact sequence length
        seq_data = encoded_uint8[:seq_len]

        # Calculate padding needed to reach a multiple of 4
        pad_len = (4 - (seq_len % 4)) % 4
        if pad_len > 0:
            padded = np.pad(seq_data, (0, pad_len), mode="constant", constant_values=0)
        else:
            padded = seq_data

        # Reshape into groups of 4 bases and pack bitwise: (b0 << 6) | (b1 << 4) | (b2 << 2) | b3
        groups = padded.reshape(-1, 4)
        packed = (
            (groups[:, 0].astype(np.uint8) << 6)
            | (groups[:, 1].astype(np.uint8) << 4)
            | (groups[:, 2].astype(np.uint8) << 2)
            | groups[:, 3].astype(np.uint8)
        )

        return EncodedTwoBit(data=packed, length=seq_len)

    @classmethod
    def unpack_8bit(
        cls,
        packed: Union[EncodedTwoBit, np.ndarray],
        length: Optional[int] = None,
    ) -> np.ndarray:
        """Unpacks a 4-bases-per-byte uint8 array back to a 1-base-per-byte uint8 array.

        Args:
            packed: EncodedTwoBit instance or 1D packed uint8 array.
            length: Target length in base pairs (required if `packed` is an ndarray).

        Returns:
            1D NumPy uint8 array of length `length` with values in {0, 1, 2, 3}.
        """
        if isinstance(packed, EncodedTwoBit):
            data = packed.data
            seq_len = packed.length
        else:
            data = packed
            if length is None:
                raise ValueError("Length must be provided when unpacking a raw NumPy array.")
            seq_len = length

        if seq_len == 0:
            return np.empty(0, dtype=np.uint8)

        # Extract 2-bit values from each byte
        b0 = (data >> 6) & 0x03
        b1 = (data >> 4) & 0x03
        b2 = (data >> 2) & 0x03
        b3 = data & 0x03

        # Interleave bases and slice to exact original length
        unpacked = np.column_stack((b0, b1, b2, b3)).reshape(-1)
        return unpacked[:seq_len].astype(np.uint8)

    @classmethod
    def pack_uint64(cls, encoded_uint8: np.ndarray, length: Optional[int] = None) -> np.ndarray:
        """Packs 32 2-bit nucleotides into single 64-bit unsigned integers.

        Args:
            encoded_uint8: 1D NumPy uint8 array with values in {0, 1, 2, 3}.
            length: Sequence length (defaults to len(encoded_uint8)).

        Returns:
            1D NumPy array with dtype=uint64.
        """
        seq_len = length if length is not None else encoded_uint8.size
        if seq_len == 0:
            return np.empty(0, dtype=np.uint64)

        seq_data = encoded_uint8[:seq_len]
        pad_len = (32 - (seq_len % 32)) % 32
        if pad_len > 0:
            padded = np.pad(seq_data, (0, pad_len), mode="constant", constant_values=0)
        else:
            padded = seq_data

        groups = padded.reshape(-1, 32).astype(np.uint64)
        shifts = np.arange(62, -1, -2, dtype=np.uint64)
        packed_u64 = np.bitwise_or.reduce(groups << shifts, axis=1)

        return packed_u64

    @classmethod
    def unpack_uint64(cls, packed: np.ndarray, length: int) -> np.ndarray:
        """Unpacks 64-bit packed words back to 1-base-per-byte uint8 array.

        Args:
            packed: 1D NumPy array of uint64 words.
            length: Exact sequence length in base pairs.

        Returns:
            1D NumPy uint8 array with values in {0, 1, 2, 3}.
        """
        if length == 0 or packed.size == 0:
            return np.empty(0, dtype=np.uint8)

        shifts = np.arange(62, -1, -2, dtype=np.uint64)
        unpacked = (packed[:, None] >> shifts) & np.uint64(0x03)
        return unpacked.ravel()[:length].astype(np.uint8)

    @classmethod
    def encode_packed(cls, sequence: str) -> EncodedTwoBit:
        """Directly encodes and packs an ASCII DNA string into an EncodedTwoBit descriptor.

        Args:
            sequence: Raw or normalized DNA nucleotide string.

        Returns:
            EncodedTwoBit descriptor with packed uint8 data and sequence length.
        """
        unpacked = cls.encode(sequence)
        return cls.pack_8bit(unpacked, length=unpacked.size)

    @classmethod
    def decode_packed(
        cls,
        packed: Union[EncodedTwoBit, np.ndarray],
        length: Optional[int] = None,
    ) -> str:
        """Directly decodes a packed 2-bit representation back to an ASCII DNA string.

        Args:
            packed: EncodedTwoBit instance or 1D packed uint8 array.
            length: Target length (required if `packed` is an ndarray).

        Returns:
            Decoded uppercase ASCII DNA string.
        """
        unpacked = cls.unpack_8bit(packed, length=length)
        return cls.decode(unpacked)
