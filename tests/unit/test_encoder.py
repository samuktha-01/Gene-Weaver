"""Comprehensive unit tests for TwoBitEncoder and EncodedTwoBit."""

import numpy as np
import pytest

from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder


class TestTwoBitEncoderBasics:
    """Tests basic encoding and decoding of canonical nucleotides and edge cases."""

    def test_encode_empty_sequence(self):
        encoded = TwoBitEncoder.encode("")
        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.uint8
        assert encoded.size == 0
        assert TwoBitEncoder.decode(encoded) == ""

    def test_encode_packed_empty_sequence(self):
        packed = TwoBitEncoder.encode_packed("")
        assert isinstance(packed, EncodedTwoBit)
        assert len(packed) == 0
        assert packed.length == 0
        assert packed.byte_size == 0
        assert TwoBitEncoder.decode_packed(packed) == ""

    @pytest.mark.parametrize(
        "nucleotide,expected_code,expected_byte",
        [
            ("A", 0, 0),       # 00 << 6 = 0
            ("C", 1, 64),      # 01 << 6 = 64 (0x40)
            ("G", 2, 128),     # 10 << 6 = 128 (0x80)
            ("T", 3, 192),     # 11 << 6 = 192 (0xC0)
        ],
    )
    def test_single_nucleotides(self, nucleotide, expected_code, expected_byte):
        # Unpacked
        unpacked = TwoBitEncoder.encode(nucleotide)
        assert len(unpacked) == 1
        assert unpacked[0] == expected_code
        assert TwoBitEncoder.decode(unpacked) == nucleotide

        # Packed
        packed = TwoBitEncoder.encode_packed(nucleotide)
        assert len(packed) == 1
        assert packed.data.size == 1
        assert packed.data[0] == expected_byte
        assert TwoBitEncoder.decode_packed(packed) == nucleotide

    def test_canonical_four_nucleotides(self):
        seq = "ACGT"
        unpacked = TwoBitEncoder.encode(seq)
        np.testing.assert_array_equal(unpacked, np.array([0, 1, 2, 3], dtype=np.uint8))
        assert TwoBitEncoder.decode(unpacked) == seq

        # In binary: 00 01 10 11 = 27 (0x1B)
        packed = TwoBitEncoder.encode_packed(seq)
        assert len(packed) == 4
        assert packed.data.size == 1
        assert packed.data[0] == 27
        assert TwoBitEncoder.decode_packed(packed) == seq

    def test_case_insensitivity(self):
        unpacked_lower = TwoBitEncoder.encode("acgt")
        unpacked_upper = TwoBitEncoder.encode("ACGT")
        np.testing.assert_array_equal(unpacked_lower, unpacked_upper)
        assert TwoBitEncoder.decode(unpacked_lower) == "ACGT"

    def test_rna_uracil_translation(self):
        # Uracil translates to Thymine (code 3)
        unpacked_rna = TwoBitEncoder.encode("AUGCAUGC")
        unpacked_dna = TwoBitEncoder.encode("ATGCTATG")
        # 'AUGC' -> A:0, T:3, G:2, C:1. So 'AUGCAUGC' -> 'ATGCTA GC'? Wait:
        # A(0) U(3) G(2) C(1) A(0) U(3) G(2) C(1) -> ATGCATGC
        assert TwoBitEncoder.decode(unpacked_rna) == "ATGCATGC"

    def test_whitespace_handling(self):
        raw = "  A C \n G T \t A   "
        unpacked = TwoBitEncoder.encode(raw)
        assert len(unpacked) == 5
        assert TwoBitEncoder.decode(unpacked) == "ACGTA"


class TestTwoBitEncoderPaddingAndNonMultiplesOfFour:
    """Tests sequences whose lengths are not multiples of four, ensuring padding is stripped."""

    @pytest.mark.parametrize("length", range(1, 16))
    def test_lengths_one_to_fifteen(self, length):
        # Generate predictable sequence of repeating ACGT
        base_pattern = "ACGT" * 4
        seq = base_pattern[:length]

        # Round-trip unpacked
        unpacked = TwoBitEncoder.encode(seq)
        assert len(unpacked) == length
        assert TwoBitEncoder.decode(unpacked) == seq

        # Round-trip packed
        packed = TwoBitEncoder.encode_packed(seq)
        expected_bytes = (length + 3) // 4
        assert len(packed) == length
        assert packed.data.size == expected_bytes
        assert packed.byte_size == expected_bytes

        # Decode directly and verify original length is preserved with zero trailing padding
        decoded = TwoBitEncoder.decode_packed(packed)
        assert len(decoded) == length
        assert decoded == seq


class TestTwoBitEncoderValidation:
    """Tests validation of invalid, ambiguous, or unsupported nucleotide characters."""

    @pytest.mark.parametrize(
        "ambiguous_char",
        ["N", "n", "R", "r", "Y", "y", "S", "s", "W", "w", "K", "k", "M", "m", "B", "b", "D", "d", "H", "h", "V", "v"],
    )
    def test_ambiguous_nucleotides_rejected(self, ambiguous_char):
        seq = f"ACGT{ambiguous_char}TGCA"
        with pytest.raises(InvalidSequenceError) as exc_info:
            TwoBitEncoder.encode(seq)
        assert f"index 4" in str(exc_info.value)
        assert ambiguous_char.upper() in str(exc_info.value) or ambiguous_char in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_char",
        ["1", "9", "@", "$", "!", "-", "?", "X", "Z"],
    )
    def test_unsupported_characters_rejected(self, invalid_char):
        seq = f"ACGT{invalid_char}ACGT"
        with pytest.raises(InvalidSequenceError) as exc_info:
            TwoBitEncoder.encode(seq)
        assert f"index 4" in str(exc_info.value)

    def test_invalid_unpacked_integer_decode_raises_value_error(self):
        # Values > 3 are invalid in 2-bit
        bad_array = np.array([0, 1, 4, 3], dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid 2-bit integer value 4"):
            TwoBitEncoder.decode(bad_array)

    def test_unpack_without_length_raises_value_error(self):
        raw_packed = np.array([27], dtype=np.uint8)
        with pytest.raises(ValueError, match="Length must be provided"):
            TwoBitEncoder.unpack_8bit(raw_packed)


class TestTwoBitEncoderLongSequences:
    """Tests long sequences and memory compression efficiency."""

    def test_long_sequence_round_trip(self):
        # 10,003 nucleotides (not a multiple of 4)
        rng = np.random.default_rng(42)
        bases = np.array(["A", "C", "G", "T"])
        random_bases = bases[rng.integers(0, 4, size=10_003)]
        long_seq = "".join(random_bases)

        # Encode and pack
        packed = TwoBitEncoder.encode_packed(long_seq)
        assert len(packed) == 10_003
        assert packed.data.size == (10_003 + 3) // 4  # 2501 bytes

        # Memory verification: 10,003 bytes ASCII compressed to 2501 bytes (approx 75% savings)
        assert packed.byte_size == 2501

        # Decode and verify exact round-trip fidelity
        decoded = TwoBitEncoder.decode_packed(packed)
        assert decoded == long_seq

    def test_packed_uint64_round_trip(self):
        # Test 64-bit word packing (32 nucleotides per uint64 word)
        seq_65 = "ACGT" * 16 + "A"  # 65 nucleotides (requires 3 uint64 words)
        unpacked = TwoBitEncoder.encode(seq_65)
        packed_u64 = TwoBitEncoder.pack_uint64(unpacked)

        assert packed_u64.dtype == np.uint64
        assert packed_u64.size == 3  # ceil(65 / 32) = 3

        unpacked_back = TwoBitEncoder.unpack_uint64(packed_u64, length=65)
        np.testing.assert_array_equal(unpacked_back, unpacked)
        assert TwoBitEncoder.decode(unpacked_back) == seq_65

    def test_packed_uint64_empty(self):
        empty_unpacked = np.empty(0, dtype=np.uint8)
        packed_u64 = TwoBitEncoder.pack_uint64(empty_unpacked)
        assert packed_u64.size == 0
        unpacked_back = TwoBitEncoder.unpack_uint64(packed_u64, length=0)
        assert unpacked_back.size == 0


class TestEncodedTwoBitDataclass:
    """Tests equality, length, and representation for the EncodedTwoBit container."""

    def test_equality(self):
        seq1 = TwoBitEncoder.encode_packed("ACGTA")
        seq2 = TwoBitEncoder.encode_packed("ACGTA")
        seq3 = TwoBitEncoder.encode_packed("ACGTT")

        assert seq1 == seq2
        assert seq1 != seq3
        assert seq1 != "ACGTA"

    def test_repr_and_len(self):
        seq = TwoBitEncoder.encode_packed("ACGTACGT")
        assert len(seq) == 8
        assert repr(seq) == "<EncodedTwoBit length=8 bp, bytes=2 B>"

    def test_decode_via_general_decode_method(self):
        seq_str = "ACGTACGTACGT"
        encoded = TwoBitEncoder.encode_packed(seq_str)
        # Passing EncodedTwoBit directly to TwoBitEncoder.decode()
        assert TwoBitEncoder.decode(encoded) == seq_str
