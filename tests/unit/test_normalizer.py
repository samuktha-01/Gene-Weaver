"""Unit tests for SequenceNormalizer."""

import pytest
from geneweaver.sequence_processor.normalizer import SequenceNormalizer


class TestSequenceNormalizer:
    """Tests sequence cleaning, case folding, RNA translation, and poly-N detection."""

    def test_normalize_whitespace_and_case(self):
        normalizer = SequenceNormalizer()
        raw = "  atgc \n tgtg \t aaaa  \r\n"
        assert normalizer.normalize(raw) == "ATGCTGTGAAAA"

    def test_normalize_rna_uracil(self):
        normalizer = SequenceNormalizer()
        assert normalizer.normalize("augcAUGC") == "ATGCATGC"

    def test_validate_canonical_bases(self):
        normalizer = SequenceNormalizer()
        is_valid, err = normalizer.validate("ACGTACGT", allow_ambiguity=False)
        assert is_valid is True
        assert err is None

    def test_validate_ambiguous_bases(self):
        normalizer = SequenceNormalizer()
        is_valid, err = normalizer.validate("ACGTNRY", allow_ambiguity=True)
        assert is_valid is True
        assert err is None

        is_valid_strict, err_strict = normalizer.validate("ACGTNRY", allow_ambiguity=False)
        assert is_valid_strict is False
        assert "Invalid nucleotide character 'N'" in err_strict

    def test_find_poly_n_regions(self):
        normalizer = SequenceNormalizer()
        seq = "ACGT" + "N" * 25 + "CGTA" + "N" * 30 + "AAAA"
        regions = normalizer.find_poly_n_regions(seq, min_length=20)
        assert len(regions) == 2
        assert regions[0] == (4, 29)
        assert regions[1] == (33, 63)

    def test_find_poly_n_regions_none(self):
        normalizer = SequenceNormalizer()
        seq = "ACGTNNNNACGT"
        assert normalizer.find_poly_n_regions(seq, min_length=10) == []
