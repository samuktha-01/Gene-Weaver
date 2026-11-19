"""Unit tests for SequenceAligner protocol, BaseSequenceAligner, and alignment data models."""

from typing import List
import numpy as np
import pytest

from geneweaver.alignment import (
    AlignmentConfig,
    AlignmentResult,
    BaseSequenceAligner,
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    SequenceAligner,
    Strand,
)
from geneweaver.genomic_parser.exceptions import InvalidSequenceError
from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker


class DummyAligner(BaseSequenceAligner):
    """Concrete mock implementation of BaseSequenceAligner for protocol verification."""

    @property
    def backend_name(self) -> str:
        return "mock-backend"

    def _align_chunk(
        self,
        chunk: ChunkDescriptor,
        guide: GuideTarget,
        effective_config: AlignmentConfig,
    ) -> AlignmentResult:
        # Simulate detecting a hit at local offset 10
        mismatch_info = MismatchInfo.from_bitmask(0b0001, spacer_length=20, seed_length=10)
        hit = RawCandidateHit(
            chunk_id=chunk.chunk_id,
            chromosome=chunk.chromosome,
            local_offset=10,
            global_start=chunk.global_start + 10,
            global_end=chunk.global_start + 10 + 23,
            genomic_position=chunk.to_1based_coordinate(10),
            strand=Strand.FORWARD,
            mismatch_info=mismatch_info,
            pam_sequence="AGG",
            target_sequence="A" * 20 + "AGG",
        )
        return AlignmentResult(
            chunk_id=chunk.chunk_id,
            chromosome=chunk.chromosome,
            hits=[hit],
            total_windows_evaluated=chunk.actual_length - 22 if chunk.actual_length >= 23 else 0,
            execution_time_ms=1.25,
            backend=self.backend_name,
            config=effective_config,
        )


class TestStrandModel:
    """Tests for the Strand integer enum and symbol conversions."""

    def test_strand_values_and_symbols(self):
        assert Strand.FORWARD == 0
        assert Strand.REVERSE == 1
        assert Strand.FORWARD.sign == "+"
        assert Strand.REVERSE.sign == "-"
        assert str(Strand.FORWARD) == "+"
        assert str(Strand.REVERSE) == "-"
        assert repr(Strand.FORWARD) == "<Strand.FORWARD: '+'>"

    @pytest.mark.parametrize(
        "val,expected",
        [
            ("+", Strand.FORWARD),
            ("fwd", Strand.FORWARD),
            ("FORWARD", Strand.FORWARD),
            (0, Strand.FORWARD),
            (Strand.FORWARD, Strand.FORWARD),
            ("-", Strand.REVERSE),
            ("rev", Strand.REVERSE),
            ("REVERSE", Strand.REVERSE),
            (1, Strand.REVERSE),
            (Strand.REVERSE, Strand.REVERSE),
        ],
    )
    def test_from_symbol_valid(self, val, expected):
        assert Strand.from_symbol(val) == expected

    def test_from_symbol_invalid(self):
        with pytest.raises(ValueError, match="Invalid strand symbol"):
            Strand.from_symbol("?")
        with pytest.raises(ValueError, match="Invalid integer for Strand"):
            Strand.from_symbol(2)
        with pytest.raises(TypeError, match="Unsupported type"):
            Strand.from_symbol([1])  # type: ignore


class TestMismatchInfoModel:
    """Tests for the MismatchInfo record and bitmask extraction."""

    def test_from_bitmask_no_mismatches(self):
        info = MismatchInfo.from_bitmask(0, spacer_length=20, seed_length=10)
        assert info.count == 0
        assert info.bitmask == 0
        assert info.positions == ()
        assert info.seed_count == 0
        assert info.is_within_tolerance(max_mismatches=4, max_seed_mismatches=2)

    def test_from_bitmask_distal_and_seed_mismatches(self):
        # Bitmask with mismatch at index 0 (distal, 1-based pos 1)
        # and index 15 (seed, 1-based pos 16)
        # Binary: (1 << 0) | (1 << 15) = 1 | 32768 = 32769
        bitmask = (1 << 0) | (1 << 15)
        info = MismatchInfo.from_bitmask(bitmask, spacer_length=20, seed_length=10)

        assert info.count == 2
        assert info.bitmask == bitmask
        assert info.positions == (1, 16)
        assert info.seed_count == 1  # Index 15 is >= 10, so it's in seed

        assert info.is_within_tolerance(max_mismatches=2, max_seed_mismatches=1)
        assert not info.is_within_tolerance(max_mismatches=1, max_seed_mismatches=1)
        assert not info.is_within_tolerance(max_mismatches=4, max_seed_mismatches=0)

    def test_invalid_mismatch_counts_raise_value_error(self):
        with pytest.raises(ValueError, match="Mismatch count must be between 0 and 20"):
            MismatchInfo(count=21, bitmask=0, positions=(), seed_count=0)
        with pytest.raises(ValueError, match="Seed mismatch count .* cannot exceed total count"):
            MismatchInfo(count=2, bitmask=0, positions=(), seed_count=3)


class TestAlignmentConfigModel:
    """Tests for AlignmentConfig defaults and threshold validation."""

    def test_default_config(self):
        config = AlignmentConfig()
        assert config.max_mismatches == 4
        assert config.seed_max_mismatches == 2
        assert config.seed_length == 10
        assert config.canonical_pam_only is False
        assert config.search_reverse_strand is True
        assert config.batch_size == 65536

    def test_custom_valid_config(self):
        config = AlignmentConfig(
            max_mismatches=5,
            seed_max_mismatches=1,
            seed_length=8,
            canonical_pam_only=True,
            search_reverse_strand=False,
            batch_size=32768,
        )
        assert config.max_mismatches == 5
        assert config.seed_max_mismatches == 1
        assert config.seed_length == 8
        assert config.canonical_pam_only is True
        assert config.search_reverse_strand is False
        assert config.batch_size == 32768

    def test_invalid_config_raises_value_error(self):
        with pytest.raises(ValueError, match="max_mismatches must be between 0 and 20"):
            AlignmentConfig(max_mismatches=-1)
        with pytest.raises(ValueError, match="max_mismatches must be between 0 and 20"):
            AlignmentConfig(max_mismatches=25)
        with pytest.raises(ValueError, match="seed_max_mismatches .* cannot exceed max_mismatches"):
            AlignmentConfig(max_mismatches=2, seed_max_mismatches=3)
        with pytest.raises(ValueError, match="seed_length must be between 1 and 20"):
            AlignmentConfig(seed_length=0)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            AlignmentConfig(batch_size=0)


class TestGuideTargetModel:
    """Tests for the GuideTarget container and validation."""

    def test_valid_guide_target(self):
        guide_str = "GAGTCCGAGCAGAAGAAGAAAGG"
        guide = GuideTarget.from_string(guide_str)

        assert len(guide) == 23
        assert guide.sequence == guide_str
        assert guide.spacer == "GAGTCCGAGCAGAAGAAGAA"
        assert guide.pam == "AGG"
        assert isinstance(guide.encoded_2bit, np.ndarray)
        assert guide.encoded_2bit.dtype == np.uint8
        assert guide.encoded_2bit.size == 23
        assert guide.encoded_packed is not None
        assert len(guide.encoded_packed) == 23

    def test_invalid_length_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid CRISPR guide length"):
            GuideTarget.from_string("GAGTCCGAGCAGAAGAAGAA")  # 20 nt, missing PAM

    def test_invalid_characters_raises_sequence_error(self):
        with pytest.raises(InvalidSequenceError):
            GuideTarget.from_string("GAGTCCGAGCAGAAGAAGAA123")


class TestAlignmentResultAndHitModels:
    """Tests for RawCandidateHit and AlignmentResult properties."""

    def test_raw_candidate_hit_properties(self):
        mismatch_info = MismatchInfo.from_bitmask(0b10, spacer_length=20, seed_length=10)
        hit = RawCandidateHit(
            chunk_id="chr1_chunk_0000",
            chromosome="chr1",
            local_offset=5,
            global_start=105,
            global_end=128,
            genomic_position=106,
            strand=Strand.REVERSE,
            mismatch_info=mismatch_info,
            pam_sequence="TGG",
            target_sequence="T" * 20 + "TGG",
        )
        assert hit.mismatch_count == 1
        assert hit.mismatch_bitmask == 2
        assert hit.strand_symbol == "-"

    def test_alignment_result_properties(self):
        config = AlignmentConfig()
        result = AlignmentResult(
            chunk_id="chr1_chunk_0000",
            chromosome="chr1",
            hits=[],
            total_windows_evaluated=1000,
            execution_time_ms=2.5,
            backend="test-backend",
            config=config,
        )
        assert result.hit_count == 0
        assert len(result) == 0
        assert "chunk='chr1_chunk_0000'" in repr(result)
        assert "backend='test-backend'" in repr(result)


class TestSequenceAlignerProtocolAndBase:
    """Tests adherence to SequenceAligner Protocol and BaseSequenceAligner abstract class."""

    def test_protocol_conformance(self):
        aligner = DummyAligner()
        assert isinstance(aligner, SequenceAligner)
        assert aligner.backend_name == "mock-backend"

    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseSequenceAligner()  # type: ignore

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class IncompleteAligner(BaseSequenceAligner):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAligner()  # type: ignore

    def test_align_with_real_chunk_descriptor(self):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        seq = "A" * 150
        chunks = chunker.chunk(seq, chromosome="chr1")
        assert len(chunks) == 2

        aligner = DummyAligner()
        guide = "GAGTCCGAGCAGAAGAAGAAAGG"

        # Align with string guide
        result = aligner.align(chunk=chunks[0], guide=guide)
        assert isinstance(result, AlignmentResult)
        assert result.chunk_id == "chr1_chunk_0000"
        assert result.chromosome == "chr1"
        assert result.hit_count == 1
        assert result.hits[0].local_offset == 10
        assert result.hits[0].global_start == 10
        assert result.hits[0].genomic_position == 11
        assert result.backend == "mock-backend"

        # Align with pre-constructed GuideTarget
        guide_target = GuideTarget.from_string(guide)
        result2 = aligner.align(chunk=chunks[0], guide=guide_target)
        assert result2.hit_count == 1

    def test_align_with_override_config(self):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk("C" * 100)[0]

        default_cfg = AlignmentConfig(max_mismatches=3)
        aligner = DummyAligner(config=default_cfg)

        override_cfg = AlignmentConfig(max_mismatches=5)
        result = aligner.align(chunk=chunk, guide="GAGTCCGAGCAGAAGAAGAAAGG", config=override_cfg)
        assert result.config.max_mismatches == 5

    def test_align_invalid_chunk_type_raises_type_error(self):
        aligner = DummyAligner()
        with pytest.raises(TypeError, match="Expected chunk to be ChunkDescriptor"):
            aligner.align(chunk="not_a_chunk", guide="GAGTCCGAGCAGAAGAAGAAAGG")  # type: ignore

    def test_align_invalid_guide_type_raises_type_error(self):
        chunker = GenomicChunker(chunk_size=100, halo_size=22)
        chunk = chunker.chunk("A" * 100)[0]
        aligner = DummyAligner()
        with pytest.raises(TypeError, match="Expected guide to be GuideTarget or str"):
            aligner.align(chunk=chunk, guide=12345)  # type: ignore
