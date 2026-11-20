"""Unit and integration tests for real genomic dataset validation (GRCh38 chr1_region)."""

from collections import Counter
from pathlib import Path
import pytest

from geneweaver.alignment import AlignmentConfig, CPUAligner
from geneweaver.genomic_parser.fasta_reader import FastaReader
from geneweaver.genomic_parser.records import FastaRecord
from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker


@pytest.fixture(scope="module")
def real_fasta_path() -> Path:
    """Returns the resolved path to the real GRCh38 chr1_region dataset."""
    path = Path(__file__).resolve().parents[2] / "data" / "raw" / "GRCh38" / "chr1_region.fasta.fa"
    if not path.exists():
        pytest.skip(f"Real dataset not found at {path}")
    return path


@pytest.fixture(scope="module")
def real_record(real_fasta_path: Path) -> FastaRecord:
    """Parses and returns the real dataset record once for the test module."""
    reader = FastaReader(real_fasta_path)
    records = reader.read_records()
    assert len(records) == 1, f"Expected 1 record in real dataset, got {len(records)}"
    return records[0]


class TestRealDatasetValidation:
    """Validates header, accession, coordinates, length, and base composition of real GRCh38 data."""

    def test_file_exists_and_size(self, real_fasta_path: Path):
        assert real_fasta_path.is_file()
        assert real_fasta_path.stat().st_size == 25_251_389

    def test_header_and_accession_extraction(self, real_record: FastaRecord):
        assert real_record.identifier == "ref|NC_000001.11|:85401937-110297579"
        assert real_record.description == "Homo sapiens chromosome 1, GRCh38.p14 Primary Assembly"
        assert real_record.accession == "NC_000001.11"
        assert real_record.coordinates == (85401937, 110297579)

    def test_sequence_length_matches_header_coordinates(self, real_record: FastaRecord):
        coords = real_record.coordinates
        assert coords is not None
        start_coord, end_coord = coords

        expected_length = end_coord - start_coord + 1
        assert expected_length == 24_895_643
        assert real_record.length == expected_length
        assert len(real_record) == expected_length

    def test_base_composition_and_counts(self, real_record: FastaRecord):
        counts = Counter(real_record.sequence)

        # Canonical bases
        assert counts["A"] == 7_705_524
        assert counts["C"] == 4_760_025
        assert counts["G"] == 4_761_494
        assert counts["T"] == 7_668_600

        # Ambiguous bases
        assert counts.get("N", 0) == 0
        iupac_ambiguous = {"R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"}
        for ambig in iupac_ambiguous:
            assert counts.get(ambig, 0) == 0

        # Exact total sum
        total_canonical = counts["A"] + counts["C"] + counts["G"] + counts["T"]
        assert total_canonical == real_record.length

    def test_gc_content(self, real_record: FastaRecord):
        counts = Counter(real_record.sequence)
        gc_count = counts["G"] + counts["C"]
        gc_pct = (gc_count / real_record.length) * 100
        # Expected GC content for this pericentromeric region of chr1 is ~38.246%
        assert 38.2 < gc_pct < 38.3

    def test_stream_chromosomes_chunking_real_dataset(self, real_fasta_path: Path):
        reader = FastaReader(real_fasta_path)
        chrom_stream = reader.stream_chromosomes(chunk_size=10_000_000)
        chrom_list = list(chrom_stream)

        assert len(chrom_list) == 1
        chrom_id, chunk_gen = chrom_list[0]
        assert chrom_id == "ref|NC_000001.11|:85401937-110297579"

        chunks = list(chunk_gen)
        assert len(chunks) == 3
        assert len(chunks[0]) == 10_000_000
        assert len(chunks[1]) == 10_000_000
        assert len(chunks[2]) == 4_895_643
        assert sum(len(c) for c in chunks) == 24_895_643

    def test_genomic_chunker_partitioning_and_coordinates(self, real_record: FastaRecord):
        chunker = GenomicChunker(chunk_size=10_000_000, halo_size=22)
        chunks = chunker.chunk(real_record)

        assert len(chunks) == 3
        c0, c1, c2 = chunks

        # Chunk 0
        assert c0.chunk_id == "ref|NC_000001.11|:85401937-110297579_chunk_0000"
        assert c0.core_bounds == (0, 10_000_000)
        assert c0.actual_length == 10_000_022
        assert c0.halo_length == 22

        # Chunk 1
        assert c1.core_bounds == (10_000_000, 20_000_000)
        assert c1.actual_length == 10_000_022
        assert c1.halo_length == 22

        # Chunk 2 (terminal chunk)
        assert c2.core_bounds == (20_000_000, 24_895_643)
        assert c2.actual_length == 4_895_643
        assert c2.halo_length == 0

        # Verify coordinate translation to global chromosomal coordinates
        start_coord, end_coord = real_record.coordinates
        # First base in Chunk 0:
        c0_first_pos = (start_coord - 1) + c0.to_1based_coordinate(0)
        assert c0_first_pos == 85_401_937

        # Last core base in Chunk 2:
        c2_last_local = c2.core_length - 1  # 4,895,642
        c2_last_pos = (start_coord - 1) + c2.to_1based_coordinate(c2_last_local)
        assert c2_last_pos == 110_297_579

    def test_alignment_on_real_dataset_subslice(self, real_record: FastaRecord):
        # Sample a 20 nt guide from the real dataset at offset 1,000,000
        sample_offset = 1_000_000
        guide_seq = real_record.sequence[sample_offset : sample_offset + 20]
        pam_seq = real_record.sequence[sample_offset + 20 : sample_offset + 23]

        chunker = GenomicChunker(chunk_size=100_000, halo_size=22)
        # Create a chunk around the target (e.g. from 950,000 to 1,050,000)
        slice_start = 950_000
        slice_end = 1_050_000
        slice_seq = real_record.sequence[slice_start:slice_end]

        chunk = ChunkDescriptor(
            chunk_id="real_test_chunk",
            chromosome=real_record.accession,
            chunk_index=0,
            global_start=slice_start,
            global_end=slice_end,
            actual_length=slice_end - slice_start,
            core_start=slice_start,
            core_end=slice_end,
            halo_start=slice_end,
            halo_end=slice_end,
            sequence_data=slice_seq,
        )

        aligner = CPUAligner(config=AlignmentConfig(max_mismatches=0, seed_max_mismatches=0))
        result = aligner.align(chunk=chunk, guide=guide_seq)

        assert result.hit_count >= 1
        # Find the hit at our exact injected offset (1,000,000 - 950,000 = 50,000)
        exact_hits = [h for h in result.hits if h.local_offset == (sample_offset - slice_start)]
        assert len(exact_hits) == 1
        hit = exact_hits[0]
        assert hit.global_start == sample_offset
        assert hit.target_sequence[:20] == guide_seq
        assert hit.pam_sequence == pam_seq
        assert hit.mismatch_count == 0
