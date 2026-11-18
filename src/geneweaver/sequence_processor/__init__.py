"""Sequence processing package for GeneWeaver.

Handles nucleotide sequence sanitization, IUPAC ambiguity code resolution, 2-bit binary
encoding (A:00, C:01, G:10, T:11), and halo-partitioned genomic chunk generation.
"""

from geneweaver.sequence_processor.chunker import ChunkDescriptor, GenomicChunker
from geneweaver.sequence_processor.encoder import EncodedTwoBit, TwoBitEncoder
from geneweaver.sequence_processor.normalizer import SequenceNormalizer

__all__ = [
    "SequenceNormalizer",
    "TwoBitEncoder",
    "EncodedTwoBit",
    "GenomicChunker",
    "ChunkDescriptor",
]
