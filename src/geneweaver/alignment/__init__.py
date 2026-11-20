"""Alignment package for GeneWeaver.

Provides alignment interface protocols, base aligner classes, strongly-typed
data models, and coordinates sequence alignment execution across CPU and GPU backends.
"""

from geneweaver.alignment.base import (
    BaseSequenceAligner,
    SequenceAligner,
)
from geneweaver.alignment.cpu_aligner import CPUAligner
from geneweaver.alignment.models import (
    AlignmentConfig,
    AlignmentResult,
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    Strand,
)

__all__ = [
    "Strand",
    "MismatchInfo",
    "AlignmentConfig",
    "GuideTarget",
    "RawCandidateHit",
    "AlignmentResult",
    "SequenceAligner",
    "BaseSequenceAligner",
    "CPUAligner",
]
