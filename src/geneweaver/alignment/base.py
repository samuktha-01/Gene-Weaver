"""Base classes, abstract interfaces, and protocols for CRISPR sequence alignment.

This module defines the architectural contract that all concrete alignment engines
(e.g., CPUAligner using NumPy, CUDAAligner using Numba CUDA) must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, Union, runtime_checkable

from geneweaver.alignment.models import (
    AlignmentConfig,
    AlignmentResult,
    GuideTarget,
    MismatchInfo,
    RawCandidateHit,
    Strand,
)
from geneweaver.sequence_processor.chunker import ChunkDescriptor


@runtime_checkable
class SequenceAligner(Protocol):
    """Structural protocol defining the contract for CRISPR off-target sequence aligners.

    Any class providing the `backend_name` property and `align` method conforms to
    this protocol, whether executed via CPU vectorization, CUDA SIMT kernels, or
    distributed worker nodes.
    """

    @property
    def backend_name(self) -> str:
        """Returns the unique identifier of the execution backend (e.g., 'cpu-numpy', 'cuda-numba')."""
        ...

    def align(
        self,
        chunk: ChunkDescriptor,
        guide: Union[GuideTarget, str],
        config: Optional[AlignmentConfig] = None,
    ) -> AlignmentResult:
        """Aligns a CRISPR target guide sequence against a genomic chunk.

        Args:
            chunk: Genomic chunk descriptor containing sequence data, halo, and coordinates.
            guide: Target guide sequence (GuideTarget instance or 23 nt ASCII string).
            config: Optional runtime configuration overriding the aligner's default settings.

        Returns:
            AlignmentResult containing detected candidate hits and execution telemetry.
        """
        ...


class BaseSequenceAligner(ABC):
    """Abstract base class providing shared parameter validation and input resolution.

    Concrete aligners (CPUAligner, CUDAAligner) inherit from this base class and
    implement the backend-specific `_align_chunk` method.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        """Initializes the aligner with default runtime configuration.

        Args:
            config: Default alignment configuration (uses AlignmentConfig() if omitted).
        """
        self.config = config or AlignmentConfig()

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Returns the execution backend name."""
        pass

    def align(
        self,
        chunk: ChunkDescriptor,
        guide: Union[GuideTarget, str],
        config: Optional[AlignmentConfig] = None,
    ) -> AlignmentResult:
        """Validates inputs, resolves guide representations, and dispatches to backend implementation.

        Args:
            chunk: ChunkDescriptor instance.
            guide: GuideTarget instance or 23 nt guide string.
            config: Optional override configuration.

        Returns:
            AlignmentResult instance containing detected candidate hits.

        Raises:
            TypeError: If chunk is not a ChunkDescriptor.
            ValueError: If guide length or sequence is invalid.
        """
        if not isinstance(chunk, ChunkDescriptor):
            raise TypeError(f"Expected chunk to be ChunkDescriptor, got {type(chunk).__name__}")

        resolved_guide = self._resolve_guide(guide)
        effective_config = config if config is not None else self.config

        return self._align_chunk(
            chunk=chunk,
            guide=resolved_guide,
            effective_config=effective_config,
        )

    @abstractmethod
    def _align_chunk(
        self,
        chunk: ChunkDescriptor,
        guide: GuideTarget,
        effective_config: AlignmentConfig,
    ) -> AlignmentResult:
        """Backend-specific alignment execution method.

        Must be implemented by concrete subclasses (e.g. CPUAligner or CUDAAligner).

        Args:
            chunk: Validated ChunkDescriptor.
            guide: Resolved GuideTarget with pre-computed 2-bit representation.
            effective_config: Validated AlignmentConfig parameters.

        Returns:
            AlignmentResult instance containing detected candidate hits.
        """
        pass

    @staticmethod
    def _resolve_guide(guide: Union[GuideTarget, str]) -> GuideTarget:
        """Resolves guide input into a typed GuideTarget instance."""
        if isinstance(guide, GuideTarget):
            return guide
        if isinstance(guide, str):
            return GuideTarget.from_string(guide)
        raise TypeError(f"Expected guide to be GuideTarget or str, got {type(guide).__name__}")


__all__ = [
    "Strand",
    "MismatchInfo",
    "AlignmentConfig",
    "GuideTarget",
    "RawCandidateHit",
    "AlignmentResult",
    "SequenceAligner",
    "BaseSequenceAligner",
]
