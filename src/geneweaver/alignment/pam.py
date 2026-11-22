"""PAM (Protospacer Adjacent Motif) validation interface for GeneWeaver alignment.

Re-exports PAM recognition and validation models from geneweaver.scoring.pam_evaluator.
"""

from geneweaver.scoring.pam_evaluator import (
    IUPACMatcher,
    NucleaseConfig,
    PAMLocation,
    PAMPattern,
    PAMValidationResult,
    PAMValidator,
    SPCAS9,
    ValidatedCandidateHit,
)

__all__ = [
    "PAMLocation",
    "IUPACMatcher",
    "PAMPattern",
    "NucleaseConfig",
    "SPCAS9",
    "PAMValidationResult",
    "ValidatedCandidateHit",
    "PAMValidator",
]
