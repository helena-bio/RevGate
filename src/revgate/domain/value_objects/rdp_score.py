# Reactivated Developmental Program score -- ssGSEA enrichment result.
# Immutable value object.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RDPClass(str, Enum):
    """Reactivated Developmental Program classification."""

    RDP_I   = "RDP-I"    # Pluripotency / Stemness
    RDP_II  = "RDP-II"   # Neural Crest
    RDP_III = "RDP-III"  # EMT (Epithelial-Mesenchymal Transition)
    RDP_IV  = "RDP-IV"   # Morphogenic
    RDP_V   = "RDP-V"    # Lineage Commitment


# Minimum NES for a program to be considered enriched
NES_THRESHOLD = 2.0
FDR_THRESHOLD = 0.05


@dataclass(frozen=True)
class RDPScore:
    """ssGSEA enrichment result for a single developmental program.

    Attributes:
        program   -- which RDP program this score belongs to
        nes       -- normalized enrichment score from ssGSEA
        fdr       -- false discovery rate
    """

    program: RDPClass
    nes: float
    fdr: float

    def __post_init__(self) -> None:
        if self.fdr < 0.0 or self.fdr > 1.0:
            raise ValueError(f"RDPScore.fdr must be in [0.0, 1.0], got {self.fdr}")

    @property
    def is_significant(self) -> bool:
        """True if NES >= threshold AND FDR < 0.05."""
        return self.nes >= NES_THRESHOLD and self.fdr < FDR_THRESHOLD

    def __repr__(self) -> str:
        return (
            f"RDPScore(program={self.program.value}, "
            f"nes={self.nes:.4f}, fdr={self.fdr:.4f}, "
            f"significant={self.is_significant})"
        )
