# TumorClassification entity -- the 4-axis DNV-TC classification output.
# Central output entity of the RevGate pipeline.

from __future__ import annotations

from dataclasses import dataclass

from revgate.domain.value_objects.cep_score import CEPClass, CEPScore
from revgate.domain.value_objects.mp_score import MPClass, MPScore
from revgate.domain.value_objects.nv_score import NVClass, NVScore
from revgate.domain.value_objects.rdp_score import RDPClass, RDPScore


@dataclass
class TumorClassification:
    """4-axis DNV-TC classification for a single cancer type.

    Axes:
        rdp_class -- which embryonic program is reactivated (RDP-I..V)
        nv_class  -- dependency concentration pattern (NV-A/B/C)
        cep_class -- cascade expansion phenotype (CEP-I/II/III)
        mp_class  -- metastatic propensity (MP-I/II/III)

    Each axis answers an independent clinical question.
    Together they map to a therapeutic strategy.
    """

    cancer_id: str
    nv_score: NVScore
    rdp_scores: list[RDPScore]
    cep_score: CEPScore
    mp_score: MPScore

    def __post_init__(self) -> None:
        if not self.cancer_id:
            raise ValueError("TumorClassification.cancer_id must not be empty")
        if not self.rdp_scores:
            raise ValueError("TumorClassification.rdp_scores must not be empty")

    @property
    def nv_class(self) -> NVClass:
        return self.nv_score.nv_class

    @property
    def rdp_class(self) -> RDPClass:
        """Assign RDP-Class as the program with the highest significant NES.
        If no program reaches significance, return the highest NES regardless.
        """
        significant = [s for s in self.rdp_scores if s.is_significant]
        pool = significant if significant else self.rdp_scores
        return max(pool, key=lambda s: s.nes).program

    @property
    def cep_class(self) -> CEPClass:
        return self.cep_score.cep_class

    @property
    def mp_class(self) -> MPClass:
        return self.mp_score.mp_class

    def summary(self) -> str:
        """Human-readable 4-axis classification summary."""
        return (
            f"{self.cancer_id}: "
            f"NV={self.nv_class.value} | "
            f"RDP={self.rdp_class.value} | "
            f"CEP={self.cep_class.value} | "
            f"MP={self.mp_class.value}"
        )

    def __repr__(self) -> str:
        return f"TumorClassification({self.summary()})"
