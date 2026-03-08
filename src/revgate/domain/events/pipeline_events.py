# Pipeline domain events -- facts emitted during pipeline execution.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from revgate.domain.events.base import DomainEvent
from revgate.domain.value_objects.nv_score import NVClass
from revgate.domain.value_objects.rdp_score import RDPClass
from revgate.domain.value_objects.cep_score import CEPClass
from revgate.domain.value_objects.mp_score import MPClass


@dataclass(frozen=True)
class PipelineStarted(DomainEvent):
    """Emitted when the validation pipeline begins execution.

    Attributes:
        cancer_ids -- list of TCGA cancer type codes being processed
    """

    cancer_ids: tuple[str, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return f"PipelineStarted(cancers={list(self.cancer_ids)}, at={self.occurred_at})"


@dataclass(frozen=True)
class DataLoadingCompleted(DomainEvent):
    """Emitted when all data sources have been downloaded and cached.

    Attributes:
        sources -- data sources successfully loaded
    """

    sources: tuple[str, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return f"DataLoadingCompleted(sources={list(self.sources)}, at={self.occurred_at})"


@dataclass(frozen=True)
class InvariantCheckPassed(DomainEvent):
    """Emitted when all Tier 0 biological invariant checks pass.

    Attributes:
        invariant_count -- number of invariants that passed
    """

    invariant_count: int = 0

    def __repr__(self) -> str:
        return f"InvariantCheckPassed(count={self.invariant_count}, at={self.occurred_at})"


@dataclass(frozen=True)
class CancerTypeClassified(DomainEvent):
    """Emitted when a single cancer type completes 4-axis classification.

    Attributes:
        cancer_id -- TCGA cancer type code
        nv_class  -- assigned NV-Class
        rdp_class -- assigned RDP-Class
        cep_class -- assigned CEP-Class
        mp_class  -- assigned MP-Class
    """

    cancer_id: str = ""
    nv_class: NVClass = NVClass.NV_B
    rdp_class: RDPClass = RDPClass.RDP_V
    cep_class: CEPClass = CEPClass.CEP_II
    mp_class: MPClass = MPClass.MP_II

    def __repr__(self) -> str:
        return (
            f"CancerTypeClassified(cancer={self.cancer_id}, "
            f"NV={self.nv_class.value}, RDP={self.rdp_class.value}, "
            f"CEP={self.cep_class.value}, MP={self.mp_class.value})"
        )


@dataclass(frozen=True)
class ClassificationCompleted(DomainEvent):
    """Emitted when all cancer types have been classified.

    Attributes:
        cancer_count -- number of cancer types successfully classified
    """

    cancer_count: int = 0

    def __repr__(self) -> str:
        return f"ClassificationCompleted(count={self.cancer_count}, at={self.occurred_at})"


@dataclass(frozen=True)
class HypothesisTestCompleted(DomainEvent):
    """Emitted when H1 / H2 / H3 statistical tests complete.

    Attributes:
        h1_passed -- whether H1 passed
        h2_passed -- whether H2 passed
        h3_passed -- whether H3 passed
    """

    h1_passed: bool = False
    h2_passed: bool = False
    h3_passed: bool = False

    @property
    def passed_count(self) -> int:
        return sum([self.h1_passed, self.h2_passed, self.h3_passed])

    def __repr__(self) -> str:
        return (
            f"HypothesisTestCompleted("
            f"H1={self.h1_passed}, H2={self.h2_passed}, H3={self.h3_passed}, "
            f"passed={self.passed_count}/3)"
        )


@dataclass(frozen=True)
class PipelineCompleted(DomainEvent):
    """Emitted when the full pipeline finishes successfully.

    Attributes:
        duration_seconds -- total wall-clock time in seconds
        output_path      -- path where results were written
    """

    duration_seconds: float = 0.0
    output_path: str = ""

    def __repr__(self) -> str:
        return (
            f"PipelineCompleted("
            f"duration={self.duration_seconds:.1f}s, "
            f"output={self.output_path!r})"
        )
