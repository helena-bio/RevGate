# DTOs for tumor classification use case.
# No domain entities cross the application boundary -- only DTOs.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClassificationRequestDTO:
    """Input DTO for ClassifyTumorUseCase.

    Attributes:
        cancer_ids -- list of TCGA cancer type codes to classify
        top_n      -- number of top dependency genes to use (default 20)
    """

    cancer_ids: list[str]
    top_n: int = 20

    def __post_init__(self) -> None:
        if not self.cancer_ids:
            raise ValueError("ClassificationRequestDTO.cancer_ids must not be empty")
        if self.top_n < 5:
            raise ValueError(f"top_n must be >= 5, got {self.top_n}")


@dataclass
class AxisResultDTO:
    """Single axis classification result for one cancer type."""

    cancer_id: str
    nv_class: str
    rdp_class: str
    cep_class: str
    mp_class: str
    nv_composite_score: float
    gini: float
    mean_pli: float
    mean_centrality: float
    cep_value: float
    invasion_zscore: float


@dataclass
class ClassificationResultDTO:
    """Output DTO for ClassifyTumorUseCase.

    Attributes:
        axes          -- per-cancer classification results
        errors        -- cancer IDs that failed with error messages
        duration_sec  -- total classification wall time
    """

    axes: list[AxisResultDTO] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    duration_sec: float = 0.0

    @property
    def success_count(self) -> int:
        return len(self.axes)

    @property
    def error_count(self) -> int:
        return len(self.errors)
