# DTOs for survival analysis use case.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SurvivalRequestDTO:
    """Input DTO for AnalyzeSurvivalUseCase.

    Attributes:
        cancer_ids       -- cancer types to include in survival analysis
        min_patients     -- minimum patients per NV-Class group for KM
        alpha            -- significance level
    """

    cancer_ids: list[str]
    min_patients: int = 20
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not self.cancer_ids:
            raise ValueError("SurvivalRequestDTO.cancer_ids must not be empty")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


@dataclass
class KMResultDTO:
    """Kaplan-Meier result for a single cancer type."""

    cancer_id: str
    logrank_p_value: float
    median_os_nv_a: float | None
    median_os_nv_b: float | None
    median_os_nv_c: float | None
    n_nv_a: int
    n_nv_b: int
    n_nv_c: int


@dataclass
class CoxResultDTO:
    """Cox proportional hazards result for a single cancer type."""

    cancer_id: str
    hazard_ratio_nvc_vs_nva: float
    ci_lower: float
    ci_upper: float
    p_value: float
    c_index_base: float
    c_index_with_nv: float
    lrt_p_value: float


@dataclass
class ROCResultDTO:
    """ROC analysis result for metastasis prediction (H3)."""

    cancer_id: str
    auc_tnm_only: float
    auc_tnm_plus_mp: float
    delong_p_value: float
    nri: float
    nri_p_value: float


@dataclass
class SurvivalResultDTO:
    """Output DTO for AnalyzeSurvivalUseCase."""

    km_results: list[KMResultDTO] = field(default_factory=list)
    cox_results: list[CoxResultDTO] = field(default_factory=list)
    roc_results: list[ROCResultDTO] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    duration_sec: float = 0.0
