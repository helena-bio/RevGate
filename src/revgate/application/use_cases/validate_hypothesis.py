# ValidateHypothesisUseCase -- main pipeline orchestrator.
# Coordinates all use cases in correct execution order.
# Application layer entry point.

from __future__ import annotations

import time
from dataclasses import dataclass, field

from revgate.application.dto.classification_dto import ClassificationRequestDTO
from revgate.application.dto.survival_dto import SurvivalRequestDTO
from revgate.application.use_cases.analyze_survival import AnalyzeSurvivalUseCase
from revgate.application.use_cases.classify_tumors import ClassifyTumorUseCase
from revgate.domain.events.pipeline_events import (
    ClassificationCompleted,
    HypothesisTestCompleted,
    PipelineCompleted,
    PipelineStarted,
)
from revgate.domain.services.hypothesis_tester import (
    H1Result,
    H2Result,
    H3Result,
    HypothesisResult,
    ValidationResult,
)


@dataclass
class ValidationRequestDTO:
    """Input DTO for the full pipeline.

    Attributes:
        cancer_ids   -- TCGA cancer type codes to process
        top_n        -- top dependency genes for NV-Score (default 20)
        min_patients -- minimum patients per group for KM (default 20)
        alpha        -- significance level (default 0.05)
        output_path  -- directory to write results
    """

    cancer_ids: list[str]
    top_n: int = 20
    min_patients: int = 20
    alpha: float = 0.05
    output_path: str = "results/"

    def __post_init__(self) -> None:
        if not self.cancer_ids:
            raise ValueError("ValidationRequestDTO.cancer_ids must not be empty")


@dataclass
class FullValidationResultDTO:
    """Output DTO for the full pipeline.

    Attributes:
        validation_result   -- H1/H2/H3 hypothesis test results
        classification_axes -- per-cancer 4-axis classification
        events              -- domain events emitted during execution
        duration_sec        -- total pipeline wall time
        errors              -- any errors encountered
    """

    validation_result: ValidationResult | None = None
    classification_axes: list = field(default_factory=list)
    events: list = field(default_factory=list)
    duration_sec: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.validation_result is not None and not self.errors


class ValidateHypothesisUseCase:
    """Main pipeline orchestrator for the DNV-TC hypothesis validation.

    Execution order (from HLD Section 4.1):
        Phase 1: Data acquisition (via repositories)
        Phase 2: Classification per cancer type (ClassifyTumorUseCase)
        Phase 3: Hypothesis testing H1/H2/H3 (AnalyzeSurvivalUseCase)
        Phase 4: Output generation (presentation layer)

    All coordination is here. No business logic in this class.
    """

    def __init__(
        self,
        classify_use_case: ClassifyTumorUseCase,
        analyze_survival_use_case: AnalyzeSurvivalUseCase,
    ) -> None:
        self._classify = classify_use_case
        self._analyze_survival = analyze_survival_use_case
        self._events: list = []

    def execute(self, request: ValidationRequestDTO) -> FullValidationResultDTO:
        """Run the full validation pipeline.

        Args:
            request: ValidationRequestDTO with cancer_ids and pipeline parameters.

        Returns:
            FullValidationResultDTO with classification results and H1/H2/H3 outcomes.
        """
        started_at = time.monotonic()
        self._events.clear()

        # Emit pipeline started event
        self._events.append(PipelineStarted(cancer_ids=tuple(request.cancer_ids)))

        # Phase 2: Classification
        classification_request = ClassificationRequestDTO(
            cancer_ids=request.cancer_ids,
            top_n=request.top_n,
        )
        classification_result = self._classify.execute(classification_request)

        if classification_result.error_count == len(request.cancer_ids):
            return FullValidationResultDTO(
                errors=classification_result.errors,
                duration_sec=time.monotonic() - started_at,
            )

        self._events.append(
            ClassificationCompleted(cancer_count=classification_result.success_count)
        )

        # Build nv_class lookup for survival analysis
        nv_class_map = {
            axis.cancer_id: axis.nv_class
            for axis in classification_result.axes
        }

        # Phase 3: Survival analysis (H2 + H3)
        survival_request = SurvivalRequestDTO(
            cancer_ids=[a.cancer_id for a in classification_result.axes],
            min_patients=request.min_patients,
            alpha=request.alpha,
        )
        survival_result = self._analyze_survival.execute(survival_request, nv_class_map)

        # Assemble H1/H2/H3 results
        # H1 result assembled from classification clustering (stub values pending)
        h1 = H1Result(
            adjusted_rand_index=0.0,
            permutation_p_value=1.0,
            permanova_r2=0.0,
            permanova_p_value=1.0,
        )

        h2 = self._assemble_h2(survival_result)
        h3 = self._assemble_h3(survival_result)

        validation_result = ValidationResult(h1=h1, h2=h2, h3=h3)

        self._events.append(
            HypothesisTestCompleted(
                h1_passed=h1.result == HypothesisResult.PASS,
                h2_passed=h2.result == HypothesisResult.PASS,
                h3_passed=h3.result == HypothesisResult.PASS,
            )
        )

        duration = time.monotonic() - started_at
        self._events.append(
            PipelineCompleted(duration_seconds=duration, output_path=request.output_path)
        )

        return FullValidationResultDTO(
            validation_result=validation_result,
            classification_axes=classification_result.axes,
            events=list(self._events),
            duration_sec=duration,
            errors={**classification_result.errors, **survival_result.errors},
        )

    def _assemble_h2(self, survival_result) -> H2Result:
        """Assemble H2Result from Cox regression outputs."""
        if not survival_result.cox_results:
            return H2Result(
                logrank_p_value=1.0,
                hazard_ratio_nvc_vs_nva=1.0,
                hazard_ratio_ci_lower=0.0,
                hazard_ratio_ci_upper=99.0,
                c_index_base=0.5,
                c_index_with_nv=0.5,
            )

        # Use first available Cox result as representative
        cox = survival_result.cox_results[0]
        km = survival_result.km_results[0] if survival_result.km_results else None

        return H2Result(
            logrank_p_value=km.logrank_p_value if km else 1.0,
            hazard_ratio_nvc_vs_nva=cox.hazard_ratio_nvc_vs_nva,
            hazard_ratio_ci_lower=cox.ci_lower,
            hazard_ratio_ci_upper=cox.ci_upper,
            c_index_base=cox.c_index_base,
            c_index_with_nv=cox.c_index_with_nv,
        )

    def _assemble_h3(self, survival_result) -> H3Result:
        """Assemble H3Result from ROC analysis outputs."""
        if not survival_result.roc_results:
            return H3Result(
                auc_tnm_only=0.5,
                auc_tnm_plus_mp=0.5,
                delong_p_value=1.0,
                nri=0.0,
                nri_p_value=1.0,
            )

        roc = survival_result.roc_results[0]

        return H3Result(
            auc_tnm_only=roc.auc_tnm_only,
            auc_tnm_plus_mp=roc.auc_tnm_plus_mp,
            delong_p_value=roc.delong_p_value,
            nri=roc.nri,
            nri_p_value=roc.nri_p_value,
        )
