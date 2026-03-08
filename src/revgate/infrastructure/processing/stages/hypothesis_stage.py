# HypothesisStage -- Phase 3: H1/H2/H3 statistical tests.

from __future__ import annotations

from typing import Any

from revgate.application.dto.survival_dto import SurvivalRequestDTO
from revgate.application.use_cases.analyze_survival import AnalyzeSurvivalUseCase
from revgate.infrastructure.processing.stages.base_stage import BaseStage


class HypothesisStage(BaseStage):
    """Runs H1, H2, H3 statistical hypothesis tests.

    Context keys consumed:
        classification_result: ClassificationResultDTO
        cancer_ids:            list[str]
        min_patients:          int
        alpha:                 float

    Context keys produced:
        survival_result: SurvivalResultDTO
    """

    def __init__(self, analyze_use_case: AnalyzeSurvivalUseCase) -> None:
        self._use_case = analyze_use_case

    @property
    def name(self) -> str:
        return "HypothesisStage"

    async def _validate(self, context: dict[str, Any]) -> None:
        if "classification_result" not in context:
            raise ValueError("HypothesisStage requires 'classification_result' in context")

    async def _process(self, context: dict[str, Any]) -> dict[str, Any]:
        classification_result = context["classification_result"]

        nv_class_map = {
            axis.cancer_id: axis.nv_class
            for axis in classification_result.axes
        }

        request = SurvivalRequestDTO(
            cancer_ids=[a.cancer_id for a in classification_result.axes],
            min_patients=context.get("min_patients", 20),
            alpha=context.get("alpha", 0.05),
        )

        result = self._use_case.execute(request, nv_class_map)
        context["survival_result"] = result

        return {
            "km_count": len(result.km_results),
            "cox_count": len(result.cox_results),
            "roc_count": len(result.roc_results),
            "error_count": len(result.errors),
        }
