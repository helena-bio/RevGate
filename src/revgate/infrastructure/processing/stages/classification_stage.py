# ClassificationStage -- Phase 2: run 4-axis classification for all cancer types.

from __future__ import annotations

from typing import Any

from revgate.application.dto.classification_dto import ClassificationRequestDTO
from revgate.application.use_cases.classify_tumors import ClassifyTumorUseCase
from revgate.infrastructure.processing.stages.base_stage import BaseStage


class ClassificationStage(BaseStage):
    """Runs 4-axis DNV-TC classification for all requested cancer types.

    Context keys consumed:
        cancer_ids: list[str]
        top_n:      int

    Context keys produced:
        classification_result: ClassificationResultDTO
    """

    def __init__(self, classify_use_case: ClassifyTumorUseCase) -> None:
        self._use_case = classify_use_case

    @property
    def name(self) -> str:
        return "ClassificationStage"

    async def _validate(self, context: dict[str, Any]) -> None:
        if "cancer_ids" not in context:
            raise ValueError("ClassificationStage requires 'cancer_ids' in context")
        if not context["cancer_ids"]:
            raise ValueError("ClassificationStage: cancer_ids must not be empty")

    async def _process(self, context: dict[str, Any]) -> dict[str, Any]:
        request = ClassificationRequestDTO(
            cancer_ids=context["cancer_ids"],
            top_n=context.get("top_n", 20),
        )
        result = self._use_case.execute(request)
        context["classification_result"] = result

        return {
            "success_count": result.success_count,
            "error_count": result.error_count,
            "duration_sec": result.duration_sec,
        }
