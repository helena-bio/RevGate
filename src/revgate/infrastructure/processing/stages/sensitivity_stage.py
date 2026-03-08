# SensitivityStage -- Phase 4: parameter sweep robustness analysis.

from __future__ import annotations

from typing import Any

from revgate.application.use_cases.run_sensitivity import (
    RunSensitivityUseCase,
    SensitivityRequestDTO,
)
from revgate.infrastructure.processing.stages.base_stage import BaseStage


class SensitivityStage(BaseStage):
    """Runs sensitivity analysis across parameter variations.

    Context keys consumed:
        cancer_ids:       list[str]
        top_n_range:      list[int]   (optional, default [10,15,20,30,50])
        string_thresholds: list[int]  (optional, default [500,700,900])
        damping_range:    list[float] (optional, default [0.70,0.80,0.85,0.90])

    Context keys produced:
        sensitivity_result: SensitivityResultDTO
    """

    def __init__(self, sensitivity_use_case: RunSensitivityUseCase) -> None:
        self._use_case = sensitivity_use_case

    @property
    def name(self) -> str:
        return "SensitivityStage"

    async def _validate(self, context: dict[str, Any]) -> None:
        if "cancer_ids" not in context:
            raise ValueError("SensitivityStage requires 'cancer_ids' in context")

    async def _process(self, context: dict[str, Any]) -> dict[str, Any]:
        request = SensitivityRequestDTO(
            cancer_ids=context["cancer_ids"],
            top_n_range=context.get("top_n_range", [10, 15, 20, 30, 50]),
            string_thresholds=context.get("string_thresholds", [500, 700, 900]),
            damping_range=context.get("damping_range", [0.70, 0.80, 0.85, 0.90]),
        )

        result = self._use_case.execute(request)
        context["sensitivity_result"] = result

        return {
            "top_n_sweep_count": len(request.top_n_range),
            "string_sweep_count": len(request.string_thresholds),
            "damping_sweep_count": len(request.damping_range),
            "duration_sec": result.duration_sec,
        }
