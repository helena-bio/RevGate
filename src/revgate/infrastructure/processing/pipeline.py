# PipelineOrchestrator -- runs all stages in sequence.
# Infrastructure layer coordination.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from revgate.infrastructure.processing.stages.base_stage import BaseStage, StageResult


@dataclass
class PipelineResult:
    """Result of a full pipeline run.

    Attributes:
        success        -- True if all stages completed without error
        stage_results  -- ordered list of StageResult per stage
        duration_sec   -- total wall time
        context        -- final pipeline context after all stages
    """

    success: bool
    stage_results: list[StageResult] = field(default_factory=list)
    duration_sec: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_stages(self) -> list[StageResult]:
        return [r for r in self.stage_results if not r.success]


class PipelineOrchestrator:
    """Runs pipeline stages sequentially with shared context.

    Stops on first stage failure by default.
    Each stage receives and can mutate the shared context dict.
    """

    def __init__(
        self,
        stages: list[BaseStage],
        stop_on_failure: bool = True,
    ) -> None:
        self._stages = stages
        self._stop_on_failure = stop_on_failure

    async def run(self, initial_context: dict[str, Any]) -> PipelineResult:
        """Execute all stages in order.

        Args:
            initial_context: Starting context dict (cancer_ids, top_n, etc.)

        Returns:
            PipelineResult with per-stage results and final context.
        """
        started_at = time.monotonic()
        context = dict(initial_context)
        stage_results: list[StageResult] = []
        overall_success = True

        for stage in self._stages:
            result = await stage.execute(context)
            stage_results.append(result)

            if not result.success:
                overall_success = False
                if self._stop_on_failure:
                    break

        return PipelineResult(
            success=overall_success,
            stage_results=stage_results,
            duration_sec=time.monotonic() - started_at,
            context=context,
        )
