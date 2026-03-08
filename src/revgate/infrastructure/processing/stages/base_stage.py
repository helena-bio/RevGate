# BaseStage -- Template Method pattern for pipeline stages.
# All processing stages inherit from this class.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    """Result of a single pipeline stage execution.

    Attributes:
        stage_name    -- name of the stage
        success       -- whether the stage completed without error
        duration_sec  -- wall time in seconds
        metadata      -- arbitrary stage-specific metadata
        error         -- error message if success is False
    """

    stage_name: str
    success: bool
    duration_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseStage(ABC):
    """Abstract base class for all RevGate pipeline stages.

    Implements Template Method pattern:
        execute() calls _validate() -> _process() -> _finalize()

    Subclasses implement _process().
    _validate() and _finalize() are optional hooks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name."""
        ...

    async def execute(self, context: dict[str, Any]) -> StageResult:
        """Execute the stage with full lifecycle management.

        Args:
            context: Shared pipeline context dict passed through all stages.

        Returns:
            StageResult with success status, timing, and metadata.
        """
        import time

        started_at = time.monotonic()

        try:
            await self._validate(context)
            metadata = await self._process(context)
            await self._finalize(context)

            return StageResult(
                stage_name=self.name,
                success=True,
                duration_sec=time.monotonic() - started_at,
                metadata=metadata or {},
            )

        except Exception as exc:
            return StageResult(
                stage_name=self.name,
                success=False,
                duration_sec=time.monotonic() - started_at,
                error=str(exc),
            )

    async def _validate(self, context: dict[str, Any]) -> None:
        """Pre-execution validation hook. Override to add checks."""
        pass

    @abstractmethod
    async def _process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Core stage logic. Must be implemented by subclasses.

        Args:
            context: Shared pipeline context.

        Returns:
            Metadata dict to include in StageResult.
        """
        ...

    async def _finalize(self, context: dict[str, Any]) -> None:
        """Post-execution cleanup hook. Override to add cleanup."""
        pass
