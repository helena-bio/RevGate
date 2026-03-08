# DataLoadingStage -- Phase 1: verify all data sources are cached.

from __future__ import annotations

from typing import Any

from revgate.domain.exceptions.data import DataNotAvailableError
from revgate.infrastructure.external.file_cache import FileCache
from revgate.infrastructure.processing.stages.base_stage import BaseStage


class DataLoadingStage(BaseStage):
    """Verifies all required data sources are present in the local cache.

    Does not download -- download is a separate CLI command (revgate download).
    Raises DataNotAvailableError if any required source is missing.

    Context keys produced:
        data_sources_verified: bool
    """

    REQUIRED_KEYS = [
        "depmap/CRISPRGeneEffect.csv",
        "depmap/Model.csv",
        "string/9606.protein.links.v12.0.txt",
        "gnomad/gnomad.v4.constraint.tsv",
    ]

    def __init__(self, cache: FileCache) -> None:
        self._cache = cache

    @property
    def name(self) -> str:
        return "DataLoadingStage"

    async def _validate(self, context: dict[str, Any]) -> None:
        missing = [key for key in self.REQUIRED_KEYS if not self._cache.exists(key)]
        if missing:
            raise DataNotAvailableError(
                f"Missing cached data sources: {missing}. "
                f"Run: revgate download --all"
            )

    async def _process(self, context: dict[str, Any]) -> dict[str, Any]:
        context["data_sources_verified"] = True
        return {"verified_sources": self.REQUIRED_KEYS}
