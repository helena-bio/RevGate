# GnomADRepository -- concrete implementation of ConstraintRepository port.
# Downloads and caches gnomAD v4 gene constraint metrics.

from __future__ import annotations

import pandas as pd

from revgate.domain.exceptions.data import DataNotAvailableError
from revgate.infrastructure.external.file_cache import FileCache


GNOMAD_CONSTRAINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad"
    "/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv"
)

GNOMAD_CACHE_KEY = "gnomad/gnomad.v4.constraint.tsv"


class GnomADRepository:
    """Concrete implementation of ConstraintRepository.

    Loads gnomAD v4 constraint metrics (pLI, LOEUF) from local cache.
    """

    def __init__(self, cache: FileCache) -> None:
        self._cache = cache
        self._df: pd.DataFrame | None = None

    def get_pli_scores(self) -> dict[str, float]:
        """Return pLI scores for all genes in gnomAD v4.

        Returns:
            Dict mapping gene symbol -> pLI [0.0, 1.0].
        """
        df = self._load()
        result: dict[str, float] = {}

        if "gene" in df.columns and "lof.pLI" in df.columns:
            for _, row in df.iterrows():
                gene = str(row["gene"])
                pli = row["lof.pLI"]
                if pd.notna(pli):
                    result[gene] = float(pli)

        return result

    def get_loeuf_scores(self) -> dict[str, float]:
        """Return LOEUF scores for all genes in gnomAD v4.

        Returns:
            Dict mapping gene symbol -> LOEUF score.
        """
        df = self._load()
        result: dict[str, float] = {}

        if "gene" in df.columns and "lof.oe_ci.upper" in df.columns:
            for _, row in df.iterrows():
                gene = str(row["gene"])
                loeuf = row["lof.oe_ci.upper"]
                if pd.notna(loeuf):
                    result[gene] = float(loeuf)

        return result

    def _load(self) -> pd.DataFrame:
        """Load constraint TSV from cache or raise DataNotAvailableError."""
        if self._df is not None:
            return self._df

        path = self._cache.get(GNOMAD_CACHE_KEY)

        if path is None:
            raise DataNotAvailableError(
                "gnomAD v4 constraint data not cached. "
                "Run: revgate download --source gnomad"
            )

        self._df = pd.read_csv(path, sep="\t", low_memory=False)
        return self._df
