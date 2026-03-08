# DepMapRepository -- concrete implementation of DependencyRepository port.
# Downloads and caches DepMap 24Q4 CRISPR Chronos data.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from revgate.domain.entities.cell_line import CellLine
from revgate.domain.exceptions.data import DataNotAvailableError, DownloadFailedError
from revgate.infrastructure.external.file_cache import FileCache


# DepMap 24Q4 public download URLs
DEPMAP_CHRONOS_URL = (
    "https://depmap.org/portal/api/download/file"
    "?file_name=public_24Q4%2FCRISPRGeneEffect.csv"
)
DEPMAP_MODEL_URL = (
    "https://depmap.org/portal/api/download/file"
    "?file_name=public_24Q4%2FModel.csv"
)

CHRONOS_CACHE_KEY = "depmap/CRISPRGeneEffect.csv"
MODEL_CACHE_KEY = "depmap/Model.csv"

# DepMap OncotreeLineage -> TCGA cancer code mapping
# Values must match OncotreeLineage column in Model.csv exactly
LINEAGE_TO_TCGA: dict[str, str] = {
    "Myeloid": "LAML",
    "Skin": "SKCM",
    "Breast": "BRCA",
    "Kidney": "KIRC",
    "Pancreas": "PAAD",
    "Lung": "LUAD",
}


class DepMapRepository:
    """Concrete implementation of DependencyRepository.

    Downloads DepMap 24Q4 CRISPRGeneEffect.csv and Model.csv,
    caches them locally, and provides the domain interface.
    """

    def __init__(self, cache: FileCache) -> None:
        self._cache = cache
        self._dep_matrix: pd.DataFrame | None = None
        self._model_df: pd.DataFrame | None = None

    def get_dependency_matrix(self) -> pd.DataFrame:
        """Return the full Chronos dependency matrix.

        Returns:
            DataFrame (n_cell_lines x n_genes).
            Index: DepMap model IDs.
            Columns: gene symbols.
            Values: Chronos scores.
        """
        if self._dep_matrix is None:
            self._dep_matrix = self._load_chronos()
        return self._dep_matrix

    def get_cell_lines_for_cancer(self, cancer_id: str) -> list[CellLine]:
        """Return DepMap cell lines mapped to the given TCGA cancer type.

        Args:
            cancer_id: TCGA abbreviation (e.g. SKCM).

        Returns:
            List of CellLine entities for this cancer type.
        """
        if self._model_df is None:
            self._model_df = self._load_model()

        # Filter by TCGA mapping
        tcga_to_lineage = {v: k for k, v in LINEAGE_TO_TCGA.items()}
        lineage = tcga_to_lineage.get(cancer_id)

        if lineage is None:
            raise DataNotAvailableError(
                f"No DepMap lineage mapping for cancer_id={cancer_id!r}"
            )

        subset = self._model_df[
            self._model_df["OncotreeLineage"].str.contains(lineage, na=False)
        ]

        cell_lines: list[CellLine] = []
        for _, row in subset.iterrows():
            cell_lines.append(
                CellLine(
                    model_id=str(row.get("ModelID", "")),
                    cell_name=str(row.get("CellLineName", "")),
                    cancer_id=cancer_id,
                    lineage=lineage,
                )
            )

        return cell_lines

    def _load_chronos(self) -> pd.DataFrame:
        """Load Chronos matrix from cache or raise DataNotAvailableError."""
        path = self._cache.get(CHRONOS_CACHE_KEY)

        if path is None:
            raise DataNotAvailableError(
                f"DepMap Chronos matrix not cached. "
                f"Run: revgate download --source depmap"
            )

        df = pd.read_csv(path, index_col=0)

        # Columns are formatted as 'SYMBOL (ENTREZ_ID)' -- extract symbol only
        df.columns = [col.split(" (")[0] for col in df.columns]

        return df

    def _load_model(self) -> pd.DataFrame:
        """Load Model metadata from cache or raise DataNotAvailableError."""
        path = self._cache.get(MODEL_CACHE_KEY)

        if path is None:
            raise DataNotAvailableError(
                f"DepMap Model metadata not cached. "
                f"Run: revgate download --source depmap"
            )

        return pd.read_csv(path)

    def get_common_essentials(self, threshold: float = 0.8) -> set[str]:
        """Return set of common essential genes present in >threshold fraction of cell lines.

        These genes are universally essential across cancer types and should be
        excluded from cancer-type-specific dependency profiling.

        Args:
            threshold: Fraction of cell lines in which a gene must be essential
                       (Chronos < -0.5) to be considered a common essential.
                       Default 0.8 (80%).

        Returns:
            Set of gene symbols that are common essentials.
        """
        dep_matrix = self.get_dependency_matrix()
        frac_essential = (dep_matrix < -0.5).mean(axis=0)
        common = frac_essential[frac_essential > threshold].index.tolist()
        return set(common)

    def get_global_mean_scores(self) -> "pd.Series":
        """Return mean Chronos score per gene across ALL cell lines.

        Used as background for cancer-type-specific differential scoring.

        Returns:
            pd.Series indexed by gene symbol, values are mean Chronos scores.
        """
        return self.get_dependency_matrix().mean(axis=0)
