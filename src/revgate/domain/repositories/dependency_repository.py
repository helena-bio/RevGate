# Port interface for DepMap dependency data access.
# Infrastructure layer implements this protocol.

from typing import Protocol, runtime_checkable

import pandas as pd

from revgate.domain.entities.cell_line import CellLine


@runtime_checkable
class DependencyRepository(Protocol):
    """Abstract interface for DepMap CRISPR Chronos dependency data.

    The concrete implementation in infrastructure downloads and caches
    the DepMap 24Q4 CRISPRGeneEffect.csv and Model.csv files.

    Methods:
        get_dependency_matrix    -- full genes x cell_lines Chronos matrix
        get_cell_lines_for_cancer -- cell lines belonging to a cancer type
    """

    def get_dependency_matrix(self) -> pd.DataFrame:
        """Return the full Chronos dependency matrix.

        Returns:
            DataFrame with shape (n_cell_lines, n_genes).
            Index: DepMap model IDs.
            Columns: gene symbols.
            Values: Chronos scores (float, negative = dependency).
        """
        ...

    def get_cell_lines_for_cancer(self, cancer_id: str) -> list[CellLine]:
        """Return all DepMap cell lines mapped to the given TCGA cancer type.

        Args:
            cancer_id: TCGA abbreviation (e.g. SKCM, LAML).

        Returns:
            List of CellLine entities for this cancer type.
        """
        ...
