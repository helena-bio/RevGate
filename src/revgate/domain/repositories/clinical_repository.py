# Port interface for TCGA clinical and expression data access.
# Infrastructure layer implements this protocol.

from typing import Protocol, runtime_checkable

import pandas as pd

from revgate.domain.entities.patient import Patient


@runtime_checkable
class ClinicalRepository(Protocol):
    """Abstract interface for TCGA expression and clinical data.

    The concrete implementation downloads and caches per-cancer
    HTSeq count matrices and clinical TSV files from NCI GDC.

    Methods:
        get_patients          -- TCGA patients with clinical outcomes
        get_expression_matrix -- RNA-seq expression matrix for a cancer type
    """

    def get_patients(self, cancer_id: str) -> list[Patient]:
        """Return TCGA patients with clinical outcome data.

        Args:
            cancer_id: TCGA abbreviation (e.g. BRCA, LUAD).

        Returns:
            List of Patient entities with OS, metastasis, TNM stage.
        """
        ...

    def get_expression_matrix(self, cancer_id: str) -> pd.DataFrame:
        """Return RNA-seq expression matrix for a cancer type.

        Args:
            cancer_id: TCGA abbreviation.

        Returns:
            DataFrame with shape (n_patients, n_genes).
            Index: TCGA patient barcodes.
            Columns: gene symbols.
            Values: normalized expression (log2 TPM or VST counts).
        """
        ...
