# TCGARepository -- concrete implementation of ClinicalRepository port.
# Loads TCGA expression and clinical data from local cache.

from __future__ import annotations

import pandas as pd

from revgate.domain.entities.patient import Patient
from revgate.domain.exceptions.data import DataNotAvailableError
from revgate.infrastructure.external.file_cache import FileCache


def _expression_cache_key(cancer_id: str) -> str:
    return f"tcga/{cancer_id}_expression.parquet"


def _clinical_cache_key(cancer_id: str) -> str:
    return f"tcga/{cancer_id}_clinical.parquet"


class TCGARepository:
    """Concrete implementation of ClinicalRepository.

    Reads pre-cached TCGA expression (parquet) and clinical (parquet) files.
    Downloads are handled by the revgate download CLI command.
    """

    def __init__(self, cache: FileCache) -> None:
        self._cache = cache
        self._expression_cache: dict[str, pd.DataFrame] = {}
        self._clinical_cache: dict[str, pd.DataFrame] = {}

    def get_patients(self, cancer_id: str) -> list[Patient]:
        """Return TCGA patients with clinical outcome data.

        Args:
            cancer_id: TCGA abbreviation (e.g. BRCA).

        Returns:
            List of Patient entities.
        """
        df = self._load_clinical(cancer_id)
        patients: list[Patient] = []

        for _, row in df.iterrows():
            try:
                patient = Patient(
                    patient_id=str(row.get("case_id", row.name)),
                    cancer_id=cancer_id,
                    overall_survival_months=float(row.get("days_to_death", 0) or 0) / 30.44,
                    is_deceased=str(row.get("vital_status", "")).lower() == "dead",
                    tnm_stage=str(row.get("ajcc_pathologic_stage", "")) or None,
                    treatment_type=str(row.get("treatment_type", "")) or None,
                    has_metastasis=self._parse_metastasis(row),
                )
                patients.append(patient)
            except (ValueError, TypeError):
                # Skip malformed rows
                continue

        return patients

    def get_expression_matrix(self, cancer_id: str) -> pd.DataFrame:
        """Return RNA-seq expression matrix for a cancer type.

        Args:
            cancer_id: TCGA abbreviation.

        Returns:
            DataFrame (n_patients x n_genes), log2-normalized.
        """
        return self._load_expression(cancer_id)

    def _load_clinical(self, cancer_id: str) -> pd.DataFrame:
        if cancer_id in self._clinical_cache:
            return self._clinical_cache[cancer_id]

        key = _clinical_cache_key(cancer_id)
        path = self._cache.get(key)

        if path is None:
            raise DataNotAvailableError(
                f"TCGA clinical data not cached for {cancer_id}. "
                f"Run: revgate download --source tcga --cancer {cancer_id}"
            )

        df = pd.read_parquet(path)
        self._clinical_cache[cancer_id] = df
        return df

    def _load_expression(self, cancer_id: str) -> pd.DataFrame:
        if cancer_id in self._expression_cache:
            return self._expression_cache[cancer_id]

        key = _expression_cache_key(cancer_id)
        path = self._cache.get(key)

        if path is None:
            raise DataNotAvailableError(
                f"TCGA expression data not cached for {cancer_id}. "
                f"Run: revgate download --source tcga --cancer {cancer_id}"
            )

        df = pd.read_parquet(path)
        self._expression_cache[cancer_id] = df
        return df

    def _parse_metastasis(self, row) -> bool | None:
        """Parse metastasis status from TCGA clinical row."""
        m_stage = str(row.get("ajcc_pathologic_m", "")).upper()
        if m_stage.startswith("M1"):
            return True
        if m_stage.startswith("M0"):
            return False
        return None
