# AnalyzeSurvivalUseCase -- runs KM, Cox, ROC for H2 and H3 validation.
# Application layer -- orchestration only, no statistical logic here.

from __future__ import annotations

import time

from revgate.application.dto.survival_dto import (
    SurvivalRequestDTO,
    SurvivalResultDTO,
    KMResultDTO,
    CoxResultDTO,
    ROCResultDTO,
)
from revgate.domain.repositories.clinical_repository import ClinicalRepository


class AnalyzeSurvivalUseCase:
    """Runs all survival analyses required for H2 and H3 validation.

    Analyses per cancer type:
        H2: KM survival by NV-Class, Cox PH multivariate, C-index comparison
        H3: ROC for metastasis prediction (MP-Class vs TNM), NRI/IDI

    Statistical computation delegated to infrastructure layer (lifelines, scipy).
    """

    def __init__(
        self,
        clinical_repo: ClinicalRepository,
    ) -> None:
        self._clinical_repo = clinical_repo

    def execute(
        self,
        request: SurvivalRequestDTO,
        classification_results: dict[str, str],
    ) -> SurvivalResultDTO:
        """Run survival analysis for all requested cancer types.

        Args:
            request:                SurvivalRequestDTO with cancer_ids and params.
            classification_results: Dict cancer_id -> nv_class string.
                                    Used to assign NV-Class to each patient.

        Returns:
            SurvivalResultDTO with KM, Cox, ROC results and any errors.
        """
        started_at = time.monotonic()
        km_results: list[KMResultDTO] = []
        cox_results: list[CoxResultDTO] = []
        roc_results: list[ROCResultDTO] = []
        errors: dict[str, str] = {}

        for cancer_id in request.cancer_ids:
            try:
                patients = self._clinical_repo.get_patients(cancer_id)
                nv_class = classification_results.get(cancer_id)

                if nv_class is None:
                    errors[cancer_id] = "No NV-Class classification available"
                    continue

                if len(patients) < request.min_patients:
                    errors[cancer_id] = (
                        f"Insufficient patients: {len(patients)} < {request.min_patients}"
                    )
                    continue

                km = self._run_km(cancer_id, patients, nv_class)
                cox = self._run_cox(cancer_id, patients, nv_class)
                roc = self._run_roc(cancer_id, patients)

                km_results.append(km)
                cox_results.append(cox)
                roc_results.append(roc)

            except Exception as exc:
                errors[cancer_id] = str(exc)

        return SurvivalResultDTO(
            km_results=km_results,
            cox_results=cox_results,
            roc_results=roc_results,
            errors=errors,
            duration_sec=time.monotonic() - started_at,
        )

    def _run_km(self, cancer_id: str, patients, nv_class: str) -> KMResultDTO:
        """Delegate KM computation to infrastructure. Stub for interface clarity."""
        raise NotImplementedError("KM analysis must be implemented in infrastructure layer")

    def _run_cox(self, cancer_id: str, patients, nv_class: str) -> CoxResultDTO:
        """Delegate Cox PH computation to infrastructure. Stub for interface clarity."""
        raise NotImplementedError("Cox PH must be implemented in infrastructure layer")

    def _run_roc(self, cancer_id: str, patients) -> ROCResultDTO:
        """Delegate ROC computation to infrastructure. Stub for interface clarity."""
        raise NotImplementedError("ROC analysis must be implemented in infrastructure layer")
