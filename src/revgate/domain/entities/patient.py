# Patient entity -- TCGA patient with clinical outcome data.

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Patient:
    """A TCGA patient with clinical outcome data for survival analysis.

    Attributes:
        patient_id             -- TCGA barcode (e.g. TCGA-ER-A19T)
        cancer_id              -- TCGA cancer type code (e.g. SKCM)
        overall_survival_months -- OS time in months
        is_deceased            -- vital status event indicator
        pfs_months             -- progression-free survival (optional)
        tnm_stage              -- TNM staging I/II/III/IV (optional)
        treatment_type         -- Targeted / Chemo / Combination / Hormonal (optional)
        has_metastasis         -- 5-year metastatic event (optional)
        molecular_subtype      -- cancer-specific subtype (optional)
    """

    patient_id: str
    cancer_id: str
    overall_survival_months: float
    is_deceased: bool
    pfs_months: float | None = None
    tnm_stage: str | None = None
    treatment_type: str | None = None
    has_metastasis: bool | None = None
    molecular_subtype: str | None = None

    def __post_init__(self) -> None:
        if not self.patient_id:
            raise ValueError("Patient.patient_id must not be empty")
        if not self.cancer_id:
            raise ValueError("Patient.cancer_id must not be empty")
        if self.overall_survival_months < 0.0:
            raise ValueError(
                f"Patient.overall_survival_months must be >= 0, "
                f"got {self.overall_survival_months}"
            )

    def __repr__(self) -> str:
        return (
            f"Patient(id={self.patient_id!r}, cancer={self.cancer_id!r}, "
            f"os={self.overall_survival_months:.1f}mo, deceased={self.is_deceased})"
        )

    def __hash__(self) -> int:
        return hash(self.patient_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Patient):
            return NotImplemented
        return self.patient_id == other.patient_id
