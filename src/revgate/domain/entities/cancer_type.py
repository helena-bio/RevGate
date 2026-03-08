# CancerType entity -- aggregate root for a single TCGA cancer type.
# Holds cell lines, patients, and the final classification.

from __future__ import annotations

from dataclasses import dataclass, field

from revgate.domain.entities.cell_line import CellLine
from revgate.domain.entities.patient import Patient
from revgate.domain.entities.tumor_classification import TumorClassification


@dataclass
class CancerType:
    """Aggregate root representing a cancer type.

    Attributes:
        cancer_id      -- TCGA abbreviation (e.g. SKCM, LAML)
        name           -- full name (e.g. Skin Cutaneous Melanoma)
        cell_lines     -- DepMap cell lines for this cancer type
        patients       -- TCGA patients with clinical outcomes
        classification -- 4-axis DNV-TC classification (set after pipeline)
    """

    cancer_id: str
    name: str
    cell_lines: list[CellLine] = field(default_factory=list)
    patients: list[Patient] = field(default_factory=list)
    classification: TumorClassification | None = None

    def __post_init__(self) -> None:
        if not self.cancer_id:
            raise ValueError("CancerType.cancer_id must not be empty")
        if not self.name:
            raise ValueError("CancerType.name must not be empty")

    @property
    def cell_line_count(self) -> int:
        return len(self.cell_lines)

    @property
    def patient_count(self) -> int:
        return len(self.patients)

    @property
    def is_classified(self) -> bool:
        return self.classification is not None

    def add_cell_line(self, cell_line: CellLine) -> None:
        if cell_line.cancer_id != self.cancer_id:
            raise ValueError(
                f"CellLine cancer_id={cell_line.cancer_id!r} "
                f"does not match CancerType cancer_id={self.cancer_id!r}"
            )
        self.cell_lines.append(cell_line)

    def add_patient(self, patient: Patient) -> None:
        if patient.cancer_id != self.cancer_id:
            raise ValueError(
                f"Patient cancer_id={patient.cancer_id!r} "
                f"does not match CancerType cancer_id={self.cancer_id!r}"
            )
        self.patients.append(patient)

    def set_classification(self, classification: TumorClassification) -> None:
        if classification.cancer_id != self.cancer_id:
            raise ValueError(
                f"Classification cancer_id={classification.cancer_id!r} "
                f"does not match CancerType cancer_id={self.cancer_id!r}"
            )
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"CancerType(id={self.cancer_id!r}, name={self.name!r}, "
            f"cell_lines={self.cell_line_count}, patients={self.patient_count}, "
            f"classified={self.is_classified})"
        )

    def __hash__(self) -> int:
        return hash(self.cancer_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CancerType):
            return NotImplemented
        return self.cancer_id == other.cancer_id
