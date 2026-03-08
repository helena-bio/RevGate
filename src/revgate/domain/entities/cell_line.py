# CellLine entity -- represents a single DepMap cancer cell line.

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CellLine:
    """A DepMap cell line entry.

    Attributes:
        model_id    -- DepMap model ID (e.g. ACH-000001)
        cell_name   -- human-readable cell line name (e.g. JURKAT)
        cancer_id   -- TCGA cancer type code (e.g. LAML)
        lineage     -- DepMap lineage annotation (e.g. Hematopoietic)
        is_mutant   -- whether a known driver mutation is annotated
        driver_gene -- symbol of the annotated driver gene (optional)
    """

    model_id: str
    cell_name: str
    cancer_id: str
    lineage: str
    is_mutant: bool = False
    driver_gene: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("CellLine.model_id must not be empty")
        if not self.cancer_id:
            raise ValueError("CellLine.cancer_id must not be empty")

    def __repr__(self) -> str:
        return (
            f"CellLine(model_id={self.model_id!r}, "
            f"name={self.cell_name!r}, cancer={self.cancer_id!r})"
        )

    def __hash__(self) -> int:
        return hash(self.model_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CellLine):
            return NotImplemented
        return self.model_id == other.model_id
