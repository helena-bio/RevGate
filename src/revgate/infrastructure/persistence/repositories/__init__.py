from revgate.infrastructure.persistence.repositories.depmap_repository import DepMapRepository
from revgate.infrastructure.persistence.repositories.gnomad_repository import GnomADRepository
from revgate.infrastructure.persistence.repositories.string_repository import STRINGRepository
from revgate.infrastructure.persistence.repositories.tcga_repository import TCGARepository

__all__ = [
    "DepMapRepository",
    "TCGARepository",
    "STRINGRepository",
    "GnomADRepository",
]
