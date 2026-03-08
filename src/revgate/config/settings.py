# Settings -- Pydantic-based configuration loader.
# Reads from config/settings.yaml with environment variable overrides.

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


# Default config directory relative to project root
CONFIG_DIR = Path(__file__).parent.parent.parent.parent.parent / "config"


class PipelineSettings(BaseSettings):
    """Pipeline execution parameters.

    Matches structure in config/settings.yaml under 'pipeline:' key.
    """

    cancer_types: list[str] = Field(
        default=["LAML", "SKCM", "BRCA", "KIRC", "PAAD", "LUAD"],
        description="TCGA cancer type codes to process",
    )
    top_n_genes: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of top dependency genes for NV-Score",
    )
    depmap_threshold: float = Field(
        default=-0.5,
        description="Chronos score threshold for essentiality",
    )


class NetworkSettings(BaseSettings):
    """STRING network parameters."""

    string_min_score: int = Field(
        default=700,
        ge=400,
        le=1000,
        description="Minimum STRING combined score for PPI edges",
    )
    pagerank_damping: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="PageRank damping factor for cascade analysis",
    )


class ClassificationSettings(BaseSettings):
    """NV-Score classification thresholds."""

    nv_threshold_a: float = Field(
        default=0.15,
        description="NV-Score above this -> NV-A",
    )
    nv_threshold_c: float = Field(
        default=0.05,
        description="NV-Score below this -> NV-C",
    )
    fdr_threshold: float = Field(
        default=0.05,
        description="ssGSEA FDR significance threshold for RDP-Class",
    )


class SurvivalSettings(BaseSettings):
    """Survival analysis parameters."""

    min_patients: int = Field(
        default=20,
        ge=10,
        description="Minimum patients per NV-Class group for KM",
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Statistical significance level",
    )


class SensitivitySettings(BaseSettings):
    """Sensitivity analysis parameter ranges."""

    top_n_range: list[int] = Field(
        default=[10, 15, 20, 30, 50],
        description="top_n values to sweep",
    )
    string_thresholds: list[int] = Field(
        default=[500, 700, 900],
        description="STRING score thresholds to sweep",
    )
    damping_range: list[float] = Field(
        default=[0.70, 0.80, 0.85, 0.90],
        description="PageRank damping values to sweep",
    )


class RevGateSettings(BaseSettings):
    """Root settings object for the RevGate pipeline.

    Loaded from config/settings.yaml.
    Individual sections can be overridden via environment variables
    prefixed with REVGATE_ (e.g. REVGATE_PIPELINE__TOP_N_GENES=30).
    """

    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    survival: SurvivalSettings = Field(default_factory=SurvivalSettings)
    sensitivity: SensitivitySettings = Field(default_factory=SensitivitySettings)

    cache_dir: Path = Field(
        default=Path.home() / ".revgate" / "cache",
        description="Local cache directory for downloaded data",
    )
    output_dir: Path = Field(
        default=Path("results"),
        description="Output directory for pipeline results",
    )

    model_config = {"env_prefix": "REVGATE_", "env_nested_delimiter": "__"}

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "RevGateSettings":
        """Load settings from YAML file with environment variable overrides.

        Args:
            path: Path to settings.yaml. Defaults to config/settings.yaml.

        Returns:
            RevGateSettings instance.
        """
        yaml_path = path or (CONFIG_DIR / "settings.yaml")

        if not yaml_path.exists():
            return cls()

        with yaml_path.open() as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)
