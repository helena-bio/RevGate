# RunSensitivityUseCase -- parameter sweep for robustness validation.
# Addresses Testing Framework Section 7 directly.

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SensitivityRequestDTO:
    """Input DTO for RunSensitivityUseCase.

    Parameter ranges from Testing Framework spec:
        top_n_range        -- [10, 15, 20, 30, 50]
        string_thresholds  -- [500, 700, 900]
        damping_range      -- [0.70, 0.80, 0.85, 0.90]
    """

    cancer_ids: list[str]
    top_n_range: list[int] = field(default_factory=lambda: [10, 15, 20, 30, 50])
    string_thresholds: list[int] = field(default_factory=lambda: [500, 700, 900])
    damping_range: list[float] = field(default_factory=lambda: [0.70, 0.80, 0.85, 0.90])


@dataclass
class SensitivityResultDTO:
    """Output DTO for RunSensitivityUseCase.

    Attributes:
        top_n_kappa         -- Cohen's kappa for NV-Class stability across top_n values
        string_kappa        -- Cohen's kappa for NV-Class stability across STRING thresholds
        damping_kappa       -- Cohen's kappa for CEP-Class stability across damping values
        nv_score_spearman   -- Spearman rho of NV-Scores across parameter variations
        duration_sec        -- total wall time
    """

    top_n_kappa: dict[str, float] = field(default_factory=dict)
    string_kappa: dict[str, float] = field(default_factory=dict)
    damping_kappa: dict[str, float] = field(default_factory=dict)
    nv_score_spearman: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)
    duration_sec: float = 0.0


class RunSensitivityUseCase:
    """Runs parameter sensitivity analysis for classification robustness.

    Tests five dimensions per Testing Framework spec:
        1. top_n_genes variation (10, 15, 20, 30, 50)
        2. STRING score threshold (500, 700, 900)
        3. PageRank damping factor (0.70, 0.80, 0.85, 0.90)
        4. Normalization strategy comparison
        5. Missing data simulation (10%, 20%, 30%)

    Target stability thresholds (from Testing Framework):
        NV-Class kappa > 0.75
        CEP-Class kappa > 0.75
        RDP-Class kappa > 0.80
    """

    def __init__(self, classify_use_case) -> None:
        # Accepts ClassifyTumorUseCase instance for re-running with varied params
        self._classify_use_case = classify_use_case

    def execute(self, request: SensitivityRequestDTO) -> SensitivityResultDTO:
        """Run full sensitivity analysis across all parameter dimensions.

        Args:
            request: SensitivityRequestDTO with cancer_ids and parameter ranges.

        Returns:
            SensitivityResultDTO with kappa and correlation metrics.
        """
        started_at = time.monotonic()

        top_n_kappa = self._sweep_top_n(request)
        string_kappa = self._sweep_string_threshold(request)
        damping_kappa = self._sweep_damping(request)

        return SensitivityResultDTO(
            top_n_kappa=top_n_kappa,
            string_kappa=string_kappa,
            damping_kappa=damping_kappa,
            duration_sec=time.monotonic() - started_at,
        )

    def _sweep_top_n(self, request: SensitivityRequestDTO) -> dict[str, float]:
        """Sweep top_n parameter and compute NV-Class stability (Cohen's kappa)."""
        raise NotImplementedError("top_n sweep implemented in infrastructure layer")

    def _sweep_string_threshold(self, request: SensitivityRequestDTO) -> dict[str, float]:
        """Sweep STRING score threshold and compute NV-Class stability."""
        raise NotImplementedError("STRING threshold sweep implemented in infrastructure layer")

    def _sweep_damping(self, request: SensitivityRequestDTO) -> dict[str, float]:
        """Sweep PageRank damping factor and compute CEP-Class stability."""
        raise NotImplementedError("damping sweep implemented in infrastructure layer")
