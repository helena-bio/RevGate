# HypothesisTester -- defines the three testable claims of the DNV-TC framework.
# Domain service -- holds test specifications and result structures.
# Actual statistical computation delegated to infrastructure (lifelines, scipy).

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HypothesisResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class H1Result:
    """H1: Dependencies cluster by developmental program.

    Success: ARI > 0.2 AND permutation p < 0.05
             PERMANOVA R2 > 0.10 AND p < 0.05
    """

    adjusted_rand_index: float
    permutation_p_value: float
    permanova_r2: float
    permanova_p_value: float

    @property
    def result(self) -> HypothesisResult:
        ari_pass = self.adjusted_rand_index > 0.2 and self.permutation_p_value < 0.05
        permanova_pass = self.permanova_r2 > 0.10 and self.permanova_p_value < 0.05
        if ari_pass and permanova_pass:
            return HypothesisResult.PASS
        if ari_pass or permanova_pass:
            return HypothesisResult.INCONCLUSIVE
        return HypothesisResult.FAIL

    def summary(self) -> str:
        return (
            f"H1 [{self.result.value}]: "
            f"ARI={self.adjusted_rand_index:.3f} (p={self.permutation_p_value:.4f}), "
            f"PERMANOVA R2={self.permanova_r2:.3f} (p={self.permanova_p_value:.4f})"
        )


@dataclass
class H2Result:
    """H2: NV-Class predicts targeted therapy response.

    Success: log-rank p < 0.05 AND HR(NV-C vs NV-A) > 1.0
             delta C-index > 0
    """

    logrank_p_value: float
    hazard_ratio_nvc_vs_nva: float
    hazard_ratio_ci_lower: float
    hazard_ratio_ci_upper: float
    c_index_base: float
    c_index_with_nv: float

    @property
    def delta_c_index(self) -> float:
        return self.c_index_with_nv - self.c_index_base

    @property
    def result(self) -> HypothesisResult:
        survival_pass = (
            self.logrank_p_value < 0.05
            and self.hazard_ratio_nvc_vs_nva > 1.0
        )
        discrimination_pass = self.delta_c_index > 0.0
        if survival_pass and discrimination_pass:
            return HypothesisResult.PASS
        if survival_pass or discrimination_pass:
            return HypothesisResult.INCONCLUSIVE
        return HypothesisResult.FAIL

    def summary(self) -> str:
        return (
            f"H2 [{self.result.value}]: "
            f"log-rank p={self.logrank_p_value:.4f}, "
            f"HR(NV-C/NV-A)={self.hazard_ratio_nvc_vs_nva:.2f} "
            f"[{self.hazard_ratio_ci_lower:.2f}-{self.hazard_ratio_ci_upper:.2f}], "
            f"delta-C={self.delta_c_index:.4f}"
        )


@dataclass
class H3Result:
    """H3: MP-Class predicts metastasis better than TNM staging.

    Success: AUC(TNM+MP) > AUC(TNM) with DeLong p < 0.05
    """

    auc_tnm_only: float
    auc_tnm_plus_mp: float
    delong_p_value: float
    nri: float
    nri_p_value: float

    @property
    def delta_auc(self) -> float:
        return self.auc_tnm_plus_mp - self.auc_tnm_only

    @property
    def result(self) -> HypothesisResult:
        auc_pass = self.delta_auc > 0.0 and self.delong_p_value < 0.05
        nri_pass = self.nri > 0.0 and self.nri_p_value < 0.05
        if auc_pass and nri_pass:
            return HypothesisResult.PASS
        if auc_pass or nri_pass:
            return HypothesisResult.INCONCLUSIVE
        return HypothesisResult.FAIL

    def summary(self) -> str:
        return (
            f"H3 [{self.result.value}]: "
            f"AUC(TNM)={self.auc_tnm_only:.3f}, "
            f"AUC(TNM+MP)={self.auc_tnm_plus_mp:.3f}, "
            f"delta={self.delta_auc:.3f} (DeLong p={self.delong_p_value:.4f}), "
            f"NRI={self.nri:.3f} (p={self.nri_p_value:.4f})"
        )


@dataclass
class ValidationResult:
    """Complete hypothesis validation result for the full pipeline."""

    h1: H1Result
    h2: H2Result
    h3: H3Result

    def overall_summary(self) -> str:
        lines = [
            "=== RevGate Hypothesis Validation Results ===",
            self.h1.summary(),
            self.h2.summary(),
            self.h3.summary(),
        ]
        results = [self.h1.result, self.h2.result, self.h3.result]
        passed = sum(1 for r in results if r == HypothesisResult.PASS)
        lines.append(f"Overall: {passed}/3 hypotheses passed")
        return "\n".join(lines)
