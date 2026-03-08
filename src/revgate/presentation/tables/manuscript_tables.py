# ManuscriptTables -- generates manuscript-ready tables from pipeline results.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from revgate.application.dto.classification_dto import ClassificationResultDTO
from revgate.application.dto.survival_dto import SurvivalResultDTO
from revgate.domain.services.hypothesis_tester import ValidationResult


class ManuscriptTables:
    """Generates all manuscript tables from pipeline results.

    Table mapping from HLD Section 6.2:
        Table 1 -- DNV-TC signatures per cancer type
        Table 2 -- Multivariate Cox regression results
        Table 3 -- ROC comparison: MP-Class vs TNM
        Table 4 -- Sensitivity analysis kappa matrix
    """

    def generate_table1(
        self,
        classification_result: ClassificationResultDTO,
        output_path: Path,
    ) -> Path:
        """Table 1: DNV-TC classification per cancer type.

        Args:
            classification_result: Output from ClassifyTumorUseCase.
            output_path:           Directory to save the table.

        Returns:
            Path to saved CSV file.
        """
        rows = []
        for axis in classification_result.axes:
            rows.append({
                "Cancer Type": axis.cancer_id,
                "NV-Class": axis.nv_class,
                "RDP-Class": axis.rdp_class,
                "CEP-Class": axis.cep_class,
                "MP-Class": axis.mp_class,
                "NV-Score": round(axis.nv_composite_score, 4),
                "Gini": round(axis.gini, 4),
                "Mean pLI": round(axis.mean_pli, 4),
                "Mean Centrality": round(axis.mean_centrality, 4),
            })

        df = pd.DataFrame(rows)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "table1_dnvtc_classification.csv"
        df.to_csv(csv_path, index=False)

        return csv_path

    def generate_table2(
        self,
        survival_result: SurvivalResultDTO,
        output_path: Path,
    ) -> Path:
        """Table 2: Multivariate Cox regression results.

        Args:
            survival_result: Output from AnalyzeSurvivalUseCase.
            output_path:     Directory to save the table.

        Returns:
            Path to saved CSV file.
        """
        rows = []
        for cox in survival_result.cox_results:
            rows.append({
                "Cancer Type": cox.cancer_id,
                "HR (NV-C vs NV-A)": round(cox.hazard_ratio_nvc_vs_nva, 3),
                "95% CI Lower": round(cox.ci_lower, 3),
                "95% CI Upper": round(cox.ci_upper, 3),
                "p-value": round(cox.p_value, 4),
                "C-index (base)": round(cox.c_index_base, 3),
                "C-index (+NV)": round(cox.c_index_with_nv, 3),
                "Delta C-index": round(cox.c_index_with_nv - cox.c_index_base, 4),
                "LRT p-value": round(cox.lrt_p_value, 4),
            })

        df = pd.DataFrame(rows)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "table2_cox_regression.csv"
        df.to_csv(csv_path, index=False)

        return csv_path

    def generate_table3(
        self,
        survival_result: SurvivalResultDTO,
        output_path: Path,
    ) -> Path:
        """Table 3: ROC comparison MP-Class vs TNM.

        Args:
            survival_result: Output from AnalyzeSurvivalUseCase.
            output_path:     Directory to save the table.

        Returns:
            Path to saved CSV file.
        """
        rows = []
        for roc in survival_result.roc_results:
            rows.append({
                "Cancer Type": roc.cancer_id,
                "AUC (TNM only)": round(roc.auc_tnm_only, 3),
                "AUC (TNM + MP)": round(roc.auc_tnm_plus_mp, 3),
                "Delta AUC": round(roc.auc_tnm_plus_mp - roc.auc_tnm_only, 4),
                "DeLong p": round(roc.delong_p_value, 4),
                "NRI": round(roc.nri, 4),
                "NRI p": round(roc.nri_p_value, 4),
            })

        df = pd.DataFrame(rows)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "table3_roc_comparison.csv"
        df.to_csv(csv_path, index=False)

        return csv_path

    def generate_summary_report(
        self,
        validation_result: ValidationResult,
        output_path: Path,
    ) -> Path:
        """Generate markdown summary of H1/H2/H3 results.

        Args:
            validation_result: Output from ValidateHypothesisUseCase.
            output_path:       Directory to save the report.

        Returns:
            Path to saved markdown file.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        md_path = output_path / "hypothesis_validation_summary.md"

        lines = [
            "# RevGate: DNV-TC Hypothesis Validation Results",
            "",
            "## Summary",
            "",
            "```",
            validation_result.overall_summary(),
            "```",
            "",
            "## H1: Developmental Clustering",
            "",
            f"- Adjusted Rand Index: {validation_result.h1.adjusted_rand_index:.4f}",
            f"- Permutation p-value: {validation_result.h1.permutation_p_value:.4f}",
            f"- PERMANOVA R2: {validation_result.h1.permanova_r2:.4f}",
            f"- PERMANOVA p-value: {validation_result.h1.permanova_p_value:.4f}",
            f"- Result: **{validation_result.h1.result.value}**",
            "",
            "## H2: NV-Class vs Survival",
            "",
            f"- Log-rank p-value: {validation_result.h2.logrank_p_value:.4f}",
            f"- HR (NV-C vs NV-A): {validation_result.h2.hazard_ratio_nvc_vs_nva:.3f}",
            f"- Delta C-index: {validation_result.h2.delta_c_index:.4f}",
            f"- Result: **{validation_result.h2.result.value}**",
            "",
            "## H3: MP-Class vs Metastasis",
            "",
            f"- AUC (TNM only): {validation_result.h3.auc_tnm_only:.3f}",
            f"- AUC (TNM + MP): {validation_result.h3.auc_tnm_plus_mp:.3f}",
            f"- Delta AUC: {validation_result.h3.delta_auc:.4f}",
            f"- DeLong p-value: {validation_result.h3.delong_p_value:.4f}",
            f"- Result: **{validation_result.h3.result.value}**",
        ]

        md_path.write_text("\n".join(lines))
        return md_path
