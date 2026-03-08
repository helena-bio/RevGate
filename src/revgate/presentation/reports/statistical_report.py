# StatisticalReport -- JSON + Markdown output for hypothesis test results.

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from revgate.domain.services.hypothesis_tester import ValidationResult


class StatisticalReport:
    """Generates machine-readable JSON and human-readable Markdown reports.

    Both formats are deterministic given the same ValidationResult --
    required for reproducibility (every run produces identical output).
    """

    def write_json(
        self,
        validation_result: ValidationResult,
        output_path: Path,
    ) -> Path:
        """Write hypothesis test results as structured JSON.

        Args:
            validation_result: H1/H2/H3 results from ValidateHypothesisUseCase.
            output_path:       Directory to save the report.

        Returns:
            Path to saved JSON file.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "statistical_report.json"

        report = {
            "H1": {
                "adjusted_rand_index": validation_result.h1.adjusted_rand_index,
                "permutation_p_value": validation_result.h1.permutation_p_value,
                "permanova_r2": validation_result.h1.permanova_r2,
                "permanova_p_value": validation_result.h1.permanova_p_value,
                "result": validation_result.h1.result.value,
            },
            "H2": {
                "logrank_p_value": validation_result.h2.logrank_p_value,
                "hazard_ratio_nvc_vs_nva": validation_result.h2.hazard_ratio_nvc_vs_nva,
                "hazard_ratio_ci_lower": validation_result.h2.hazard_ratio_ci_lower,
                "hazard_ratio_ci_upper": validation_result.h2.hazard_ratio_ci_upper,
                "c_index_base": validation_result.h2.c_index_base,
                "c_index_with_nv": validation_result.h2.c_index_with_nv,
                "delta_c_index": validation_result.h2.delta_c_index,
                "result": validation_result.h2.result.value,
            },
            "H3": {
                "auc_tnm_only": validation_result.h3.auc_tnm_only,
                "auc_tnm_plus_mp": validation_result.h3.auc_tnm_plus_mp,
                "delta_auc": validation_result.h3.delta_auc,
                "delong_p_value": validation_result.h3.delong_p_value,
                "nri": validation_result.h3.nri,
                "nri_p_value": validation_result.h3.nri_p_value,
                "result": validation_result.h3.result.value,
            },
        }

        json_path.write_text(json.dumps(report, indent=2))
        return json_path

    def write_markdown(
        self,
        validation_result: ValidationResult,
        output_path: Path,
    ) -> Path:
        """Write hypothesis test results as human-readable Markdown.

        Args:
            validation_result: H1/H2/H3 results.
            output_path:       Directory to save the report.

        Returns:
            Path to saved Markdown file.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        md_path = output_path / "statistical_report.md"

        content = validation_result.overall_summary()
        md_path.write_text(content)

        return md_path
