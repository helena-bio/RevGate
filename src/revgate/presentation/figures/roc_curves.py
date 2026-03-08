# ROCCurves -- ROC comparison MP-Class vs TNM for metastasis prediction.
# Maps to manuscript Figure 5 / Supplementary Figure S7.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ROCCurvesFigure:
    """Generates ROC curves comparing MP-Class vs TNM for H3 validation."""

    def __init__(self, figsize: tuple[int, int] = (7, 7)) -> None:
        self._figsize = figsize

    def plot(
        self,
        roc_data: dict[str, dict],
        cancer_id: str,
        delong_p: float,
        output_path: Path,
    ) -> Path:
        """Generate and save ROC curve comparison.

        Args:
            roc_data:    Dict model_name -> {'fpr': list, 'tpr': list, 'auc': float}.
                         Expected keys: 'TNM only', 'MP-Class only', 'TNM + MP-Class'.
            cancer_id:   TCGA cancer code for title.
            delong_p:    DeLong test p-value for AUC comparison annotation.
            output_path: Directory to save the figure.

        Returns:
            Path to saved figure file.
        """
        colors = {
            "TNM only": "#1f77b4",
            "MP-Class only": "#ff7f0e",
            "TNM + MP-Class": "#d62728",
        }

        fig, ax = plt.subplots(figsize=self._figsize)

        for model_name, data in roc_data.items():
            color = colors.get(model_name, "#7f7f7f")
            auc = data.get("auc", 0.0)
            ax.plot(
                data["fpr"],
                data["tpr"],
                color=color,
                linewidth=2.0,
                label=f"{model_name} (AUC = {auc:.3f})",
            )

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Random")

        # DeLong p-value annotation
        p_text = f"DeLong p = {delong_p:.4f}" if delong_p >= 0.0001 else "DeLong p < 0.0001"
        ax.text(
            0.98, 0.05, p_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
        )

        ax.set_title(
            f"ROC: Metastasis Prediction — {cancer_id}",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_path.mkdir(parents=True, exist_ok=True)
        fig_path = output_path / f"figure5_roc_{cancer_id.lower()}.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig_path
