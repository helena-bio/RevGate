# SensitivityPlots -- parameter sweep visualizations.
# Maps to Supplementary Figures S8-S9.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class SensitivityPlots:
    """Generates sensitivity analysis visualizations.

    Shows classification stability (Cohen's kappa) across
    parameter variations for top_n, STRING threshold, and
    PageRank damping factor.
    """

    def __init__(self, figsize: tuple[int, int] = (12, 4)) -> None:
        self._figsize = figsize

    def plot_top_n_sweep(
        self,
        top_n_values: list[int],
        kappa_scores: list[float],
        output_path: Path,
    ) -> Path:
        """Plot NV-Class stability across top_n parameter values.

        Args:
            top_n_values:  List of top_n values swept.
            kappa_scores:  Cohen's kappa for each top_n value.
            output_path:   Directory to save the figure.

        Returns:
            Path to saved figure file.
        """
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            top_n_values,
            kappa_scores,
            marker="o",
            color="#1f77b4",
            linewidth=2.0,
            markersize=7,
        )

        # Target threshold line from Testing Framework spec
        ax.axhline(y=0.75, color="red", linestyle="--", linewidth=1.0, label="Target kappa = 0.75")

        ax.set_title("NV-Class Stability: top_n Parameter Sweep", fontsize=12)
        ax.set_xlabel("top_n (number of dependency genes)", fontsize=10)
        ax.set_ylabel("Cohen's kappa", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_path.mkdir(parents=True, exist_ok=True)
        fig_path = output_path / "figureS8_sensitivity_top_n.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig_path

    def plot_string_threshold_sweep(
        self,
        thresholds: list[int],
        kappa_scores: list[float],
        output_path: Path,
    ) -> Path:
        """Plot NV-Class stability across STRING score thresholds."""
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            thresholds,
            kappa_scores,
            marker="s",
            color="#ff7f0e",
            linewidth=2.0,
            markersize=7,
        )

        ax.axhline(y=0.75, color="red", linestyle="--", linewidth=1.0, label="Target kappa = 0.75")

        ax.set_title("NV-Class Stability: STRING Score Threshold Sweep", fontsize=12)
        ax.set_xlabel("STRING Combined Score Threshold", fontsize=10)
        ax.set_ylabel("Cohen's kappa", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_path.mkdir(parents=True, exist_ok=True)
        fig_path = output_path / "figureS9_sensitivity_string_threshold.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig_path
