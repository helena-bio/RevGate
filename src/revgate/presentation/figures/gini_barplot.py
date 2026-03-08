# GiniBarplot -- Gini coefficient comparison per cancer type.
# Maps to manuscript Figure 2 (empirical NV rationale).

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# NV-Class color palette
NV_COLORS = {
    "NV-A": "#d62728",  # Red -- concentrated, high vulnerability
    "NV-B": "#ff7f0e",  # Orange -- moderate
    "NV-C": "#1f77b4",  # Blue -- distributed, low vulnerability
}


class GiniBarplot:
    """Generates Gini coefficient bar plot for manuscript Figure 2.

    Visualizes dependency concentration per cancer type,
    color-coded by NV-Class assignment.
    """

    def __init__(self, figsize: tuple[int, int] = (10, 5)) -> None:
        self._figsize = figsize

    def plot(
        self,
        gini_scores: dict[str, float],
        nv_classes: dict[str, str],
        output_path: Path,
    ) -> Path:
        """Generate and save the Gini coefficient bar plot.

        Args:
            gini_scores:  Dict cancer_id -> Gini coefficient value.
            nv_classes:   Dict cancer_id -> NV-Class string.
            output_path:  Directory to save the figure.

        Returns:
            Path to saved figure file.
        """
        cancer_ids = sorted(gini_scores.keys(), key=lambda c: gini_scores[c], reverse=True)
        values = [gini_scores[c] for c in cancer_ids]
        colors = [NV_COLORS.get(nv_classes.get(c, "NV-B"), "#aec7e8") for c in cancer_ids]

        fig, ax = plt.subplots(figsize=self._figsize)

        bars = ax.bar(cancer_ids, values, color=colors, edgecolor="white", linewidth=0.8)

        # NV threshold lines from HLD spec
        ax.axhline(y=0.7, color="black", linestyle="--", linewidth=1.0, label="NV-A threshold (0.70)")
        ax.axhline(y=0.4, color="grey", linestyle="--", linewidth=1.0, label="NV-C threshold (0.40)")

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Legend for NV-Class colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=NV_COLORS["NV-A"], label="NV-A (concentrated)"),
            Patch(facecolor=NV_COLORS["NV-B"], label="NV-B (moderate)"),
            Patch(facecolor=NV_COLORS["NV-C"], label="NV-C (distributed)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        ax.set_title(
            "Dependency Concentration (Gini Coefficient) by Cancer Type",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Cancer Type", fontsize=10)
        ax.set_ylabel("Gini Coefficient", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()

        output_path.mkdir(parents=True, exist_ok=True)
        fig_path = output_path / "figure2_gini_barplot.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig_path
