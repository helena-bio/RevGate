# KaplanMeier -- KM survival curves by NV-Class.
# Maps to manuscript Figure 3 (OS by NV-Class).

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


NV_COLORS = {
    "NV-A": "#d62728",
    "NV-B": "#ff7f0e",
    "NV-C": "#1f77b4",
}


class KaplanMeierFigure:
    """Generates Kaplan-Meier survival curves stratified by NV-Class.

    Requires lifelines KaplanMeierFitter results passed as pre-computed
    data structures to keep this class free of statistical logic.
    """

    def __init__(self, figsize: tuple[int, int] = (10, 6)) -> None:
        self._figsize = figsize

    def plot(
        self,
        km_data: dict[str, dict],
        cancer_id: str,
        logrank_p: float,
        output_path: Path,
    ) -> Path:
        """Generate and save KM curves for a single cancer type.

        Args:
            km_data:     Dict NV-Class -> {'timeline': list, 'survival': list,
                         'ci_lower': list, 'ci_upper': list, 'n': int,
                         'median_os': float}.
            cancer_id:   TCGA cancer code for title.
            logrank_p:   Log-rank test p-value for annotation.
            output_path: Directory to save the figure.

        Returns:
            Path to saved figure file.
        """
        fig, ax = plt.subplots(figsize=self._figsize)

        for nv_class, data in km_data.items():
            color = NV_COLORS.get(nv_class, "#7f7f7f")
            n = data.get("n", 0)
            median = data.get("median_os", float("nan"))

            ax.step(
                data["timeline"],
                data["survival"],
                where="post",
                color=color,
                linewidth=2.0,
                label=f"{nv_class} (n={n}, median={median:.1f}mo)",
            )

            # Confidence interval shading
            if "ci_lower" in data and "ci_upper" in data:
                ax.fill_between(
                    data["timeline"],
                    data["ci_lower"],
                    data["ci_upper"],
                    alpha=0.15,
                    color=color,
                    step="post",
                )

        # P-value annotation
        p_text = f"Log-rank p = {logrank_p:.4f}" if logrank_p >= 0.0001 else "Log-rank p < 0.0001"
        ax.text(
            0.98, 0.98, p_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
        )

        ax.set_title(
            f"Overall Survival by NV-Class — {cancer_id}",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        output_path.mkdir(parents=True, exist_ok=True)
        fig_path = output_path / f"figure3_km_{cancer_id.lower()}.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig_path
