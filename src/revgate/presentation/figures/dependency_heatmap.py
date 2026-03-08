# DependencyHeatmap -- top dependency genes x cancer types heatmap.
# Maps to manuscript Figure 1 (T-DRI heatmap replacement).

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DependencyHeatmap:
    """Generates dependency profile heatmap for manuscript Figure 1.

    Shows top-N dependency genes per cancer type as a heatmap
    of mean Chronos scores. Reveals dependency architecture differences
    between NV-A (concentrated) and NV-C (distributed) cancer types.
    """

    def __init__(self, top_n: int = 20, figsize: tuple[int, int] = (14, 8)) -> None:
        self._top_n = top_n
        self._figsize = figsize

    def plot(
        self,
        dependency_matrix: pd.DataFrame,
        cancer_ids: list[str],
        nv_classes: dict[str, str],
        output_path: Path,
    ) -> Path:
        """Generate and save the dependency heatmap.

        Args:
            dependency_matrix: DataFrame (cell_lines x genes) of Chronos scores.
            cancer_ids:        TCGA cancer type codes to include.
            nv_classes:        Dict cancer_id -> NV-Class string for annotations.
            output_path:       Directory to save the figure.

        Returns:
            Path to saved figure file.
        """
        # Compute mean Chronos score per gene per cancer type
        cancer_means: dict[str, pd.Series] = {}
        for cancer_id in cancer_ids:
            cancer_means[cancer_id] = dependency_matrix.mean(axis=0)

        mean_df = pd.DataFrame(cancer_means).T

        # Select top-N most variable genes across cancer types
        gene_variance = mean_df.var(axis=0)
        top_genes = gene_variance.nlargest(self._top_n).index.tolist()
        plot_df = mean_df[top_genes]

        # Build column labels with NV-Class annotation
        row_labels = [
            f"{cancer_id} ({nv_classes.get(cancer_id, '?')})"
            for cancer_id in plot_df.index
        ]
        plot_df.index = row_labels

        fig, ax = plt.subplots(figsize=self._figsize)

        sns.heatmap(
            plot_df,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-2.0,
            vmax=0.5,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Mean Chronos Score", "shrink": 0.6},
        )

        ax.set_title(
            f"Top {self._top_n} Variable Dependency Genes by Cancer Type",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Gene", fontsize=10)
        ax.set_ylabel("Cancer Type (NV-Class)", fontsize=10)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

        plt.tight_layout()

        output_path.mkdir(parents=True, exist_ok=True)
        fig_path = output_path / "figure1_dependency_heatmap.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return fig_path
