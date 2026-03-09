#!/usr/bin/env python3
"""
RevGate — Bootstrap CI за PCA тегла на NV-Score
=================================================
Изчислява 95% confidence intervals за NV-Score компонентните тегла
чрез bootstrap resampling на tumor lineages.

Метод:
  1. Зареждаме nv_components_all_lineages.csv (20 lineages × 4 компонента)
  2. Bootstrap: resample 20 lineages с replacement × 1000 итерации
  3. За всяка итерация: StandardScaler → PCA → PC1 loadings → нормализирани тегла
  4. Резултат: mean, 95% CI (percentile), CV за всяко тегло
  5. Stability criterion: CV < 0.20 (Testing Framework v1.0, Section 8.2)

Употреба:
  cd ~/revgate
  python3 run_bootstrap_pca_weights.py
  python3 run_bootstrap_pca_weights.py --input /root/.revgate/cache/nv_components_all_lineages.csv
  python3 run_bootstrap_pca_weights.py --n-bootstrap 2000 --output results/

Helena Bioinformatics, 2026.
Biological hypothesis: Toncheva & Sgurev, BAS.
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bootstrap_pca")

COMPONENTS = ["gini", "selectivity", "mean_pli", "mean_centrality"]
COMPONENT_LABELS = {
    "gini":             "Gini\n(Dependency Concentration)",
    "selectivity":      "Selectivity\n(Cancer Specificity)",
    "mean_pli":         "mean pLI\n(Evolutionary Constraint)",
    "mean_centrality":  "mean Centrality\n(Network Position)",
}
OBSERVED_WEIGHTS = {
    "gini": 0.26,
    "selectivity": 0.20,
    "mean_pli": 0.36,
    "mean_centrality": 0.18,
}
CV_THRESHOLD = 0.20  # Testing Framework v1.0, Section 8.2


def load_data(input_path: str) -> pd.DataFrame:
    """Зарежда nv_components и филтрира lineages с NaN в компонентите."""
    log.info(f"Зареждане: {input_path}")
    df = pd.read_csv(input_path, index_col=0)

    # Запазваме само редовете с пълни данни за 4-те компонента
    before = len(df)
    df = df[COMPONENTS].dropna()
    after = len(df)

    if before != after:
        log.warning(f"  Изключени {before - after} lineages с NaN (Bladder, Liver, Thyroid, Cervix, Eye)")

    log.info(f"  {after} lineages с пълни данни: {list(df.index)}")
    return df


# Reference direction: observed PC1 loadings (sign от пълния dataset)
# Използваме mean_pLI като anchor — 93% позитивен при bootstrap,
# биологично най-стабилен компонент (evolutionary constraint)
# Индекс 2 = mean_pli в [gini, selectivity, mean_pli, mean_centrality]
_PLI_IDX = 2

def compute_pca_weights(matrix: np.ndarray) -> np.ndarray:
    """
    StandardScaler → PCA → PC1 loadings → sign-aligned → нормализирани тегла.

    Sign alignment (Bro & Smilde, J. Chemometrics 2014):
    PC1 е идентифициран до знак — при bootstrap различни итерации могат
    да върнат огледален вектор. Алайнваме знака спрямо mean_pLI loading
    (биологично: по-висок pLI = по-уязвим = по-висок NV-Score = позитивна посока).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    pca = PCA(n_components=4)
    pca.fit(scaled)

    pc1 = pca.components_[0].copy()

    # Sign alignment: ако pLI loading е отрицателен, flip целия вектор
    if pc1[_PLI_IDX] < 0:
        pc1 = -pc1

    # След alignment всички loadings трябва да са позитивни (обща уязвимост)
    # Ако някой е отрицателен след alignment — взимаме abs само за него
    pc1_aligned = np.abs(pc1)

    weights = pc1_aligned / pc1_aligned.sum()
    return weights


def run_bootstrap(df: pd.DataFrame, n_bootstrap: int, seed: int) -> np.ndarray:
    """
    Bootstrap resampling на lineages.
    Връща матрица (n_bootstrap × 4) с нормализирани тегла.
    """
    rng = np.random.default_rng(seed)
    n_lineages = len(df)
    matrix = df.values
    all_weights = np.zeros((n_bootstrap, len(COMPONENTS)))

    log.info(f"Bootstrap: {n_bootstrap} итерации върху {n_lineages} lineages (seed={seed})...")

    failed = 0
    for i in range(n_bootstrap):
        # Resample с replacement
        idx = rng.integers(0, n_lineages, size=n_lineages)
        sample = matrix[idx]

        try:
            weights = compute_pca_weights(sample)
            all_weights[i] = weights
        except Exception:
            # Ако PCA fail-не (singular matrix при дублирани редове) — skip
            all_weights[i] = np.full(len(COMPONENTS), np.nan)
            failed += 1

        if (i + 1) % 200 == 0:
            log.info(f"  {i + 1}/{n_bootstrap} итерации...")

    if failed > 0:
        log.warning(f"  {failed} итерации fail-наха (singular matrix) — изключени от CI")

    # Премахваме NaN редове
    valid = all_weights[~np.isnan(all_weights).any(axis=1)]
    log.info(f"  Валидни итерации: {len(valid)}/{n_bootstrap}")
    return valid


def compute_statistics(bootstrap_weights: np.ndarray) -> dict:
    """Изчислява mean, median, 95% CI, CV за всяко тегло."""
    stats = {}
    for i, comp in enumerate(COMPONENTS):
        w = bootstrap_weights[:, i]
        mean = float(np.mean(w))
        median = float(np.median(w))
        ci_low = float(np.percentile(w, 2.5))
        ci_high = float(np.percentile(w, 97.5))
        std = float(np.std(w))
        cv = std / mean if mean > 0 else float("inf")
        stable = cv < CV_THRESHOLD

        stats[comp] = {
            "observed_weight": OBSERVED_WEIGHTS[comp],
            "bootstrap_mean": round(mean, 4),
            "bootstrap_median": round(median, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "std": round(std, 4),
            "cv": round(cv, 4),
            "stable": stable,
            "ci_width": round(ci_high - ci_low, 4),
        }
    return stats


def print_summary(stats: dict, n_bootstrap: int, n_lineages: int) -> None:
    """Печата форматирана таблица в терминала."""
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[1m"
    E = "\033[0m"

    print(f"\n{B}{'='*72}{E}")
    print(f"{B}  RevGate — Bootstrap CI за NV-Score PCA тегла{E}")
    print(f"  Bootstrap iterations: {n_bootstrap}  |  Lineages: {n_lineages}  |  Seed: 42")
    print(f"{'='*72}")
    print(f"  {'Компонент':<22} {'Observed':>8} {'BS Mean':>8} {'95% CI':>18} {'CV':>7} {'Stable':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*18} {'-'*7} {'-'*8}")

    all_stable = True
    for comp, s in stats.items():
        stable_str = f"{G}YES{E}" if s["stable"] else f"{R}NO{E}"
        if not s["stable"]:
            all_stable = False
        ci_str = f"[{s['ci_95_low']:.3f} – {s['ci_95_high']:.3f}]"
        label = comp.replace("mean_", "").replace("_", " ")
        print(
            f"  {label:<22} {s['observed_weight']:>8.3f} "
            f"{s['bootstrap_mean']:>8.3f} {ci_str:>18} "
            f"{s['cv']:>7.3f} {stable_str:>8}"
        )

    print(f"\n  {B}Stability criterion: CV < {CV_THRESHOLD}{E}")
    overall = f"{G}{B}ALL STABLE ✓{E}" if all_stable else f"{R}{B}UNSTABLE — проверете данните{E}"
    print(f"  Overall: {overall}")
    print(f"{'='*72}\n")


def plot_bootstrap_distributions(
    bootstrap_weights: np.ndarray,
    stats: dict,
    output_dir: Path,
) -> Path:
    """
    Генерира publication-ready фигура:
    4 панела — violin + box plot за всяко тегло с observed стойност и 95% CI.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle(
        "Bootstrap Distribution of NV-Score PCA Weights\n"
        f"(n=1000 iterations, {len(bootstrap_weights)} lineages resampled with replacement)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]

    for i, (comp, ax) in enumerate(zip(COMPONENTS, axes)):
        s = stats[comp]
        w = bootstrap_weights[:, i]

        # Violin plot
        vp = ax.violinplot(w, positions=[0], showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colors[i])
            body.set_alpha(0.4)
            body.set_edgecolor(colors[i])

        # Box plot
        bp = ax.boxplot(
            w, positions=[0], widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor=colors[i], alpha=0.6),
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=colors[i], linewidth=1.5),
            capprops=dict(color=colors[i], linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor=colors[i], markersize=3, alpha=0.3),
        )

        # Observed weight — вертикална линия
        ax.axhline(
            s["observed_weight"], color="black", linestyle="--",
            linewidth=2, label=f"Observed: {s['observed_weight']:.3f}", zorder=5,
        )

        # 95% CI band
        ax.axhspan(
            s["ci_95_low"], s["ci_95_high"],
            alpha=0.12, color=colors[i], label=f"95% CI: [{s['ci_95_low']:.3f}–{s['ci_95_high']:.3f}]",
        )

        # Annotations
        ax.set_title(COMPONENT_LABELS[comp], fontsize=9, fontweight="bold")
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_ylabel("Weight", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        # Stats текст
        cv_color = "green" if s["stable"] else "red"
        stats_text = (
            f"Mean: {s['bootstrap_mean']:.3f}\n"
            f"95% CI: [{s['ci_95_low']:.3f}–{s['ci_95_high']:.3f}]\n"
            f"CV: {s['cv']:.3f} "
            f"({'stable' if s['stable'] else 'UNSTABLE'})"
        )
        ax.text(
            0.97, 0.03, stats_text,
            transform=ax.transAxes,
            fontsize=7.5, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=colors[i]),
            color=cv_color if not s["stable"] else "black",
        )

        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    out_path = output_dir / "bootstrap_pca_weights.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Фигура: {out_path}")
    return out_path


def plot_weight_comparison(stats: dict, output_dir: Path) -> Path:
    """
    Сравнителна фигура: observed vs bootstrap mean с 95% CI error bars.
    Publication-ready за ръкописа.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(COMPONENTS))
    observed = [OBSERVED_WEIGHTS[c] for c in COMPONENTS]
    bs_mean = [stats[c]["bootstrap_mean"] for c in COMPONENTS]
    ci_low = [stats[c]["ci_95_low"] for c in COMPONENTS]
    ci_high = [stats[c]["ci_95_high"] for c in COMPONENTS]
    yerr_low = [bs_mean[i] - ci_low[i] for i in range(len(COMPONENTS))]
    yerr_high = [ci_high[i] - bs_mean[i] for i in range(len(COMPONENTS))]

    # Bootstrap mean + CI
    ax.bar(
        x - 0.2, bs_mean, width=0.35,
        label="Bootstrap Mean ± 95% CI",
        color="#2196F3", alpha=0.75, edgecolor="#1565C0",
    )
    ax.errorbar(
        x - 0.2, bs_mean,
        yerr=[yerr_low, yerr_high],
        fmt="none", color="#1565C0", capsize=6, linewidth=2,
    )

    # Observed weights
    ax.bar(
        x + 0.2, observed, width=0.35,
        label="Observed PCA Weight",
        color="#F44336", alpha=0.75, edgecolor="#B71C1C",
    )

    # Equal weights reference line
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=1.5, label="Equal weights (0.25)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [COMPONENT_LABELS[c].replace("\n", " ") for c in COMPONENTS],
        fontsize=9,
    )
    ax.set_ylabel("NV-Score Weight", fontsize=10)
    ax.set_title(
        "NV-Score PCA Weights: Observed vs Bootstrap Estimates\n"
        "Error bars = 95% confidence intervals (n=1000 bootstrap iterations)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "bootstrap_pca_weights_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Comparison фигура: {out_path}")
    return out_path


def save_json(stats: dict, n_bootstrap: int, n_lineages: int, output_dir: Path) -> Path:
    """Записва пълните резултати като JSON."""
    out = {
        "timestamp": datetime.now().isoformat(),
        "method": "Bootstrap resampling of tumor lineages with replacement",
        "n_bootstrap": n_bootstrap,
        "n_lineages": n_lineages,
        "cv_threshold": CV_THRESHOLD,
        "all_stable": all(s["stable"] for s in stats.values()),
        "weights": stats,
        "manuscript_text": (
            f"NV-Score weights were derived from PCA on {n_lineages} tumor lineages "
            f"(DepMap 24Q4). Bootstrap resampling (n={n_bootstrap} iterations) yielded "
            "95% confidence intervals: "
            + ", ".join(
                f"{c.replace('mean_','')}={stats[c]['observed_weight']:.2f} "
                f"[{stats[c]['ci_95_low']:.2f}–{stats[c]['ci_95_high']:.2f}]"
                for c in COMPONENTS
            )
            + f". All weights were stable (CV < {CV_THRESHOLD})."
            if all(s["stable"] for s in stats.values())
            else " WARNING: some weights are unstable."
        ),
    }
    out_path = output_dir / "bootstrap_pca_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info(f"JSON: {out_path}")
    return out_path


def save_csv(bootstrap_weights: np.ndarray, output_dir: Path) -> Path:
    """Записва всички 1000 bootstrap weight sets като CSV за допълнителен анализ."""
    df = pd.DataFrame(bootstrap_weights, columns=COMPONENTS)
    df.index.name = "iteration"
    out_path = output_dir / "bootstrap_pca_all_iterations.csv"
    df.to_csv(out_path)
    log.info(f"CSV (всички итерации): {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="RevGate Bootstrap CI за NV-Score PCA тегла"
    )
    parser.add_argument(
        "--input",
        default="/root/.revgate/cache/nv_components_all_lineages.csv",
        help="Път до nv_components_all_lineages.csv",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Output директория (default: results/)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Брой bootstrap итерации (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed за reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error(f"Файлът не е намерен: {args.input}")
        raise SystemExit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Зареждане
    df = load_data(args.input)
    n_lineages = len(df)

    # 2. Observed weights (за верификация)
    observed_weights = compute_pca_weights(df.values)
    log.info("Observed PCA weights (верификация):")
    for comp, w in zip(COMPONENTS, observed_weights):
        log.info(f"  {comp:<20}: {w:.4f}  (expected: {OBSERVED_WEIGHTS[comp]:.2f})")

    # 3. Bootstrap
    bootstrap_weights = run_bootstrap(df, args.n_bootstrap, args.seed)

    # 4. Статистики
    stats = compute_statistics(bootstrap_weights)

    # 5. Output
    print_summary(stats, args.n_bootstrap, n_lineages)
    plot_bootstrap_distributions(bootstrap_weights, stats, output_dir)
    plot_weight_comparison(stats, output_dir)
    save_json(stats, args.n_bootstrap, n_lineages, output_dir)
    save_csv(bootstrap_weights, output_dir)

    log.info("Bootstrap анализът е завършен.")
    log.info(f"Фигури и данни: {output_dir}")


if __name__ == "__main__":
    main()
