#!/usr/bin/env python3
"""
RevGate — Разширен Sensitivity Analysis на NV-Score
====================================================
Три нива на валидация:

  LEVEL 1 — Threshold sensitivity
    Sweep на NV-A/NV-B праговете ±0.10 стъпка 0.01
    Колко lineages сменят клас? Кои са граничните случаи?

  LEVEL 2 — Alternative weighting schemes
    Сравнява 4 модела:
      - PCA weights     (0.26 / 0.20 / 0.36 / 0.18)
      - Equal weights   (0.25 / 0.25 / 0.25 / 0.25)
      - Theory-driven   (0.15 / 0.15 / 0.50 / 0.20) — pLI доминира
      - Clinical anchor (от Finding_01 Cox резултати)
    Класификационно съгласие между модели + Cohen's kappa

  LEVEL 3 — Jackknife (leave-one-out)
    По-подходящ от bootstrap при n=20
    Премахваме по един lineage → PCA → тегла
    Pseudo-values → jackknife SE и 95% CI

Употреба:
  cd ~/revgate
  python3 run_sensitivity_analysis.py
  python3 run_sensitivity_analysis.py --input /root/.revgate/cache/nv_components_all_lineages.csv

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sensitivity")

COMPONENTS  = ["gini", "selectivity", "mean_pli", "mean_centrality"]
COMP_SHORT  = ["Gini", "Select.", "pLI", "Centr."]

# Четири weighting схеми
WEIGHT_SCHEMES = {
    "PCA":      [0.26, 0.20, 0.36, 0.18],
    "Equal":    [0.25, 0.25, 0.25, 0.25],
    "Theory":   [0.15, 0.15, 0.50, 0.20],  # pLI доминира биологично
    "Clinical": [0.30, 0.25, 0.30, 0.15],  # BRCA/KIRC Finding_01 anchored
}

# Базови прагове
BASE_THRESH_A = 0.45
BASE_THRESH_B = 0.35


# ── Helpers ─────────────────────────────────────────────────────────────────

def nv_class(score: float, thresh_a: float, thresh_b: float) -> str:
    if score >= thresh_a: return "NV-A"
    if score >= thresh_b: return "NV-B"
    return "NV-C"


def compute_nv_scores(scaled_df: pd.DataFrame, weights: list) -> pd.Series:
    """Изчислява NV-Score за всеки lineage по дадени тегла."""
    return scaled_df[COMPONENTS].apply(
        lambda row: sum(weights[i] * row[c] for i, c in enumerate(COMPONENTS)),
        axis=1,
    )


def scale_components(df: pd.DataFrame) -> pd.DataFrame:
    """MinMaxScaler върху 4-те компонента за сравнимост между схеми."""
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[COMPONENTS]),
        index=df.index,
        columns=COMPONENTS,
    )
    return scaled


def compute_pca_weights_aligned(matrix: np.ndarray) -> np.ndarray:
    """PCA с sign alignment по pLI (индекс 2)."""
    sc = StandardScaler()
    pca = PCA(n_components=4)
    pca.fit(sc.fit_transform(matrix))
    pc1 = pca.components_[0].copy()
    if pc1[2] < 0:  # pLI anchor
        pc1 = -pc1
    pc1_abs = np.abs(pc1)
    return pc1_abs / pc1_abs.sum()


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa между две класификации."""
    classes = ["NV-A", "NV-B", "NV-C"]
    n = len(a)
    po = (a == b).sum() / n
    pe = sum(
        (a == c).sum() / n * (b == c).sum() / n
        for c in classes
    )
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ── LEVEL 1: Threshold Sensitivity ──────────────────────────────────────────

def level1_threshold_sensitivity(scaled_df: pd.DataFrame) -> dict:
    """
    Sweep на праговете NV-A и NV-B ±0.10 (стъпка 0.01).
    За всяка комбинация: брой lineages в NV-A/B/C и брой смени спрямо базовото.
    """
    log.info("LEVEL 1: Threshold sensitivity analysis...")

    base_scores = compute_nv_scores(scaled_df, WEIGHT_SCHEMES["PCA"])
    base_classes = base_scores.apply(lambda s: nv_class(s, BASE_THRESH_A, BASE_THRESH_B))

    results = []
    thresholds_a = np.arange(BASE_THRESH_A - 0.10, BASE_THRESH_A + 0.11, 0.01)
    thresholds_b = np.arange(BASE_THRESH_B - 0.10, BASE_THRESH_B + 0.11, 0.01)

    for ta in thresholds_a:
        for tb in thresholds_b:
            if tb >= ta:
                continue  # невалидна комбинация
            classes = base_scores.apply(lambda s: nv_class(s, ta, tb))
            changes = (classes != base_classes).sum()
            results.append({
                "thresh_a": round(ta, 3),
                "thresh_b": round(tb, 3),
                "n_NV-A":   int((classes == "NV-A").sum()),
                "n_NV-B":   int((classes == "NV-B").sum()),
                "n_NV-C":   int((classes == "NV-C").sum()),
                "n_changes": int(changes),
                "pct_stable": round(100 * (1 - changes / len(base_classes)), 1),
            })

    df_res = pd.DataFrame(results)

    # Граничните lineages — тези с score близо до праговете
    border_a = scaled_df.index[
        (base_scores - BASE_THRESH_A).abs() < 0.05
    ].tolist()
    border_b = scaled_df.index[
        (base_scores - BASE_THRESH_B).abs() < 0.05
    ].tolist()

    # Средна стабилност
    mean_stable = df_res["pct_stable"].mean()
    min_stable  = df_res["pct_stable"].min()

    log.info(f"  Средна стабилност при threshold sweep: {mean_stable:.1f}%")
    log.info(f"  Минимална стабилност: {min_stable:.1f}%")
    log.info(f"  Гранични lineages (±0.05 от NV-A прага): {border_a}")
    log.info(f"  Гранични lineages (±0.05 от NV-B прага): {border_b}")

    return {
        "sweep_results": df_res,
        "mean_stability_pct": round(mean_stable, 2),
        "min_stability_pct":  round(min_stable, 2),
        "border_lineages_a":  border_a,
        "border_lineages_b":  border_b,
        "base_scores":        base_scores,
        "base_classes":       base_classes,
    }


# ── LEVEL 2: Alternative Weighting Schemes ──────────────────────────────────

def level2_weighting_schemes(scaled_df: pd.DataFrame) -> dict:
    """
    Сравнява 4 weighting схеми.
    За всяка схема: NV-Score, NV-Class, Cohen's kappa спрямо PCA.
    """
    log.info("LEVEL 2: Alternative weighting schemes...")

    all_scores  = {}
    all_classes = {}

    for name, weights in WEIGHT_SCHEMES.items():
        scores  = compute_nv_scores(scaled_df, weights)
        classes = scores.apply(lambda s: nv_class(s, BASE_THRESH_A, BASE_THRESH_B))
        all_scores[name]  = scores
        all_classes[name] = classes

    # Pairwise Cohen's kappa
    scheme_names = list(WEIGHT_SCHEMES.keys())
    kappa_matrix = pd.DataFrame(index=scheme_names, columns=scheme_names, dtype=float)
    agreement_matrix = pd.DataFrame(index=scheme_names, columns=scheme_names, dtype=float)

    for s1 in scheme_names:
        for s2 in scheme_names:
            kappa = cohen_kappa(all_classes[s1], all_classes[s2])
            agree = (all_classes[s1] == all_classes[s2]).mean() * 100
            kappa_matrix.loc[s1, s2]     = round(kappa, 3)
            agreement_matrix.loc[s1, s2] = round(agree, 1)

    # Консенсусна класификация (мнозинство от 4 схеми)
    classes_df = pd.DataFrame(all_classes)
    consensus  = classes_df.apply(
        lambda row: row.value_counts().index[0], axis=1
    )

    # Lineages с несъгласие
    unanimous = classes_df.apply(lambda row: row.nunique() == 1, axis=1)
    contested = scaled_df.index[~unanimous].tolist()

    log.info(f"  Unanimous classification: {unanimous.sum()}/20 lineages")
    log.info(f"  Contested lineages: {contested}")
    log.info(f"  Cohen's kappa (PCA vs Equal): {kappa_matrix.loc['PCA','Equal']:.3f}")

    return {
        "scores":           pd.DataFrame(all_scores).round(4),
        "classes":          classes_df,
        "consensus":        consensus,
        "kappa_matrix":     kappa_matrix,
        "agreement_matrix": agreement_matrix,
        "unanimous_count":  int(unanimous.sum()),
        "contested":        contested,
    }


# ── LEVEL 3: Jackknife Leave-One-Out ────────────────────────────────────────

def level3_jackknife(df: pd.DataFrame) -> dict:
    """
    Jackknife (LOO) върху lineages.
    За всяко изключено lineage → PCA → тегла.
    Pseudo-values → jackknife estimate, SE, 95% CI.
    """
    log.info("LEVEL 3: Jackknife leave-one-out...")

    n = len(df)
    matrix = df.values
    loo_weights = np.zeros((n, len(COMPONENTS)))

    for i in range(n):
        loo_matrix = np.delete(matrix, i, axis=0)
        try:
            loo_weights[i] = compute_pca_weights_aligned(loo_matrix)
        except Exception as e:
            log.warning(f"  LOO i={i} fail: {e}")
            loo_weights[i] = np.full(len(COMPONENTS), np.nan)

    # Observed weights (пълен dataset)
    observed = compute_pca_weights_aligned(matrix)

    # Jackknife pseudo-values: θ_i = n*θ - (n-1)*θ_{-i}
    pseudo = n * observed - (n - 1) * loo_weights

    # Jackknife estimate и SE
    jk_estimate = pseudo.mean(axis=0)
    jk_se       = np.sqrt(((n - 1) / n) * ((pseudo - jk_estimate) ** 2).sum(axis=0))

    # 95% CI (normal approximation — подходящо за n=20)
    z = 1.96
    ci_low  = jk_estimate - z * jk_se
    ci_high = jk_estimate + z * jk_se

    stats = {}
    for i, comp in enumerate(COMPONENTS):
        stats[comp] = {
            "observed":    round(float(observed[i]), 4),
            "jk_estimate": round(float(jk_estimate[i]), 4),
            "jk_se":       round(float(jk_se[i]), 4),
            "ci_95_low":   round(float(ci_low[i]), 4),
            "ci_95_high":  round(float(ci_high[i]), 4),
            "cv":          round(float(jk_se[i] / jk_estimate[i]) if jk_estimate[i] > 0 else 999, 4),
            "stable":      bool(jk_se[i] / jk_estimate[i] < 0.20 if jk_estimate[i] > 0 else False),
        }

    # Influential lineages — кои най-много влияят на теглата
    influence = np.abs(loo_weights - observed).sum(axis=1)
    influential = [
        {"lineage": df.index[i], "total_influence": round(float(influence[i]), 4)}
        for i in np.argsort(influence)[::-1][:5]
    ]

    log.info("  Jackknife 95% CI:")
    for comp, s in stats.items():
        stable = "✓" if s["stable"] else "✗"
        log.info(
            f"    {comp:<20}: {s['observed']:.4f}  "
            f"[{s['ci_95_low']:.4f} – {s['ci_95_high']:.4f}]  "
            f"SE={s['jk_se']:.4f}  {stable}"
        )
    log.info(f"  Най-влиятелни lineages: {[x['lineage'] for x in influential[:3]]}")

    return {
        "stats":         stats,
        "loo_weights":   pd.DataFrame(loo_weights, index=df.index, columns=COMPONENTS),
        "influential":   influential,
        "observed":      observed,
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_threshold_heatmap(level1: dict, output_dir: Path) -> Path:
    """Heatmap на % стабилност при threshold sweep."""
    df_sweep = level1["sweep_results"]

    # Pivot: rows=thresh_a, cols=thresh_b, values=pct_stable
    pivot = df_sweep.pivot_table(
        index="thresh_a", columns="thresh_b", values="pct_stable"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=60, vmax=100)
    plt.colorbar(im, ax=ax, label="% Stable classifications")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=7)
    ax.set_xlabel("NV-B threshold", fontsize=10)
    ax.set_ylabel("NV-A threshold", fontsize=10)
    ax.set_title(
        "NV-Score Classification Stability under Threshold Perturbation\n"
        "(% lineages retaining same class vs. base thresholds 0.45/0.35)",
        fontsize=10, fontweight="bold",
    )

    # Маркираме базовия threshold
    base_a_idx = list(pivot.index).index(
        min(pivot.index, key=lambda x: abs(x - BASE_THRESH_A))
    )
    base_b_idx = list(pivot.columns).index(
        min(pivot.columns, key=lambda x: abs(x - BASE_THRESH_B))
    )
    ax.plot(base_b_idx, base_a_idx, "k*", markersize=12, label="Base (0.45/0.35)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = output_dir / "sensitivity_threshold_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Heatmap: {out}")
    return out


def plot_weighting_comparison(level2: dict, output_dir: Path) -> Path:
    """Dot plot на NV-Score за 4 схеми × 20 lineages."""
    scores_df  = level2["scores"]
    classes_df = level2["classes"]
    contested  = level2["contested"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Panel A: Score comparison
    ax = axes[0]
    colors = {"PCA": "#2196F3", "Equal": "#4CAF50", "Theory": "#F44336", "Clinical": "#FF9800"}
    markers = {"PCA": "o", "Equal": "s", "Theory": "^", "Clinical": "D"}
    lineages = list(scores_df.index)
    y = np.arange(len(lineages))

    for scheme in WEIGHT_SCHEMES:
        ax.scatter(
            scores_df[scheme], y,
            color=colors[scheme], marker=markers[scheme],
            s=60, alpha=0.8, label=scheme, zorder=3,
        )

    ax.axvline(BASE_THRESH_A, color="red",  linestyle="--", linewidth=1, alpha=0.7, label="NV-A threshold")
    ax.axvline(BASE_THRESH_B, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="NV-B threshold")

    # Highlight contested
    for lineage in contested:
        idx = lineages.index(lineage)
        ax.axhspan(idx - 0.4, idx + 0.4, alpha=0.1, color="yellow")

    ax.set_yticks(y)
    ax.set_yticklabels(lineages, fontsize=8)
    ax.set_xlabel("NV-Score", fontsize=10)
    ax.set_title("NV-Score by Weighting Scheme", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Panel B: Kappa heatmap
    ax2 = axes[1]
    kappa = level2["kappa_matrix"].astype(float)
    im = ax2.imshow(kappa.values, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, label="Cohen's κ")

    schemes = list(WEIGHT_SCHEMES.keys())
    ax2.set_xticks(range(len(schemes)))
    ax2.set_yticks(range(len(schemes)))
    ax2.set_xticklabels(schemes, fontsize=10)
    ax2.set_yticklabels(schemes, fontsize=10)
    ax2.set_title("Pairwise Cohen's κ\nbetween Weighting Schemes", fontsize=10, fontweight="bold")

    for i in range(len(schemes)):
        for j in range(len(schemes)):
            ax2.text(j, i, f"{kappa.values[i,j]:.2f}",
                     ha="center", va="center", fontsize=11,
                     color="white" if kappa.values[i,j] > 0.6 else "black")

    plt.tight_layout()
    out = output_dir / "sensitivity_weighting_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Weighting comparison: {out}")
    return out


def plot_jackknife(level3: dict, output_dir: Path) -> Path:
    """Forest plot на jackknife CI за 4 компонента."""
    stats = level3["stats"]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
    y = np.arange(len(COMPONENTS))

    for i, comp in enumerate(COMPONENTS):
        s = stats[comp]
        color = colors[i]

        # CI line
        ax.plot(
            [s["ci_95_low"], s["ci_95_high"]], [i, i],
            color=color, linewidth=3, alpha=0.6,
        )
        # Jackknife estimate
        ax.scatter(s["jk_estimate"], i, color=color, s=100, zorder=5, label=f"{COMP_SHORT[i]} JK")
        # Observed
        ax.scatter(s["observed"], i, color=color, s=80, marker="D", zorder=6,
                   edgecolors="black", linewidths=1.5)

        # Annotation
        ax.text(
            s["ci_95_high"] + 0.005, i,
            f"[{s['ci_95_low']:.3f}–{s['ci_95_high']:.3f}]  SE={s['jk_se']:.3f}",
            va="center", fontsize=8,
        )

    ax.axvline(0.25, color="gray", linestyle=":", linewidth=1.5, label="Equal weight (0.25)")

    ax.set_yticks(y)
    ax.set_yticklabels([COMP_SHORT[i] for i in range(len(COMPONENTS))], fontsize=10)
    ax.set_xlabel("NV-Score Weight", fontsize=10)
    ax.set_title(
        "Jackknife (Leave-One-Out) Confidence Intervals for NV-Score PCA Weights\n"
        "Circles = jackknife estimate; Diamonds = observed; Lines = 95% CI",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlim(-0.05, 0.80)
    ax.grid(axis="x", alpha=0.3)

    obs_patch   = mpatches.Patch(color="gray", label="◆ Observed weight")
    jk_patch    = mpatches.Patch(color="gray", label="● Jackknife estimate")
    equal_line  = plt.Line2D([0], [0], color="gray", linestyle=":", label="Equal weight (0.25)")
    ax.legend(handles=[obs_patch, jk_patch, equal_line], fontsize=8, loc="upper right")

    plt.tight_layout()
    out = output_dir / "sensitivity_jackknife_ci.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Jackknife CI plot: {out}")
    return out


# ── Summary и JSON ────────────────────────────────────────────────────────────

def print_full_summary(level1: dict, level2: dict, level3: dict) -> None:
    G = "\033[92m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"

    print(f"\n{B}{'='*70}{E}")
    print(f"{B}  RevGate — Sensitivity Analysis Summary{E}")
    print(f"{'='*70}")

    print(f"\n{B}LEVEL 1 — Threshold Sensitivity{E}")
    print(f"  Средна стабилност при ±0.10 sweep: {level1['mean_stability_pct']:.1f}%")
    print(f"  Минимална стабилност:               {level1['min_stability_pct']:.1f}%")
    print(f"  Гранични lineages (NV-A):  {level1['border_lineages_a']}")
    print(f"  Гранични lineages (NV-B):  {level1['border_lineages_b']}")

    print(f"\n{B}LEVEL 2 — Alternative Weighting Schemes{E}")
    print(f"  Unanimous classifications: {level2['unanimous_count']}/20")
    print(f"  Contested lineages:        {level2['contested']}")
    print(f"  Cohen's κ matrix:")
    print(level2["kappa_matrix"].to_string())

    print(f"\n{B}LEVEL 3 — Jackknife Leave-One-Out{E}")
    all_stable = all(s["stable"] for s in level3["stats"].values())
    print(f"  {'Компонент':<20} {'Observed':>8} {'JK Est.':>8} {'95% CI':>22} {'SE':>7} {'Stable':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*22} {'-'*7} {'-'*8}")
    for comp, s in level3["stats"].items():
        stable_str = f"{G}YES{E}" if s["stable"] else f"{R}NO{E}"
        ci_str = f"[{s['ci_95_low']:.4f} – {s['ci_95_high']:.4f}]"
        print(f"  {comp:<20} {s['observed']:>8.4f} {s['jk_estimate']:>8.4f} {ci_str:>22} {s['jk_se']:>7.4f} {stable_str:>8}")

    overall = f"{G}{B}STABLE ✓{E}" if all_stable else f"{R}{B}UNSTABLE{E}"
    print(f"\n  Jackknife stability: {overall}")
    print(f"{'='*70}\n")


def save_results(level1, level2, level3, output_dir: Path) -> None:
    # Jackknife JSON
    jk_out = {
        "timestamp": datetime.now().isoformat(),
        "method": "Jackknife leave-one-out on tumor lineages",
        "n_lineages": 20,
        "stats": level3["stats"],
        "influential_lineages": level3["influential"],
        "all_stable": all(s["stable"] for s in level3["stats"].values()),
    }
    with open(output_dir / "sensitivity_jackknife.json", "w") as f:
        json.dump(jk_out, f, indent=2, ensure_ascii=False)

    # Weighting schemes CSV
    level2["scores"].to_csv(output_dir / "sensitivity_weighting_scores.csv")
    level2["classes"].to_csv(output_dir / "sensitivity_weighting_classes.csv")
    level2["kappa_matrix"].to_csv(output_dir / "sensitivity_kappa_matrix.csv")

    # Threshold sweep CSV
    level1["sweep_results"].to_csv(
        output_dir / "sensitivity_threshold_sweep.csv", index=False
    )

    # LOO weights CSV
    level3["loo_weights"].to_csv(output_dir / "sensitivity_loo_weights.csv")

    log.info(f"Резултати записани в {output_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RevGate Sensitivity Analysis — 3 нива"
    )
    parser.add_argument(
        "--input",
        default="/root/.revgate/cache/nv_components_all_lineages.csv",
        help="Път до nv_components_all_lineages.csv",
    )
    parser.add_argument(
        "--output", default="results/",
        help="Output директория (default: results/)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error(f"Файлът не е намерен: {args.input}")
        raise SystemExit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Зареждане и scaling
    df_raw = pd.read_csv(args.input, index_col=0)
    df_raw = df_raw[COMPONENTS].dropna()
    log.info(f"Заредени {len(df_raw)} lineages с пълни данни")

    scaled_df = scale_components(df_raw)

    # Три нива
    level1 = level1_threshold_sensitivity(scaled_df)
    level2 = level2_weighting_schemes(scaled_df)
    level3 = level3_jackknife(df_raw)

    # Plots
    plot_threshold_heatmap(level1, output_dir)
    plot_weighting_comparison(level2, output_dir)
    plot_jackknife(level3, output_dir)

    # Summary
    print_full_summary(level1, level2, level3)

    # Save
    save_results(level1, level2, level3, output_dir)

    log.info("Sensitivity analysis завършен.")


if __name__ == "__main__":
    main()
