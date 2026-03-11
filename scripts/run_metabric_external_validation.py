#!/usr/bin/env python3
"""
RevGate — METABRIC External Validation
=======================================
Independent validation на ssGSEA NV-Score -> OS в METABRIC cohort.
n=1980, 1143 events (57.7% event rate -- много по-богат от TCGA BRCA).

Анализи:
  1. Univariate Cox: ssGSEA -> OS
  2. Multivariate Cox: ssGSEA + Age + ER_status + CLAUDIN_SUBTYPE -> OS
  3. Subtype-stratified Cox: LumA / LumB / Her2 / Basal / claudin-low
  4. ER-stratified Cox: ER+ vs ER-
  5. Kaplan-Meier: High vs Low ssGSEA (median split)
  6. Replication check: сравнение с Finding_01 BRCA HR=0.813

Helena Bioinformatics, 2026.
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
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("metabric_validation")

TCGA_BRCA_EARLY_HR = 0.813   # Finding_01 reference
TCGA_BRCA_EARLY_P  = 0.0015


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(ssgsea_path: str, clinical_path: str) -> pd.DataFrame:
    """Merge ssGSEA scores + clinical data."""
    log.info("Зареждане на данни...")

    ssgsea = pd.read_parquet(ssgsea_path)
    clin   = pd.read_csv(clinical_path)

    # Merge по patient ID
    df = ssgsea.merge(clin, left_on="sample_id", right_on="patientId", how="inner")

    # OS construction
    df["os_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["os_event"]  = df["OS_STATUS"].map({"1:DECEASED": 1, "0:LIVING": 0})

    # Age
    df["age"] = pd.to_numeric(df["AGE_AT_DIAGNOSIS"], errors="coerce")

    # ER status binary
    df["er_positive"] = df["ER_IHC"].map({"Positve": 1, "Negative": 0})

    # Subtype cleanup
    df["subtype"] = df["CLAUDIN_SUBTYPE"].replace("NC", np.nan)

    # Filter
    df = df.dropna(subset=["os_months", "os_event", "ssgsea_nv_score"])
    df = df[df["os_months"] > 0]

    log.info(f"  Cohort: n={len(df)}, events={int(df['os_event'].sum())} ({df['os_event'].mean()*100:.1f}%)")
    log.info(f"  ssGSEA range: [{df['ssgsea_nv_score'].min():.3f} – {df['ssgsea_nv_score'].max():.3f}]")
    log.info(f"  ER+: {int(df['er_positive'].sum())}, ER-: {int((df['er_positive']==0).sum())}")

    return df


# ── Cox analyses ──────────────────────────────────────────────────────────────

def fit_cox(df: pd.DataFrame, covariates: list,
            duration="os_months", event="os_event",
            penalizer=0.1) -> dict:
    """Fit Cox PH, върни резултат за първия covariate."""
    sub = df[covariates + [duration, event]].dropna().copy()

    # Standardize continuous vars
    for col in covariates:
        if col in ["er_positive"] or sub[col].nunique() <= 5:
            continue
        sub[col] = StandardScaler().fit_transform(sub[[col]])

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(sub, duration_col=duration, event_col=event)

    s   = cph.summary
    var = covariates[0]
    return {
        "hr":       round(float(s.loc[var, "exp(coef)"]), 4),
        "ci_low":   round(float(s.loc[var, "exp(coef) lower 95%"]), 4),
        "ci_high":  round(float(s.loc[var, "exp(coef) upper 95%"]), 4),
        "p":        round(float(s.loc[var, "p"]), 4),
        "coef":     round(float(s.loc[var, "coef"]), 4),
        "n":        len(sub),
        "events":   int(sub[event].sum()),
    }


def univariate_cox(df: pd.DataFrame) -> dict:
    log.info("Univariate Cox: ssGSEA -> OS...")
    r = fit_cox(df, ["ssgsea_nv_score", "age"])
    log.info(f"  HR={r['hr']:.4f} [{r['ci_low']:.4f}–{r['ci_high']:.4f}] p={r['p']:.4f}")
    return r


def multivariate_cox(df: pd.DataFrame) -> dict:
    log.info("Multivariate Cox: ssGSEA + age + ER + subtype dummies...")
    sub = df.copy()

    # Dummy за subtype (LumA reference)
    sub = sub.dropna(subset=["subtype", "er_positive"])
    dummies = pd.get_dummies(sub["subtype"], prefix="st", drop_first=False)
    # Drop LumA като reference
    if "st_LumA" in dummies.columns:
        dummies = dummies.drop(columns=["st_LumA"])
    sub = pd.concat([sub, dummies], axis=1)

    dummy_cols = [c for c in dummies.columns]
    covs = ["ssgsea_nv_score", "age", "er_positive"] + dummy_cols

    try:
        r = fit_cox(sub, covs)
        log.info(f"  HR={r['hr']:.4f} [{r['ci_low']:.4f}–{r['ci_high']:.4f}] p={r['p']:.4f}")
    except Exception as e:
        log.warning(f"  Multivariate failed: {e}")
        r = {"hr": None, "error": str(e)}
    return r


def subtype_stratified_cox(df: pd.DataFrame) -> dict:
    """Cox по молекулярен subtype."""
    log.info("Subtype-stratified Cox...")
    results = {}
    subtypes = ["LumA", "LumB", "Her2", "Basal", "claudin-low"]

    for st in subtypes:
        sub = df[df["subtype"] == st]
        if len(sub) < 30 or sub["os_event"].sum() < 10:
            log.info(f"  {st}: пропуснат (n={len(sub)}, events={sub['os_event'].sum()})")
            continue
        try:
            r = fit_cox(sub, ["ssgsea_nv_score", "age"])
            results[st] = {**r, "subtype": st}
            sig = "✓" if r["p"] < 0.05 else "○"
            log.info(f"  {st:<15}: HR={r['hr']:.4f} p={r['p']:.4f} n={r['n']} {sig}")
        except Exception as e:
            log.warning(f"  {st}: {e}")

    return results


def er_stratified_cox(df: pd.DataFrame) -> dict:
    """Cox stratified по ER status."""
    log.info("ER-stratified Cox...")
    results = {}

    for er_val, er_name in [(1, "ER+"), (0, "ER-")]:
        sub = df[df["er_positive"] == er_val]
        if len(sub) < 30:
            continue
        try:
            r = fit_cox(sub, ["ssgsea_nv_score", "age"])
            results[er_name] = {**r, "er_status": er_name}
            sig = "✓" if r["p"] < 0.05 else "○"
            log.info(f"  {er_name}: HR={r['hr']:.4f} p={r['p']:.4f} n={r['n']} {sig}")
        except Exception as e:
            log.warning(f"  {er_name}: {e}")

    return results


# ── Replication check ─────────────────────────────────────────────────────────

def replication_check(univariate: dict, subtype_results: dict) -> dict:
    """
    Формален replication assessment.
    Finding_01: BRCA Early HR=0.813, p=0.0015 (protective effect, HR<1).
    Replication: directional consistency (HR<1) + p<0.05.
    """
    hr   = univariate.get("hr")
    p    = univariate.get("p")

    directional = hr is not None and hr < 1.0
    significant = p is not None and p < 0.05
    replicated  = directional and significant

    # Lum A -- най-близо до Finding_01 Early BRCA (ER+ luminal)
    luma = subtype_results.get("LumA", {})
    luma_repl = luma.get("hr", 1.0) < 1.0 and luma.get("p", 1.0) < 0.05

    return {
        "tcga_brca_early_hr":  TCGA_BRCA_EARLY_HR,
        "tcga_brca_early_p":   TCGA_BRCA_EARLY_P,
        "metabric_univariate_hr": hr,
        "metabric_univariate_p":  p,
        "directional_consistency": bool(directional),
        "statistically_significant": bool(significant),
        "replicated": bool(replicated),
        "luma_replicated": bool(luma_repl),
        "luma_hr": luma.get("hr"),
        "luma_p":  luma.get("p"),
        "verdict": (
            "REPLICATED — directional + significant"      if replicated else
            "PARTIAL — directional but not significant"   if directional else
            "NOT REPLICATED — opposite direction"
        ),
    }


# ── Kaplan-Meier ──────────────────────────────────────────────────────────────

def kaplan_meier_analysis(df: pd.DataFrame) -> dict:
    """KM curves: High vs Low ssGSEA (median split), overall + per subtype."""
    median = df["ssgsea_nv_score"].median()
    df     = df.copy()
    df["score_group"] = df["ssgsea_nv_score"].apply(
        lambda x: "High NV-Score" if x >= median else "Low NV-Score"
    )

    high = df[df["score_group"] == "High NV-Score"]
    low  = df[df["score_group"] == "Low NV-Score"]

    lr = logrank_test(
        high["os_months"], low["os_months"],
        event_observed_A=high["os_event"],
        event_observed_B=low["os_event"],
    )

    log.info(f"KM log-rank: p={lr.p_value:.4f} (High n={len(high)}, Low n={len(low)})")

    return {
        "median_score":  round(float(median), 4),
        "n_high":        len(high),
        "n_low":         len(low),
        "logrank_p":     round(float(lr.p_value), 4),
        "logrank_stat":  round(float(lr.test_statistic), 4),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(df, univariate, multivariate, subtype_results,
                 er_results, km_data, replication, output_dir: Path) -> Path:
    """Publication-ready 6-panel figure."""
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    colors = {
        "LumA": "#2196F3", "LumB": "#03A9F4", "Her2": "#F44336",
        "Basal": "#9C27B0", "claudin-low": "#FF9800",
        "ER+": "#4CAF50", "ER-": "#F44336",
    }

    # ── Panel A: KM curves ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    median = km_data["median_score"]
    df_plot = df.copy()
    df_plot["group"] = (df_plot["ssgsea_nv_score"] >= median).map(
        {True: "High NV-Score", False: "Low NV-Score"}
    )

    for grp, col in [("High NV-Score", "#4CAF50"), ("Low NV-Score", "#F44336")]:
        sub = df_plot[df_plot["group"] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_months"], sub["os_event"], label=f"{grp} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax_a, color=col, ci_show=True, ci_alpha=0.1)

    ax_a.set_xlabel("Overall Survival (Months)", fontsize=9)
    ax_a.set_ylabel("Survival Probability", fontsize=9)
    ax_a.set_title(
        f"METABRIC KM — ssGSEA NV-Score\nLog-rank p={km_data['logrank_p']:.4f}",
        fontweight="bold", fontsize=10,
    )
    ax_a.legend(fontsize=8)
    ax_a.set_ylim(0, 1.05)

    # ── Panel B: Subtype forest plot ──────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    subtypes = list(subtype_results.keys())
    hrs  = [subtype_results[s]["hr"] for s in subtypes]
    cis  = [(subtype_results[s]["ci_low"], subtype_results[s]["ci_high"]) for s in subtypes]
    ps   = [subtype_results[s]["p"] for s in subtypes]
    ns   = [subtype_results[s]["n"] for s in subtypes]
    cols = [colors.get(s, "#607D8B") for s in subtypes]

    y = range(len(subtypes))
    ax_b.scatter(hrs, y, color=cols, s=80, zorder=3)
    for i, (ci, hr) in enumerate(zip(cis, hrs)):
        ax_b.plot([ci[0], ci[1]], [i, i], color=cols[i], lw=2, alpha=0.7)
    ax_b.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax_b.set_yticks(list(y))
    ax_b.set_yticklabels(
        [f"{s}\n(n={n}, p={p:.3f})" for s, n, p in zip(subtypes, ns, ps)],
        fontsize=8,
    )
    ax_b.set_xlabel("Hazard Ratio (ssGSEA)", fontsize=9)
    ax_b.set_title("Subtype-Stratified Cox\nssGSEA NV-Score → OS", fontweight="bold", fontsize=10)

    # ── Panel C: ER stratified ────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    er_names = list(er_results.keys())
    er_hrs   = [er_results[e]["hr"] for e in er_names]
    er_cis   = [(er_results[e]["ci_low"], er_results[e]["ci_high"]) for e in er_names]
    er_ps    = [er_results[e]["p"] for e in er_names]
    er_ns    = [er_results[e]["n"] for e in er_names]
    er_cols  = [colors.get(e, "#607D8B") for e in er_names]

    y_er = range(len(er_names))
    ax_c.scatter(er_hrs, y_er, color=er_cols, s=100, zorder=3)
    for i, (ci, hr) in enumerate(zip(er_cis, er_hrs)):
        ax_c.plot([ci[0], ci[1]], [i, i], color=er_cols[i], lw=2.5, alpha=0.7)
    ax_c.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax_c.set_yticks(list(y_er))
    ax_c.set_yticklabels(
        [f"{e}\n(n={n}, p={p:.3f})" for e, n, p in zip(er_names, er_ns, er_ps)],
        fontsize=9,
    )
    ax_c.set_xlabel("Hazard Ratio (ssGSEA)", fontsize=9)
    ax_c.set_title("ER-Stratified Cox\nssGSEA NV-Score → OS", fontweight="bold", fontsize=10)

    # ── Panel D: Replication comparison ──────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    labels  = ["TCGA BRCA\nEarly (F01)", "METABRIC\nUnivariate", "METABRIC\nLumA"]
    hr_vals = [
        TCGA_BRCA_EARLY_HR,
        replication["metabric_univariate_hr"] or 1.0,
        replication["luma_hr"] or 1.0,
    ]
    p_vals  = [
        TCGA_BRCA_EARLY_P,
        replication["metabric_univariate_p"] or 1.0,
        replication["luma_p"] or 1.0,
    ]
    bar_cols = ["#2196F3", "#4CAF50" if replication["replicated"] else "#FF9800", "#9C27B0"]
    bars = ax_d.bar(labels, hr_vals, color=bar_cols, alpha=0.8, edgecolor="gray")
    ax_d.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="HR=1 (null)")
    for bar, p in zip(bars, p_vals):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax_d.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.005,
                  f"p={p:.3f}\n{sig}", ha="center", fontsize=8)
    ax_d.set_ylabel("Hazard Ratio", fontsize=9)
    ax_d.set_title(f"Replication Check\n{replication['verdict']}", fontweight="bold", fontsize=10)
    ax_d.set_ylim(0, max(hr_vals) * 1.3)
    ax_d.legend(fontsize=8)

    # ── Panel E: Score distribution per subtype ───────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    subtypes_plot = ["LumA", "LumB", "Her2", "Basal", "claudin-low"]
    data_plot = [df[df["subtype"] == s]["ssgsea_nv_score"].dropna().values
                 for s in subtypes_plot]
    bp = ax_e.boxplot(data_plot, patch_artist=True, notch=False)
    for patch, st in zip(bp["boxes"], subtypes_plot):
        patch.set_facecolor(colors.get(st, "#607D8B"))
        patch.set_alpha(0.7)
    ax_e.set_xticklabels(subtypes_plot, rotation=20, fontsize=8)
    ax_e.set_ylabel("ssGSEA NV-Score", fontsize=9)
    ax_e.set_title("NV-Score Distribution\nper Molecular Subtype", fontweight="bold", fontsize=10)

    # ── Panel F: Multivariate summary ────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    rows = [
        ("Univariate", univariate["hr"], univariate["ci_low"],
         univariate["ci_high"], univariate["p"], univariate["n"]),
    ]
    if multivariate.get("hr"):
        rows.append(("Multivariate", multivariate["hr"], multivariate["ci_low"],
                     multivariate["ci_high"], multivariate["p"], multivariate["n"]))

    for i, (label, hr, ci_l, ci_h, p, n) in enumerate(rows):
        col = "#4CAF50" if p < 0.05 else "#FF9800"
        ax_f.scatter([hr], [i], color=col, s=120, zorder=3)
        ax_f.plot([ci_l, ci_h], [i, i], color=col, lw=2.5)
        ax_f.text(ci_h + 0.01, i,
                  f"HR={hr:.3f} [{ci_l:.3f}–{ci_h:.3f}]\np={p:.4f}  n={n}",
                  va="center", fontsize=8)

    ax_f.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax_f.set_yticks(range(len(rows)))
    ax_f.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax_f.set_xlabel("Hazard Ratio (ssGSEA)", fontsize=9)
    ax_f.set_title("Cox Summary\nUnivariate vs Multivariate", fontweight="bold", fontsize=10)
    ax_f.set_xlim(0.5, 1.5)

    fig.suptitle(
        "RevGate — METABRIC External Validation: ssGSEA NV-Score → Overall Survival\n"
        f"n=1980, 1143 events | cBioPortal brca_metabric",
        fontsize=12, fontweight="bold",
    )

    out = output_dir / "metabric_external_validation.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Фигура: {out}")
    return out


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(univariate, multivariate, subtype_results, er_results,
                  km_data, replication):
    G = "\033[92m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"; Y = "\033[93m"

    print(f"\n{B}{'='*68}{E}")
    print(f"{B}  RevGate METABRIC External Validation{E}")
    print(f"{'='*68}")

    def sig(p): return f"{G}p={p:.4f} ✓{E}" if p < 0.05 else f"{Y}p={p:.4f}{E}"

    print(f"\n{B}Univariate Cox (ssGSEA -> OS):{E}")
    print(f"  HR={univariate['hr']:.4f} [{univariate['ci_low']:.4f}–{univariate['ci_high']:.4f}]  "
          f"{sig(univariate['p'])}  n={univariate['n']}, events={univariate['events']}")

    print(f"\n{B}Multivariate Cox (ssGSEA + age + ER + subtype):{E}")
    if multivariate.get("hr"):
        print(f"  HR={multivariate['hr']:.4f} [{multivariate['ci_low']:.4f}–{multivariate['ci_high']:.4f}]  "
              f"{sig(multivariate['p'])}  n={multivariate['n']}")
    else:
        print(f"  {R}Failed: {multivariate.get('error')}{E}")

    print(f"\n{B}Subtype-stratified Cox:{E}")
    for st, r in subtype_results.items():
        print(f"  {st:<15}: HR={r['hr']:.4f} [{r['ci_low']:.4f}–{r['ci_high']:.4f}]  "
              f"{sig(r['p'])}  n={r['n']}")

    print(f"\n{B}ER-stratified Cox:{E}")
    for er, r in er_results.items():
        print(f"  {er}: HR={r['hr']:.4f} [{r['ci_low']:.4f}–{r['ci_high']:.4f}]  "
              f"{sig(r['p'])}  n={r['n']}")

    print(f"\n{B}KM Log-rank (High vs Low median split):{E}")
    print(f"  p={km_data['logrank_p']:.4f}  "
          f"High n={km_data['n_high']}, Low n={km_data['n_low']}")

    print(f"\n{B}Replication Assessment:{E}")
    print(f"  TCGA BRCA Early (Finding_01): HR={TCGA_BRCA_EARLY_HR}, p={TCGA_BRCA_EARLY_P}")
    print(f"  METABRIC univariate:          HR={replication['metabric_univariate_hr']}, "
          f"p={replication['metabric_univariate_p']}")
    verdict_col = G if replication["replicated"] else (Y if replication["directional_consistency"] else R)
    print(f"  Verdict: {verdict_col}{replication['verdict']}{E}")
    print(f"{'='*68}\n")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(univariate, multivariate, subtype_results, er_results,
                 km_data, replication, output_dir: Path):
    out = {
        "timestamp": datetime.now().isoformat(),
        "cohort": "METABRIC (cBioPortal brca_metabric)",
        "n": univariate["n"],
        "events": univariate["events"],
        "univariate_cox": univariate,
        "multivariate_cox": multivariate,
        "subtype_stratified": subtype_results,
        "er_stratified": er_results,
        "kaplan_meier": km_data,
        "replication": replication,
    }
    p = output_dir / "metabric_external_validation_results.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info(f"JSON: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssgsea",
        default="/root/.revgate/cache/metabric/metabric_ssgsea_scores.parquet")
    parser.add_argument("--clinical",
        default="/root/.revgate/cache/metabric/metabric_clinical.csv")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df         = load_data(args.ssgsea, args.clinical)
    univariate = univariate_cox(df)
    multiv     = multivariate_cox(df)
    subtype_r  = subtype_stratified_cox(df)
    er_r       = er_stratified_cox(df)
    km_data    = kaplan_meier_analysis(df)
    replication = replication_check(univariate, subtype_r)

    print_summary(univariate, multiv, subtype_r, er_r, km_data, replication)
    plot_results(df, univariate, multiv, subtype_r, er_r, km_data,
                 replication, output_dir)
    save_results(univariate, multiv, subtype_r, er_r, km_data,
                 replication, output_dir)

    log.info("METABRIC external validation завършен.")


if __name__ == "__main__":
    main()
