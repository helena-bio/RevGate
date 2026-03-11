#!/usr/bin/env python3
"""
RevGate — Finding 12: BRCA Subtype Stratification with Restricted Follow-up
============================================================================
Адресира: Тончева §2.2, LumA парадокс от Finding 11, ER+/ER- разделение.

Въпроси:
  1. NV-Score ефектът специфичен ли е за ER+ тумори?
  2. LumA 36mo HR=1.784 артефакт ли е (37 events) или реален?
  3. Пролиферативният подтип (THREEGENE) обяснява ли посоката?
  4. Как се държи ефектът при HER2+ vs HER2-?

Анализи:
  A. ER-stratified Cox с restricted follow-up (36, 48, 60, 84, 120mo)
  B. PAM50 subtype Cox с restricted follow-up (36, 60, 84, 120mo)
  C. THREEGENE stratification (ER+/HER2- Low vs High Prolif)
  D. HER2-stratified Cox
  E. ER+ subtype decomposition (LumA vs LumB within ER+)
  F. Interaction: NV-Score × ER status (formal test)

Данни: METABRIC n=1,979
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
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brca_subtype")

file_handler = logging.FileHandler("results/brca_subtype_stratification.log", mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(file_handler)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(ssgsea_path: str, clinical_path: str) -> pd.DataFrame:
    """Merge ssGSEA + clinical."""
    log.info("Зареждане на данни...")

    ssgsea = pd.read_parquet(ssgsea_path)
    clin = pd.read_csv(clinical_path)
    df = ssgsea.merge(clin, left_on="sample_id", right_on="patientId", how="inner")

    df["os_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["os_event"] = df["OS_STATUS"].map({"1:DECEASED": 1, "0:LIVING": 0})
    df["age"] = pd.to_numeric(df["AGE_AT_DIAGNOSIS"], errors="coerce")
    df["er_positive"] = df["ER_IHC"].map({"Positve": 1, "Negative": 0})
    df["her2_gain"] = df["HER2_SNP6"].map({"GAIN": 1, "NEUTRAL": 0, "LOSS": 0, "UNDEF": np.nan})
    df["subtype"] = df["CLAUDIN_SUBTYPE"].replace({"NC": np.nan, "Normal": np.nan})
    df["threegene"] = df["THREEGENE"]
    df["npi"] = pd.to_numeric(df["NPI"], errors="coerce")

    # Z-score
    df["nv_zscore"] = StandardScaler().fit_transform(df[["ssgsea_nv_score"]])

    df = df.dropna(subset=["os_months", "os_event", "ssgsea_nv_score"])
    df = df[df["os_months"] > 0]

    log.info(f"  Cohort: n={len(df)}, events={int(df['os_event'].sum())}")
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────

def restricted(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    out = df.copy()
    mask = out["os_months"] > cutoff
    out.loc[mask, "os_months"] = cutoff
    out.loc[mask, "os_event"] = 0
    return out


def cox_univariate(df: pd.DataFrame, label: str = "", penalizer: float = 0.01) -> dict:
    """Univariate Cox: nv_zscore + age -> OS."""
    sub = df[["nv_zscore", "age", "os_months", "os_event"]].dropna()
    sub = sub[sub["os_months"] > 0]

    if len(sub) < 30 or sub["os_event"].sum() < 15:
        return {"label": label, "n": len(sub), "events": int(sub["os_event"].sum()),
                "error": "insufficient_data"}

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(sub, duration_col="os_months", event_col="os_event")

    s = cph.summary
    return {
        "label": label,
        "n": len(sub),
        "events": int(sub["os_event"].sum()),
        "hr": round(float(s.loc["nv_zscore", "exp(coef)"]), 4),
        "ci_low": round(float(s.loc["nv_zscore", "exp(coef) lower 95%"]), 4),
        "ci_high": round(float(s.loc["nv_zscore", "exp(coef) upper 95%"]), 4),
        "p": round(float(s.loc["nv_zscore", "p"]), 6),
        "concordance": round(float(cph.concordance_index_), 4),
    }


def cox_interaction_er(df: pd.DataFrame, label: str = "") -> dict:
    """Cox с NV × ER interaction term."""
    sub = df[["nv_zscore", "er_positive", "age", "os_months", "os_event"]].dropna()
    sub = sub[sub["os_months"] > 0]
    sub["nv_x_er"] = sub["nv_zscore"] * sub["er_positive"]

    if len(sub) < 50:
        return {"label": label, "error": "insufficient_data", "n": len(sub)}

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(sub, duration_col="os_months", event_col="os_event")
    s = cph.summary

    result = {"label": label, "n": len(sub), "events": int(sub["os_event"].sum()),
              "concordance": round(float(cph.concordance_index_), 4)}

    for var in ["nv_zscore", "er_positive", "nv_x_er", "age"]:
        if var in s.index:
            result[f"{var}_hr"] = round(float(s.loc[var, "exp(coef)"]), 4)
            result[f"{var}_p"] = round(float(s.loc[var, "p"]), 6)

    return result


def sig_marker(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.10: return "~"
    return ""


# ── Analysis A: ER-stratified restricted follow-up ───────────────────────────

def analysis_er_restricted(df: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("ANALYSIS A: ER-Stratified × Restricted Follow-up")
    log.info("=" * 60)

    results = {}
    cutoffs = [36, 48, 60, 84, 120, 180]

    for er_val, er_name in [(1, "ER+"), (0, "ER-")]:
        results[er_name] = {}
        sub = df[df["er_positive"] == er_val]

        for cutoff in cutoffs:
            r_df = restricted(sub, cutoff)
            r = cox_univariate(r_df, label=f"{er_name} {cutoff}mo")
            results[er_name][f"{cutoff}mo"] = r

            if "error" not in r:
                sm = sig_marker(r["p"])
                log.info(f"  {er_name} {cutoff:3d}mo: HR={r['hr']:.4f} "
                         f"[{r['ci_low']:.4f}-{r['ci_high']:.4f}] p={r['p']:.6f} "
                         f"n={r['n']} ev={r['events']} {sm}")
            else:
                log.info(f"  {er_name} {cutoff:3d}mo: {r.get('error')} (n={r['n']}, ev={r['events']})")

        # Full follow-up
        r = cox_univariate(sub, label=f"{er_name} full")
        results[er_name]["full"] = r
        if "error" not in r:
            log.info(f"  {er_name} full:  HR={r['hr']:.4f} p={r['p']:.6f} n={r['n']} {sig_marker(r['p'])}")

    return results


# ── Analysis B: PAM50 subtype restricted follow-up ───────────────────────────

def analysis_subtype_restricted(df: pd.DataFrame) -> dict:
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS B: PAM50 Subtype × Restricted Follow-up")
    log.info("=" * 60)

    results = {}
    subtypes = ["LumA", "LumB", "Her2", "Basal", "claudin-low"]
    cutoffs = [36, 60, 84, 120]

    for st in subtypes:
        results[st] = {}
        sub = df[df["subtype"] == st]

        for cutoff in cutoffs:
            r_df = restricted(sub, cutoff)
            r = cox_univariate(r_df, label=f"{st} {cutoff}mo")
            results[st][f"{cutoff}mo"] = r

            if "error" not in r:
                sm = sig_marker(r["p"])
                log.info(f"  {st:<12} {cutoff:3d}mo: HR={r['hr']:.4f} "
                         f"[{r['ci_low']:.4f}-{r['ci_high']:.4f}] p={r['p']:.6f} "
                         f"n={r['n']} ev={r['events']} {sm}")
            else:
                log.info(f"  {st:<12} {cutoff:3d}mo: {r.get('error')} (n={r['n']}, ev={r['events']})")

        # Full
        r = cox_univariate(sub, label=f"{st} full")
        results[st]["full"] = r
        if "error" not in r:
            log.info(f"  {st:<12} full:  HR={r['hr']:.4f} p={r['p']:.6f} {sig_marker(r['p'])}")

    return results


# ── Analysis C: THREEGENE stratification ─────────────────────────────────────

def analysis_threegene(df: pd.DataFrame) -> dict:
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS C: THREEGENE Stratification")
    log.info("  ER+/HER2- Low Prolif vs High Prolif — пролиферативна хипотеза")
    log.info("=" * 60)

    results = {}
    groups = ["ER+/HER2- Low Prolif", "ER+/HER2- High Prolif", "ER-/HER2-", "HER2+"]
    cutoffs = [36, 60, 84, 120]

    for grp in groups:
        results[grp] = {}
        sub = df[df["threegene"] == grp]

        for cutoff in cutoffs:
            r_df = restricted(sub, cutoff)
            r = cox_univariate(r_df, label=f"{grp} {cutoff}mo")
            results[grp][f"{cutoff}mo"] = r

            if "error" not in r:
                sm = sig_marker(r["p"])
                log.info(f"  {grp:<28} {cutoff:3d}mo: HR={r['hr']:.4f} "
                         f"p={r['p']:.6f} n={r['n']} ev={r['events']} {sm}")
            else:
                log.info(f"  {grp:<28} {cutoff:3d}mo: {r.get('error')} (n={r['n']}, ev={r['events']})")

    return results


# ── Analysis D: HER2-stratified ──────────────────────────────────────────────

def analysis_her2_stratified(df: pd.DataFrame) -> dict:
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS D: HER2-Stratified × Restricted Follow-up")
    log.info("=" * 60)

    results = {}
    cutoffs = [36, 60, 84]

    for her2_val, her2_name in [(1, "HER2+"), (0, "HER2-")]:
        results[her2_name] = {}
        sub = df[df["her2_gain"] == her2_val]

        for cutoff in cutoffs:
            r_df = restricted(sub, cutoff)
            r = cox_univariate(r_df, label=f"{her2_name} {cutoff}mo")
            results[her2_name][f"{cutoff}mo"] = r

            if "error" not in r:
                sm = sig_marker(r["p"])
                log.info(f"  {her2_name} {cutoff:3d}mo: HR={r['hr']:.4f} p={r['p']:.6f} "
                         f"n={r['n']} ev={r['events']} {sm}")

    return results


# ── Analysis E: ER+ decomposition (LumA vs LumB within ER+) ─────────────────

def analysis_erpos_decomposition(df: pd.DataFrame) -> dict:
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS E: ER+ Subtype Decomposition (LumA vs LumB)")
    log.info("=" * 60)

    results = {}
    erpos = df[df["er_positive"] == 1]
    cutoffs = [36, 60, 84, 120]

    for st in ["LumA", "LumB"]:
        results[st] = {}
        sub = erpos[erpos["subtype"] == st]

        for cutoff in cutoffs:
            r_df = restricted(sub, cutoff)
            r = cox_univariate(r_df, label=f"ER+ {st} {cutoff}mo")
            results[st][f"{cutoff}mo"] = r

            if "error" not in r:
                sm = sig_marker(r["p"])
                log.info(f"  ER+ {st:<6} {cutoff:3d}mo: HR={r['hr']:.4f} p={r['p']:.6f} "
                         f"n={r['n']} ev={r['events']} {sm}")
            else:
                log.info(f"  ER+ {st:<6} {cutoff:3d}mo: {r.get('error')} (n={r['n']}, ev={r['events']})")

    return results


# ── Analysis F: NV × ER interaction ──────────────────────────────────────────

def analysis_nv_er_interaction(df: pd.DataFrame) -> dict:
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS F: NV-Score × ER Status Interaction")
    log.info("=" * 60)

    results = {}
    cutoffs = [36, 60, 84, 120]

    for cutoff in cutoffs:
        r_df = restricted(df, cutoff)
        r = cox_interaction_er(r_df, label=f"NV×ER {cutoff}mo")
        results[f"{cutoff}mo"] = r

        if "error" not in r:
            int_hr = r.get("nv_x_er_hr", 1.0)
            int_p = r.get("nv_x_er_p", 1.0)
            nv_hr = r.get("nv_zscore_hr", 1.0)
            sm = sig_marker(int_p)
            log.info(f"  {cutoff:3d}mo: NV HR={nv_hr:.4f}, NV×ER HR={int_hr:.4f}, "
                     f"int_p={int_p:.6f} {sm}  (n={r['n']}, ev={r['events']})")

    # Full
    r = cox_interaction_er(df, label="NV×ER full")
    results["full"] = r
    if "error" not in r:
        log.info(f"  full: NV×ER HR={r.get('nv_x_er_hr', 'NA')}, p={r.get('nv_x_er_p', 'NA')}")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(er_results, subtype_results, threegene_results,
                 her2_results, erpos_decomp, nv_er_interaction,
                 output_dir: Path) -> Path:
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)

    # ── Panel A: ER+ vs ER- HR gradient ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for er_name, col, marker in [("ER+", "#4CAF50", "o"), ("ER-", "#F44336", "s")]:
        cutoffs_plot = []
        hrs_plot = []
        for key in ["36mo", "48mo", "60mo", "84mo", "120mo", "180mo"]:
            r = er_results.get(er_name, {}).get(key, {})
            if "error" not in r and r:
                months = int(key.replace("mo", ""))
                cutoffs_plot.append(months)
                hrs_plot.append(r["hr"])
        if cutoffs_plot:
            ax_a.plot(cutoffs_plot, hrs_plot, f"{marker}-", color=col, lw=2,
                      label=f"{er_name}", markersize=7)
    ax_a.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax_a.set_xlabel("Follow-up Cutoff (Months)")
    ax_a.set_ylabel("Hazard Ratio (NV-Score)")
    ax_a.set_title("ER+ vs ER-\nHR gradient by follow-up", fontweight="bold", fontsize=10)
    ax_a.legend(fontsize=9)
    ax_a.set_ylim(0.5, 1.5)

    # ── Panel B: PAM50 subtype heatmap ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    subtypes = ["LumA", "LumB", "Her2", "Basal", "claudin-low"]
    cutoffs_sub = ["36mo", "60mo", "84mo", "120mo"]

    hr_matrix = []
    for st in subtypes:
        row = []
        for c in cutoffs_sub:
            r = subtype_results.get(st, {}).get(c, {})
            row.append(r.get("hr", np.nan))
        hr_matrix.append(row)

    hr_arr = np.array(hr_matrix)
    im = ax_b.imshow(hr_arr, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
    ax_b.set_xticks(range(len(cutoffs_sub)))
    ax_b.set_xticklabels(cutoffs_sub, fontsize=9)
    ax_b.set_yticks(range(len(subtypes)))
    ax_b.set_yticklabels(subtypes, fontsize=9)

    for i in range(len(subtypes)):
        for j in range(len(cutoffs_sub)):
            val = hr_arr[i, j]
            r = subtype_results.get(subtypes[i], {}).get(cutoffs_sub[j], {})
            p = r.get("p", 1.0)
            sm = sig_marker(p) if p else ""
            if not np.isnan(val):
                ax_b.text(j, i, f"{val:.2f}{sm}", ha="center", va="center",
                         fontsize=8, fontweight="bold" if p and p < 0.05 else "normal")

    plt.colorbar(im, ax=ax_b, label="HR", shrink=0.8)
    ax_b.set_title("PAM50 Subtype × Follow-up\nHR heatmap", fontweight="bold", fontsize=10)

    # ── Panel C: THREEGENE — Low vs High Prolif ──────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    tg_groups = ["ER+/HER2- Low Prolif", "ER+/HER2- High Prolif", "ER-/HER2-", "HER2+"]
    tg_colors = ["#2E7D32", "#81C784", "#E53935", "#FF9800"]

    for grp, col in zip(tg_groups, tg_colors):
        cutoffs_plot = []
        hrs_plot = []
        for key in ["36mo", "60mo", "84mo", "120mo"]:
            r = threegene_results.get(grp, {}).get(key, {})
            if "error" not in r and r:
                cutoffs_plot.append(int(key.replace("mo", "")))
                hrs_plot.append(r["hr"])
        if cutoffs_plot:
            short_label = grp.replace("ER+/HER2- ", "").replace("ER-/HER2-", "ER-/HER2-")
            ax_c.plot(cutoffs_plot, hrs_plot, "o-", color=col, lw=2,
                      label=short_label, markersize=6)

    ax_c.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax_c.set_xlabel("Follow-up Cutoff (Months)")
    ax_c.set_ylabel("HR")
    ax_c.set_title("THREEGENE Classification\nLow vs High Proliferation",
                    fontweight="bold", fontsize=10)
    ax_c.legend(fontsize=7)
    ax_c.set_ylim(0.4, 2.0)

    # ── Panel D: ER+ LumA vs LumB ───────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    for st, col, marker in [("LumA", "#1565C0", "o"), ("LumB", "#42A5F5", "s")]:
        cutoffs_plot = []
        hrs_plot = []
        for key in ["36mo", "60mo", "84mo", "120mo"]:
            r = erpos_decomp.get(st, {}).get(key, {})
            if "error" not in r and r:
                cutoffs_plot.append(int(key.replace("mo", "")))
                hrs_plot.append(r["hr"])
        if cutoffs_plot:
            ax_d.plot(cutoffs_plot, hrs_plot, f"{marker}-", color=col, lw=2,
                      label=f"ER+ {st}", markersize=7)

    ax_d.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax_d.set_xlabel("Follow-up Cutoff (Months)")
    ax_d.set_ylabel("HR")
    ax_d.set_title("ER+ Decomposition\nLumA vs LumB within ER+",
                    fontweight="bold", fontsize=10)
    ax_d.legend(fontsize=9)

    # ── Panel E: HER2+ vs HER2- ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    for her2_name, col in [("HER2+", "#FF6F00"), ("HER2-", "#1565C0")]:
        cutoffs_plot = []
        hrs_plot = []
        for key in ["36mo", "60mo", "84mo"]:
            r = her2_results.get(her2_name, {}).get(key, {})
            if "error" not in r and r:
                cutoffs_plot.append(int(key.replace("mo", "")))
                hrs_plot.append(r["hr"])
        if cutoffs_plot:
            ax_e.plot(cutoffs_plot, hrs_plot, "o-", color=col, lw=2,
                      label=her2_name, markersize=7)

    ax_e.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax_e.set_xlabel("Follow-up Cutoff (Months)")
    ax_e.set_ylabel("HR")
    ax_e.set_title("HER2-Stratified\nNV-Score HR by HER2 status",
                    fontweight="bold", fontsize=10)
    ax_e.legend(fontsize=9)

    # ── Panel F: NV × ER interaction over time ───────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    cutoffs_plot = []
    int_hrs = []
    int_ps = []
    nv_hrs = []

    for key in ["36mo", "60mo", "84mo", "120mo"]:
        r = nv_er_interaction.get(key, {})
        if "error" not in r and r:
            cutoffs_plot.append(int(key.replace("mo", "")))
            int_hrs.append(r.get("nv_x_er_hr", 1.0))
            int_ps.append(r.get("nv_x_er_p", 1.0))
            nv_hrs.append(r.get("nv_zscore_hr", 1.0))

    if cutoffs_plot:
        ax_f.plot(cutoffs_plot, int_hrs, "o-", color="#E91E63", lw=2,
                  label="NV×ER interaction", markersize=8)
        ax_f.plot(cutoffs_plot, nv_hrs, "s--", color="#2196F3", lw=1.5,
                  label="NV main effect", markersize=6)
        for x, p in zip(cutoffs_plot, int_ps):
            if p < 0.05:
                idx = cutoffs_plot.index(x)
                ax_f.annotate("*", (x, int_hrs[idx]),
                             fontsize=16, ha="center", va="bottom", color="#E91E63")

    ax_f.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax_f.set_xlabel("Follow-up Cutoff (Months)")
    ax_f.set_ylabel("HR")
    ax_f.set_title("NV × ER Interaction\nover follow-up duration",
                    fontweight="bold", fontsize=10)
    ax_f.legend(fontsize=8)

    # ── Panel G: Summary forest plot — key strata at 60mo ───────────────────
    ax_g = fig.add_subplot(gs[2, :])
    forest_data = []

    # Collect key 60mo results
    strata = [
        ("ER+ 60mo", er_results.get("ER+", {}).get("60mo", {}), "#4CAF50"),
        ("ER- 60mo", er_results.get("ER-", {}).get("60mo", {}), "#F44336"),
        ("LumA 60mo", subtype_results.get("LumA", {}).get("60mo", {}), "#1565C0"),
        ("LumB 60mo", subtype_results.get("LumB", {}).get("60mo", {}), "#42A5F5"),
        ("Her2 60mo", subtype_results.get("Her2", {}).get("60mo", {}), "#FF6F00"),
        ("Basal 60mo", subtype_results.get("Basal", {}).get("60mo", {}), "#9C27B0"),
        ("Low Prolif 60mo", threegene_results.get("ER+/HER2- Low Prolif", {}).get("60mo", {}), "#2E7D32"),
        ("High Prolif 60mo", threegene_results.get("ER+/HER2- High Prolif", {}).get("60mo", {}), "#81C784"),
    ]

    valid_strata = [(label, r, col) for label, r, col in strata if r and "error" not in r]

    for i, (label, r, col) in enumerate(valid_strata):
        ax_g.scatter([r["hr"]], [i], color=col, s=100, zorder=3)
        ax_g.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, lw=2.5)
        sm = sig_marker(r["p"])
        ax_g.text(r["ci_high"] + 0.02, i,
                  f"HR={r['hr']:.3f} [{r['ci_low']:.3f}-{r['ci_high']:.3f}] "
                  f"p={r['p']:.4f} n={r['n']} {sm}",
                  va="center", fontsize=8)

    ax_g.axvline(1.0, color="black", ls="--", lw=1)
    ax_g.set_yticks(range(len(valid_strata)))
    ax_g.set_yticklabels([v[0] for v in valid_strata], fontsize=9)
    ax_g.set_xlabel("Hazard Ratio (NV-Score → OS)", fontsize=10)
    ax_g.set_title("Summary Forest Plot — Key Strata at 60 Months",
                    fontweight="bold", fontsize=11)

    fig.suptitle(
        "RevGate Finding 12 — BRCA Subtype Stratification with Restricted Follow-up\n"
        "METABRIC n=1,979 | PAM50, ER, HER2, THREEGENE, NV×ER interaction",
        fontsize=13, fontweight="bold",
    )

    out = output_dir / "brca_subtype_stratification.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Фигура: {out}")
    return out


# ── Print summary ────────────────────────────────────────────────────────────

def print_summary(er_res, sub_res, tg_res, her2_res, erp_res, inter_res):
    G = "\033[92m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"; Y = "\033[93m"

    print(f"\n{B}{'='*72}{E}")
    print(f"{B}  RevGate Finding 12 — BRCA Subtype Stratification{E}")
    print(f"{'='*72}")

    def fmt(r):
        if not r or "error" in r:
            return "n/a"
        p = r["p"]
        col = G if p < 0.05 else (Y if p < 0.10 else "")
        end = E if col else ""
        sm = sig_marker(p)
        return f"{col}HR={r['hr']:.3f} p={p:.4f} n={r['n']}{sm}{end}"

    print(f"\n{B}A. ER-Stratified × Restricted Follow-up:{E}")
    for er_name in ["ER+", "ER-"]:
        line = f"  {er_name}: "
        for c in ["36mo", "60mo", "84mo", "120mo", "full"]:
            r = er_res.get(er_name, {}).get(c, {})
            line += f"{c}={fmt(r)}  "
        print(line)

    print(f"\n{B}B. PAM50 Subtype × Restricted Follow-up:{E}")
    for st in ["LumA", "LumB", "Her2", "Basal", "claudin-low"]:
        line = f"  {st:<12}: "
        for c in ["36mo", "60mo", "84mo", "120mo"]:
            r = sub_res.get(st, {}).get(c, {})
            line += f"{c}={fmt(r)}  "
        print(line)

    print(f"\n{B}C. THREEGENE (Low vs High Proliferation):{E}")
    for grp in ["ER+/HER2- Low Prolif", "ER+/HER2- High Prolif", "ER-/HER2-", "HER2+"]:
        line = f"  {grp:<28}: "
        for c in ["36mo", "60mo", "84mo"]:
            r = tg_res.get(grp, {}).get(c, {})
            line += f"{c}={fmt(r)}  "
        print(line)

    print(f"\n{B}E. ER+ Decomposition (LumA vs LumB):{E}")
    for st in ["LumA", "LumB"]:
        line = f"  ER+ {st}: "
        for c in ["36mo", "60mo", "84mo", "120mo"]:
            r = erp_res.get(st, {}).get(c, {})
            line += f"{c}={fmt(r)}  "
        print(line)

    print(f"\n{B}F. NV × ER Interaction:{E}")
    for c in ["36mo", "60mo", "84mo", "120mo", "full"]:
        r = inter_res.get(c, {})
        if r and "error" not in r:
            int_hr = r.get("nv_x_er_hr", 1.0)
            int_p = r.get("nv_x_er_p", 1.0)
            nv_hr = r.get("nv_zscore_hr", 1.0)
            sm = sig_marker(int_p)
            print(f"  {c:>5}: NV HR={nv_hr:.4f}, NV×ER HR={int_hr:.4f}, int_p={int_p:.6f} {sm}")

    print(f"{'='*72}\n")


# ── Save ─────────────────────────────────────────────────────────────────────

def save_results(er_res, sub_res, tg_res, her2_res, erp_res, inter_res, output_dir: Path):
    out = {
        "timestamp": datetime.now().isoformat(),
        "finding": "Finding_12",
        "description": "BRCA Subtype Stratification with Restricted Follow-up",
        "cohort": "METABRIC n=1979",
        "er_stratified": er_res,
        "pam50_subtype": sub_res,
        "threegene": tg_res,
        "her2_stratified": her2_res,
        "erpos_decomposition": erp_res,
        "nv_er_interaction": inter_res,
    }
    p = output_dir / "brca_subtype_stratification_results.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"JSON: {p}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Finding 12: BRCA Subtype Stratification")
    parser.add_argument("--ssgsea",
                        default="/root/.revgate/cache/metabric/metabric_ssgsea_scores.parquet")
    parser.add_argument("--clinical",
                        default="/root/.revgate/cache/metabric/metabric_clinical.csv")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.ssgsea, args.clinical)

    er_res = analysis_er_restricted(df)
    sub_res = analysis_subtype_restricted(df)
    tg_res = analysis_threegene(df)
    her2_res = analysis_her2_stratified(df)
    erp_res = analysis_erpos_decomposition(df)
    inter_res = analysis_nv_er_interaction(df)

    print_summary(er_res, sub_res, tg_res, her2_res, erp_res, inter_res)
    plot_results(er_res, sub_res, tg_res, her2_res, erp_res, inter_res, output_dir)
    save_results(er_res, sub_res, tg_res, her2_res, erp_res, inter_res, output_dir)

    log.info("Finding 12 завършен.")


if __name__ == "__main__":
    main()
