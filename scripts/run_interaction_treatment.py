#!/usr/bin/env python3
"""
RevGate — Finding 11: NV-Score × Treatment Interaction Test
============================================================
Критичен анализ: преминаване от PROGNOSTIC към PREDICTIVE biomarker.

Хипотеза (Тончева §5.1):
  При BRCA METABRIC, пациентите с висок NV-Score имат допълнителна полза
  от ендокринна терапия (HORMONE_THERAPY) над и извън основния прогностичен
  ефект. Ако interaction p < 0.05 — NV-Score е PREDICTIVE, не само prognostic.

Модел (Cox PH):
  h(t) = h₀(t) · exp(β₁·NV + β₂·HT + β₃·NV×HT + γ·Covariates)
  β₃ < 0 и p < 0.05 → пациентите с висок NV-Score + HT имат допълнителна полза.

Анализи:
  1. Interaction test: NV-Score × HORMONE_THERAPY (Cox PH)
  2. Interaction test: NV-Score × CHEMOTHERAPY (Cox PH)
  3. Treatment-stratified Cox: HR при HT-only vs CHEMO-only vs HT+CHEMO
  4. PAM50 субтип стратификация с restricted follow-up (36mo, 60mo, 84mo)
  5. ER-stratified interaction: ER+ и ER- поотделно
  6. Sensitivity: restricted follow-up (36, 60, 84 месеца)

Данни: METABRIC (cBioPortal brca_metabric), n=1980
ssGSEA scores: предварително изчислени (Finding 01/09)

Helena Bioinformatics, 2026.
Biological hypothesis: Academician Draga Toncheva, DSc & Academician Vassil Sgurev, DSc, BAS.
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
log = logging.getLogger("interaction_treatment")

# Файлов лог
file_handler = logging.FileHandler("results/interaction_treatment.log", mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(file_handler)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_merge(ssgsea_path: str, clinical_path: str) -> pd.DataFrame:
    """Merge ssGSEA scores + clinical data с treatment annotation."""
    log.info("Зареждане на данни...")

    ssgsea = pd.read_parquet(ssgsea_path)
    clin = pd.read_csv(clinical_path)

    df = ssgsea.merge(clin, left_on="sample_id", right_on="patientId", how="inner")

    # OS
    df["os_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["os_event"] = df["OS_STATUS"].map({"1:DECEASED": 1, "0:LIVING": 0})

    # Age
    df["age"] = pd.to_numeric(df["AGE_AT_DIAGNOSIS"], errors="coerce")

    # ER status binary
    df["er_positive"] = df["ER_IHC"].map({"Positve": 1, "Negative": 0})

    # HER2 status
    df["her2_positive"] = df["HER2_SNP6"].map({
        "GAIN": 1, "NEUTRAL": 0, "LOSS": 0, "UNDEF": np.nan
    })

    # Treatment binary
    df["chemotherapy"] = df["CHEMOTHERAPY"].map({"YES": 1, "NO": 0})
    df["radio_therapy"] = df["RADIO_THERAPY"].map({"YES": 1, "NO": 0})

    # PAM50 subtype
    df["subtype"] = df["CLAUDIN_SUBTYPE"].replace("NC", np.nan)

    # Lymph nodes
    df["lymph_nodes_pos"] = pd.to_numeric(
        df["LYMPH_NODES_EXAMINED_POSITIVE"], errors="coerce"
    )

    # NPI (Nottingham Prognostic Index) — proxy за stage
    df["npi"] = pd.to_numeric(df["NPI"], errors="coerce")

    # ssGSEA z-score (стандартизация)
    df["nv_zscore"] = StandardScaler().fit_transform(df[["ssgsea_nv_score"]])

    # Treatment groups
    df["treatment_group"] = "Other"
    ht = df["hormone_therapy"] == 1
    ct = df["chemotherapy"] == 1
    df.loc[ht & ~ct, "treatment_group"] = "HT_only"
    df.loc[~ht & ct, "treatment_group"] = "Chemo_only"
    df.loc[ht & ct, "treatment_group"] = "HT+Chemo"
    df.loc[~ht & ~ct, "treatment_group"] = "Neither"

    # Филтриране
    df = df.dropna(subset=["os_months", "os_event", "ssgsea_nv_score"])
    df = df[df["os_months"] > 0]

    log.info(f"  Cohort: n={len(df)}, events={int(df['os_event'].sum())}")
    log.info(f"  Treatment groups:")
    for grp, cnt in df["treatment_group"].value_counts().items():
        events = int(df.loc[df["treatment_group"] == grp, "os_event"].sum())
        log.info(f"    {grp}: n={cnt}, events={events}")
    log.info(f"  ER+: {int(df['er_positive'].sum())}, ER-: {int((df['er_positive']==0).sum())}")
    log.info(f"  Subtypes: {df['subtype'].value_counts().to_dict()}")

    return df


# ── Restricted follow-up helper ──────────────────────────────────────────────

def apply_restricted_followup(df: pd.DataFrame, cutoff_months: float) -> pd.DataFrame:
    """Administrative censoring at cutoff."""
    out = df.copy()
    mask = out["os_months"] > cutoff_months
    out.loc[mask, "os_months"] = cutoff_months
    out.loc[mask, "os_event"] = 0
    return out


# ── Cox helpers ──────────────────────────────────────────────────────────────

def fit_cox_interaction(df: pd.DataFrame, treatment_col: str,
                        covariates: list = None,
                        penalizer: float = 0.01,
                        label: str = "") -> dict:
    """
    Cox PH с interaction term: NV-Score × Treatment.

    Model: OS ~ nv_zscore + treatment + nv_zscore:treatment + covariates
    """
    sub = df.copy()

    # Interaction term
    sub["nv_x_treatment"] = sub["nv_zscore"] * sub[treatment_col]

    # Covariates
    cols = ["nv_zscore", treatment_col, "nv_x_treatment"]
    if covariates:
        cols.extend(covariates)
    cols += ["os_months", "os_event"]

    sub = sub[cols].dropna()
    sub = sub[sub["os_months"] > 0]

    if len(sub) < 50 or sub["os_event"].sum() < 20:
        return {"label": label, "n": len(sub), "events": int(sub["os_event"].sum()),
                "error": "insufficient_data"}

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(sub, duration_col="os_months", event_col="os_event")

    s = cph.summary
    result = {"label": label, "n": len(sub), "events": int(sub["os_event"].sum()),
              "concordance": round(float(cph.concordance_index_), 4)}

    # Извличаме всички коефициенти
    for var in ["nv_zscore", treatment_col, "nv_x_treatment"]:
        if var in s.index:
            result[f"{var}_hr"] = round(float(s.loc[var, "exp(coef)"]), 4)
            result[f"{var}_ci_low"] = round(float(s.loc[var, "exp(coef) lower 95%"]), 4)
            result[f"{var}_ci_high"] = round(float(s.loc[var, "exp(coef) upper 95%"]), 4)
            result[f"{var}_p"] = round(float(s.loc[var, "p"]), 6)
            result[f"{var}_coef"] = round(float(s.loc[var, "coef"]), 4)

    return result


def fit_cox_stratified(df: pd.DataFrame, label: str = "",
                       penalizer: float = 0.01) -> dict:
    """Univariate Cox: nv_zscore -> OS в subset."""
    sub = df[["nv_zscore", "os_months", "os_event", "age"]].dropna()
    sub = sub[sub["os_months"] > 0]

    if len(sub) < 30 or sub["os_event"].sum() < 10:
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
    }


# ── Analysis 1: Primary interaction test ─────────────────────────────────────

def analysis_interaction_primary(df: pd.DataFrame) -> dict:
    """Primary interaction: NV-Score × HORMONE_THERAPY, adjusted for age."""
    log.info("=" * 60)
    log.info("ANALYSIS 1: Primary Interaction Test")
    log.info("  Model: OS ~ NV + HT + NV×HT + age")
    log.info("=" * 60)

    result = fit_cox_interaction(
        df, "hormone_therapy", covariates=["age"],
        label="NV × Hormone Therapy (adjusted for age)"
    )

    int_p = result.get("nv_x_treatment_p", 1.0)
    int_hr = result.get("nv_x_treatment_hr", 1.0)
    nv_hr = result.get("nv_zscore_hr", 1.0)
    ht_hr = result.get("hormone_therapy_hr", 1.0)

    log.info(f"  NV main effect:   HR={nv_hr:.4f}, p={result.get('nv_zscore_p', 'NA')}")
    log.info(f"  HT main effect:   HR={ht_hr:.4f}, p={result.get('hormone_therapy_p', 'NA')}")
    log.info(f"  NV×HT interaction: HR={int_hr:.4f}, p={int_p:.6f}")
    log.info(f"  n={result['n']}, events={result['events']}, C={result.get('concordance', 'NA')}")

    if int_p < 0.05:
        log.info("  ★★★ INTERACTION SIGNIFICANT — NV-Score is PREDICTIVE ★★★")
    elif int_p < 0.10:
        log.info("  ★★ Interaction borderline (p<0.10) — suggestive ★★")
    else:
        log.info("  ○ Interaction not significant — NV-Score is prognostic only")

    return result


# ── Analysis 2: Chemo interaction ────────────────────────────────────────────

def analysis_interaction_chemo(df: pd.DataFrame) -> dict:
    """Secondary interaction: NV-Score × CHEMOTHERAPY."""
    log.info("\nANALYSIS 2: Chemotherapy Interaction Test")

    result = fit_cox_interaction(
        df, "chemotherapy", covariates=["age"],
        label="NV × Chemotherapy (adjusted for age)"
    )

    int_p = result.get("nv_x_treatment_p", 1.0)
    log.info(f"  NV×Chemo interaction: HR={result.get('nv_x_treatment_hr', 'NA')}, p={int_p:.6f}")
    return result


# ── Analysis 3: Treatment-stratified Cox ─────────────────────────────────────

def analysis_treatment_stratified(df: pd.DataFrame) -> dict:
    """Cox в всяка treatment group поотделно."""
    log.info("\nANALYSIS 3: Treatment-Stratified Cox")
    log.info("  HR per treatment group (NV-Score → OS)")

    results = {}
    groups = ["HT_only", "Chemo_only", "HT+Chemo", "Neither"]

    for grp in groups:
        sub = df[df["treatment_group"] == grp]
        r = fit_cox_stratified(sub, label=f"Treatment: {grp}")
        results[grp] = r

        if "error" not in r:
            sig = "✓" if r["p"] < 0.05 else ("~" if r["p"] < 0.10 else "○")
            log.info(f"  {grp:<12}: HR={r['hr']:.4f} [{r['ci_low']:.4f}–{r['ci_high']:.4f}] "
                     f"p={r['p']:.4f} n={r['n']} {sig}")
        else:
            log.info(f"  {grp:<12}: {r['error']} (n={r['n']})")

    return results


# ── Analysis 4: PAM50 subtype + restricted follow-up ────────────────────────

def analysis_subtype_restricted(df: pd.DataFrame) -> dict:
    """PAM50 субтип стратификация с restricted follow-up."""
    log.info("\nANALYSIS 4: PAM50 Subtype × Restricted Follow-up")

    results = {}
    subtypes = ["LumA", "LumB", "Her2", "Basal", "claudin-low"]
    cutoffs = [36, 60, 84]

    for st in subtypes:
        sub = df[df["subtype"] == st]
        results[st] = {}

        for cutoff in cutoffs:
            restricted = apply_restricted_followup(sub, cutoff)
            r = fit_cox_stratified(restricted, label=f"{st} {cutoff}mo")
            results[st][f"{cutoff}mo"] = r

            if "error" not in r:
                sig = "✓" if r["p"] < 0.05 else ("~" if r["p"] < 0.10 else "○")
                log.info(f"  {st:<12} {cutoff:3d}mo: HR={r['hr']:.4f} "
                         f"[{r['ci_low']:.4f}–{r['ci_high']:.4f}] p={r['p']:.4f} {sig}")
            else:
                log.info(f"  {st:<12} {cutoff:3d}mo: {r.get('error', 'NA')} (n={r['n']})")

    return results


# ── Analysis 5: ER-stratified interaction ────────────────────────────────────

def analysis_er_stratified_interaction(df: pd.DataFrame) -> dict:
    """Interaction test стратифициран по ER статус."""
    log.info("\nANALYSIS 5: ER-Stratified Interaction Test")

    results = {}
    for er_val, er_name in [(1, "ER+"), (0, "ER-")]:
        sub = df[df["er_positive"] == er_val]
        r = fit_cox_interaction(
            sub, "hormone_therapy", covariates=["age"],
            label=f"NV × HT in {er_name}"
        )
        results[er_name] = r

        int_p = r.get("nv_x_treatment_p", 1.0)
        int_hr = r.get("nv_x_treatment_hr", 1.0)
        log.info(f"  {er_name}: NV×HT HR={int_hr:.4f}, p={int_p:.6f}, n={r['n']}")

    return results


# ── Analysis 6: Restricted follow-up interaction ─────────────────────────────

def analysis_restricted_interaction(df: pd.DataFrame) -> dict:
    """Interaction test при различни follow-up cutoffs."""
    log.info("\nANALYSIS 6: Restricted Follow-up Interaction")

    results = {}
    cutoffs = [36, 48, 60, 84, 120]

    for cutoff in cutoffs:
        restricted = apply_restricted_followup(df, cutoff)
        r = fit_cox_interaction(
            restricted, "hormone_therapy", covariates=["age"],
            label=f"NV × HT at {cutoff}mo"
        )
        results[f"{cutoff}mo"] = r

        int_p = r.get("nv_x_treatment_p", 1.0)
        int_hr = r.get("nv_x_treatment_hr", 1.0)
        nv_hr = r.get("nv_zscore_hr", 1.0)
        sig = "✓" if int_p < 0.05 else ("~" if int_p < 0.10 else "○")
        log.info(f"  {cutoff:3d}mo: NV HR={nv_hr:.4f}, NV×HT HR={int_hr:.4f}, "
                 f"int_p={int_p:.6f} {sig}  (n={r['n']}, events={r['events']})")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(df, primary_interaction, treatment_stratified,
                 subtype_restricted, restricted_interaction,
                 er_interaction, output_dir: Path) -> Path:
    """Publication-ready 6-panel figure."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

    # ── Panel A: Treatment-stratified forest plot ────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    groups = ["HT_only", "Chemo_only", "HT+Chemo", "Neither"]
    group_labels = ["Hormone therapy\nonly", "Chemotherapy\nonly", "HT + Chemo", "Neither"]
    colors_treat = ["#4CAF50", "#F44336", "#FF9800", "#9E9E9E"]

    valid = [(g, l, c) for g, l, c in zip(groups, group_labels, colors_treat)
             if "error" not in treatment_stratified.get(g, {"error": True})]

    for i, (grp, label, col) in enumerate(valid):
        r = treatment_stratified[grp]
        ax_a.scatter([r["hr"]], [i], color=col, s=100, zorder=3)
        ax_a.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, lw=2.5)
        sig = "★" if r["p"] < 0.05 else ""
        ax_a.text(r["ci_high"] + 0.02, i,
                  f"HR={r['hr']:.3f} p={r['p']:.3f} n={r['n']} {sig}",
                  va="center", fontsize=8)

    ax_a.axvline(1.0, color="black", ls="--", lw=1)
    ax_a.set_yticks(range(len(valid)))
    ax_a.set_yticklabels([v[1] for v in valid], fontsize=9)
    ax_a.set_xlabel("Hazard Ratio (NV-Score)", fontsize=9)
    ax_a.set_title("Treatment-Stratified Cox\nNV-Score → OS per treatment group",
                    fontweight="bold", fontsize=10)

    # ── Panel B: Interaction coefficients ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    int_labels = ["NV main\neffect", "HT main\neffect", "NV × HT\ninteraction"]
    int_vars = ["nv_zscore", "hormone_therapy", "nv_x_treatment"]
    int_colors = ["#2196F3", "#4CAF50", "#E91E63"]

    for i, (var, label, col) in enumerate(zip(int_vars, int_labels, int_colors)):
        hr = primary_interaction.get(f"{var}_hr", 1.0)
        ci_l = primary_interaction.get(f"{var}_ci_low", 1.0)
        ci_h = primary_interaction.get(f"{var}_ci_high", 1.0)
        p = primary_interaction.get(f"{var}_p", 1.0)
        ax_b.scatter([hr], [i], color=col, s=120, zorder=3)
        ax_b.plot([ci_l, ci_h], [i, i], color=col, lw=2.5)
        sig = "★" if p < 0.05 else ("~" if p < 0.10 else "")
        ax_b.text(ci_h + 0.02, i, f"HR={hr:.3f} p={p:.4f} {sig}",
                  va="center", fontsize=8)

    ax_b.axvline(1.0, color="black", ls="--", lw=1)
    ax_b.set_yticks(range(len(int_labels)))
    ax_b.set_yticklabels(int_labels, fontsize=9)
    ax_b.set_xlabel("Hazard Ratio", fontsize=9)
    ax_b.set_title("Interaction Model Coefficients\nOS ~ NV + HT + NV×HT + age",
                    fontweight="bold", fontsize=10)

    # ── Panel C: KM по NV-Score × Treatment ──────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    median_nv = df["nv_zscore"].median()
    df_km = df.copy()
    df_km["nv_group"] = np.where(df_km["nv_zscore"] >= median_nv, "High NV", "Low NV")

    km_groups = [
        ("High NV + HT", (df_km["nv_group"] == "High NV") & (df_km["hormone_therapy"] == 1), "#1B5E20"),
        ("High NV − HT", (df_km["nv_group"] == "High NV") & (df_km["hormone_therapy"] == 0), "#81C784"),
        ("Low NV + HT",  (df_km["nv_group"] == "Low NV") & (df_km["hormone_therapy"] == 1), "#B71C1C"),
        ("Low NV − HT",  (df_km["nv_group"] == "Low NV") & (df_km["hormone_therapy"] == 0), "#EF9A9A"),
    ]

    kmf = KaplanMeierFitter()
    for label, mask, col in km_groups:
        sub = df_km[mask].dropna(subset=["os_months", "os_event"])
        if len(sub) < 10:
            continue
        kmf.fit(sub["os_months"], sub["os_event"], label=f"{label} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax_c, color=col, ci_show=False)

    ax_c.set_xlabel("Overall Survival (Months)", fontsize=9)
    ax_c.set_ylabel("Survival Probability", fontsize=9)
    ax_c.set_title("KM: NV-Score × Hormone Therapy\n4-group stratification",
                    fontweight="bold", fontsize=10)
    ax_c.legend(fontsize=7, loc="lower left")
    ax_c.set_ylim(0, 1.05)

    # ── Panel D: Restricted follow-up interaction gradient ───────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    cutoffs_plot = []
    int_hrs_plot = []
    int_ps_plot = []
    nv_hrs_plot = []

    for key in sorted(restricted_interaction.keys(),
                       key=lambda x: int(x.replace("mo", ""))):
        r = restricted_interaction[key]
        if "error" not in r:
            months = int(key.replace("mo", ""))
            cutoffs_plot.append(months)
            int_hrs_plot.append(r.get("nv_x_treatment_hr", 1.0))
            int_ps_plot.append(r.get("nv_x_treatment_p", 1.0))
            nv_hrs_plot.append(r.get("nv_zscore_hr", 1.0))

    ax_d.plot(cutoffs_plot, int_hrs_plot, "o-", color="#E91E63", lw=2,
              label="NV×HT interaction HR", markersize=8)
    ax_d.plot(cutoffs_plot, nv_hrs_plot, "s--", color="#2196F3", lw=1.5,
              label="NV main effect HR", markersize=6)
    ax_d.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)

    for x, p in zip(cutoffs_plot, int_ps_plot):
        if p < 0.05:
            ax_d.annotate("★", (x, int_hrs_plot[cutoffs_plot.index(x)]),
                         fontsize=14, ha="center", va="bottom", color="#E91E63")

    ax_d.set_xlabel("Follow-up Cutoff (Months)", fontsize=9)
    ax_d.set_ylabel("Hazard Ratio", fontsize=9)
    ax_d.set_title("Interaction HR × Follow-up Duration\nTime-dependent predictive effect",
                    fontweight="bold", fontsize=10)
    ax_d.legend(fontsize=8)

    # ── Panel E: PAM50 subtype HR heatmap (restricted follow-up) ─────────────
    ax_e = fig.add_subplot(gs[1, 1])
    subtypes = ["LumA", "LumB", "Her2", "Basal", "claudin-low"]
    cutoffs_sub = ["36mo", "60mo", "84mo"]

    hr_matrix = []
    for st in subtypes:
        row = []
        for c in cutoffs_sub:
            r = subtype_restricted.get(st, {}).get(c, {})
            hr_val = r.get("hr", np.nan)
            row.append(hr_val)
        hr_matrix.append(row)

    hr_arr = np.array(hr_matrix)
    im = ax_e.imshow(hr_arr, cmap="RdBu_r", vmin=0.6, vmax=1.4, aspect="auto")
    ax_e.set_xticks(range(len(cutoffs_sub)))
    ax_e.set_xticklabels(["36mo", "60mo", "84mo"], fontsize=9)
    ax_e.set_yticks(range(len(subtypes)))
    ax_e.set_yticklabels(subtypes, fontsize=9)

    for i in range(len(subtypes)):
        for j in range(len(cutoffs_sub)):
            val = hr_arr[i, j]
            r = subtype_restricted.get(subtypes[i], {}).get(cutoffs_sub[j], {})
            p = r.get("p", 1.0)
            sig = "★" if p < 0.05 else ("~" if p < 0.10 else "")
            if not np.isnan(val):
                ax_e.text(j, i, f"{val:.2f}{sig}", ha="center", va="center",
                         fontsize=8, fontweight="bold" if p < 0.05 else "normal")

    plt.colorbar(im, ax=ax_e, label="HR", shrink=0.8)
    ax_e.set_title("PAM50 Subtype × Restricted Follow-up\nNV-Score HR heatmap",
                    fontweight="bold", fontsize=10)

    # ── Panel F: ER-stratified interaction ────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    er_labels = []
    er_int_hrs = []
    er_int_cis = []
    er_int_ps = []
    er_colors = ["#4CAF50", "#F44336"]

    for i, (er_name, col) in enumerate(zip(["ER+", "ER-"], er_colors)):
        r = er_interaction.get(er_name, {})
        if "error" not in r:
            int_hr = r.get("nv_x_treatment_hr", 1.0)
            ci_l = r.get("nv_x_treatment_ci_low", 1.0)
            ci_h = r.get("nv_x_treatment_ci_high", 1.0)
            p = r.get("nv_x_treatment_p", 1.0)

            ax_f.scatter([int_hr], [i], color=col, s=120, zorder=3)
            ax_f.plot([ci_l, ci_h], [i, i], color=col, lw=2.5)
            sig = "★" if p < 0.05 else ""
            ax_f.text(ci_h + 0.02, i,
                      f"HR={int_hr:.3f} p={p:.4f} n={r['n']} {sig}",
                      va="center", fontsize=8)
            er_labels.append(f"{er_name}\n(interaction)")

    ax_f.axvline(1.0, color="black", ls="--", lw=1)
    ax_f.set_yticks(range(len(er_labels)))
    ax_f.set_yticklabels(er_labels, fontsize=9)
    ax_f.set_xlabel("Interaction HR (NV×HT)", fontsize=9)
    ax_f.set_title("ER-Stratified Interaction\nNV×HT effect by ER status",
                    fontweight="bold", fontsize=10)

    # ── Title ────────────────────────────────────────────────────────────────
    int_p = primary_interaction.get("nv_x_treatment_p", 1.0)
    verdict = ("★ PREDICTIVE (interaction p<0.05)" if int_p < 0.05
               else "~ Suggestive (p<0.10)" if int_p < 0.10
               else "○ Prognostic only")

    fig.suptitle(
        f"RevGate Finding 11 — NV-Score × Treatment Interaction\n"
        f"METABRIC n={primary_interaction['n']} | {verdict}",
        fontsize=13, fontweight="bold",
    )

    out = output_dir / "interaction_treatment.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Фигура: {out}")
    return out


# ── Print summary ────────────────────────────────────────────────────────────

def print_summary(primary, chemo, treat_strat, subtype_restr, er_inter, restr_inter):
    G = "\033[92m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"
    Y = "\033[93m"; M = "\033[95m"

    print(f"\n{B}{'='*70}{E}")
    print(f"{B}  RevGate Finding 11 — NV-Score × Treatment Interaction{E}")
    print(f"{'='*70}")

    # Primary result
    int_p = primary.get("nv_x_treatment_p", 1.0)
    int_hr = primary.get("nv_x_treatment_hr", 1.0)
    nv_hr = primary.get("nv_zscore_hr", 1.0)

    color = G if int_p < 0.05 else (Y if int_p < 0.10 else R)
    print(f"\n{B}PRIMARY: NV-Score × Hormone Therapy{E}")
    print(f"  NV main effect:    HR={nv_hr:.4f}  p={primary.get('nv_zscore_p', 'NA')}")
    print(f"  HT main effect:    HR={primary.get('hormone_therapy_hr', 'NA')}  "
          f"p={primary.get('hormone_therapy_p', 'NA')}")
    print(f"  {color}NV×HT interaction:  HR={int_hr:.4f}  p={int_p:.6f}{E}")
    print(f"  n={primary['n']}, events={primary['events']}")

    if int_p < 0.05:
        print(f"\n  {G}★★★ NV-Score is PREDICTIVE — patients with high NV-Score{E}")
        print(f"  {G}    benefit MORE from hormone therapy than low NV-Score patients ★★★{E}")
    elif int_p < 0.10:
        print(f"\n  {Y}★★ Suggestive predictive signal (p<0.10){E}")
    else:
        print(f"\n  {R}○ NV-Score is prognostic but not predictive of HT benefit{E}")

    # Treatment-stratified
    print(f"\n{B}TREATMENT-STRATIFIED Cox:{E}")
    for grp, r in treat_strat.items():
        if "error" not in r:
            sig = G + "✓" + E if r["p"] < 0.05 else Y + "~" + E if r["p"] < 0.10 else "○"
            print(f"  {grp:<12}: HR={r['hr']:.4f} [{r['ci_low']:.4f}–{r['ci_high']:.4f}] "
                  f"p={r['p']:.4f} n={r['n']} {sig}")

    # Subtype restricted
    print(f"\n{B}PAM50 SUBTYPE × RESTRICTED FOLLOW-UP:{E}")
    for st in ["LumA", "LumB", "Her2", "Basal", "claudin-low"]:
        line = f"  {st:<12}: "
        for c in ["36mo", "60mo", "84mo"]:
            r = subtype_restr.get(st, {}).get(c, {})
            if "error" not in r:
                sig = "✓" if r["p"] < 0.05 else "~" if r["p"] < 0.10 else "○"
                line += f"{c} HR={r['hr']:.3f}{sig}  "
            else:
                line += f"{c} n/a  "
        print(line)

    # Restricted interaction
    print(f"\n{B}RESTRICTED FOLLOW-UP INTERACTION:{E}")
    for key in sorted(restr_inter.keys(), key=lambda x: int(x.replace("mo", ""))):
        r = restr_inter[key]
        if "error" not in r:
            int_p = r.get("nv_x_treatment_p", 1.0)
            int_hr = r.get("nv_x_treatment_hr", 1.0)
            sig = G + "✓" + E if int_p < 0.05 else Y + "~" + E if int_p < 0.10 else "○"
            print(f"  {key:>5}: NV×HT HR={int_hr:.4f}  p={int_p:.6f}  {sig}")

    print(f"{'='*70}\n")


# ── Save ─────────────────────────────────────────────────────────────────────

def save_results(primary, chemo, treat_strat, subtype_restr,
                 er_inter, restr_inter, output_dir: Path):
    out = {
        "timestamp": datetime.now().isoformat(),
        "finding": "Finding_11",
        "description": "NV-Score × Treatment Interaction Test",
        "cohort": "METABRIC (cBioPortal brca_metabric)",
        "primary_interaction_hormone_therapy": primary,
        "secondary_interaction_chemotherapy": chemo,
        "treatment_stratified_cox": treat_strat,
        "pam50_subtype_restricted_followup": subtype_restr,
        "er_stratified_interaction": er_inter,
        "restricted_followup_interaction": restr_inter,
    }
    p = output_dir / "interaction_treatment_results.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"JSON: {p}")

    # CSV summary
    rows = []
    for key, r in restr_inter.items():
        if "error" not in r:
            rows.append({
                "cutoff": key,
                "nv_hr": r.get("nv_zscore_hr"),
                "nv_p": r.get("nv_zscore_p"),
                "ht_hr": r.get("hormone_therapy_hr"),
                "ht_p": r.get("hormone_therapy_p"),
                "interaction_hr": r.get("nv_x_treatment_hr"),
                "interaction_p": r.get("nv_x_treatment_p"),
                "n": r["n"],
                "events": r["events"],
            })
    csv_path = output_dir / "interaction_treatment_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info(f"CSV: {csv_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Finding 11: NV-Score × Treatment Interaction")
    parser.add_argument("--ssgsea",
                        default="/root/.revgate/cache/metabric/metabric_ssgsea_scores.parquet")
    parser.add_argument("--clinical",
                        default="/root/.revgate/cache/metabric/metabric_clinical.csv")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_and_merge(args.ssgsea, args.clinical)

    # Run all analyses
    primary = analysis_interaction_primary(df)
    chemo = analysis_interaction_chemo(df)
    treat_strat = analysis_treatment_stratified(df)
    subtype_restr = analysis_subtype_restricted(df)
    er_inter = analysis_er_stratified_interaction(df)
    restr_inter = analysis_restricted_interaction(df)

    # Output
    print_summary(primary, chemo, treat_strat, subtype_restr, er_inter, restr_inter)
    plot_results(df, primary, treat_strat, subtype_restr, restr_inter,
                 er_inter, output_dir)
    save_results(primary, chemo, treat_strat, subtype_restr, er_inter,
                 restr_inter, output_dir)

    log.info("Finding 11 завършен.")


if __name__ == "__main__":
    main()
