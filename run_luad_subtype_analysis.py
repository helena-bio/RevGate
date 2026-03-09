#!/usr/bin/env python3
"""
RevGate — LUAD Subtype Analysis
================================
Обяснение на обратния HR=1.289 в Finding_01 LUAD Early.

Хипотеза: LUAD е хетерогенен тумор. Обратният HR отразява
доминиране на KRAS/housekeeping-driven NV-Score в majority,
докато EGFR-mutant subset би показал HR<1 (концентрирана dependency).

Анализи:
  1. Mutation-stratified Cox: EGFR-mutant vs KRAS-mutant vs WT
  2. Expression subtype Cox: Bronchioid vs Squamoid vs Magnoid
  3. Smoking-stratified Cox: Never vs Ever smoker
  4. Stage-stratified Cox: Early (I-II) vs Advanced (III-IV)
  5. Top dependency genes: PSMB5/YRDC housekeeping vs oncogene deps
  6. NV-Score distribution per subtype

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
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("luad_subtype")

FINDING01_LUAD_EARLY_HR = 1.289
FINDING01_LUAD_EARLY_P  = 0.0008

STAGE_MAP = {
    "Stage I": 1, "Stage IA": 1, "Stage IB": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3,
    "Stage IV": 4,
}

# Smoking: 1=never, 2=former light, 3=former heavy, 4=current, 5=NOS
NEVER_SMOKER    = {1}
EVER_SMOKER     = {2, 3, 4, 5}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(clinical_path: str, ssgsea_path: str) -> pd.DataFrame:
    log.info("Зареждане на LUAD данни...")
    clin   = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    ssgsea = pd.read_parquet(ssgsea_path)
    luad   = ssgsea[ssgsea["cancer_id"] == "LUAD"].copy()

    # ID normalization
    clin["patient_id_12"] = clin["sampleID"].str[:12]
    luad["patient_id_12"] = luad["patient_id"].str[:12]

    df = clin.merge(luad[["patient_id_12","ssgsea_nv_score"]],
                    on="patient_id_12", how="inner")

    # OS
    df["os_time"] = np.where(
        df["vital_status"] == "DECEASED",
        df["days_to_death"],
        df["days_to_last_followup"],
    )
    df["os_event"] = (df["vital_status"] == "DECEASED").astype(int)

    # Stage
    df["stage_num"]    = df["pathologic_stage"].map(STAGE_MAP)
    df["stage_binary"] = df["stage_num"].apply(
        lambda x: 0 if x in {1,2} else (1 if x in {3,4} else np.nan)
    )

    # Age
    df["age"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"],
                               errors="coerce")

    # EGFR mutation status
    def egfr_status(row):
        val = str(row.get("EGFR","")).strip().lower()
        if val in ["nan","none",""]:
            return np.nan
        return 1 if val != "none" else 0

    def kras_status(row):
        val = str(row.get("KRAS","")).strip().lower()
        if val in ["nan","none",""]:
            return np.nan
        return 1 if val != "none" else 0

    df["egfr_mut"] = df.apply(egfr_status, axis=1)
    df["kras_mut"] = df.apply(kras_status, axis=1)

    # Mutual exclusivity group
    def mut_group(row):
        if pd.isna(row["egfr_mut"]) and pd.isna(row["kras_mut"]):
            return np.nan
        if row["egfr_mut"] == 1:
            return "EGFR-mutant"
        if row["kras_mut"] == 1:
            return "KRAS-mutant"
        return "WT/Other"

    df["mut_group"] = df.apply(mut_group, axis=1)

    # Expression subtype
    df["expr_subtype"] = df["Expression_Subtype"].replace("", np.nan)

    # Smoking
    df["smoking"] = pd.to_numeric(df["tobacco_smoking_history"], errors="coerce")
    df["never_smoker"] = df["smoking"].apply(
        lambda x: 1 if x in NEVER_SMOKER else (0 if x in EVER_SMOKER else np.nan)
    )

    # Filter
    df = df.dropna(subset=["os_time","os_event","ssgsea_nv_score"])
    df = df[df["os_time"] > 0]

    log.info(f"  Total: n={len(df)}, events={int(df.os_event.sum())}")
    log.info(f"  EGFR-mut: {int((df.egfr_mut==1).sum())}, "
             f"KRAS-mut: {int((df.kras_mut==1).sum())}, "
             f"WT: {int(((df.egfr_mut==0)&(df.kras_mut==0)).sum())}")
    log.info(f"  Expr subtypes: {df.expr_subtype.value_counts().to_dict()}")
    log.info(f"  Never smoker: {int((df.never_smoker==1).sum())}, "
             f"Ever: {int((df.never_smoker==0).sum())}")
    return df


# ── Cox helper ────────────────────────────────────────────────────────────────

def fit_cox(df: pd.DataFrame, covs: list,
            dur="os_time", ev="os_event") -> dict | None:
    sub = df[covs + [dur, ev]].dropna().copy()
    if len(sub) < 20 or sub[ev].sum() < 10:
        return None
    for col in covs:
        if sub[col].nunique() <= 3:
            continue
        sub[col] = StandardScaler().fit_transform(sub[[col]])
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(sub, duration_col=dur, event_col=ev)
        s = cph.summary
        v = covs[0]
        return {
            "hr":      round(float(s.loc[v,"exp(coef)"]), 4),
            "ci_low":  round(float(s.loc[v,"exp(coef) lower 95%"]), 4),
            "ci_high": round(float(s.loc[v,"exp(coef) upper 95%"]), 4),
            "p":       round(float(s.loc[v,"p"]), 4),
            "coef":    round(float(s.loc[v,"coef"]), 4),
            "n":       len(sub),
            "events":  int(sub[ev].sum()),
        }
    except Exception as e:
        log.warning(f"Cox failed: {e}")
        return None


# ── Analyses ──────────────────────────────────────────────────────────────────

def overall_cox(df):
    log.info("Overall Cox (репликация на Finding_01)...")
    # Early only
    early = df[df["stage_binary"]==0]
    r_early = fit_cox(early, ["ssgsea_nv_score","age"])
    r_all   = fit_cox(df,    ["ssgsea_nv_score","age"])
    for label, r in [("All stages", r_all), ("Early (I-II)", r_early)]:
        if r:
            sig = "✓" if r["p"]<0.05 else "○"
            log.info(f"  {label}: HR={r['hr']:.4f} p={r['p']:.4f} "
                     f"n={r['n']} {sig}")
    return {"all": r_all, "early": r_early}


def mutation_stratified(df):
    log.info("Mutation-stratified Cox (EGFR / KRAS / WT)...")
    results = {}
    for grp in ["EGFR-mutant", "KRAS-mutant", "WT/Other"]:
        sub = df[df["mut_group"]==grp]
        r   = fit_cox(sub, ["ssgsea_nv_score","age"])
        if r:
            results[grp] = r
            sig = "✓" if r["p"]<0.05 else "○"
            dir_arrow = "↓ protective" if r["hr"]<1 else "↑ harmful"
            log.info(f"  {grp:<15}: HR={r['hr']:.4f} p={r['p']:.4f} "
                     f"n={r['n']} {sig} {dir_arrow}")
        else:
            log.info(f"  {grp}: insufficient data")
    return results


def expression_subtype_stratified(df):
    log.info("Expression subtype Cox (Bronchioid/Squamoid/Magnoid)...")
    results = {}
    for st in ["Bronchioid","Squamoid","Magnoid"]:
        sub = df[df["expr_subtype"]==st]
        r   = fit_cox(sub, ["ssgsea_nv_score","age"])
        if r:
            results[st] = r
            sig = "✓" if r["p"]<0.05 else "○"
            dir_arrow = "↓ protective" if r["hr"]<1 else "↑ harmful"
            log.info(f"  {st:<12}: HR={r['hr']:.4f} p={r['p']:.4f} "
                     f"n={r['n']} {sig} {dir_arrow}")
        else:
            log.info(f"  {st}: insufficient data")
    return results


def smoking_stratified(df):
    log.info("Smoking-stratified Cox...")
    results = {}
    for val, label in [(1,"Never smoker"), (0,"Ever smoker")]:
        sub = df[df["never_smoker"]==val]
        r   = fit_cox(sub, ["ssgsea_nv_score","age"])
        if r:
            results[label] = r
            sig = "✓" if r["p"]<0.05 else "○"
            dir_arrow = "↓ protective" if r["hr"]<1 else "↑ harmful"
            log.info(f"  {label:<15}: HR={r['hr']:.4f} p={r['p']:.4f} "
                     f"n={r['n']} {sig} {dir_arrow}")
    return results


def stage_stratified(df):
    log.info("Stage-stratified Cox...")
    results = {}
    for val, label in [(0,"Early (I-II)"), (1,"Advanced (III-IV)")]:
        sub = df[df["stage_binary"]==val]
        r   = fit_cox(sub, ["ssgsea_nv_score","age"])
        if r:
            results[label] = r
            sig = "✓" if r["p"]<0.05 else "○"
            dir_arrow = "↓ protective" if r["hr"]<1 else "↑ harmful"
            log.info(f"  {label}: HR={r['hr']:.4f} p={r['p']:.4f} "
                     f"n={r['n']} {sig} {dir_arrow}")
    return results


def score_distribution(df):
    log.info("Score distribution per subtype...")
    results = {}

    # Per mutation group
    for grp in ["EGFR-mutant","KRAS-mutant","WT/Other"]:
        sub = df[df["mut_group"]==grp]["ssgsea_nv_score"].dropna()
        if len(sub) > 5:
            results[f"mut_{grp}"] = {
                "mean": round(float(sub.mean()),4),
                "std":  round(float(sub.std()),4),
                "cv":   round(float(sub.std()/sub.mean()),4),
                "n":    len(sub),
            }
            log.info(f"  {grp}: mean={sub.mean():.3f} std={sub.std():.3f} "
                     f"CV={sub.std()/sub.mean():.3f}")

    # Per expression subtype
    for st in ["Bronchioid","Squamoid","Magnoid"]:
        sub = df[df["expr_subtype"]==st]["ssgsea_nv_score"].dropna()
        if len(sub) > 5:
            results[f"expr_{st}"] = {
                "mean": round(float(sub.mean()),4),
                "std":  round(float(sub.std()),4),
                "cv":   round(float(sub.std()/sub.mean()),4),
                "n":    len(sub),
            }
            log.info(f"  {st}: mean={sub.mean():.3f} CV={sub.std()/sub.mean():.3f}")
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(df, overall, mut_results, expr_results,
                 smoking_results, stage_results, output_dir: Path):
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

    mut_colors  = {"EGFR-mutant":"#4CAF50","KRAS-mutant":"#F44336","WT/Other":"#9E9E9E"}
    expr_colors = {"Bronchioid":"#2196F3","Squamoid":"#FF9800","Magnoid":"#9C27B0"}

    # ── Panel A: Forest plot — mutation groups ────────────────────────────────
    ax_a = fig.add_subplot(gs[0,0])
    all_groups = {}
    all_groups.update({f"MUT: {k}": v for k,v in mut_results.items()})
    all_groups.update({f"EXPR: {k}": v for k,v in expr_results.items()})
    if overall.get("early"):
        all_groups["Overall Early"] = overall["early"]

    labels = list(all_groups.keys())
    hrs    = [all_groups[l]["hr"]     for l in labels]
    ci_l   = [all_groups[l]["ci_low"] for l in labels]
    ci_h   = [all_groups[l]["ci_high"]for l in labels]
    ps     = [all_groups[l]["p"]      for l in labels]
    ns     = [all_groups[l]["n"]      for l in labels]

    cols = []
    for l in labels:
        if "EGFR" in l:   cols.append("#4CAF50")
        elif "KRAS" in l: cols.append("#F44336")
        elif "Bronch" in l: cols.append("#2196F3")
        elif "Squam" in l:  cols.append("#FF9800")
        elif "Magnoid" in l:cols.append("#9C27B0")
        else: cols.append("#607D8B")

    y = range(len(labels))
    ax_a.scatter(hrs, list(y), color=cols, s=80, zorder=3)
    for i,(cl,ch,hr,c) in enumerate(zip(ci_l,ci_h,hrs,cols)):
        ax_a.plot([cl,ch],[i,i], color=c, lw=2, alpha=0.7)
    ax_a.axvline(1.0, color="black", linestyle="--", lw=1)
    ax_a.set_yticks(list(y))
    ax_a.set_yticklabels(
        [f"{l}\n(n={n}, p={p:.3f})" for l,n,p in zip(labels,ns,ps)],
        fontsize=7.5)
    ax_a.set_xlabel("Hazard Ratio (ssGSEA)", fontsize=9)
    ax_a.set_title("Forest Plot — All Subgroups\nssGSEA → OS", fontweight="bold", fontsize=10)

    # ── Panel B: KM — EGFR vs KRAS vs WT ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0,1])
    median = df["ssgsea_nv_score"].median()
    for grp, col in [("EGFR-mutant","#4CAF50"),("KRAS-mutant","#F44336"),("WT/Other","#9E9E9E")]:
        sub = df[df["mut_group"]==grp].dropna(subset=["os_time","os_event"])
        if len(sub) < 10: continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_time"]/30.44, sub["os_event"],
                label=f"{grp} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax_b, color=col, ci_show=False)
    ax_b.set_xlabel("OS (Months)", fontsize=9)
    ax_b.set_ylabel("Survival Probability", fontsize=9)
    ax_b.set_title("KM Curves by\nMutation Group", fontweight="bold", fontsize=10)
    ax_b.legend(fontsize=7)

    # ── Panel C: Score distribution per mutation group ────────────────────────
    ax_c = fig.add_subplot(gs[0,2])
    grps = ["EGFR-mutant","KRAS-mutant","WT/Other"]
    data = [df[df["mut_group"]==g]["ssgsea_nv_score"].dropna().values for g in grps]
    bp = ax_c.boxplot(data, patch_artist=True)
    for patch, grp in zip(bp["boxes"], grps):
        patch.set_facecolor(mut_colors[grp])
        patch.set_alpha(0.7)
    ax_c.set_xticklabels(grps, rotation=15, fontsize=8)
    ax_c.set_ylabel("ssGSEA NV-Score", fontsize=9)
    ax_c.set_title("Score Distribution\nby Mutation Group", fontweight="bold", fontsize=10)

    # ── Panel D: Stage + Smoking forest ──────────────────────────────────────
    ax_d = fig.add_subplot(gs[1,0])
    combined = {}
    combined.update({f"Stage: {k}": v for k,v in stage_results.items()})
    combined.update({f"Smoke: {k}": v for k,v in smoking_results.items()})
    if combined:
        lbls2 = list(combined.keys())
        hrs2  = [combined[l]["hr"]     for l in lbls2]
        cl2   = [combined[l]["ci_low"] for l in lbls2]
        ch2   = [combined[l]["ci_high"]for l in lbls2]
        ps2   = [combined[l]["p"]      for l in lbls2]
        ns2   = [combined[l]["n"]      for l in lbls2]
        cols2 = ["#2196F3" if "Stage" in l else "#FF9800" for l in lbls2]
        y2    = range(len(lbls2))
        ax_d.scatter(hrs2, list(y2), color=cols2, s=80, zorder=3)
        for i,(cl,ch,c) in enumerate(zip(cl2,ch2,cols2)):
            ax_d.plot([cl,ch],[i,i],color=c,lw=2,alpha=0.7)
        ax_d.axvline(1.0,color="black",linestyle="--",lw=1)
        ax_d.set_yticks(list(y2))
        ax_d.set_yticklabels(
            [f"{l}\n(n={n}, p={p:.3f})" for l,n,p in zip(lbls2,ns2,ps2)],
            fontsize=8)
    ax_d.set_xlabel("Hazard Ratio", fontsize=9)
    ax_d.set_title("Stage & Smoking\nStratified Cox", fontweight="bold", fontsize=10)

    # ── Panel E: HR comparison — Finding_01 vs subtypes ──────────────────────
    ax_e = fig.add_subplot(gs[1,1])
    ref_groups = {
        "F01: All Early": FINDING01_LUAD_EARLY_HR,
    }
    for grp, r in mut_results.items():
        ref_groups[f"MUT: {grp}"] = r["hr"]
    bar_cols = ["#607D8B"] + [mut_colors.get(k.replace("MUT: ",""),"#9E9E9E")
                               for k in list(ref_groups.keys())[1:]]
    bars = ax_e.bar(range(len(ref_groups)),
                    list(ref_groups.values()),
                    color=bar_cols, alpha=0.8)
    ax_e.axhline(1.0, color="red", linestyle="--", lw=1.5, label="HR=1 (null)")
    ax_e.set_xticks(range(len(ref_groups)))
    ax_e.set_xticklabels(list(ref_groups.keys()), rotation=20, fontsize=7.5)
    ax_e.set_ylabel("Hazard Ratio", fontsize=9)
    ax_e.set_title("HR Decomposition\nFinding_01 vs Subtypes", fontweight="bold", fontsize=10)
    ax_e.legend(fontsize=8)

    # ── Panel F: KM Never vs Ever smoker ─────────────────────────────────────
    ax_f = fig.add_subplot(gs[1,2])
    smoke_colors = {1: "#4CAF50", 0: "#F44336"}
    smoke_labels = {1: "Never smoker", 0: "Ever smoker"}
    for val, col in smoke_colors.items():
        sub = df[df["never_smoker"]==val].dropna(subset=["os_time","os_event"])
        if len(sub) < 10: continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_time"]/30.44, sub["os_event"],
                label=f"{smoke_labels[val]} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax_f, color=col, ci_show=False)

    # Log-rank
    never = df[df["never_smoker"]==1].dropna(subset=["os_time","os_event"])
    ever  = df[df["never_smoker"]==0].dropna(subset=["os_time","os_event"])
    if len(never)>10 and len(ever)>10:
        lr = logrank_test(never["os_time"], ever["os_time"],
                          event_observed_A=never["os_event"],
                          event_observed_B=ever["os_event"])
        ax_f.set_title(f"KM — Smoking Status\nLog-rank p={lr.p_value:.4f}",
                       fontweight="bold", fontsize=10)
    ax_f.set_xlabel("OS (Months)", fontsize=9)
    ax_f.set_ylabel("Survival Probability", fontsize=9)
    ax_f.legend(fontsize=8)

    fig.suptitle(
        "RevGate LUAD — Subtype Decomposition of HR=1.289 (Finding_01)\n"
        "ssGSEA NV-Score × Mutation / Expression / Smoking Stratification",
        fontsize=12, fontweight="bold",
    )
    out = output_dir / "luad_subtype_analysis.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Фигура: {out}")
    return out


# ── Save & Summary ────────────────────────────────────────────────────────────

def save_results(overall, mut_r, expr_r, smoking_r, stage_r,
                 score_dist, output_dir: Path):
    out = {
        "timestamp": datetime.now().isoformat(),
        "cancer": "LUAD",
        "finding01_reference": {
            "early_hr": FINDING01_LUAD_EARLY_HR,
            "early_p":  FINDING01_LUAD_EARLY_P,
        },
        "overall_cox": overall,
        "mutation_stratified": mut_r,
        "expression_subtype": expr_r,
        "smoking_stratified": smoking_r,
        "stage_stratified": stage_r,
        "score_distribution": score_dist,
    }
    p = output_dir / "luad_subtype_results.json"
    with open(p,"w",encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info(f"JSON: {p}")


def print_summary(overall, mut_r, expr_r, smoking_r, stage_r):
    B="\033[1m"; E="\033[0m"; G="\033[92m"; R="\033[91m"; Y="\033[93m"

    print(f"\n{B}{'='*68}{E}")
    print(f"{B}  RevGate LUAD Subtype Analysis — Декомпозиция на HR=1.289{E}")
    print(f"{'='*68}")

    def fmt(r, label):
        if not r: return f"  {label}: insufficient data"
        sig  = f"{G}✓{E}" if r["p"]<0.05 else "○"
        col  = G if r["hr"]<1 else R
        arr  = "↓ protective" if r["hr"]<1 else "↑ harmful"
        return (f"  {label:<20}: {col}HR={r['hr']:.4f}{E} "
                f"[{r['ci_low']:.3f}–{r['ci_high']:.3f}] "
                f"p={r['p']:.4f} {sig} {arr} n={r['n']}")

    print(f"\n{B}Overall (Finding_01 репликация):{E}")
    print(fmt(overall.get("early"), "Early (I-II)"))
    print(fmt(overall.get("all"),   "All stages"))

    print(f"\n{B}Mutation-stratified:{E}")
    for grp, r in mut_r.items():
        print(fmt(r, grp))

    print(f"\n{B}Expression subtype:{E}")
    for st, r in expr_r.items():
        print(fmt(r, st))

    print(f"\n{B}Smoking-stratified:{E}")
    for label, r in smoking_r.items():
        print(fmt(r, label))

    print(f"\n{B}Stage-stratified:{E}")
    for label, r in stage_r.items():
        print(fmt(r, label))

    print(f"\n{B}Ключов въпрос — обяснява ли се HR=1.289?{E}")
    egfr = mut_r.get("EGFR-mutant",{})
    kras = mut_r.get("KRAS-mutant",{})
    if egfr.get("hr",1.0) < 1.0 and kras.get("hr",0.0) > 1.0:
        print(f"  {G}✓ ДА — EGFR-mutant показва HR<1, KRAS-mutant HR>1{E}")
        print(f"  Обратният ефект е артефакт от смесване на биологично различни subtypes")
    elif egfr.get("hr",1.0) < 1.0:
        print(f"  {Y}ЧАСТИЧНО — EGFR-mutant показва HR<1{E}")
    else:
        print(f"  {R}НЕ — нито EGFR subtype показва HR<1{E}")
        print(f"  Необходимо е алтернативно обяснение")
    print(f"{'='*68}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical",
        default="/root/.revgate/cache/tcga/LUAD_clinical_raw.tsv")
    parser.add_argument("--ssgsea",
        default="/root/.revgate/cache/tcga/per_patient_ssgsea_scores.parquet")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df         = load_data(args.clinical, args.ssgsea)
    overall    = overall_cox(df)
    mut_r      = mutation_stratified(df)
    expr_r     = expression_subtype_stratified(df)
    smoking_r  = smoking_stratified(df)
    stage_r    = stage_stratified(df)
    score_dist = score_distribution(df)

    print_summary(overall, mut_r, expr_r, smoking_r, stage_r)
    plot_results(df, overall, mut_r, expr_r, smoking_r, stage_r, output_dir)
    save_results(overall, mut_r, expr_r, smoking_r, stage_r, score_dist, output_dir)
    log.info("LUAD subtype analysis завършен.")


if __name__ == "__main__":
    main()
