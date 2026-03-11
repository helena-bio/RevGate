#!/usr/bin/env python3
"""
RevGate — Finding 14: Sel+pLI Threshold Optimization
=====================================================
Formal optimization of the Dependency Architecture Principle threshold.

Analyses:
  A. Youden index sweep (0.50-1.50, step 0.01)
  B. Two-criterion gate: Sel+pLI > T AND selectivity > S
  C. Bootstrap CI for optimal threshold (n=1000)
  D. Sensitivity/specificity curve
  E. Comparison: single threshold vs two-criterion gate

Uses Finding 13 scorecard as input.

Helena Bioinformatics, 2026.
"""

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

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("threshold_opt")

file_handler = logging.FileHandler("results/threshold_optimization.log", mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(file_handler)

# Load NV components
NV_CSV = Path("/root/.revgate/cache/nv_components_all_lineages.csv")

# Finding 13 scorecard with Cox results
CANCER_TO_LINEAGE = {
    "BRCA":"Breast","KIRC":"Kidney","LUAD":"Lung","SKCM":"Skin",
    "PAAD":"Pancreas","LAML":"Myeloid","COAD":"Bowel",
    "OV":"Ovary/Fallopian Tube","HNSC":"Head and Neck","UCEC":"Uterus",
    "GBM":"CNS/Brain","LGG":"CNS/Brain","DLBC":"Lymphoid",
    "STAD":"Esophagus/Stomach","LIHC":"Liver",
    "BLCA":"Bladder/Urinary Tract","THCA":"Thyroid",
    "PRAD":"Prostate","CESC":"Cervix","SARC":"Soft Tissue",
}

# Observed Cox results from Finding 13 (best stratum per cancer)
COX_RESULTS = {
    "BRCA": {"hr":0.829,"p":0.0034,"dir":"protective"},
    "KIRC": {"hr":0.883,"p":0.0112,"dir":"protective"},
    "LUAD": {"hr":1.318,"p":0.0005,"dir":"harmful"},
    "HNSC": {"hr":1.147,"p":0.0463,"dir":"harmful"},
    "LGG":  {"hr":1.330,"p":0.0241,"dir":"harmful"},
    "LIHC": {"hr":1.453,"p":0.0001,"dir":"harmful"},
    "PAAD": {"hr":1.396,"p":0.0570,"dir":"harmful"},
    "SKCM": {"hr":0.890,"p":0.2703,"dir":"protective"},
    "LAML": {"hr":0.994,"p":0.9549,"dir":"protective"},
    "COAD": {"hr":0.834,"p":0.1657,"dir":"protective"},
    "OV":   {"hr":0.899,"p":0.1207,"dir":"protective"},
    "UCEC": {"hr":1.430,"p":0.1484,"dir":"harmful"},
    "GBM":  {"hr":1.015,"p":0.8674,"dir":"harmful"},
    "STAD": {"hr":1.208,"p":0.1579,"dir":"harmful"},
    "BLCA": {"hr":1.047,"p":0.5748,"dir":"harmful"},
    "CESC": {"hr":0.925,"p":0.5166,"dir":"protective"},
    "SARC": {"hr":0.856,"p":0.1587,"dir":"protective"},
}


def load_nv_components():
    """Load lineage-level NV components."""
    df = pd.read_csv(NV_CSV)
    comp = {}
    for _, row in df.iterrows():
        sel = float(row["selectivity"]) if pd.notna(row.get("selectivity")) else 0.0
        pli = float(row["mean_pli"]) if pd.notna(row.get("mean_pli")) else 0.0
        comp[row["lineage"]] = {"selectivity": sel, "mean_pli": pli, "sel_pli": sel + pli}
    return comp


def build_validation_set(nv_comp, p_threshold=0.05):
    """Build validation set: cancers with p < threshold."""
    rows = []
    for cancer, lineage in CANCER_TO_LINEAGE.items():
        cox = COX_RESULTS.get(cancer)
        if cox is None or cox["p"] > p_threshold:
            continue
        c = nv_comp.get(lineage, {})
        rows.append({
            "cancer": cancer, "lineage": lineage,
            "sel_pli": c.get("sel_pli", 0),
            "selectivity": c.get("selectivity", 0),
            "mean_pli": c.get("mean_pli", 0),
            "hr": cox["hr"], "p": cox["p"],
            "observed": cox["dir"],
            "true_label": 1 if cox["dir"] == "protective" else 0,
        })
    return pd.DataFrame(rows)


# ── Analysis A: Youden Index Sweep ───────────────────────────────────────────

def analysis_youden_sweep(df):
    """Sweep Sel+pLI threshold, compute sensitivity/specificity/Youden."""
    log.info("=" * 60)
    log.info("ANALYSIS A: Youden Index Sweep (Sel+pLI threshold)")
    log.info("=" * 60)

    thresholds = np.arange(0.50, 1.50, 0.01)
    results = []

    for t in thresholds:
        pred = (df["sel_pli"] > t).astype(int)
        tp = ((pred == 1) & (df["true_label"] == 1)).sum()
        tn = ((pred == 0) & (df["true_label"] == 0)).sum()
        fp = ((pred == 1) & (df["true_label"] == 0)).sum()
        fn = ((pred == 0) & (df["true_label"] == 1)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sens + spec - 1
        acc = (tp + tn) / len(df) if len(df) > 0 else 0

        results.append({
            "threshold": round(t, 2), "sensitivity": round(sens, 3),
            "specificity": round(spec, 3), "youden": round(youden, 3),
            "accuracy": round(acc, 3), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    res_df = pd.DataFrame(results)
    best_idx = res_df["youden"].idxmax()
    best = res_df.iloc[best_idx]

    log.info(f"  Best threshold: {best['threshold']:.2f}")
    log.info(f"  Sensitivity: {best['sensitivity']:.3f}")
    log.info(f"  Specificity: {best['specificity']:.3f}")
    log.info(f"  Youden J: {best['youden']:.3f}")
    log.info(f"  Accuracy: {best['accuracy']:.3f}")

    # Also check threshold = 1.0 (current)
    t10 = res_df[res_df["threshold"] == 1.0].iloc[0]
    log.info(f"\n  Current threshold (1.0): Acc={t10['accuracy']:.3f} "
             f"Sens={t10['sensitivity']:.3f} Spec={t10['specificity']:.3f} Y={t10['youden']:.3f}")

    return res_df, best.to_dict()


# ── Analysis B: Two-Criterion Gate ───────────────────────────────────────────

def analysis_two_criterion(df):
    """Two-criterion gate: Sel+pLI > T1 AND selectivity > T2."""
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS B: Two-Criterion Gate")
    log.info("=" * 60)

    results = []
    for t1 in np.arange(0.80, 1.30, 0.05):
        for t2 in np.arange(0.20, 0.50, 0.05):
            pred = ((df["sel_pli"] > t1) & (df["selectivity"] > t2)).astype(int)
            tp = ((pred == 1) & (df["true_label"] == 1)).sum()
            tn = ((pred == 0) & (df["true_label"] == 0)).sum()
            acc = (tp + tn) / len(df)
            results.append({
                "sel_pli_threshold": round(t1, 2),
                "selectivity_threshold": round(t2, 2),
                "accuracy": round(acc, 3), "tp": tp, "tn": tn,
            })

    res_df = pd.DataFrame(results)
    best_idx = res_df["accuracy"].idxmax()
    best = res_df.iloc[best_idx]

    log.info(f"  Best two-criterion gate:")
    log.info(f"    Sel+pLI > {best['sel_pli_threshold']:.2f} AND selectivity > {best['selectivity_threshold']:.2f}")
    log.info(f"    Accuracy: {best['accuracy']:.3f}")

    # Check specific gates
    for t1, t2 in [(1.0, 0.35), (1.0, 0.30), (0.95, 0.35), (1.05, 0.35)]:
        pred = ((df["sel_pli"] > t1) & (df["selectivity"] > t2)).astype(int)
        tp = ((pred == 1) & (df["true_label"] == 1)).sum()
        tn = ((pred == 0) & (df["true_label"] == 0)).sum()
        acc = (tp + tn) / len(df)
        log.info(f"    Gate Sel+pLI>{t1:.2f} & Sel>{t2:.2f}: acc={acc:.3f} tp={tp} tn={tn}")

    return res_df, best.to_dict()


# ── Analysis C: Bootstrap CI ─────────────────────────────────────────────────

def analysis_bootstrap_ci(df, n_boot=1000):
    """Bootstrap CI for optimal threshold."""
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS C: Bootstrap CI for Optimal Threshold")
    log.info("=" * 60)

    np.random.seed(42)
    opt_thresholds = []

    for i in range(n_boot):
        boot = df.sample(n=len(df), replace=True)
        best_acc = 0
        best_t = 1.0
        for t in np.arange(0.50, 1.50, 0.01):
            pred = (boot["sel_pli"] > t).astype(int)
            acc = (pred == boot["true_label"]).mean()
            if acc >= best_acc:
                best_acc = acc
                best_t = t
        opt_thresholds.append(best_t)

    opt_arr = np.array(opt_thresholds)
    ci_low = np.percentile(opt_arr, 2.5)
    ci_high = np.percentile(opt_arr, 97.5)
    median_t = np.median(opt_arr)

    log.info(f"  Median optimal threshold: {median_t:.2f}")
    log.info(f"  95% CI: [{ci_low:.2f} - {ci_high:.2f}]")
    log.info(f"  Mean: {opt_arr.mean():.2f}, SD: {opt_arr.std():.2f}")

    return {
        "median": round(float(median_t), 2),
        "ci_low": round(float(ci_low), 2),
        "ci_high": round(float(ci_high), 2),
        "mean": round(float(opt_arr.mean()), 2),
        "sd": round(float(opt_arr.std()), 2),
        "distribution": opt_arr.tolist(),
    }


# ── Analysis D: Extended validation sets ─────────────────────────────────────

def analysis_extended_validation(nv_comp):
    """Validate at p<0.05, p<0.10, and all cancers."""
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS D: Extended Validation at Multiple p Thresholds")
    log.info("=" * 60)

    results = {}
    for p_thresh in [0.05, 0.10, 0.20, 1.0]:
        df = build_validation_set(nv_comp, p_threshold=p_thresh)
        if len(df) == 0:
            continue

        # Test at threshold=1.0
        pred = (df["sel_pli"] > 1.0).astype(int)
        acc = (pred == df["true_label"]).mean()
        n_correct = (pred == df["true_label"]).sum()

        # Test two-criterion (1.0, 0.35)
        pred2 = ((df["sel_pli"] > 1.0) & (df["selectivity"] > 0.35)).astype(int)
        acc2 = (pred2 == df["true_label"]).mean()
        n_correct2 = (pred2 == df["true_label"]).sum()

        results[f"p<{p_thresh}"] = {
            "n_cancers": len(df),
            "single_threshold_acc": round(float(acc), 3),
            "single_threshold_correct": f"{n_correct}/{len(df)}",
            "two_criterion_acc": round(float(acc2), 3),
            "two_criterion_correct": f"{n_correct2}/{len(df)}",
        }
        log.info(f"  p<{p_thresh}: n={len(df)}, "
                 f"single={n_correct}/{len(df)} ({acc:.1%}), "
                 f"two-criterion={n_correct2}/{len(df)} ({acc2:.1%})")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(youden_df, best_youden, bootstrap, two_crit_df, best_two,
                 val_set, nv_comp, output_dir):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel A: Youden curve
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(youden_df["threshold"], youden_df["youden"], "b-", lw=2, label="Youden J")
    ax_a.plot(youden_df["threshold"], youden_df["accuracy"], "g--", lw=1.5, label="Accuracy")
    ax_a.axvline(best_youden["threshold"], color="red", ls="--", lw=1.5,
                 label=f"Optimal: {best_youden['threshold']:.2f}")
    ax_a.axvline(1.0, color="gray", ls=":", lw=1, label="Current: 1.00")
    ax_a.set_xlabel("Sel+pLI Threshold")
    ax_a.set_ylabel("Score")
    ax_a.set_title(f"Youden Index Optimization\nOptimal: {best_youden['threshold']:.2f} "
                    f"(J={best_youden['youden']:.2f})", fontweight="bold")
    ax_a.legend(fontsize=8)

    # Panel B: Sensitivity/Specificity
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(youden_df["threshold"], youden_df["sensitivity"], "b-", lw=2, label="Sensitivity")
    ax_b.plot(youden_df["threshold"], youden_df["specificity"], "r-", lw=2, label="Specificity")
    ax_b.axvline(best_youden["threshold"], color="purple", ls="--", lw=1.5)
    ax_b.axvline(1.0, color="gray", ls=":", lw=1)
    ax_b.set_xlabel("Sel+pLI Threshold")
    ax_b.set_ylabel("Rate")
    ax_b.set_title("Sensitivity & Specificity\nvs Threshold", fontweight="bold")
    ax_b.legend(fontsize=9)

    # Panel C: Bootstrap distribution
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.hist(bootstrap["distribution"], bins=30, color="#42A5F5", alpha=0.7, edgecolor="white")
    ax_c.axvline(bootstrap["median"], color="red", ls="--", lw=2,
                 label=f"Median: {bootstrap['median']:.2f}")
    ax_c.axvline(bootstrap["ci_low"], color="orange", ls=":", lw=1.5,
                 label=f"95% CI: [{bootstrap['ci_low']:.2f}-{bootstrap['ci_high']:.2f}]")
    ax_c.axvline(bootstrap["ci_high"], color="orange", ls=":", lw=1.5)
    ax_c.axvline(1.0, color="gray", ls="--", lw=1, label="Current: 1.00")
    ax_c.set_xlabel("Optimal Threshold")
    ax_c.set_ylabel("Count")
    ax_c.set_title("Bootstrap Distribution (n=1000)\nof Optimal Threshold", fontweight="bold")
    ax_c.legend(fontsize=8)

    # Panel D: Sel+pLI vs HR with threshold lines
    ax_d = fig.add_subplot(gs[1, 0])
    for _, row in val_set.iterrows():
        color = "#2E7D32" if row["observed"] == "protective" else "#C62828"
        ax_d.scatter(row["sel_pli"], row["hr"], s=120, color=color, zorder=3, edgecolors="white")
        ax_d.annotate(row["cancer"], (row["sel_pli"], row["hr"]),
                      fontsize=7, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")

    ax_d.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax_d.axvline(1.0, color="#E91E63", ls="--", lw=1.5, label="Threshold=1.0")
    ax_d.axvline(best_youden["threshold"], color="blue", ls=":", lw=1.5,
                 label=f"Youden optimal={best_youden['threshold']:.2f}")
    ax_d.set_xlabel("Sel+pLI")
    ax_d.set_ylabel("HR")
    ax_d.set_title("Sel+pLI vs HR\nSignificant cancer types (p<0.05)", fontweight="bold")
    ax_d.legend(fontsize=8)

    # Panel E: Two-criterion heatmap
    ax_e = fig.add_subplot(gs[1, 1])
    pivot = two_crit_df.pivot_table(index="selectivity_threshold", columns="sel_pli_threshold", values="accuracy")
    im = ax_e.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto",
                      extent=[pivot.columns.min()-0.025, pivot.columns.max()+0.025,
                              pivot.index.min()-0.025, pivot.index.max()+0.025],
                      origin="lower")
    ax_e.set_xlabel("Sel+pLI Threshold")
    ax_e.set_ylabel("Selectivity Threshold")
    ax_e.set_title(f"Two-Criterion Gate Accuracy\nBest: Sel+pLI>{best_two['sel_pli_threshold']:.2f} & "
                    f"Sel>{best_two['selectivity_threshold']:.2f}", fontweight="bold")
    plt.colorbar(im, ax=ax_e, label="Accuracy", shrink=0.8)

    # Panel F: Comparison summary
    ax_f = fig.add_subplot(gs[1, 2])
    labels = ["Current\n(1.0)", f"Youden\n({best_youden['threshold']:.2f})",
              f"Two-crit\n({best_two['sel_pli_threshold']:.2f}+{best_two['selectivity_threshold']:.2f})"]
    accs = [
        youden_df[youden_df["threshold"] == 1.0].iloc[0]["accuracy"],
        best_youden["accuracy"],
        best_two["accuracy"],
    ]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50"]
    bars = ax_f.bar(labels, accs, color=colors, alpha=0.8, edgecolor="gray")
    for bar, acc in zip(bars, accs):
        ax_f.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{acc:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax_f.set_ylabel("Accuracy")
    ax_f.set_title("Threshold Comparison\nSingle vs Two-Criterion", fontweight="bold")
    ax_f.set_ylim(0, 1.15)

    fig.suptitle("RevGate Finding 14 \u2014 Sel+pLI Threshold Optimization\n"
                 f"Youden optimal: {best_youden['threshold']:.2f} | "
                 f"Bootstrap 95% CI: [{bootstrap['ci_low']:.2f}\u2013{bootstrap['ci_high']:.2f}]",
                 fontsize=13, fontweight="bold")

    out = output_dir / "threshold_optimization.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Figure: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    nv_comp = load_nv_components()
    val_005 = build_validation_set(nv_comp, p_threshold=0.05)
    val_010 = build_validation_set(nv_comp, p_threshold=0.10)

    log.info(f"Validation set (p<0.05): {len(val_005)} cancers")
    log.info(f"Validation set (p<0.10): {len(val_010)} cancers")
    for _, r in val_005.iterrows():
        log.info(f"  {r['cancer']:5s} Sel+pLI={r['sel_pli']:.3f} sel={r['selectivity']:.3f} "
                 f"HR={r['hr']:.3f} dir={r['observed']}")

    youden_df, best_youden = analysis_youden_sweep(val_005)
    two_crit_df, best_two = analysis_two_criterion(val_005)
    bootstrap = analysis_bootstrap_ci(val_005)
    extended = analysis_extended_validation(nv_comp)

    # Print summary
    G = "\033[92m"; B = "\033[1m"; E = "\033[0m"
    print(f"\n{B}{'='*60}{E}")
    print(f"{B}  Finding 14 — Threshold Optimization{E}")
    print(f"{'='*60}")
    print(f"\n  {B}Youden optimal:{E} {best_youden['threshold']:.2f} "
          f"(J={best_youden['youden']:.2f}, acc={best_youden['accuracy']:.1%})")
    print(f"  {B}Current (1.0):{E} acc={youden_df[youden_df['threshold']==1.0].iloc[0]['accuracy']:.1%}")
    print(f"  {B}Bootstrap 95% CI:{E} [{bootstrap['ci_low']:.2f} \u2013 {bootstrap['ci_high']:.2f}]")
    print(f"  {B}Two-criterion best:{E} Sel+pLI>{best_two['sel_pli_threshold']:.2f} & "
          f"Sel>{best_two['selectivity_threshold']:.2f} (acc={best_two['accuracy']:.1%})")
    print(f"\n  {B}Extended validation:{E}")
    for k, v in extended.items():
        print(f"    {k}: single={v['single_threshold_correct']} ({v['single_threshold_acc']:.1%}), "
              f"two-crit={v['two_criterion_correct']} ({v['two_criterion_acc']:.1%})")
    print(f"{'='*60}\n")

    plot_results(youden_df, best_youden, bootstrap, two_crit_df, best_two,
                 val_005, nv_comp, output_dir)

    # Save
    out = {
        "timestamp": datetime.now().isoformat(),
        "finding": "Finding_14",
        "youden_optimal": best_youden,
        "current_threshold_accuracy": float(youden_df[youden_df["threshold"]==1.0].iloc[0]["accuracy"]),
        "bootstrap_ci": {k: v for k, v in bootstrap.items() if k != "distribution"},
        "two_criterion_best": best_two,
        "extended_validation": extended,
    }
    with open(output_dir / "threshold_optimization_results.json", "w") as f:
        json.dump(out, f, indent=2)
    youden_df.to_csv(output_dir / "threshold_youden_sweep.csv", index=False)

    log.info("Finding 14 complete.")


if __name__ == "__main__":
    main()
