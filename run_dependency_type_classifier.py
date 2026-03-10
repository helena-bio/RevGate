#!/usr/bin/env python3
"""
RevGate — Dependency Type Classifier v2 (Finding_07 → Приоритет 2)
===================================================================
Dependency Architecture Ratio (DAR) — oncogene vs housekeeping.

КРИТИЧНА ПРОМЯНА спрямо v1:
  v1 грешка: top-20 по absolute median Chronos = pan-essential гени навсякъде.
  v2 fix: top-20 по DIFFERENTIAL dependency = cancer_median - global_median.
  Това са cancer-SPECIFIC гени — същата методология като ssGSEA gene set derivation.

DAR = fraction of top-20 differential deps with pLI > 0.5
    DAR > 0.50 → oncogene-driven → HR < 1 (protective)
    DAR < 0.50 → housekeeping-driven → HR > 1 (harmful)

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
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dep_type_classifier")

# Pan-essential гени — изключваме преди ranking
PAN_ESSENTIAL_PREFIXES = ("RPL", "RPS")
PAN_ESSENTIAL_GENES = {
    "RAN", "SNRPD3", "HSPE1", "RRM1", "CHMP4B", "GPX4", "HSPA9",
    "PLK1", "RRM2", "SNRPA1", "SNRPF", "PSMB3", "PSMA3", "PSMA6",
    "SMU1", "SRSF3", "SRSF1", "CDC20", "PCNA",
    "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "TOP2A", "TYMS", "NOP56", "NOP58", "FBL",
}

# DAR класификация: pLI > 0.5 = constrained (oncogene-типичен)
PLI_THRESHOLD = 0.5

# Валидация
VALIDATION = {
    "Breast":  {"hr": 0.813, "direction": "protective", "finding": "F01 BRCA Early"},
    "Kidney":  {"hr": 0.872, "direction": "protective", "finding": "F01 KIRC Adv"},
    "Lung":    {"hr": 1.289, "direction": "harmful",    "finding": "F01 LUAD Early"},
    "Skin":    {"hr": None,  "direction": "protective", "finding": "Expected (BRAF)"},
    "Myeloid": {"hr": None,  "direction": "protective", "finding": "Expected (MYB/ABL1)"},
}

LINEAGE_TO_TCGA = {
    "Lung": "LUAD", "Breast": "BRCA", "Kidney": "KIRC",
    "Skin": "SKCM", "Myeloid": "LAML", "Pancreas": "PAAD",
}


def is_pan_essential(gene: str) -> bool:
    if any(gene.startswith(p) for p in PAN_ESSENTIAL_PREFIXES):
        return True
    return gene in PAN_ESSENTIAL_GENES


def load_data(cache_dir: str) -> tuple:
    cache = Path(cache_dir)

    log.info("Зареждане DepMap Chronos...")
    chronos = pd.read_csv(cache / "depmap/CRISPRGeneEffect.csv", index_col=0)
    chronos.columns = [c.split(" (")[0] for c in chronos.columns]
    log.info(f"  {chronos.shape[0]} cell lines x {chronos.shape[1]} genes")

    log.info("Зареждане Model metadata...")
    meta = pd.read_csv(cache / "depmap/Model.csv")

    log.info("Зареждане gnomAD pLI...")
    gnomad = pd.read_csv(cache / "gnomad/gnomad.v4.constraint.tsv",
                         sep="\t", low_memory=False)
    pli_dict = dict(zip(gnomad["gene"], gnomad["lof_hc_lc.pLI"]))
    log.info(f"  {len(pli_dict)} genes с pLI scores")

    nv_comp = pd.read_csv(cache / "nv_components_all_lineages.csv", index_col=0)

    return chronos, meta, pli_dict, nv_comp


def get_lineage_cell_lines(chronos, meta, lineage):
    id_col = "ModelID" if "ModelID" in meta.columns else "DepMap_ID"
    mask = meta["OncotreeLineage"].str.contains(lineage, case=False, na=False)
    ids = meta.loc[mask, id_col]
    available = chronos.index.intersection(ids)
    return chronos.loc[available]


def classify_lineage(lineage_chronos, all_chronos, pli_dict,
                     lineage_name, top_n=20):
    """
    v2: Класификация по DIFFERENTIAL dependency гени.
    diff_score = lineage_median - global_median (nsmallest = най-специфичните)
    За всеки от top-N differential гени: pLI > 0.5 = oncogene-driven.
    """
    if lineage_chronos.empty:
        return {"lineage": lineage_name, "error": "no cell lines"}

    # Филтриране на pan-essential
    non_pan = [g for g in lineage_chronos.columns if not is_pan_essential(g)]
    filtered = lineage_chronos[non_pan]

    # КЛЮЧОВА ПРОМЯНА: differential score, не absolute ranking
    lineage_medians = filtered.median(axis=0)
    global_medians = all_chronos[non_pan].median(axis=0)
    diff_scores = lineage_medians - global_medians  # по-негативно = по-специфично

    # Top-N най-специфични гени за този lineage
    top_diff = diff_scores.nsmallest(top_n)
    top_genes = list(top_diff.index)

    # Класификация по pLI
    gene_details = []
    n_oncogene = 0
    n_housekeeping = 0

    for gene in top_genes:
        pli = pli_dict.get(gene, np.nan)
        diff = float(diff_scores[gene])
        lin_med = float(lineage_medians.get(gene, 0))
        glob_med = float(global_medians.get(gene, 0))

        has_high_pli = not np.isnan(pli) and pli > PLI_THRESHOLD

        if has_high_pli:
            dep_type = "oncogene-driven"
            n_oncogene += 1
        else:
            dep_type = "housekeeping"
            n_housekeeping += 1

        gene_details.append({
            "gene": gene,
            "lineage_median": round(lin_med, 4),
            "global_median": round(glob_med, 4),
            "diff_score": round(diff, 4),
            "pLI": round(float(pli), 4) if not np.isnan(pli) else None,
            "dep_type": dep_type,
        })

    dar = n_oncogene / top_n if top_n > 0 else 0.0
    predicted_direction = "protective" if dar > 0.50 else "harmful"

    return {
        "lineage": lineage_name,
        "n_cell_lines": len(lineage_chronos),
        "top_n": top_n,
        "n_oncogene_driven": n_oncogene,
        "n_housekeeping": n_housekeeping,
        "dar": round(dar, 4),
        "predicted_hr_direction": predicted_direction,
        "top5_differential": [g["gene"] for g in gene_details[:5]],
        "top3_oncogene": [g["gene"] for g in gene_details
                          if g["dep_type"] == "oncogene-driven"][:3],
        "top3_housekeeping": [g["gene"] for g in gene_details
                              if g["dep_type"] == "housekeeping"][:3],
        "gene_details": gene_details,
    }


def validate_predictions(results):
    validations = []
    correct = 0
    total = 0

    for r in results:
        lineage = r["lineage"]
        if lineage not in VALIDATION:
            continue
        expected = VALIDATION[lineage]
        predicted = r["predicted_hr_direction"]
        match = predicted == expected["direction"]
        if match:
            correct += 1
        total += 1

        validations.append({
            "lineage": lineage,
            "dar": r["dar"],
            "predicted": predicted,
            "observed_hr": expected["hr"],
            "observed_direction": expected["direction"],
            "finding_ref": expected["finding"],
            "correct": match,
        })

    accuracy = correct / total if total > 0 else 0
    return {
        "validations": validations,
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
    }


def plot_results(results, validation, output_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

    results_sorted = sorted(results, key=lambda x: x["dar"], reverse=True)

    # ── Panel A: DAR bar chart ───────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    lineages = [r["lineage"] for r in results_sorted]
    dars = [r["dar"] for r in results_sorted]
    colors = ["#4CAF50" if d > 0.50 else ("#FF9800" if d > 0.35 else "#F44336")
              for d in dars]

    bars = ax_a.bar(range(len(lineages)), dars, color=colors, alpha=0.8,
                    edgecolor="gray", linewidth=0.5)
    ax_a.axhline(0.50, color="black", linestyle="--", linewidth=2,
                 label="DAR=0.50 (oncogene/housekeeping boundary)")
    ax_a.set_xticks(range(len(lineages)))
    ax_a.set_xticklabels(lineages, rotation=50, ha="right", fontsize=7.5)
    ax_a.set_ylabel("DAR (fraction of top-20 diff. genes with pLI > 0.5)", fontsize=9)
    ax_a.set_title(
        "Dependency Architecture Ratio (DAR) — Differential Dependencies\n"
        "Green: oncogene-driven (DAR>0.50) | Orange: mixed | Red: housekeeping (DAR<0.35)",
        fontsize=10, fontweight="bold",
    )
    ax_a.set_ylim(0, 1.0)
    ax_a.legend(fontsize=9)

    for i, r in enumerate(results_sorted):
        if r["lineage"] in VALIDATION and VALIDATION[r["lineage"]]["hr"]:
            hr = VALIDATION[r["lineage"]]["hr"]
            color = "#1B5E20" if hr < 1 else "#B71C1C"
            ax_a.text(i, r["dar"] + 0.02, f"HR={hr:.3f}",
                      ha="center", fontsize=7, color=color, fontweight="bold")

        # Top gene annotation
        if r.get("top5_differential"):
            ax_a.text(i, r["dar"] - 0.04,
                      r["top5_differential"][0],
                      ha="center", fontsize=5.5, color="gray", rotation=90)

    # ── Panel B: Stacked composition ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    n_onco = [r["n_oncogene_driven"] for r in results_sorted]
    n_house = [r["n_housekeeping"] for r in results_sorted]
    x = range(len(lineages))

    ax_b.bar(x, n_onco, color="#4CAF50", alpha=0.8, label="Oncogene-driven (pLI>0.5)")
    ax_b.bar(x, n_house, bottom=n_onco, color="#F44336", alpha=0.8,
             label="Housekeeping (pLI≤0.5)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(lineages, rotation=50, ha="right", fontsize=6.5)
    ax_b.set_ylabel("Number of top-20 differential genes", fontsize=9)
    ax_b.set_title("Gene Composition per Lineage\n(Differential Dependencies)",
                    fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=8)

    # ── Panel C: Validation table ────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.axis("off")

    val_data = validation["validations"]
    if val_data:
        col_labels = ["Lineage", "DAR", "Predicted", "Obs. HR", "Match"]
        table_data = []
        for v in val_data:
            match_str = "✓" if v["correct"] else "✗"
            hr_str = f"{v['observed_hr']:.3f}" if v["observed_hr"] else "N/A"
            table_data.append([
                v["lineage"], f"{v['dar']:.3f}", v["predicted"],
                hr_str, match_str,
            ])

        table = ax_c.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        for i, v in enumerate(val_data):
            color = "#E8F5E9" if v["correct"] else "#FFEBEE"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

    acc_pct = validation["accuracy"] * 100
    ax_c.set_title(
        f"Validation vs Finding_01/07: {validation['correct']}/{validation['total']} "
        f"({acc_pct:.0f}%)",
        fontsize=10, fontweight="bold",
    )

    fig.suptitle(
        "RevGate — Dependency Architecture Principle v2 (Differential Dependencies)\n"
        "DAR = fraction of cancer-specific top-20 genes with pLI > 0.5",
        fontsize=12, fontweight="bold",
    )

    out = output_dir / "dependency_type_classifier.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Фигура: {out}")
    return out


def print_summary(results, validation):
    G = "\033[92m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"; Y = "\033[93m"

    print(f"\n{B}{'='*78}{E}")
    print(f"{B}  RevGate — Dependency Type Classifier v2 (Differential Dependencies){E}")
    print(f"{'='*78}")

    results_sorted = sorted(results, key=lambda x: x["dar"], reverse=True)

    print(f"\n{B}{'Lineage':<25} {'DAR':>6} {'Onco':>5} {'House':>5} {'Predicted':>12} {'Top5 Differential Genes':<40}{E}")
    print(f"{'-'*25} {'-'*6} {'-'*5} {'-'*5} {'-'*12} {'-'*40}")

    for r in results_sorted:
        dar_col = G if r["dar"] > 0.50 else (Y if r["dar"] > 0.35 else R)
        pred = r["predicted_hr_direction"]
        pred_col = G if pred == "protective" else R
        top5 = ", ".join(r.get("top5_differential", [])[:5])
        print(
            f"  {r['lineage']:<23} {dar_col}{r['dar']:>6.3f}{E} "
            f"{r['n_oncogene_driven']:>5} {r['n_housekeeping']:>5} "
            f"{pred_col}{pred:>12}{E} {top5:<40}"
        )

    print(f"\n{B}Валидация срещу Finding_01/07:{E}")
    for v in validation["validations"]:
        match_str = f"{G}✓ CORRECT{E}" if v["correct"] else f"{R}✗ WRONG{E}"
        hr_str = f"HR={v['observed_hr']:.3f}" if v["observed_hr"] else "HR=N/A"
        print(
            f"  {v['lineage']:<15} DAR={v['dar']:.3f} → {v['predicted']:<12} "
            f"| Observed: {hr_str} ({v['observed_direction']}) "
            f"| {match_str}  [{v['finding_ref']}]"
        )

    acc = validation["accuracy"]
    acc_col = G if acc >= 0.80 else (Y if acc >= 0.60 else R)
    print(f"\n{B}  Accuracy: {acc_col}{validation['correct']}/{validation['total']} "
          f"({acc*100:.0f}%){E}")
    print(f"{'='*78}\n")


def save_results(results, validation, output_dir):
    out = {
        "timestamp": datetime.now().isoformat(),
        "version": "v2 — differential dependencies",
        "method": "DAR = fraction of top-20 differential deps with pLI > 0.5",
        "criteria": {
            "pLI_threshold": PLI_THRESHOLD,
            "gene_selection": "diff_score = lineage_median - global_median, nsmallest(20)",
            "classification_rule": "pLI > 0.5 = oncogene-driven, else housekeeping",
        },
        "results": [{k: v for k, v in r.items() if k != "gene_details"}
                    for r in results],
        "gene_details": {r["lineage"]: r["gene_details"] for r in results},
        "validation": validation,
        "scientific_principle": (
            "Dependency Architecture Principle (Finding_07, v2): "
            "NV-Score predicts favorable survival (HR<1) in cancer types whose "
            "cancer-SPECIFIC dependencies are dominated by high-pLI (constrained) genes. "
            "In cancers where specific dependencies are low-pLI (housekeeping/bypassable), "
            "NV-Score reflects proliferation and predicts worse survival (HR>1)."
        ),
    }
    p = output_dir / "dependency_type_classifier_results.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info(f"JSON: {p}")

    summary = pd.DataFrame([
        {
            "lineage": r["lineage"],
            "n_cell_lines": r["n_cell_lines"],
            "dar": r["dar"],
            "n_oncogene": r["n_oncogene_driven"],
            "n_housekeeping": r["n_housekeeping"],
            "predicted_direction": r["predicted_hr_direction"],
            "top5_differential": "; ".join(r.get("top5_differential", [])),
            "top3_oncogene": "; ".join(r["top3_oncogene"]),
            "top3_housekeeping": "; ".join(r["top3_housekeeping"]),
        }
        for r in results
    ])
    csv_path = output_dir / "dependency_type_classifier_summary.csv"
    summary.to_csv(csv_path, index=False)
    log.info(f"CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RevGate Dependency Type Classifier v2 (Differential Dependencies)"
    )
    parser.add_argument("--cache", default="/root/.revgate/cache")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    chronos, meta, pli_dict, nv_comp = load_data(args.cache)

    # Глобални медиани за differential scoring
    non_pan_all = [g for g in chronos.columns if not is_pan_essential(g)]

    results = []
    lineages_to_analyze = nv_comp.index.tolist()

    log.info(f"\nКласификация на {len(lineages_to_analyze)} lineages (DIFFERENTIAL method)...")

    for lineage in lineages_to_analyze:
        lineage_df = get_lineage_cell_lines(chronos, meta, lineage)
        if lineage_df.empty:
            log.warning(f"  {lineage}: няма cell lines — skip")
            continue

        result = classify_lineage(lineage_df, chronos, pli_dict,
                                  lineage, args.top_n)
        results.append(result)

        top5 = ", ".join(result.get("top5_differential", [])[:3])
        log.info(
            f"  {lineage:<25}: DAR={result['dar']:.3f} "
            f"(onco={result['n_oncogene_driven']}, house={result['n_housekeeping']}) "
            f"→ {result['predicted_hr_direction']:<11} [{top5}]"
        )

    validation = validate_predictions(results)

    print_summary(results, validation)
    plot_results(results, validation, output_dir)
    save_results(results, validation, output_dir)

    log.info("Dependency Type Classifier v2 завършен.")


if __name__ == "__main__":
    main()
