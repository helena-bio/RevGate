#!/usr/bin/env python3
# run_stage_stratified_cox.py -- Stage-stratified Cox PH анализ
#
# Въпрос: Предсказва ли ssGSEA NV-Score OS в рамките на един stage?
# Ако да -> независим биомаркер (публикационен стандарт)
# Ако не -> сигналът е изцяло медииран от stage
#
# Стратификация:
#   Early:    Stage I + II  (локализирано заболяване)
#   Advanced: Stage III + IV (локално авансирало / метастатично)
#
# Output:
#   results/stage_stratified_cox_results.csv
#   results/stage_stratified_forest.png

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage_stratified_cox")

REVGATE_ROOT  = Path("/root/revgate")
CACHE_DIR     = Path("/root/.revgate/cache")
CLINICAL_DIR  = CACHE_DIR / "tcga"
SSGSEA_SCORES = CACHE_DIR / "tcga/per_patient_ssgsea_scores.parquet"
RESULTS_DIR   = REVGATE_ROOT / "results"

CANCER_IDS = ["BRCA", "KIRC", "LUAD", "SKCM", "PAAD"]  # LAML няма TNM stage

STAGE_MAP: dict[str, float] = {
    "stage i":    1.0, "stage ia":   1.0, "stage ib":   1.0,
    "stage ii":   2.0, "stage iia":  2.0, "stage iib":  2.0, "stage iic":  2.0,
    "stage iii":  3.0, "stage iiia": 3.0, "stage iiib": 3.0, "stage iiic": 3.0,
    "stage iv":   4.0,
}


def load_clinical(cancer_id: str) -> pd.DataFrame:
    """Зареди клинични данни с OS, vital status, stage и patient ID."""
    path = CLINICAL_DIR / f"{cancer_id}_clinical_raw.tsv"
    df = pd.read_csv(path, sep="\t", low_memory=False)

    days     = pd.to_numeric(df.get("days_to_death", pd.Series(dtype=float)), errors="coerce")
    followup = pd.to_numeric(df.get("days_to_last_followup", pd.Series(dtype=float)), errors="coerce")
    os_months = days.where(days.notna(), followup) / 30.44

    vital      = df.get("vital_status", pd.Series(dtype=str)).astype(str).str.lower()
    is_deceased = vital.isin(["dead", "deceased"]).astype(int)

    # Patient ID -- универсален lookup
    if "_PATIENT" in df.columns and df["_PATIENT"].astype(str).str.startswith("TCGA").any():
        pid = df["_PATIENT"].astype(str).str[:12]
    elif "bcr_patient_barcode" in df.columns:
        pid = df["bcr_patient_barcode"].astype(str).str[:12]
    elif "patient_id" in df.columns:
        pid = df["patient_id"].astype(str).str[:12]
    else:
        pid = df.index.astype(str)

    stage_raw = df.get("pathologic_stage", pd.Series(dtype=str))
    stage_num = stage_raw.astype(str).str.lower().str.strip().map(STAGE_MAP)

    return pd.DataFrame({
        "patient_id_short": pid,
        "os_months":        os_months,
        "is_deceased":      is_deceased,
        "stage":            stage_num,
    })


def cox_univariate_ssgsea(df: pd.DataFrame, label: str) -> dict:
    """Univariate Cox: ssgsea_nv_score -> OS в рамките на stratum.

    Univariate тук е правилен -- stage вече е фиксиран чрез стратификацията.
    Единственият predictor е ssGSEA score.
    """
    from lifelines import CoxPHFitter

    cox_df = df[["os_months", "is_deceased", "ssgsea_nv_score"]].dropna()
    cox_df = cox_df[cox_df["os_months"] > 0].copy()

    if len(cox_df) < 20:
        return {"label": label, "n": len(cox_df), "error": "insufficient_data"}

    # Стандартизирай ssGSEA score
    mu, sd = cox_df["ssgsea_nv_score"].mean(), cox_df["ssgsea_nv_score"].std()
    if sd > 0:
        cox_df["ssgsea_nv_score"] = (cox_df["ssgsea_nv_score"] - mu) / sd

    # Penalizer за стабилност при малки групи
    penalizer = 0.1 if len(cox_df) < 100 else 0.0

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cox_df, duration_col="os_months", event_col="is_deceased")
        row = cph.summary.loc["ssgsea_nv_score"]
        return {
            "label":       label,
            "n":           len(cox_df),
            "n_events":    int(cox_df["is_deceased"].sum()),
            "HR":          float(row["exp(coef)"]),
            "HR_lower":    float(row["exp(coef) lower 95%"]),
            "HR_upper":    float(row["exp(coef) upper 95%"]),
            "p":           float(row["p"]),
            "concordance": float(cph.concordance_index_),
        }
    except Exception as exc:
        return {"label": label, "n": len(cox_df), "error": str(exc)}


def plot_stage_forest(results_df: pd.DataFrame, output_path: Path) -> None:
    """Forest plot: Early vs Advanced stage за всеки cancer тип."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Само редове без грешка
    df = results_df[results_df["error"].isna()].copy()
    df = df.sort_values(["cancer_id", "stage_group"])

    n = len(df)
    if n == 0:
        logger.warning("Няма данни за forest plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.7)))

    # Цветове по stage group
    color_map = {"Early (I-II)": "#1976D2", "Advanced (III-IV)": "#D32F2F"}
    marker_map = {"Early (I-II)": "o", "Advanced (III-IV)": "s"}

    for i, (_, row) in enumerate(df.iterrows()):
        color  = color_map.get(row["stage_group"], "#9E9E9E")
        marker = marker_map.get(row["stage_group"], "o")
        alpha  = 1.0 if row["p"] < 0.05 else 0.45

        ax.errorbar(
            x=row["HR"], y=i,
            xerr=[[row["HR"] - row["HR_lower"]], [row["HR_upper"] - row["HR"]]],
            fmt=marker, color=color, ecolor=color,
            capsize=4, markersize=8, alpha=alpha, zorder=3,
        )

        # p-value анотация
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else "p<0.001"
        sig   = "**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else "")
        ax.text(
            ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 3.0,
            i, f" {sig}{p_str}", va="center", fontsize=8, color=color,
        )

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)

    labels = [
        f"{row['cancer_id']} {row['stage_group']}  "
        f"(n={row['n']}, events={row['n_events']})  "
        f"HR={row['HR']:.2f} [{row['HR_lower']:.2f}-{row['HR_upper']:.2f}]"
        for _, row in df.iterrows()
    ]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (ssGSEA NV-Score, standardized, log scale)", fontsize=11)
    ax.set_title(
        "Stage-Stratified Cox PH — ssGSEA NV-Score within Stage Groups\n"
        "Solid markers = p<0.05 | Faded = p≥0.05",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1976D2", markersize=9, label="Early stage (I-II)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#D32F2F", markersize=9, label="Advanced stage (III-IV)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Stage forest plot записан: {output_path}")


def main() -> int:
    sys.path.insert(0, str(REVGATE_ROOT / "src"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SSGSEA_SCORES.exists():
        logger.error(f"Липсват ssGSEA scores: {SSGSEA_SCORES}")
        logger.error("Пусни първо: python3 run_ssgsea_analysis.py")
        return 1

    scores_df = pd.read_parquet(SSGSEA_SCORES)
    scores_df["patient_id_short"] = scores_df["patient_id"].str[:12]

    all_results: list[dict] = []

    for cancer_id in CANCER_IDS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Обработвам {cancer_id}...")

        clin_path = CLINICAL_DIR / f"{cancer_id}_clinical_raw.tsv"
        if not clin_path.exists():
            logger.warning(f"Липсва клиничен файл за {cancer_id}")
            continue

        clin_df   = load_clinical(cancer_id)
        cancer_sc = scores_df[scores_df["cancer_id"] == cancer_id]
        merged    = cancer_sc.merge(clin_df, on="patient_id_short", how="inner")
        merged    = merged[merged["os_months"].notna() & (merged["os_months"] > 0)]

        logger.info(f"{cancer_id}: {len(merged)} пациента след merge")

        # Stage разпределение
        stage_counts = merged["stage"].value_counts().sort_index()
        logger.info(f"Stage разпределение:\n{stage_counts.to_string()}")

        # --- Early stage: I + II ---
        early = merged[merged["stage"].isin([1.0, 2.0])].copy()
        res_early = cox_univariate_ssgsea(early, f"{cancer_id} Early (I-II)")
        res_early["cancer_id"]    = cancer_id
        res_early["stage_group"]  = "Early (I-II)"
        res_early["error"]        = res_early.get("error")
        all_results.append(res_early)

        # --- Advanced stage: III + IV ---
        advanced = merged[merged["stage"].isin([3.0, 4.0])].copy()
        res_adv = cox_univariate_ssgsea(advanced, f"{cancer_id} Advanced (III-IV)")
        res_adv["cancer_id"]   = cancer_id
        res_adv["stage_group"] = "Advanced (III-IV)"
        res_adv["error"]       = res_adv.get("error")
        all_results.append(res_adv)

        # Лог резултатите
        for res in [res_early, res_adv]:
            if res.get("error"):
                logger.warning(f"  {res['label']}: {res['error']}")
            else:
                logger.info(
                    f"  {res['label']}: n={res['n']}, events={res['n_events']}, "
                    f"HR={res['HR']:.3f} [{res['HR_lower']:.3f}-{res['HR_upper']:.3f}], "
                    f"p={res['p']:.4f}, C={res['concordance']:.3f}"
                )

    # --- Запази резултатите ---
    results_df = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "stage_stratified_cox_results.csv"
    results_df.to_csv(out_csv, index=False)

    # --- Финален summary ---
    logger.info(f"\n{'='*60}")
    logger.info("=== STAGE-STRATIFIED COX ФИНАЛНИ РЕЗУЛТАТИ ===")
    valid = results_df[results_df["error"].isna()]
    logger.info(f"\n{valid[['cancer_id','stage_group','n','n_events','HR','HR_lower','HR_upper','p','concordance']].to_string()}")

    # Значими резултати
    sig = valid[valid["p"] < 0.05]
    if len(sig) > 0:
        logger.info(f"\n✅ ЗНАЧИМИ РЕЗУЛТАТИ (p<0.05):")
        logger.info(f"\n{sig[['cancer_id','stage_group','HR','p']].to_string()}")
    else:
        logger.info("\n⚠️  Няма значими резултати при p<0.05")

    # --- Forest plot ---
    plot_stage_forest(results_df, RESULTS_DIR / "stage_stratified_forest.png")

    logger.info(f"\nЗаписано: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
