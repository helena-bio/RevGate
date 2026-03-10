#!/usr/bin/env python3
# run_ssgsea_analysis.py -- Приоритет 1: Per-patient ssGSEA NV-Score + Cox + KM
#
# Употреба:
#   source /root/helix/venv/bin/activate
#   cd /root/revgate
#   python3 run_ssgsea_analysis.py
#
# Output:
#   /root/.revgate/cache/tcga/per_patient_ssgsea_scores.parquet
#   /root/revgate/results/ssgsea_kaplan_meier.png
#   /root/revgate/results/ssgsea_cox_results.csv

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ssgsea_analysis")

# --- Пътища ---
REVGATE_ROOT    = Path("/root/revgate")
CACHE_DIR       = Path("/root/.revgate/cache")
DEPMAP_CHRONOS  = CACHE_DIR / "depmap/CRISPRGeneEffect.csv"
DEPMAP_MODEL    = CACHE_DIR / "depmap/Model.csv"
EXPRESSION_DIR  = CACHE_DIR / "tcga"
CLINICAL_DIR    = CACHE_DIR / "tcga"
SSGSEA_OUTPUT   = CACHE_DIR / "tcga/per_patient_ssgsea_scores.parquet"
RESULTS_DIR     = REVGATE_ROOT / "results"

# Cancer типове с клинични данни
CANCER_IDS = ["BRCA", "KIRC", "LUAD", "LAML", "SKCM", "PAAD"]

# NV-Score прагове от handover
NV_A_THRESHOLD = 0.50  # recalibrated with equal weights, Finding_04
NV_C_THRESHOLD = 0.35  # recalibrated with equal weights, Finding_04


def load_clinical(cancer_id: str) -> pd.DataFrame:
    """Зареди TCGA клинични данни от raw TSV файл.

    Колони: days_to_death, days_to_last_followup, vital_status.
    Връща DataFrame с колони: patient_id, os_months, is_deceased.
    """
    path = CLINICAL_DIR / f"{cancer_id}_clinical_raw.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Липсва клиничен файл: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Изчисли OS в месеци
    # Приоритет: days_to_death -> days_to_last_followup
    days = pd.to_numeric(df.get("days_to_death", pd.Series(dtype=float)), errors="coerce")
    followup = pd.to_numeric(df.get("days_to_last_followup", pd.Series(dtype=float)), errors="coerce")

    # За живи пациенти: използваме followup
    os_days = days.where(days.notna(), followup)
    os_months = os_days / 30.44

    # Vital status
    vital = df.get("vital_status", pd.Series(dtype=str)).astype(str).str.lower()
    is_deceased = vital.isin(["dead", "deceased"])

    # Patient ID
    if "case_submitter_id" in df.columns:
        patient_id = df["case_submitter_id"].astype(str)
    elif "bcr_patient_barcode" in df.columns:
        patient_id = df["bcr_patient_barcode"].astype(str)
    else:
        patient_id = df.index.astype(str)

    result = pd.DataFrame({
        "patient_id": patient_id,
        "os_months": os_months,
        "is_deceased": is_deceased,
        "cancer_id": cancer_id,
    })

    # Премахни редове с невалидни OS стойности
    result = result[result["os_months"].notna() & (result["os_months"] >= 0)]
    logger.info(f"{cancer_id}: {len(result)} пациента с валидни клинични данни")
    return result


def assign_nv_class_from_ssgsea(scores: pd.Series) -> pd.Series:
    """Присвои NV клас на база ssGSEA NES score.

    NES се нормализира per-cancer-type към [0, 1] диапазона на
    NV-Score прагове чрез min-max скалиране.
    """
    # Min-max нормализация към [0.3, 0.65] — реалистичен NV-Score диапазон
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        normalized = pd.Series(0.5, index=scores.index)
    else:
        normalized = 0.3 + (scores - s_min) / (s_max - s_min) * 0.35

    classes = pd.cut(
        normalized,
        bins=[-np.inf, NV_C_THRESHOLD, NV_A_THRESHOLD, np.inf],
        labels=["NV-C", "NV-B", "NV-A"],
    )
    return classes


def run_cox(df: pd.DataFrame, cancer_id: str) -> dict:
    """Cox PH модел: ssGSEA NES score -> OS.

    Continuous predictor: ssgsea_nv_score.
    Връща: coef, HR, p_value, concordance.
    """
    from lifelines import CoxPHFitter

    cox_df = df[["os_months", "is_deceased", "ssgsea_nv_score"]].dropna()
    cox_df = cox_df[cox_df["os_months"] > 0]

    if len(cox_df) < 20:
        logger.warning(f"{cancer_id}: недостатъчно пациента за Cox ({len(cox_df)})")
        return {"cancer_id": cancer_id, "error": "insufficient_data"}

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="os_months", event_col="is_deceased")

    summary = cph.summary
    row = summary.loc["ssgsea_nv_score"]

    result = {
        "cancer_id": cancer_id,
        "n": len(cox_df),
        "coef": float(row["coef"]),
        "HR": float(row["exp(coef)"]),
        "HR_ci_lower": float(row["exp(coef) lower 95%"]),
        "HR_ci_upper": float(row["exp(coef) upper 95%"]),
        "p_value": float(row["p"]),
        "concordance": float(cph.concordance_index_),
    }

    logger.info(
        f"{cancer_id} Cox: HR={result['HR']:.3f} "
        f"[{result['HR_ci_lower']:.3f}-{result['HR_ci_upper']:.3f}], "
        f"p={result['p_value']:.4f}, C={result['concordance']:.3f}"
    )
    return result


def plot_kaplan_meier(combined: pd.DataFrame, output_path: Path) -> None:
    """Kaplan-Meier криви по NV клас от ssGSEA scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    cancer_ids = combined["cancer_id"].unique()
    n_cancers = len(cancer_ids)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = {"NV-A": "#2196F3", "NV-B": "#FF9800", "NV-C": "#F44336"}

    for idx, cancer_id in enumerate(sorted(cancer_ids)):
        ax = axes[idx]
        subset = combined[combined["cancer_id"] == cancer_id].dropna(
            subset=["os_months", "is_deceased", "nv_class"]
        )

        if len(subset) < 10:
            ax.set_title(f"{cancer_id} (недостатъчно данни)")
            continue

        kmf = KaplanMeierFitter()
        plotted_classes = []

        for nv_class in ["NV-A", "NV-B", "NV-C"]:
            group = subset[subset["nv_class"] == nv_class]
            if len(group) < 5:
                continue
            kmf.fit(
                group["os_months"],
                event_observed=group["is_deceased"],
                label=f"{nv_class} (n={len(group)})",
            )
            kmf.plot_survival_function(ax=ax, color=colors[nv_class], ci_show=True)
            plotted_classes.append(nv_class)

        # Log-rank test NV-A vs NV-C
        nva = subset[subset["nv_class"] == "NV-A"]
        nvc = subset[subset["nv_class"] == "NV-C"]
        p_str = ""
        if len(nva) >= 5 and len(nvc) >= 5:
            lr = logrank_test(
                nva["os_months"], nvc["os_months"],
                event_observed_A=nva["is_deceased"],
                event_observed_B=nvc["is_deceased"],
            )
            p_val = lr.p_value
            p_str = f"\nNV-A vs NV-C: p={p_val:.4f}"

        ax.set_title(f"{cancer_id} — ssGSEA NV-Class{p_str}", fontsize=10)
        ax.set_xlabel("Месеци")
        ax.set_ylabel("Преживяемост")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    # Скрий незапълнени axes
    for idx in range(n_cancers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Kaplan-Meier — Per-patient ssGSEA NV-Score (Приоритет 1)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"KM фигура записана: {output_path}")


def main() -> int:
    sys.path.insert(0, str(REVGATE_ROOT / "src"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Стъпка 1: ssGSEA scoring ---
    if SSGSEA_OUTPUT.exists():
        logger.info(f"Намерен кеш: {SSGSEA_OUTPUT} — зареждам директно")
        scores_df = pd.read_parquet(SSGSEA_OUTPUT)
    else:
        logger.info("Стартирам ssGSEA scoring...")
        from revgate.infrastructure.persistence.repositories.ssgsea_scorer import (
            run_ssgsea_scoring,
        )
        scores_df = run_ssgsea_scoring(
            dep_matrix_path=DEPMAP_CHRONOS,
            model_csv_path=DEPMAP_MODEL,
            expression_dir=EXPRESSION_DIR,
            output_path=SSGSEA_OUTPUT,
            cancer_ids=CANCER_IDS,
        )

    logger.info(f"ssGSEA scores: {len(scores_df)} пациента, колони: {scores_df.columns.tolist()}")

    # --- Стъпка 2: Обедини с клинични данни ---
    all_clinical: list[pd.DataFrame] = []
    for cancer_id in CANCER_IDS:
        try:
            clin = load_clinical(cancer_id)
            all_clinical.append(clin)
        except FileNotFoundError as e:
            logger.warning(str(e))

    if not all_clinical:
        logger.error("Няма клинични данни — прекратявам")
        return 1

    clinical_df = pd.concat(all_clinical, ignore_index=True)

    # Merge: scores_df.patient_id <-> clinical_df.patient_id
    # TCGA баркодовете могат да се различават по дължина — match по първите 12 символа
    scores_df["patient_id_short"] = scores_df["patient_id"].str[:12]
    clinical_df["patient_id_short"] = clinical_df["patient_id"].str[:12]

    combined = scores_df.merge(
        clinical_df[["patient_id_short", "os_months", "is_deceased"]],
        on="patient_id_short",
        how="inner",
    )

    logger.info(f"След merge: {len(combined)} пациента с expression + клинични данни")

    if len(combined) < 50:
        logger.error(f"Твърде малко пациента след merge ({len(combined)}) — проверете patient IDs")
        # Debug: покажи примерни IDs
        logger.info(f"Scores IDs пример: {scores_df['patient_id'].head(3).tolist()}")
        logger.info(f"Clinical IDs пример: {clinical_df['patient_id'].head(3).tolist()}")
        return 1

    # --- Стъпка 3: Присвои NV клас per cancer ---
    nv_classes: list[pd.Series] = []
    for cancer_id, group in combined.groupby("cancer_id"):
        classes = assign_nv_class_from_ssgsea(group["ssgsea_nv_score"])
        nv_classes.append(classes)

    combined["nv_class"] = pd.concat(nv_classes)

    logger.info("\n=== NV клас разпределение ===")
    logger.info(combined.groupby(["cancer_id", "nv_class"]).size().to_string())

    # --- Стъпка 4: Cox PH per cancer ---
    cox_results: list[dict] = []
    for cancer_id, group in combined.groupby("cancer_id"):
        result = run_cox(group.copy(), cancer_id)
        cox_results.append(result)

    cox_df = pd.DataFrame(cox_results)
    cox_output = RESULTS_DIR / "ssgsea_cox_results.csv"
    cox_df.to_csv(cox_output, index=False)
    logger.info(f"\n=== Cox резултати ===\n{cox_df.to_string()}")
    logger.info(f"Cox CSV записан: {cox_output}")

    # --- Стъпка 5: Kaplan-Meier ---
    km_output = RESULTS_DIR / "ssgsea_kaplan_meier.png"
    plot_kaplan_meier(combined, km_output)

    logger.info("\n=== ГОТОВО ===")
    logger.info(f"ssGSEA scores:  {SSGSEA_OUTPUT}")
    logger.info(f"Cox резултати:  {cox_output}")
    logger.info(f"KM фигура:      {km_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
