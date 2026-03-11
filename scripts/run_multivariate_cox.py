#!/usr/bin/env python3
# run_multivariate_cox.py -- Публикационен стандарт: Multivariate Cox PH
#
# Cox модел контролиран за: age, pathologic_stage, молекулярен подтип
# Predictor: ssgsea_nv_score (от Приоритет 1)
#
# Употреба:
#   source /root/helix/venv/bin/activate
#   cd /root/revgate
#   python3 run_multivariate_cox.py
#
# Output:
#   results/multivariate_cox_results.csv
#   results/multivariate_cox_forest.png

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
logger = logging.getLogger("run_multivariate_cox")

# --- Пътища ---
REVGATE_ROOT  = Path("/root/revgate")
CACHE_DIR     = Path("/root/.revgate/cache")
CLINICAL_DIR  = CACHE_DIR / "tcga"
SSGSEA_SCORES = CACHE_DIR / "tcga/per_patient_ssgsea_scores.parquet"
RESULTS_DIR   = REVGATE_ROOT / "results"

CANCER_IDS = ["BRCA", "KIRC", "LUAD", "LAML", "SKCM", "PAAD"]


# ---------------------------------------------------------------------------
# Stage encoding -- публикационен стандарт: ordinal I->1 .. IV->4
# ---------------------------------------------------------------------------
STAGE_MAP: dict[str, float] = {
    "stage i":   1.0, "stage ia":  1.0, "stage ib":  1.0,
    "stage ii":  2.0, "stage iia": 2.0, "stage iib": 2.0, "stage iic": 2.0,
    "stage iii": 3.0, "stage iiia":3.0, "stage iiib":3.0, "stage iiic":3.0,
    "stage iv":  4.0,
}


def encode_stage(series: pd.Series) -> pd.Series:
    """Ordinal encoding на pathologic_stage. Невалидни -> NaN."""
    return series.str.lower().str.strip().map(STAGE_MAP)


def load_clinical_with_covariates(cancer_id: str) -> pd.DataFrame:
    """Зареди клинични данни с всички covariates за multivariate Cox.

    Връща DataFrame с колони:
        patient_id_short, os_months, is_deceased,
        age (стандартизирано), stage (ordinal 1-4),
        + cancer-специфични covariates
    """
    path = CLINICAL_DIR / f"{cancer_id}_clinical_raw.tsv"
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # --- OS ---
    days = pd.to_numeric(df.get("days_to_death", pd.Series(dtype=float)), errors="coerce")
    followup = pd.to_numeric(df.get("days_to_last_followup", pd.Series(dtype=float)), errors="coerce")
    os_days = days.where(days.notna(), followup)
    os_months = os_days / 30.44

    # --- Vital status ---
    vital = df.get("vital_status", pd.Series(dtype=str)).astype(str).str.lower()
    is_deceased = vital.isin(["dead", "deceased"]).astype(int)

    # --- Patient ID (първи 12 символа за merge) ---
    # Приоритет: _PATIENT (LAML формат) -> bcr_patient_barcode -> patient_id -> index
    if "_PATIENT" in df.columns and df["_PATIENT"].astype(str).str.startswith("TCGA").any():
        pid = df["_PATIENT"].astype(str).str[:12]
    elif "bcr_patient_barcode" in df.columns:
        pid = df["bcr_patient_barcode"].astype(str).str[:12]
    elif "patient_id" in df.columns:
        pid = df["patient_id"].astype(str).str[:12]
    else:
        pid = df.index.astype(str)

    # --- Age (continuous, ще се стандартизира по-долу) ---
    age = pd.to_numeric(df.get("age_at_initial_pathologic_diagnosis", pd.Series(dtype=float)), errors="coerce")

    # --- Stage (ordinal) ---
    stage = encode_stage(df.get("pathologic_stage", pd.Series(dtype=str)))

    result = pd.DataFrame({
        "patient_id_short": pid,
        "os_months": os_months,
        "is_deceased": is_deceased,
        "age": age,
        "stage": stage,
    })

    # --- Cancer-специфични covariates ---
    if cancer_id == "BRCA":
        # ER статус: Positive=1, иначе=0
        er = df.get("ER_Status_nature2012", pd.Series(dtype=str)).astype(str)
        result["er_positive"] = (er.str.lower() == "positive").astype(float)
        result["er_positive"] = result["er_positive"].where(~er.str.lower().isin(["nan", ""]), np.nan)

        # HER2 статус: Positive=1, иначе=0
        her2 = df.get("HER2_Final_Status_nature2012", pd.Series(dtype=str)).astype(str)
        result["her2_positive"] = (her2.str.lower() == "positive").astype(float)
        result["her2_positive"] = result["her2_positive"].where(~her2.str.lower().isin(["nan", ""]), np.nan)

    if cancer_id == "LUAD":
        # Expression subtype: dummy encoding (Bronchioid е референция)
        subtype = df.get("Expression_Subtype", pd.Series(dtype=str)).astype(str).str.lower().str.strip()
        result["subtype_squamoid"] = (subtype == "squamoid").astype(float)
        result["subtype_magnoid"]  = (subtype == "magnoid").astype(float)
        # NaN за непознати subtypes
        unknown_mask = subtype.isin(["nan", "", "unknown"])
        result.loc[unknown_mask, "subtype_squamoid"] = np.nan
        result.loc[unknown_mask, "subtype_magnoid"]  = np.nan

    if cancer_id == "LAML":
        # AML няма TNM stage -- използваме CALGB цитогенетичен риск
        # Favorable=0, Intermediate=1, Poor/N.D.=NaN (референция: Favorable)
        risk = df.get(
            "acute_myeloid_leukemia_calgb_cytogenetics_risk_category",
            pd.Series(dtype=str),
        ).astype(str).str.lower().str.strip()
        result["cytogenetic_intermediate"] = (risk.str.contains("intermediate", na=False)).astype(float)
        result["cytogenetic_poor"]         = (risk.str.contains("poor", na=False)).astype(float)
        # NaN за непознати стойности
        unknown_risk = risk.isin(["nan", "", "n.d.", "not reported"])
        result.loc[unknown_risk, "cytogenetic_intermediate"] = np.nan
        result.loc[unknown_risk, "cytogenetic_poor"]         = np.nan
        # LAML няма pathologic_stage -- изчисти stage колоната
        result["stage"] = np.nan

    return result


def run_multivariate_cox(
    merged: pd.DataFrame,
    cancer_id: str,
    covariates: list[str],
) -> dict:
    """Multivariate Cox PH модел.

    Args:
        merged:     DataFrame с os_months, is_deceased, ssgsea_nv_score + covariates.
        cancer_id:  За logging.
        covariates: Списък с covariate колони за включване.

    Returns:
        Dict с резултати за всеки predictor.
    """
    from lifelines import CoxPHFitter

    # Избери само нужните колони + drop NaN
    cols = ["os_months", "is_deceased", "ssgsea_nv_score"] + covariates
    available_cols = [c for c in cols if c in merged.columns]
    cox_df = merged[available_cols].dropna()
    cox_df = cox_df[cox_df["os_months"] > 0]

    if len(cox_df) < 30:
        logger.warning(f"{cancer_id}: недостатъчно пациента за multivariate Cox ({len(cox_df)})")
        return {"cancer_id": cancer_id, "n": len(cox_df), "error": "insufficient_data"}

    # Стандартизирай age и ssgsea_nv_score за сравними коефициенти
    for col in ["age", "ssgsea_nv_score"]:
        if col in cox_df.columns:
            mu, sd = cox_df[col].mean(), cox_df[col].std()
            if sd > 0:
                cox_df[col] = (cox_df[col] - mu) / sd

    used_covariates = [c for c in covariates if c in cox_df.columns]
    logger.info(f"{cancer_id}: n={len(cox_df)}, covariates={used_covariates}")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="os_months", event_col="is_deceased")

    summary = cph.summary
    results = {"cancer_id": cancer_id, "n": len(cox_df), "concordance": float(cph.concordance_index_)}

    # Извлечи резултата за ssgsea_nv_score
    if "ssgsea_nv_score" in summary.index:
        row = summary.loc["ssgsea_nv_score"]
        results.update({
            "ssgsea_coef":       float(row["coef"]),
            "ssgsea_HR":         float(row["exp(coef)"]),
            "ssgsea_HR_lower":   float(row["exp(coef) lower 95%"]),
            "ssgsea_HR_upper":   float(row["exp(coef) upper 95%"]),
            "ssgsea_p":          float(row["p"]),
            "ssgsea_z":          float(row["z"]),
        })

    # Извлечи всички covariates за forest plot
    covariate_rows = []
    for var in summary.index:
        r = summary.loc[var]
        covariate_rows.append({
            "cancer_id": cancer_id,
            "variable":  var,
            "coef":      float(r["coef"]),
            "HR":        float(r["exp(coef)"]),
            "HR_lower":  float(r["exp(coef) lower 95%"]),
            "HR_upper":  float(r["exp(coef) upper 95%"]),
            "p":         float(r["p"]),
        })
    results["covariate_detail"] = covariate_rows

    logger.info(
        f"{cancer_id} multivariate Cox: "
        f"ssGSEA HR={results.get('ssgsea_HR', 'N/A'):.3f} "
        f"[{results.get('ssgsea_HR_lower', 0):.3f}-{results.get('ssgsea_HR_upper', 0):.3f}], "
        f"p={results.get('ssgsea_p', 1):.4f}, C={results['concordance']:.3f}"
    )

    return results


def plot_forest(all_detail: list[dict], output_path: Path) -> None:
    """Forest plot на ssGSEA HR за всички cancer типове (публикационен стандарт)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Филтрирай само ssgsea_nv_score редовете
    rows = [r for r in all_detail if r["variable"] == "ssgsea_nv_score"]
    if not rows:
        logger.warning("Няма данни за forest plot")
        return

    df = pd.DataFrame(rows).sort_values("HR")
    n = len(df)

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.8)))

    y_pos = range(n)
    colors = ["#2196F3" if p < 0.05 else "#9E9E9E" for p in df["p"]]

    # Хоризонтални error bars (CI)
    ax.errorbar(
        x=df["HR"],
        y=list(y_pos),
        xerr=[df["HR"] - df["HR_lower"], df["HR_upper"] - df["HR"]],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=4,
        markersize=7,
        zorder=3,
    )

    # Оцвети точките по значимост
    for i, (_, row) in enumerate(df.iterrows()):
        color = "#D32F2F" if row["p"] < 0.05 and row["HR"] > 1 else \
                "#1976D2" if row["p"] < 0.05 and row["HR"] < 1 else "#9E9E9E"
        ax.plot(row["HR"], i, "o", color=color, markersize=8, zorder=4)

    # Референтна линия HR=1
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Labels
    labels = [
        f"{row['cancer_id']}  HR={row['HR']:.2f} [{row['HR_lower']:.2f}-{row['HR_upper']:.2f}]  p={row['p']:.3f}"
        for _, row in df.iterrows()
    ]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel("Hazard Ratio (ssGSEA NV-Score, standardized)", fontsize=11)
    ax.set_title(
        "Multivariate Cox PH — ssGSEA NV-Score\n"
        "(adjusted for age, stage, molecular subtype)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3)

    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1976D2", markersize=8, label="p<0.05, HR<1 (protective)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D32F2F", markersize=8, label="p<0.05, HR>1 (risk)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E", markersize=8, label="p≥0.05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Forest plot записан: {output_path}")


def main() -> int:
    sys.path.insert(0, str(REVGATE_ROOT / "src"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Зареди ssGSEA scores от Приоритет 1 ---
    if not SSGSEA_SCORES.exists():
        logger.error(f"Липсват ssGSEA scores: {SSGSEA_SCORES}")
        logger.error("Пусни първо: python3 run_ssgsea_analysis.py")
        return 1

    scores_df = pd.read_parquet(SSGSEA_SCORES)
    scores_df["patient_id_short"] = scores_df["patient_id"].str[:12]

    # --- Covariates per cancer ---
    # Общи за всички: age + stage
    # Специфични: BRCA -> er_positive, her2_positive; LUAD -> subtype dummies
    BASE_COVARIATES = ["age", "stage"]
    CANCER_EXTRA_COVARIATES: dict[str, list[str]] = {
        "BRCA": ["er_positive", "her2_positive"],
        "LUAD": ["subtype_squamoid", "subtype_magnoid"],
        # LAML: CALGB цитогенетичен риск вместо stage (stage=NaN за LAML)
        "LAML": ["cytogenetic_intermediate", "cytogenetic_poor"],
    }

    summary_rows: list[dict] = []
    all_detail:   list[dict] = []

    for cancer_id in CANCER_IDS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Обработвам {cancer_id}...")

        # Зареди клинични данни
        clin_path = CLINICAL_DIR / f"{cancer_id}_clinical_raw.tsv"
        if not clin_path.exists():
            logger.warning(f"Липсва клиничен файл за {cancer_id} — пропускам")
            continue

        try:
            clin_df = load_clinical_with_covariates(cancer_id)
        except Exception as exc:
            logger.error(f"{cancer_id}: грешка при зареждане на клинични данни — {exc}")
            continue

        # Merge с ssGSEA scores
        cancer_scores = scores_df[scores_df["cancer_id"] == cancer_id]
        merged = cancer_scores.merge(clin_df, on="patient_id_short", how="inner")

        logger.info(f"{cancer_id}: {len(merged)} пациента след merge")

        if len(merged) < 30:
            logger.warning(f"{cancer_id}: твърде малко пациента ({len(merged)}) — пропускам")
            continue

        # Covariates за този cancer тип
        # LAML: няма pathologic_stage (хематологичен тумор) -- изключваме stage
        base = [c for c in BASE_COVARIATES if not (cancer_id == "LAML" and c == "stage")]
        covariates = base + CANCER_EXTRA_COVARIATES.get(cancer_id, [])

        # Пусни multivariate Cox
        result = run_multivariate_cox(merged, cancer_id, covariates)
        summary_rows.append(result)

        if "covariate_detail" in result:
            all_detail.extend(result.pop("covariate_detail"))

    if not summary_rows:
        logger.error("Няма резултати — прекратявам")
        return 1

    # --- Запази summary CSV ---
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "covariate_detail"}
        for r in summary_rows
    ])
    summary_path = RESULTS_DIR / "multivariate_cox_results.csv"
    summary_df.to_csv(summary_path, index=False)

    # --- Запази детайлен CSV (всички covariates) ---
    detail_df = pd.DataFrame(all_detail)
    detail_path = RESULTS_DIR / "multivariate_cox_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    logger.info(f"\n{'='*50}")
    logger.info("=== MULTIVARIATE COX РЕЗУЛТАТИ ===")
    logger.info(f"\n{summary_df[['cancer_id','n','ssgsea_HR','ssgsea_HR_lower','ssgsea_HR_upper','ssgsea_p','concordance']].to_string()}")

    # --- Forest plot ---
    forest_path = RESULTS_DIR / "multivariate_cox_forest.png"
    plot_forest(all_detail, forest_path)

    logger.info(f"\nФайлове записани:")
    logger.info(f"  {summary_path}")
    logger.info(f"  {detail_path}")
    logger.info(f"  {forest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
