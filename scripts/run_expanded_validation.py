#!/usr/bin/env python3
# run_expanded_validation.py -- Finding_10: Expanded Sel+pLI Validation
#
# PRE-REGISTRATION (записано преди анализа):
#   COAD (Bowel,  Sel+pLI=1.046) -> predicted: protective (HR<1)
#   OV   (Ovary,  Sel+pLI=0.978) -> predicted: harmful/null (HR>=1)
#   HNSC (H&N,    Sel+pLI=0.857) -> predicted: harmful (HR>1)
#   UCEC (Uterus, Sel+pLI=0.882) -> predicted: harmful/null (HR>=1)
#   KIRC restricted follow-up    -> predicted: protective at 36-60mo
#   SKCM primary-only            -> predicted: protective (HR<1)
#
# Употреба:
#   source /root/helix/venv/bin/activate
#   cd /root/revgate
#   python3 run_expanded_validation.py 2>&1 | tee results/expanded_validation.log
#
# Автор: Helena Bioinformatics
# Дата: 2026-03-10
# Биологична хипотеза: Тончева и Сгурев, БАН

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("expanded_validation")

# ---------------------------------------------------------------------------
# Пътища (консистентни с run_ssgsea_analysis.py)
# ---------------------------------------------------------------------------
REVGATE_ROOT   = Path("/root/revgate")
CACHE_DIR      = Path("/root/.revgate/cache")
DEPMAP_CHRONOS = CACHE_DIR / "depmap/CRISPRGeneEffect.csv"
DEPMAP_MODEL   = CACHE_DIR / "depmap/Model.csv"
TCGA_DIR       = CACHE_DIR / "tcga"
PAN_CANCER_GZ  = TCGA_DIR / "tcga_RSEM_gene_tpm.gz"
ENSEMBL_MAP    = TCGA_DIR / "ensembl_to_symbol.json"
RESULTS_DIR    = REVGATE_ROOT / "results"

# ssGSEA параметри (идентични с ssgsea_scorer.py)
TOP_N_GENES       = 50
MIN_GENES_OVERLAP = 10

# ---------------------------------------------------------------------------
# TCGA project -> DepMap lineage mapping
# ---------------------------------------------------------------------------
NEW_CANCERS = {
    "COAD": "Bowel",
    "OV":   "Ovary/Fallopian Tube",
    "HNSC": "Head and Neck",
    "UCEC": "Uterus",
}

# Sel+pLI стойности от nv_components_all_lineages.csv
SEL_PLI = {
    "BRCA": 1.158, "KIRC": 1.218, "SKCM": 1.303,
    "LAML": 1.194, "LUAD": 0.599, "PAAD": 0.979,
    "COAD": 1.046, "OV": 0.978, "HNSC": 0.857, "UCEC": 0.882,
}

# Pre-registered предсказания
PREDICTIONS = {
    "COAD": "protective",
    "OV":   "harmful/null",
    "HNSC": "harmful",
    "UCEC": "harmful/null",
}

# Stage mapping (идентичен с run_stage_stratified_cox.py)
STAGE_MAP = {
    "stage i": 1.0, "stage ia": 1.0, "stage ib": 1.0, "stage ic": 1.0,
    "stage ii": 2.0, "stage iia": 2.0, "stage iib": 2.0, "stage iic": 2.0,
    "stage iii": 3.0, "stage iiia": 3.0, "stage iiib": 3.0, "stage iiic": 3.0,
    "stage iv": 4.0, "stage iva": 4.0, "stage ivb": 4.0, "stage ivc": 4.0,
}

# TCGA barcode -> project mapping (за извличане от pan-cancer файл)
# Формат: TCGA-XX-XXXX-01, XX е TSS code
# Ползваме по-надежден подход: GDC sample sheet или hardcoded mapping
# Тъй като pan-cancer файлът има 10535 samples, ще matchнем по known TCGA barcodes
TCGA_PROJECT_CODES = {
    "COAD": ["TCGA-A6", "TCGA-AA", "TCGA-AD", "TCGA-AF", "TCGA-AG",
             "TCGA-AY", "TCGA-AZ", "TCGA-CA", "TCGA-CK", "TCGA-CM",
             "TCGA-D5", "TCGA-DM", "TCGA-F4", "TCGA-G4", "TCGA-NH",
             "TCGA-QG", "TCGA-RU", "TCGA-T9", "TCGA-WS"],
    "OV":   ["TCGA-04", "TCGA-09", "TCGA-10", "TCGA-13", "TCGA-20",
             "TCGA-23", "TCGA-24", "TCGA-25", "TCGA-29", "TCGA-30",
             "TCGA-31", "TCGA-36", "TCGA-42", "TCGA-57", "TCGA-59",
             "TCGA-61", "TCGA-WR"],
    "HNSC": ["TCGA-BA", "TCGA-BB", "TCGA-CN", "TCGA-CQ", "TCGA-CR",
             "TCGA-CV", "TCGA-D6", "TCGA-DQ", "TCGA-F7", "TCGA-H7",
             "TCGA-HD", "TCGA-IQ", "TCGA-MZ", "TCGA-P3", "TCGA-QK",
             "TCGA-RS", "TCGA-T2", "TCGA-TN", "TCGA-UP", "TCGA-UF"],
    "UCEC": ["TCGA-A5", "TCGA-AJ", "TCGA-AP", "TCGA-AX", "TCGA-B5",
             "TCGA-BG", "TCGA-BK", "TCGA-BS", "TCGA-D1", "TCGA-DI",
             "TCGA-E6", "TCGA-EO", "TCGA-EY", "TCGA-FI", "TCGA-FL",
             "TCGA-N5", "TCGA-N7", "TCGA-N8", "TCGA-N9", "TCGA-PG",
             "TCGA-QC", "TCGA-QF", "TCGA-SL"],
}


# ===================================================================
# ФАЗА А: Извличане на expression данни от pan-cancer файл
# ===================================================================
def extract_expression_for_cancer(
    cancer_id: str,
    pan_cancer_path: Path,
    ensembl_map_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Извлечи expression матрица за cancer тип от pan-cancer RSEM файл.

    Идентична логика с оригиналното създаване на BRCA_expression.parquet.
    """
    if output_path.exists():
        logger.info(f"{cancer_id}: expression parquet вече съществува -> {output_path}")
        return pd.read_parquet(output_path)

    logger.info(f"{cancer_id}: извличам expression от pan-cancer файл...")

    # Зареди ensembl -> symbol mapping
    with open(ensembl_map_path) as f:
        ensembl_to_symbol = json.load(f)

    # Прочети header за sample IDs
    with gzip.open(pan_cancer_path, "rt") as f:
        header_line = f.readline().strip()
    all_samples = header_line.split("\t")[1:]  # първата колона е gene ID

    # Намери samples за cancer типа по TSS code prefix
    tss_codes = TCGA_PROJECT_CODES.get(cancer_id, [])
    if not tss_codes:
        raise ValueError(f"Няма TSS codes за {cancer_id}")

    # Индекси на колоните за cancer типа (0-based, +1 за gene колоната)
    cancer_sample_indices = []
    cancer_sample_ids = []
    for i, sample_id in enumerate(all_samples):
        prefix = "-".join(sample_id.split("-")[:2])  # TCGA-XX
        if prefix in tss_codes:
            cancer_sample_indices.append(i + 1)  # +1 защото колона 0 е gene ID
            cancer_sample_ids.append(sample_id)

    logger.info(f"{cancer_id}: намерени {len(cancer_sample_ids)} samples от {len(all_samples)} total")

    if len(cancer_sample_ids) < 20:
        raise ValueError(f"{cancer_id}: твърде малко samples ({len(cancer_sample_ids)})")

    # Прочети само релевантните колони
    usecols = [0] + cancer_sample_indices
    logger.info(f"{cancer_id}: четене на pan-cancer файл (може да отнеме 1-2 мин)...")

    df = pd.read_csv(
        pan_cancer_path,
        sep="\t",
        usecols=usecols,
        index_col=0,
        compression="gzip",
    )

    # Конвертирай Ensembl IDs -> gene symbols
    # Ensembl ID формат: ENSG00000242268.2 -> махаме версията
    df.index = [eid.split(".")[0] for eid in df.index]
    df.index = [ensembl_to_symbol.get(eid, eid) for eid in df.index]

    # Премахни дублирани gene symbols (запази първия)
    df = df[~df.index.duplicated(keep="first")]

    # Премахни гени без символ (останали като ENSG...)
    df = df[~df.index.str.startswith("ENSG")]

    # Транспонирай: rows=patients, cols=genes
    df = df.T

    # log2(TPM+1) трансформация -- стойностите вече са log2
    # RSEM TPM файлът от UCSC Xena е log2(TPM+1) -- проверяваме
    # Ако мин. стойност е силно отрицателна (-9.96), данните вече са log2
    min_val = df.min().min()
    if min_val < -5:
        logger.info(f"{cancer_id}: данните са вече log2 трансформирани (min={min_val:.2f})")
    else:
        logger.info(f"{cancer_id}: прилагам log2(x+1) трансформация")
        df = np.log2(df + 1)

    # Запази като parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"{cancer_id}: expression записан -> {output_path} ({df.shape[0]} x {df.shape[1]})")

    return df


# ===================================================================
# ФАЗА А: Извличане на клинични данни от GDC API
# ===================================================================
def download_clinical_from_gdc(cancer_id: str, output_path: Path) -> pd.DataFrame:
    """Свали клинични данни от GDC API за cancer тип.

    Използва GDC cases endpoint с филтър по project.
    Връща TSV файл консистентен с BRCA_clinical_raw.tsv формата.
    """
    if output_path.exists():
        logger.info(f"{cancer_id}: clinical файл вече съществува -> {output_path}")
        return pd.read_csv(output_path, sep="\t", low_memory=False)

    import requests

    logger.info(f"{cancer_id}: свалям клинични данни от GDC API...")

    project_id = f"TCGA-{cancer_id}"
    endpoint = "https://api.gdc.cancer.gov/cases"

    # GDC API: вземаме всички клинични полета
    fields = [
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.days_to_last_follow_up",
        "demographic.gender",
        "demographic.age_at_index",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n",
        "diagnoses.ajcc_pathologic_m",
        "diagnoses.figo_stage",
        "diagnoses.primary_diagnosis",
        "diagnoses.tumor_stage",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.tissue_or_organ_of_origin",
    ]

    params = {
        "filters": json.dumps({
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": [project_id],
            },
        }),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "2000",
    }

    response = requests.get(endpoint, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hits = data["data"]["hits"]
    logger.info(f"{cancer_id}: получени {len(hits)} cases от GDC")

    # Парсираме в плосък DataFrame
    rows = []
    for case in hits:
        row = {}
        row["_PATIENT"] = case.get("submitter_id", "")

        demo = case.get("demographic", {}) or {}
        row["vital_status"] = demo.get("vital_status", "")
        row["days_to_death"] = demo.get("days_to_death", "")
        row["days_to_last_followup"] = demo.get("days_to_last_follow_up", "")
        row["gender"] = demo.get("gender", "")
        row["age_at_initial_pathologic_diagnosis"] = demo.get("age_at_index", "")

        # Diagnoses -- вземаме първия (primary)
        diagnoses = case.get("diagnoses", []) or []
        if diagnoses:
            diag = diagnoses[0]
            # Stage: приоритет AJCC -> FIGO (за OV/UCEC)
            stage = diag.get("ajcc_pathologic_stage", "")
            if not stage:
                stage = diag.get("figo_stage", "")
            row["pathologic_stage"] = stage
            row["ajcc_pathologic_t"] = diag.get("ajcc_pathologic_t", "")
            row["ajcc_pathologic_n"] = diag.get("ajcc_pathologic_n", "")
            row["ajcc_pathologic_m"] = diag.get("ajcc_pathologic_m", "")
            row["primary_diagnosis"] = diag.get("primary_diagnosis", "")
            row["site_of_resection_or_biopsy"] = diag.get("site_of_resection_or_biopsy", "")
            row["tissue_or_organ_of_origin"] = diag.get("tissue_or_organ_of_origin", "")
        else:
            row["pathologic_stage"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)

    # Запази като TSV (консистентно с _clinical_raw.tsv формат)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"{cancer_id}: clinical записан -> {output_path} ({len(df)} cases)")

    return df


# ===================================================================
# ФАЗА Б: ssGSEA scoring (идентичен с ssgsea_scorer.py)
# ===================================================================
def get_top_dependency_genes(
    dep_matrix: pd.DataFrame,
    lineage_cell_lines: list[str],
    top_n: int = TOP_N_GENES,
) -> list[str]:
    """Top-N диференциални гени. Идентично с ssgsea_scorer._get_top_dependency_genes."""
    global_mean = dep_matrix.mean(axis=0)
    cancer_subset = dep_matrix.loc[dep_matrix.index.intersection(lineage_cell_lines)]

    if cancer_subset.empty:
        return []

    cancer_mean = cancer_subset.mean(axis=0)
    diff_score = cancer_mean - global_mean
    return diff_score.nsmallest(top_n).index.tolist()


def run_ssgsea_for_cancer(
    cancer_id: str,
    lineage: str,
    dep_matrix: pd.DataFrame,
    model_df: pd.DataFrame,
    expression_path: Path,
) -> pd.DataFrame | None:
    """ssGSEA scoring за един cancer тип. Идентично с ssgsea_scorer.run_ssgsea_scoring."""
    import gseapy as gp

    # Намери cell lines за lineage
    lineage_mask = model_df["OncotreeLineage"].str.contains(lineage, na=False)
    lineage_model_ids = model_df.loc[lineage_mask, "ModelID"].tolist()

    # Top-50 диференциални гени
    top_genes = get_top_dependency_genes(dep_matrix, lineage_model_ids, TOP_N_GENES)
    if len(top_genes) < MIN_GENES_OVERLAP:
        logger.warning(f"{cancer_id}: само {len(top_genes)} гени -- пропускам")
        return None

    logger.info(f"{cancer_id}: gene set = {len(top_genes)} гени, top3 = {top_genes[:3]}")

    # Зареди expression
    expr_df = pd.read_parquet(expression_path)
    expr_T = expr_df.T  # genes x samples за gseapy

    # Overlap
    available_genes = set(expr_T.index)
    gene_set_filtered = [g for g in top_genes if g in available_genes]

    if len(gene_set_filtered) < MIN_GENES_OVERLAP:
        logger.warning(f"{cancer_id}: само {len(gene_set_filtered)} гени в expression -- пропускам")
        return None

    logger.info(f"{cancer_id}: overlap {len(gene_set_filtered)}/{len(top_genes)} гени")

    # ssGSEA
    gene_sets = {f"NV_proxy_{cancer_id}": gene_set_filtered}

    try:
        ssgsea_result = gp.ssgsea(
            data=expr_T,
            gene_sets=gene_sets,
            outdir=None,
            no_plot=True,
            processes=1,
            verbose=False,
        )
        scores = ssgsea_result.res2d
        cancer_scores = (
            scores[["Name", "NES"]]
            .rename(columns={"Name": "patient_id", "NES": "ssgsea_nv_score"})
            .copy()
        )
        cancer_scores["cancer_id"] = cancer_id
        cancer_scores["n_genes_in_set"] = len(gene_set_filtered)

        logger.info(
            f"{cancer_id}: {len(cancer_scores)} пациента scored, "
            f"mean NES={cancer_scores['ssgsea_nv_score'].mean():.3f}"
        )
        return cancer_scores

    except Exception as exc:
        logger.error(f"{cancer_id}: ssGSEA грешка -- {exc}")
        return None


# ===================================================================
# ФАЗА В: Stage-stratified Cox PH (идентичен с run_stage_stratified_cox.py)
# ===================================================================
def load_clinical_for_cox(cancer_id: str) -> pd.DataFrame:
    """Зареди клинични данни за Cox анализ."""
    path = TCGA_DIR / f"{cancer_id}_clinical_raw.tsv"
    df = pd.read_csv(path, sep="\t", low_memory=False)

    days = pd.to_numeric(df.get("days_to_death", pd.Series(dtype=float)), errors="coerce")
    followup = pd.to_numeric(df.get("days_to_last_followup", pd.Series(dtype=float)), errors="coerce")
    os_months = days.where(days.notna(), followup) / 30.44

    vital = df.get("vital_status", pd.Series(dtype=str)).astype(str).str.lower()
    is_deceased = vital.isin(["dead", "deceased"]).astype(int)

    # Patient ID
    if "_PATIENT" in df.columns and df["_PATIENT"].astype(str).str.startswith("TCGA").any():
        pid = df["_PATIENT"].astype(str).str[:12]
    elif "bcr_patient_barcode" in df.columns:
        pid = df["bcr_patient_barcode"].astype(str).str[:12]
    elif "patient_id" in df.columns:
        pid = df["patient_id"].astype(str).str[:12]
    else:
        pid = df.index.astype(str)

    # Stage parsing -- AJCC и FIGO
    stage_raw = df.get("pathologic_stage", pd.Series(dtype=str))
    stage_num = stage_raw.astype(str).str.lower().str.strip().map(STAGE_MAP)

    return pd.DataFrame({
        "patient_id_short": pid,
        "os_months": os_months,
        "is_deceased": is_deceased,
        "stage": stage_num,
    })


def cox_univariate(df: pd.DataFrame, label: str) -> dict:
    """Univariate Cox PH. Идентично с run_stage_stratified_cox.cox_univariate_ssgsea."""
    from lifelines import CoxPHFitter

    cox_df = df[["os_months", "is_deceased", "ssgsea_nv_score"]].dropna()
    cox_df = cox_df[cox_df["os_months"] > 0].copy()

    if len(cox_df) < 20:
        return {"label": label, "n": len(cox_df), "error": "insufficient_data"}

    # Стандартизация (z-score)
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
            "label": label,
            "n": len(cox_df),
            "n_events": int(cox_df["is_deceased"].sum()),
            "HR": float(row["exp(coef)"]),
            "HR_lower": float(row["exp(coef) lower 95%"]),
            "HR_upper": float(row["exp(coef) upper 95%"]),
            "p": float(row["p"]),
            "concordance": float(cph.concordance_index_),
        }
    except Exception as exc:
        return {"label": label, "n": len(cox_df), "error": str(exc)}


def cox_restricted_followup(df: pd.DataFrame, cancer_id: str, cutoff_months: float) -> dict:
    """Cox с административно цензуриране при cutoff. Идентично с Finding_09."""
    restricted = df.copy()
    # Административно цензуриране: пациенти с os_months > cutoff се цензурират при cutoff
    mask_over = restricted["os_months"] > cutoff_months
    restricted.loc[mask_over, "os_months"] = cutoff_months
    restricted.loc[mask_over, "is_deceased"] = 0

    label = f"{cancer_id} {int(cutoff_months)}mo"
    return cox_univariate(restricted, label)


# ===================================================================
# ФАЗА В: Forest plot
# ===================================================================
def plot_expanded_forest(results: list[dict], output_path: Path) -> None:
    """Forest plot за всички cancer типове."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results if "error" not in r or r.get("error") is None]
    if not valid:
        logger.warning("Няма валидни резултати за forest plot")
        return

    n = len(valid)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.55)))

    for i, r in enumerate(valid):
        color = "#1976D2" if r["HR"] < 1 else "#D32F2F"
        alpha = 1.0 if r.get("p", 1) < 0.05 else 0.4
        marker = "D" if r.get("is_new", False) else "o"

        ax.errorbar(
            x=r["HR"], y=i,
            xerr=[[r["HR"] - r["HR_lower"]], [r["HR_upper"] - r["HR"]]],
            fmt=marker, color=color, ecolor=color,
            capsize=4, markersize=8, alpha=alpha, zorder=3,
        )

        p_str = f"p={r['p']:.4f}" if r["p"] >= 0.001 else "p<0.001"
        sig = "**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "")
        ax.text(max(r["HR_upper"] + 0.05, 1.6), i, f" {sig}{p_str}", va="center", fontsize=8)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)

    labels = [
        f"{r['label']}  n={r['n']}, ev={r['n_events']}  "
        f"HR={r['HR']:.3f} [{r['HR_lower']:.3f}-{r['HR_upper']:.3f}]"
        for r in valid
    ]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hazard Ratio (ssGSEA NV-Score, standardized)", fontsize=11)
    ax.set_title(
        "Finding_10: Expanded Dependency Architecture Validation\n"
        "Diamonds=new cancers | Circles=original | Solid=p<0.05 | Faded=ns",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Forest plot zapisano: {output_path}")


# ===================================================================
# MAIN
# ===================================================================
def main() -> int:
    start_time = time.time()
    sys.path.insert(0, str(REVGATE_ROOT / "src"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Finding_10: EXPANDED DEPENDENCY ARCHITECTURE VALIDATION")
    logger.info("=" * 70)

    # --- Проверка на prerequisites ---
    for path in [DEPMAP_CHRONOS, DEPMAP_MODEL, PAN_CANCER_GZ, ENSEMBL_MAP]:
        if not path.exists():
            logger.error(f"Липсва: {path}")
            return 1

    # ==================================================================
    # ФАЗА А: Извличане на expression + clinical за нови cancer типове
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FAZA A: Извличане на TCGA данни за нови cancer типове")
    logger.info("=" * 70)

    for cancer_id in NEW_CANCERS:
        expr_path = TCGA_DIR / f"{cancer_id}_expression.parquet"
        clin_path = TCGA_DIR / f"{cancer_id}_clinical_raw.tsv"

        # Expression от pan-cancer файл
        try:
            extract_expression_for_cancer(cancer_id, PAN_CANCER_GZ, ENSEMBL_MAP, expr_path)
        except Exception as exc:
            logger.error(f"{cancer_id}: expression грешка -- {exc}")
            continue

        # Clinical от GDC API
        try:
            download_clinical_from_gdc(cancer_id, clin_path)
        except Exception as exc:
            logger.error(f"{cancer_id}: clinical грешка -- {exc}")
            logger.info(f"{cancer_id}: ще опитаме Cox само с expression данни")

    # ==================================================================
    # ФАЗА Б: ssGSEA scoring за нови cancer типове
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FAZA B: ssGSEA scoring за нови cancer типове")
    logger.info("=" * 70)

    # Зареди DepMap матрица веднъж
    logger.info("Зареждам DepMap CRISPR матрица...")
    dep_df = pd.read_csv(DEPMAP_CHRONOS, index_col=0)
    dep_df.columns = [col.split(" (")[0] for col in dep_df.columns]

    logger.info("Зареждам DepMap Model metadata...")
    model_df = pd.read_csv(DEPMAP_MODEL)

    new_scores_list = []
    for cancer_id, lineage in NEW_CANCERS.items():
        expr_path = TCGA_DIR / f"{cancer_id}_expression.parquet"
        if not expr_path.exists():
            logger.warning(f"{cancer_id}: липсва expression -- пропускам ssGSEA")
            continue

        scores = run_ssgsea_for_cancer(cancer_id, lineage, dep_df, model_df, expr_path)
        if scores is not None:
            new_scores_list.append(scores)

    if new_scores_list:
        new_scores_df = pd.concat(new_scores_list, ignore_index=True)
        # Запази отделно (не презаписваме оригиналния per_patient_ssgsea_scores.parquet)
        new_scores_path = TCGA_DIR / "per_patient_ssgsea_expanded.parquet"
        new_scores_df.to_parquet(new_scores_path, index=False)
        logger.info(f"Нови ssGSEA scores записани: {new_scores_path} ({len(new_scores_df)} пациента)")
    else:
        logger.error("Нито един нов cancer тип не беше успешно scored")
        new_scores_df = pd.DataFrame()

    # ==================================================================
    # ФАЗА В: Stage-stratified Cox за нови cancer типове
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FAZA V: Stage-stratified Cox PH за нови cancer типове")
    logger.info("=" * 70)

    all_results = []

    # --- Нови cancer типове ---
    for cancer_id in NEW_CANCERS:
        clin_path = TCGA_DIR / f"{cancer_id}_clinical_raw.tsv"
        if not clin_path.exists():
            logger.warning(f"{cancer_id}: липсва clinical -- пропускам Cox")
            continue

        cancer_scores = new_scores_df[new_scores_df["cancer_id"] == cancer_id] if len(new_scores_df) > 0 else pd.DataFrame()
        if cancer_scores.empty:
            logger.warning(f"{cancer_id}: липсват ssGSEA scores -- пропускам")
            continue

        clin_df = load_clinical_for_cox(cancer_id)
        cancer_scores = cancer_scores.copy()
        cancer_scores["patient_id_short"] = cancer_scores["patient_id"].str[:12]
        merged = cancer_scores.merge(clin_df, on="patient_id_short", how="inner")
        merged = merged[merged["os_months"].notna() & (merged["os_months"] > 0)]

        logger.info(f"\n{cancer_id}: {len(merged)} пациента след merge, events={int(merged['is_deceased'].sum())}")

        # Stage distribution
        stage_counts = merged["stage"].value_counts().sort_index()
        logger.info(f"Stage разпределение:\n{stage_counts.to_string()}")

        sel_pli = SEL_PLI.get(cancer_id, 0)
        predicted = PREDICTIONS.get(cancer_id, "unknown")

        # All stages (unstratified)
        res_all = cox_univariate(merged, f"{cancer_id} All")
        res_all["cancer_id"] = cancer_id
        res_all["stage_group"] = "All"
        res_all["sel_pli"] = sel_pli
        res_all["predicted"] = predicted
        res_all["is_new"] = True
        all_results.append(res_all)

        # Early (I-II)
        early = merged[merged["stage"].isin([1.0, 2.0])].copy()
        if len(early) >= 20:
            res_early = cox_univariate(early, f"{cancer_id} Early (I-II)")
            res_early["cancer_id"] = cancer_id
            res_early["stage_group"] = "Early (I-II)"
            res_early["sel_pli"] = sel_pli
            res_early["predicted"] = predicted
            res_early["is_new"] = True
            all_results.append(res_early)

        # Advanced (III-IV)
        advanced = merged[merged["stage"].isin([3.0, 4.0])].copy()
        if len(advanced) >= 20:
            res_adv = cox_univariate(advanced, f"{cancer_id} Advanced (III-IV)")
            res_adv["cancer_id"] = cancer_id
            res_adv["stage_group"] = "Advanced (III-IV)"
            res_adv["sel_pli"] = sel_pli
            res_adv["predicted"] = predicted
            res_adv["is_new"] = True
            all_results.append(res_adv)

    # ==================================================================
    # ФАЗА В.2: KIRC restricted follow-up (по модела на Finding_09)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FAZA V.2: KIRC restricted follow-up")
    logger.info("=" * 70)

    orig_scores_path = TCGA_DIR / "per_patient_ssgsea_scores.parquet"
    if orig_scores_path.exists():
        orig_scores = pd.read_parquet(orig_scores_path)

        # KIRC restricted follow-up
        kirc_scores = orig_scores[orig_scores["cancer_id"] == "KIRC"].copy()
        kirc_clin_path = TCGA_DIR / "KIRC_clinical_raw.tsv"

        if kirc_clin_path.exists() and len(kirc_scores) > 0:
            kirc_clin = load_clinical_for_cox("KIRC")
            kirc_scores["patient_id_short"] = kirc_scores["patient_id"].str[:12]
            kirc_merged = kirc_scores.merge(kirc_clin, on="patient_id_short", how="inner")
            kirc_merged = kirc_merged[kirc_merged["os_months"].notna() & (kirc_merged["os_months"] > 0)]

            logger.info(f"KIRC: {len(kirc_merged)} пациента, events={int(kirc_merged['is_deceased'].sum())}")

            for cutoff in [36, 48, 60, 84, 120]:
                res = cox_restricted_followup(kirc_merged, "KIRC", cutoff)
                res["cancer_id"] = "KIRC"
                res["stage_group"] = f"restricted_{cutoff}mo"
                res["sel_pli"] = SEL_PLI["KIRC"]
                res["predicted"] = "protective"
                res["is_new"] = False
                all_results.append(res)

        # ==================================================================
        # ФАЗА В.3: SKCM primary-only subset
        # ==================================================================
        logger.info("\n" + "=" * 70)
        logger.info("FAZA V.3: SKCM primary-only subset")
        logger.info("=" * 70)

        skcm_scores = orig_scores[orig_scores["cancer_id"] == "SKCM"].copy()
        skcm_clin_path = TCGA_DIR / "SKCM_clinical_raw.tsv"

        if skcm_clin_path.exists() and len(skcm_scores) > 0:
            skcm_clin_raw = pd.read_csv(skcm_clin_path, sep="\t", low_memory=False)

            # TCGA SKCM: sample type 01 = primary, 06 = metastatic
            # Баркод формат: TCGA-XX-XXXX-06 (позиция 13-14 е sample type)
            skcm_scores["sample_type"] = skcm_scores["patient_id"].str[13:15]
            primary_mask = skcm_scores["sample_type"] == "01"
            skcm_primary = skcm_scores[primary_mask].copy()

            logger.info(f"SKCM: {len(skcm_scores)} total, {len(skcm_primary)} primary (01)")

            if len(skcm_primary) >= 20:
                skcm_clin = load_clinical_for_cox("SKCM")
                skcm_primary["patient_id_short"] = skcm_primary["patient_id"].str[:12]
                skcm_merged = skcm_primary.merge(skcm_clin, on="patient_id_short", how="inner")
                skcm_merged = skcm_merged[skcm_merged["os_months"].notna() & (skcm_merged["os_months"] > 0)]

                logger.info(f"SKCM primary: {len(skcm_merged)} след merge, events={int(skcm_merged['is_deceased'].sum())}")

                res = cox_univariate(skcm_merged, "SKCM Primary-only")
                res["cancer_id"] = "SKCM"
                res["stage_group"] = "Primary-only"
                res["sel_pli"] = SEL_PLI["SKCM"]
                res["predicted"] = "protective"
                res["is_new"] = False
                all_results.append(res)
            else:
                logger.warning(f"SKCM: само {len(skcm_primary)} primary samples -- недостатъчно")

    # ==================================================================
    # ФАЗА Г: Резултати и валидация на Dependency Architecture Principle
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FAZA G: REZULTATI I VALIDACIA")
    logger.info("=" * 70)

    results_df = pd.DataFrame(all_results)

    # Запази пълни резултати
    results_csv = RESULTS_DIR / "expanded_validation_results.csv"
    results_df.to_csv(results_csv, index=False)

    # JSON за machine-readable
    results_json_path = RESULTS_DIR / "expanded_validation_results.json"
    results_df.to_json(results_json_path, orient="records", indent=2)

    # --- Принтирай резултати ---
    valid = results_df[~results_df.get("error", pd.Series(dtype=str)).notna() | (results_df.get("error", "") == "")]

    # По-безопасна филтрация
    if "error" in results_df.columns:
        valid = results_df[results_df["error"].isna() | (results_df["error"] == "")]
    else:
        valid = results_df

    logger.info("\n=== ПЪЛНИ РЕЗУЛТАТИ ===")
    display_cols = ["cancer_id", "stage_group", "n", "n_events", "HR", "HR_lower", "HR_upper", "p", "sel_pli", "predicted"]
    available_cols = [c for c in display_cols if c in valid.columns]
    logger.info(f"\n{valid[available_cols].to_string()}")

    # --- Значими резултати ---
    if "p" in valid.columns:
        sig = valid[valid["p"] < 0.05]
        if len(sig) > 0:
            logger.info(f"\n=== ЗНАЧИМИ РЕЗУЛТАТИ (p<0.05) ===")
            logger.info(f"\n{sig[available_cols].to_string()}")
        else:
            logger.info("\nНяма значими резултати при p<0.05")

    # --- Sel+pLI validation за НОВИТЕ cancer типове ---
    logger.info("\n=== SEL+PLI DEPENDENCY ARCHITECTURE VALIDATION ===")
    logger.info(f"{'Cancer':<8} {'Sel+pLI':>8} {'Predicted':<15} {'Observed HR':>12} {'Direction':<12} {'p':>8} {'Match':<6}")
    logger.info("-" * 75)

    # Взимаме най-информативния stratum per cancer (Early ако има, иначе All)
    for cancer_id in list(NEW_CANCERS.keys()):
        cancer_results = valid[valid["cancer_id"] == cancer_id]
        if cancer_results.empty:
            logger.info(f"{cancer_id:<8} {'N/A':>8} {'N/A':<15} {'N/A':>12} {'N/A':<12} {'N/A':>8} {'N/A':<6}")
            continue

        # Приоритет: Early > All > Advanced
        for pref_stage in ["Early (I-II)", "All", "Advanced (III-IV)"]:
            row = cancer_results[cancer_results["stage_group"] == pref_stage]
            if len(row) > 0:
                row = row.iloc[0]
                break
        else:
            row = cancer_results.iloc[0]

        sel_pli = row.get("sel_pli", 0)
        predicted = row.get("predicted", "unknown")
        hr = row.get("HR", float("nan"))
        p = row.get("p", float("nan"))

        if hr < 1:
            observed = "protective"
        elif hr > 1:
            observed = "harmful"
        else:
            observed = "null"

        if p > 0.10:
            match_str = "null"
        elif predicted == "protective" and observed == "protective":
            match_str = "YES"
        elif predicted.startswith("harmful") and observed == "harmful":
            match_str = "YES"
        elif predicted == "protective" and observed == "harmful":
            match_str = "NO"
        elif predicted.startswith("harmful") and observed == "protective":
            match_str = "NO"
        else:
            match_str = "?"

        logger.info(f"{cancer_id:<8} {sel_pli:>8.3f} {predicted:<15} {hr:>12.3f} {observed:<12} {p:>8.4f} {match_str:<6}")

    # --- Forest plot ---
    plot_results = [r for r in all_results if r.get("error") is None and "HR" in r]
    if plot_results:
        plot_expanded_forest(plot_results, RESULTS_DIR / "expanded_validation_forest.png")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"GOTOVO. Vreme: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"Rezultati: {results_csv}")
    logger.info(f"JSON: {results_json_path}")
    logger.info(f"Forest plot: {RESULTS_DIR / 'expanded_validation_forest.png'}")
    logger.info(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
