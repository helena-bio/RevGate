#!/usr/bin/env python3
# run_metabric_validation.py -- Independent validation в METABRIC cohort
#
# Cohort: METABRIC (Nature 2012 & Nat Commun 2016), n=1904
# Data: cBioPortal REST API (HTTPS)
# Analyses: ssGSEA NV-Score -> Cox PH + stage-stratified Cox
#
# Output:
#   /root/.revgate/cache/metabric/metabric_expression.parquet
#   /root/.revgate/cache/metabric/metabric_clinical.parquet
#   results/metabric_cox_results.csv
#   results/metabric_stage_stratified.csv
#   results/metabric_kaplan_meier.png

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("metabric_validation")

REVGATE_ROOT = Path("/root/revgate")
CACHE_DIR    = Path("/root/.revgate/cache/metabric")
RESULTS_DIR  = REVGATE_ROOT / "results"
DEPMAP_DIR   = Path("/root/.revgate/cache/depmap")

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
STUDY_ID        = "brca_metabric"
PROFILE_ID      = "brca_metabric_mrna"

# BRCA top-50 DepMap dependency гени -- ще се изчислят динамично
# Референция: от run_ssgsea_analysis.py (FOXA1, YRDC, PPP1R15B...)


def api_get(url: str, retries: int = 3, pause: float = 2.0) -> list | dict:
    """GET към cBioPortal API с retry логика."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read())
        except Exception as exc:
            logger.warning(f"API грешка (опит {attempt+1}/{retries}): {exc}")
            if attempt < retries - 1:
                time.sleep(pause)
    raise RuntimeError(f"API недостъпен след {retries} опита: {url}")


def fetch_clinical_data() -> pd.DataFrame:
    """Изтегли клинични данни за всички пациенти + samples от METABRIC.

    PATIENT level: OS, age, ER, HER2, PAM50
    SAMPLE level:  TUMOR_STAGE (идва от sample attributes в cBioPortal)
    """
    logger.info("Изтеглям клинични данни от cBioPortal...")

    records: dict[str, dict] = {}

    # Изтегли PATIENT level данни
    for data_type in ["PATIENT", "SAMPLE"]:
        page = 0
        page_size = 10000
        while True:
            url = (
                f"{CBIOPORTAL_BASE}/studies/{STUDY_ID}/clinical-data"
                f"?clinicalDataType={data_type}&pageSize={page_size}&pageNumber={page}"
            )
            data = api_get(url)
            if not data:
                break
            for item in data:
                pid = item["patientId"]
                if pid not in records:
                    records[pid] = {"patient_id": pid}
                records[pid][item["clinicalAttributeId"]] = item["value"]
            logger.info(f"  {data_type} страница {page}: {len(data)} записа, {len(records)} уникални пациента")
            if len(data) < page_size:
                break
            page += 1

    df = pd.DataFrame.from_dict(records, orient="index").reset_index(drop=True)
    logger.info(f"Клинични данни: {len(df)} пациента, колони: {sorted(df.columns.tolist())}")
    return df


def get_entrez_ids(gene_symbols: list[str]) -> dict[str, int]:
    """Конвертирай Hugo gene symbols -> Entrez IDs чрез cBioPortal API.

    Returns:
        Dict {hugo_symbol: entrez_id}
    """
    logger.info(f"Конвертирам {len(gene_symbols)} gene symbols -> Entrez IDs...")
    mapping = {}
    batch_size = 50
    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i:i + batch_size]
        url = f"{CBIOPORTAL_BASE}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL"
        payload = json.dumps(batch).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            genes = json.loads(resp.read())
            for g in genes:
                mapping[g["hugoGeneSymbol"]] = g["entrezGeneId"]
            time.sleep(0.3)
        except Exception as exc:
            logger.warning(f"Gene batch грешка: {exc}")
    logger.info(f"Намерени {len(mapping)}/{len(gene_symbols)} Entrez IDs")
    return mapping


def fetch_all_sample_ids() -> list[str]:
    """Вземи всички sample IDs от METABRIC."""
    url = f"{CBIOPORTAL_BASE}/studies/{STUDY_ID}/samples?pageSize=2500"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    resp = urllib.request.urlopen(req, timeout=30)
    samples = json.loads(resp.read())
    ids = [s["sampleId"] for s in samples]
    logger.info(f"Намерени {len(ids)} samples в METABRIC")
    return ids


def fetch_expression_data(gene_list: list[str]) -> pd.DataFrame:
    """Изтегли expression данни за специфични гени от METABRIC.

    Args:
        gene_list: Списък с Hugo gene symbols.

    Returns:
        DataFrame (samples x genes).
    """
    logger.info(f"Изтеглям expression данни за {len(gene_list)} гени...")

    # Стъпка 1: Hugo -> Entrez mapping
    entrez_map = get_entrez_ids(gene_list)
    entrez_ids = list(entrez_map.values())

    if not entrez_ids:
        raise RuntimeError("Няма Entrez IDs за нито един ген")

    # Стъпка 2: Вземи всички sample IDs
    all_sample_ids = fetch_all_sample_ids()

    # Стъпка 3: Fetch expression -- батчове по samples (500 наведнъж)
    url = f"{CBIOPORTAL_BASE}/molecular-profiles/{PROFILE_ID}/molecular-data/fetch"
    all_rows = []
    sample_batch_size = 500

    for i in range(0, len(all_sample_ids), sample_batch_size):
        sample_batch = all_sample_ids[i:i + sample_batch_size]
        payload = json.dumps({
            "sampleIds":    sample_batch,
            "entrezGeneIds": entrez_ids,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            batch_data = json.loads(resp.read())
            all_rows.extend(batch_data)
            logger.info(f"  Samples батч {i//sample_batch_size + 1}: {len(batch_data)} записа")
            time.sleep(0.5)
        except Exception as exc:
            logger.warning(f"  Samples батч {i//sample_batch_size + 1} грешка: {exc}")

    if not all_rows:
        raise RuntimeError("Няма expression данни от cBioPortal")

    # Обърни entrez -> hugo за четливост
    entrez_to_hugo = {v: k for k, v in entrez_map.items()}

    rows_parsed = []
    for item in all_rows:
        rows_parsed.append({
            "gene":      entrez_to_hugo.get(item["entrezGeneId"], str(item["entrezGeneId"])),
            "sample_id": item["sampleId"],
            "value":     float(item["value"]) if item.get("value") not in ("NA", None) else np.nan,
        })

    df_long = pd.DataFrame(rows_parsed)
    df_wide = df_long.pivot_table(index="sample_id", columns="gene", values="value")
    logger.info(f"Expression матрица: {df_wide.shape[0]} samples x {df_wide.shape[1]} гени")
    return df_wide


def get_brca_top50_genes() -> tuple[list[str], list[str]]:
    """Изчисли top-50 BRCA гени + top-500 background гени от DepMap.

    ssGSEA изисква background матрица >> gene set size.
    Връща: (top50_gene_set, top500_background)
    """
    logger.info("Изчислявам BRCA top-50 гени от DepMap...")

    dep_df = pd.read_csv(DEPMAP_DIR / "CRISPRGeneEffect.csv", index_col=0)
    dep_df.columns = [col.split(" (")[0] for col in dep_df.columns]

    model_df = pd.read_csv(DEPMAP_DIR / "Model.csv")
    breast_mask = model_df["OncotreeLineage"].str.contains("Breast", na=False)
    breast_ids = model_df.loc[breast_mask, "ModelID"].tolist()

    global_mean = dep_df.mean(axis=0)
    breast_subset = dep_df.loc[dep_df.index.intersection(breast_ids)]
    cancer_mean = breast_subset.mean(axis=0)
    diff_score = cancer_mean - global_mean

    # Top-50 gene set
    top50 = diff_score.nsmallest(50).index.tolist()
    # Top-500 background -- за ssGSEA контекст
    top500 = diff_score.nsmallest(500).index.tolist()

    logger.info(f"Top-3 BRCA гени: {top50[:3]}")
    logger.info(f"Background гени: {len(top500)}")
    return top50, top500


def run_ssgsea_on_metabric(expr_df: pd.DataFrame, gene_set: list[str]) -> pd.Series:
    """Пусни ssGSEA върху METABRIC expression матрица.

    Args:
        expr_df:  DataFrame (samples x genes).
        gene_set: BRCA top-50 DepMap dependency гени.

    Returns:
        Series с NES score per sample.
    """
    import gseapy as gp

    logger.info("Пускам ssGSEA върху METABRIC...")

    # Филтрирай gene set до налични гени
    available = set(expr_df.columns)
    filtered_set = [g for g in gene_set if g in available]
    logger.info(f"Gene set overlap: {len(filtered_set)}/{len(gene_set)} гени")

    if len(filtered_set) < 10:
        raise RuntimeError(f"Недостатъчен overlap: само {len(filtered_set)} гени")

    # ssGSEA изисква genes x samples -- премахни index.name за gseapy съвместимост
    # ВАЖНО: използваме ЦЯЛАТА expression матрица (транспонирана), не само filtered_set
    # gseapy търси gene set гените в пълния индекс на матрицата
    expr_T = expr_df.T
    expr_T.index.name = None
    expr_T = expr_T.fillna(0)

    result = gp.ssgsea(
        data=expr_T,
        gene_sets={"NV_proxy_BRCA": filtered_set},
        outdir=None,
        no_plot=True,
        min_size=5,
        max_size=500,
        threads=1,
        verbose=False,
    )

    scores = result.res2d[["Name", "NES"]].set_index("Name")["NES"]
    logger.info(f"ssGSEA: {len(scores)} samples scored, mean NES={scores.mean():.3f}")
    return scores


def parse_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Парсни клиничните данни за Cox анализ."""
    result = pd.DataFrame()
    result["patient_id"] = df["patient_id"]

    # OS
    result["os_months"]   = pd.to_numeric(df.get("OS_MONTHS", pd.Series(dtype=float)), errors="coerce")
    os_status             = df.get("OS_STATUS", pd.Series(dtype=str)).astype(str).str.upper()
    # METABRIC формат: '0:LIVING' или '1:DECEASED'
    result["is_deceased"] = os_status.str.contains("DECEASED|DEAD|^1:", na=False).astype(int)

    # Age
    result["age"] = pd.to_numeric(df.get("AGE_AT_DIAGNOSIS", pd.Series(dtype=float)), errors="coerce")

    # Tumor stage -- METABRIC използва числов stage (0-4)
    result["stage"] = pd.to_numeric(df.get("TUMOR_STAGE", pd.Series(dtype=float)), errors="coerce")

    # ER статус -- METABRIC използва ER_IHC колона
    er_col = "ER_STATUS" if "ER_STATUS" in df.columns else "ER_IHC"
    er = df[er_col].astype(str).str.upper().str.strip() if er_col in df.columns else pd.Series("", index=df.index)
    result["er_positive"] = er.map(lambda x: 1.0 if x == "POS" else (0.0 if x in ["NEG","NEGATIVE"] else np.nan))

    # HER2 статус -- METABRIC използва HER2_SNP6 колона
    her2_col = "HER2_STATUS" if "HER2_STATUS" in df.columns else "HER2_SNP6"
    her2 = df[her2_col].astype(str).str.upper().str.strip() if her2_col in df.columns else pd.Series("", index=df.index)
    result["her2_positive"] = her2.map(lambda x: 1.0 if x in ["POS","AMPLIFIED"] else (0.0 if x in ["NEG","NEGATIVE","NEUTRAL","UNAMPLIFIED"] else np.nan))

    # PAM50 subtype -- dummy encoding (LumA е референция)
    subtype = df.get("CLAUDIN_SUBTYPE", pd.Series(dtype=str)).astype(str).str.lower()
    result["subtype_lumb"]    = (subtype == "lumb").astype(float)
    result["subtype_her2"]    = (subtype == "her2").astype(float)
    result["subtype_basal"]   = (subtype == "basal").astype(float)
    result["subtype_claudin"] = (subtype == "claudin-low").astype(float)
    unknown_subtype_mask = subtype.isin(["na", "", "nan", "nc"]).values
    for col in ["subtype_lumb", "subtype_her2", "subtype_basal", "subtype_claudin"]:
        result.loc[unknown_subtype_mask, col] = np.nan

    return result


def run_cox(merged: pd.DataFrame, covariates: list[str], label: str) -> dict:
    """Cox PH модел."""
    from lifelines import CoxPHFitter

    cols = ["os_months", "is_deceased", "ssgsea_nv_score"] + covariates
    available = [c for c in cols if c in merged.columns]
    cox_df = merged[available].dropna()
    cox_df = cox_df[cox_df["os_months"] > 0].copy()

    if len(cox_df) < 30:
        return {"label": label, "n": len(cox_df), "error": "insufficient_data"}

    # Стандартизация
    for col in ["age", "ssgsea_nv_score"]:
        if col in cox_df.columns:
            mu, sd = cox_df[col].mean(), cox_df[col].std()
            if sd > 0:
                cox_df[col] = (cox_df[col] - mu) / sd

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="os_months", event_col="is_deceased")

    row = cph.summary.loc["ssgsea_nv_score"]
    result = {
        "label":       label,
        "n":           len(cox_df),
        "n_events":    int(cox_df["is_deceased"].sum()),
        "HR":          float(row["exp(coef)"]),
        "HR_lower":    float(row["exp(coef) lower 95%"]),
        "HR_upper":    float(row["exp(coef) upper 95%"]),
        "p":           float(row["p"]),
        "concordance": float(cph.concordance_index_),
    }
    logger.info(
        f"{label}: n={result['n']}, events={result['n_events']}, "
        f"HR={result['HR']:.3f} [{result['HR_lower']:.3f}-{result['HR_upper']:.3f}], "
        f"p={result['p']:.4f}, C={result['concordance']:.3f}"
    )
    return result


def plot_km(merged: pd.DataFrame, output_path: Path) -> None:
    """Kaplan-Meier по NV клас в METABRIC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Присвои NV клас чрез min-max нормализация
    s = merged["ssgsea_nv_score"]
    normalized = 0.3 + (s - s.min()) / (s.max() - s.min()) * 0.35
    merged = merged.copy()
    merged["nv_class"] = pd.cut(
        normalized,
        bins=[-np.inf, 0.41, 0.53, np.inf],
        labels=["NV-C", "NV-B", "NV-A"],
    )

    colors = {"NV-A": "#1976D2", "NV-B": "#FF9800", "NV-C": "#D32F2F"}

    for ax_idx, (stage_label, stage_filter) in enumerate([
        ("All stages", merged),
        ("Early (I-II)", merged[merged["stage"].isin([1.0, 2.0])]),
    ]):
        ax = axes[ax_idx]
        df = stage_filter.dropna(subset=["os_months", "is_deceased", "nv_class"])

        kmf = KaplanMeierFitter()
        for nv_class in ["NV-A", "NV-B", "NV-C"]:
            group = df[df["nv_class"] == nv_class]
            if len(group) < 5:
                continue
            kmf.fit(group["os_months"], event_observed=group["is_deceased"],
                    label=f"{nv_class} (n={len(group)})")
            kmf.plot_survival_function(ax=ax, color=colors[nv_class], ci_show=True)

        # Log-rank NV-A vs NV-C
        nva = df[df["nv_class"] == "NV-A"]
        nvc = df[df["nv_class"] == "NV-C"]
        p_str = ""
        if len(nva) >= 5 and len(nvc) >= 5:
            lr = logrank_test(nva["os_months"], nvc["os_months"],
                              event_observed_A=nva["is_deceased"],
                              event_observed_B=nvc["is_deceased"])
            p_str = f"\nNV-A vs NV-C: p={lr.p_value:.4f}"

        ax.set_title(f"METABRIC — {stage_label}{p_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Месеци")
        ax.set_ylabel("Преживяемост")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

    plt.suptitle("METABRIC Independent Validation — ssGSEA NV-Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"KM фигура записана: {output_path}")


def main() -> int:
    sys.path.insert(0, str(REVGATE_ROOT / "src"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Стъпка 1: Top-50 BRCA гени + background от DepMap ---
    top50_genes, background_genes = get_brca_top50_genes()

    # --- Стъпка 2: Изтегли expression данни от cBioPortal ---
    expr_cache = CACHE_DIR / "metabric_expression.parquet"
    if expr_cache.exists():
        logger.info(f"Зареждам expression от кеш: {expr_cache}")
        expr_df = pd.read_parquet(expr_cache)
    else:
        expr_df = fetch_expression_data(background_genes)
        expr_df.to_parquet(expr_cache)
        logger.info(f"Expression кеширан: {expr_cache}")

    # --- Стъпка 3: Изтегли клинични данни ---
    clin_cache = CACHE_DIR / "metabric_clinical.parquet"
    if clin_cache.exists():
        logger.info(f"Зареждам клинични от кеш: {clin_cache}")
        raw_clin = pd.read_parquet(clin_cache)
    else:
        raw_clin = fetch_clinical_data()
        raw_clin.to_parquet(clin_cache)
        logger.info(f"Клинични кеширани: {clin_cache}")

    # --- Стъпка 4: ssGSEA scoring ---
    ssgsea_cache = CACHE_DIR / "metabric_ssgsea_scores.parquet"
    if ssgsea_cache.exists():
        logger.info(f"Зареждам ssGSEA scores от кеш: {ssgsea_cache}")
        scores = pd.read_parquet(ssgsea_cache)["ssgsea_nv_score"]
        scores.index = pd.read_parquet(ssgsea_cache)["sample_id"]
    else:
        scores = run_ssgsea_on_metabric(expr_df, top50_genes)
        pd.DataFrame({"sample_id": scores.index, "ssgsea_nv_score": scores.values}).to_parquet(ssgsea_cache)

    # --- Стъпка 5: Парсни клинични данни ---
    clin_df = parse_clinical(raw_clin)

    # --- Стъпка 6: Merge ---
    # METABRIC: sample_id = patient_id (1 sample per patient)
    scores_df = pd.DataFrame({"sample_id": scores.index, "ssgsea_nv_score": scores.values})
    # sample_id формат: METABRIC_2_..._### -- вземи patient_id от клиничните
    # cBioPortal sample IDs съответстват на patient IDs в METABRIC
    scores_df["patient_id"] = scores_df["sample_id"]
    merged = scores_df.merge(clin_df, on="patient_id", how="inner")
    merged = merged[merged["os_months"].notna() & (merged["os_months"] > 0)]

    logger.info(f"След merge: {len(merged)} пациента")
    logger.info(f"Stage разпределение:\n{merged['stage'].value_counts().sort_index().to_string()}")

    # --- Стъпка 7: Cox анализи ---
    cox_results = []

    # Univariate
    res_uni = run_cox(merged, [], "METABRIC Univariate")
    res_uni["model"] = "univariate"
    cox_results.append(res_uni)

    # Multivariate: age + stage + PAM50 subtype
    # НЕ включваме ER/HER2 -- силно корелирани с PAM50 (multicollinearity)
    covariates_mv = ["age", "stage", "subtype_lumb", "subtype_her2",
                     "subtype_basal", "subtype_claudin"]
    res_mv = run_cox(merged, covariates_mv, "METABRIC Multivariate")
    res_mv["model"] = "multivariate"
    cox_results.append(res_mv)

    # Stage-stratified: Early (I-II)
    early = merged[merged["stage"].isin([1.0, 2.0])].copy()
    res_early = run_cox(early, [], "METABRIC Early (I-II)")
    res_early["model"] = "stage_early"
    cox_results.append(res_early)

    # Stage-stratified: Advanced (III-IV)
    advanced = merged[merged["stage"].isin([3.0, 4.0])].copy()
    res_adv = run_cox(advanced, [], "METABRIC Advanced (III-IV)")
    res_adv["model"] = "stage_advanced"
    cox_results.append(res_adv)

    # --- Запази резултати ---
    cox_df = pd.DataFrame(cox_results)
    cox_path = RESULTS_DIR / "metabric_cox_results.csv"
    cox_df.to_csv(cox_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info("=== METABRIC VALIDATION РЕЗУЛТАТИ ===")
    logger.info(f"\n{cox_df[['label','n','n_events','HR','HR_lower','HR_upper','p','concordance']].to_string()}")

    # --- KM plot ---
    plot_km(merged, RESULTS_DIR / "metabric_kaplan_meier.png")

    logger.info(f"\nФайлове записани в: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
