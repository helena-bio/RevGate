#!/usr/bin/env python3
"""
RevGate — Finding 13: Expansion to 20 Cancer Types
===================================================
Dependency Architecture Principle validation at scale.

Phase 1: Download expression + clinical for 10 new cancer types
Phase 2: Compute NV components (Gini, Selectivity, pLI, Centrality) per lineage
Phase 3: ssGSEA per-patient scoring for all 20 types
Phase 4: Cox PH (stage-stratified + restricted follow-up) for all 20 types
Phase 5: Dependency Architecture Principle validation (Sel+pLI accuracy)
Phase 6: LOOCV of Sel+pLI threshold

Data: DepMap 24Q4, TCGA GDC, gnomAD v4, STRING v12
Helena Bioinformatics, 2026.
"""

import argparse
import gzip
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
import requests
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("20cancer")

file_handler = logging.FileHandler("results/20cancer_expansion.log", mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("/root/.revgate/cache")
DEPMAP_CHRONOS = CACHE_DIR / "depmap/CRISPRGeneEffect.csv"
DEPMAP_MODEL = CACHE_DIR / "depmap/Model.csv"
GNOMAD_PATH = CACHE_DIR / "gnomad/gnomad.v4.constraint.tsv"
STRING_PATH = CACHE_DIR / "string/9606.protein.links.v12.0.txt"
STRING_INFO = CACHE_DIR / "string/9606.protein.info.v12.0.txt"
PAN_CANCER_GZ = CACHE_DIR / "tcga/tcga_RSEM_gene_tpm.gz"
ENSEMBL_MAP = CACHE_DIR / "tcga/ensembl_to_symbol.json"
TCGA_DIR = CACHE_DIR / "tcga"
RESULTS_DIR = Path("results")

# ── Complete mapping: 20 TCGA codes -> DepMap lineages ───────────────────────
CANCER_TO_LINEAGE = {
    # Existing 10
    "BRCA": "Breast", "KIRC": "Kidney", "LUAD": "Lung", "SKCM": "Skin",
    "PAAD": "Pancreas", "LAML": "Myeloid", "COAD": "Bowel",
    "OV": "Ovary/Fallopian Tube", "HNSC": "Head and Neck", "UCEC": "Uterus",
    # New 10
    "GBM": "CNS/Brain", "LGG": "CNS/Brain", "DLBC": "Lymphoid",
    "STAD": "Esophagus/Stomach", "LIHC": "Liver",
    "BLCA": "Bladder/Urinary Tract", "THCA": "Thyroid",
    "PRAD": "Prostate", "CESC": "Cervix", "SARC": "Soft Tissue",
}

# TSS codes for expression extraction from pan-cancer file
TCGA_PROJECT_CODES = {
    # Existing (from run_expanded_validation.py)
    "BRCA": ["TCGA-3C", "TCGA-5L", "TCGA-5T", "TCGA-A1", "TCGA-A2", "TCGA-A7",
             "TCGA-A8", "TCGA-AC", "TCGA-AN", "TCGA-AO", "TCGA-AQ", "TCGA-AR",
             "TCGA-AT", "TCGA-B6", "TCGA-BH", "TCGA-C8", "TCGA-D8", "TCGA-E2",
             "TCGA-E9", "TCGA-EW", "TCGA-GI", "TCGA-GM", "TCGA-HN", "TCGA-JL",
             "TCGA-LD", "TCGA-LL", "TCGA-LQ", "TCGA-OL", "TCGA-PE", "TCGA-PL",
             "TCGA-S3", "TCGA-UL", "TCGA-UU", "TCGA-WT", "TCGA-XX"],
    "KIRC": ["TCGA-A3", "TCGA-AK", "TCGA-B0", "TCGA-B2", "TCGA-B4", "TCGA-B8",
             "TCGA-BP", "TCGA-CJ", "TCGA-CW", "TCGA-CZ", "TCGA-DV", "TCGA-EU",
             "TCGA-F9", "TCGA-G7", "TCGA-HE", "TCGA-IA", "TCGA-IZ", "TCGA-KL",
             "TCGA-KM", "TCGA-KN", "TCGA-KO", "TCGA-MH", "TCGA-MM", "TCGA-MW",
             "TCGA-SX", "TCGA-T7", "TCGA-UZ", "TCGA-Y8"],
    "LUAD": ["TCGA-05", "TCGA-17", "TCGA-33", "TCGA-35", "TCGA-38", "TCGA-44",
             "TCGA-46", "TCGA-49", "TCGA-4B", "TCGA-50", "TCGA-53", "TCGA-55",
             "TCGA-62", "TCGA-64", "TCGA-67", "TCGA-69", "TCGA-6A", "TCGA-71",
             "TCGA-73", "TCGA-75", "TCGA-78", "TCGA-80", "TCGA-83", "TCGA-86",
             "TCGA-91", "TCGA-93", "TCGA-95", "TCGA-97", "TCGA-99", "TCGA-J2",
             "TCGA-L4", "TCGA-L9", "TCGA-MN", "TCGA-MP", "TCGA-NJ", "TCGA-NQ",
             "TCGA-S2"],
    "SKCM": ["TCGA-BF", "TCGA-D3", "TCGA-D9", "TCGA-DA", "TCGA-EB", "TCGA-EE",
             "TCGA-ER", "TCGA-FR", "TCGA-FS", "TCGA-FW", "TCGA-GF", "TCGA-GN",
             "TCGA-HR", "TCGA-LH", "TCGA-QB", "TCGA-RP", "TCGA-RZ", "TCGA-VD",
             "TCGA-WC", "TCGA-WE", "TCGA-XV", "TCGA-YG"],
    "PAAD": ["TCGA-2J", "TCGA-2L", "TCGA-3A", "TCGA-3E", "TCGA-F2", "TCGA-FB",
             "TCGA-H6", "TCGA-H8", "TCGA-HV", "TCGA-HZ", "TCGA-IB", "TCGA-LB",
             "TCGA-M8", "TCGA-OE", "TCGA-PZ", "TCGA-Q3", "TCGA-RB", "TCGA-RL",
             "TCGA-S4", "TCGA-US", "TCGA-XD", "TCGA-XN", "TCGA-YB", "TCGA-YH",
             "TCGA-YY", "TCGA-Z5"],
    "LAML": ["TCGA-AB"],
    "COAD": ["TCGA-3L", "TCGA-4N", "TCGA-4T", "TCGA-5M", "TCGA-A6", "TCGA-AA",
             "TCGA-AD", "TCGA-AU", "TCGA-AY", "TCGA-AZ", "TCGA-CA", "TCGA-CK",
             "TCGA-CM", "TCGA-D5", "TCGA-DM", "TCGA-DY", "TCGA-F4", "TCGA-F5",
             "TCGA-G4", "TCGA-GH", "TCGA-NH", "TCGA-QG", "TCGA-RU", "TCGA-T9",
             "TCGA-WS"],
    "OV": ["TCGA-04", "TCGA-09", "TCGA-10", "TCGA-13", "TCGA-20", "TCGA-23",
           "TCGA-24", "TCGA-25", "TCGA-29", "TCGA-30", "TCGA-31", "TCGA-36",
           "TCGA-42", "TCGA-57", "TCGA-59", "TCGA-61", "TCGA-WR"],
    "HNSC": ["TCGA-BA", "TCGA-BB", "TCGA-CN", "TCGA-CQ", "TCGA-CR", "TCGA-CV",
             "TCGA-D6", "TCGA-DQ", "TCGA-F7", "TCGA-H7", "TCGA-HD", "TCGA-IQ",
             "TCGA-MZ", "TCGA-P3", "TCGA-QK", "TCGA-RS", "TCGA-T2", "TCGA-TN",
             "TCGA-UP", "TCGA-UF"],
    "UCEC": ["TCGA-A5", "TCGA-AJ", "TCGA-AP", "TCGA-AX", "TCGA-B5", "TCGA-BG",
             "TCGA-BK", "TCGA-BS", "TCGA-D1", "TCGA-DI", "TCGA-E6", "TCGA-EO",
             "TCGA-EY", "TCGA-FI", "TCGA-FL", "TCGA-N5", "TCGA-N7", "TCGA-N8",
             "TCGA-N9", "TCGA-PG", "TCGA-QC", "TCGA-QF", "TCGA-SL"],
    # New 10
    "GBM": ["TCGA-02", "TCGA-06", "TCGA-08", "TCGA-12", "TCGA-14", "TCGA-15",
            "TCGA-16", "TCGA-19", "TCGA-26", "TCGA-27", "TCGA-28", "TCGA-32",
            "TCGA-41", "TCGA-76", "TCGA-87"],
    "LGG": ["TCGA-CS", "TCGA-DB", "TCGA-DH", "TCGA-DU", "TCGA-E1", "TCGA-FG",
            "TCGA-HT", "TCGA-HW", "TCGA-P5", "TCGA-QH", "TCGA-R8", "TCGA-S9",
            "TCGA-TM", "TCGA-TQ", "TCGA-VM", "TCGA-VV", "TCGA-WH", "TCGA-WY"],
    "DLBC": ["TCGA-FA", "TCGA-FF", "TCGA-FM", "TCGA-GR", "TCGA-GS", "TCGA-RC",
             "TCGA-RG", "TCGA-RI"],
    "STAD": ["TCGA-B7", "TCGA-BR", "TCGA-CD", "TCGA-CG", "TCGA-D7", "TCGA-FP",
             "TCGA-HU", "TCGA-IN", "TCGA-KB", "TCGA-MX", "TCGA-R5", "TCGA-R6",
             "TCGA-RD", "TCGA-RE", "TCGA-VQ"],
    "LIHC": ["TCGA-2V", "TCGA-2Y", "TCGA-BC", "TCGA-BD", "TCGA-CC", "TCGA-DD",
             "TCGA-ED", "TCGA-EP", "TCGA-ES", "TCGA-FV", "TCGA-G3", "TCGA-GJ",
             "TCGA-HP", "TCGA-K7", "TCGA-KR", "TCGA-LG", "TCGA-MI", "TCGA-NI",
             "TCGA-O8", "TCGA-PD", "TCGA-QA", "TCGA-RC", "TCGA-T1", "TCGA-UB",
             "TCGA-WX", "TCGA-XR", "TCGA-YA", "TCGA-ZP", "TCGA-ZS"],
    "BLCA": ["TCGA-2F", "TCGA-4Z", "TCGA-BL", "TCGA-BT", "TCGA-C4", "TCGA-CF",
             "TCGA-CU", "TCGA-DK", "TCGA-E7", "TCGA-FD", "TCGA-FJ", "TCGA-FT",
             "TCGA-G2", "TCGA-GC", "TCGA-GD", "TCGA-GU", "TCGA-GV", "TCGA-HQ",
             "TCGA-K4", "TCGA-KQ", "TCGA-LC", "TCGA-MV", "TCGA-NT", "TCGA-PQ",
             "TCGA-SY", "TCGA-XF", "TCGA-YC", "TCGA-ZF"],
    "THCA": ["TCGA-BJ", "TCGA-DE", "TCGA-DJ", "TCGA-DO", "TCGA-EL", "TCGA-EM",
             "TCGA-ET", "TCGA-FE", "TCGA-FK", "TCGA-H2", "TCGA-IM", "TCGA-IH",
             "TCGA-J8", "TCGA-KS", "TCGA-L6", "TCGA-MK", "TCGA-ML", "TCGA-N3",
             "TCGA-OH", "TCGA-QY"],
    "PRAD": ["TCGA-CH", "TCGA-EJ", "TCGA-FC", "TCGA-G9", "TCGA-HC", "TCGA-HI",
             "TCGA-J4", "TCGA-J9", "TCGA-KC", "TCGA-KK", "TCGA-M7", "TCGA-MC",
             "TCGA-MG", "TCGA-SU", "TCGA-TK", "TCGA-V1", "TCGA-VP", "TCGA-WW",
             "TCGA-X4", "TCGA-XJ", "TCGA-XK", "TCGA-Y6", "TCGA-YJ", "TCGA-YL"],
    "CESC": ["TCGA-2W", "TCGA-BI", "TCGA-C5", "TCGA-DS", "TCGA-EA", "TCGA-EK",
             "TCGA-FU", "TCGA-HG", "TCGA-HM", "TCGA-IR", "TCGA-JW", "TCGA-JX",
             "TCGA-MA", "TCGA-MY", "TCGA-Q1", "TCGA-R3", "TCGA-RA", "TCGA-VS",
             "TCGA-ZJ"],
    "SARC": ["TCGA-3B", "TCGA-DX", "TCGA-FX", "TCGA-HB", "TCGA-IF", "TCGA-IS",
             "TCGA-IV", "TCGA-JS", "TCGA-K2", "TCGA-LI", "TCGA-MB", "TCGA-MJ",
             "TCGA-MO", "TCGA-NB", "TCGA-PC", "TCGA-QQ", "TCGA-QU", "TCGA-SI",
             "TCGA-VG", "TCGA-WK", "TCGA-WL", "TCGA-X2", "TCGA-X6", "TCGA-Z4"],
}

NEW_CANCER_TYPES = ["GBM", "LGG", "DLBC", "STAD", "LIHC", "BLCA", "THCA", "PRAD", "CESC", "SARC"]
ALL_CANCER_TYPES = list(CANCER_TO_LINEAGE.keys())


# ── Data functions (reused from run_expanded_validation.py) ──────────────────

def extract_expression_for_cancer(cancer_id, pan_cancer_path, ensembl_map_path, output_path):
    """Extract expression from pan-cancer RSEM file."""
    if output_path.exists():
        logger.info(f"{cancer_id}: expression cached -> {output_path}")
        return pd.read_parquet(output_path)

    logger.info(f"{cancer_id}: extracting expression...")

    with open(ensembl_map_path) as f:
        ensembl_to_symbol = json.load(f)

    with gzip.open(pan_cancer_path, "rt") as f:
        header_line = f.readline().strip()
    all_samples = header_line.split("\t")[1:]

    tss_codes = TCGA_PROJECT_CODES.get(cancer_id, [])
    if not tss_codes:
        raise ValueError(f"No TSS codes for {cancer_id}")

    cancer_sample_indices = []
    cancer_sample_ids = []
    for i, sid in enumerate(all_samples):
        prefix = "-".join(sid.split("-")[:2])
        if prefix in tss_codes:
            cancer_sample_indices.append(i + 1)
            cancer_sample_ids.append(sid)

    logger.info(f"{cancer_id}: {len(cancer_sample_ids)} samples from {len(all_samples)} total")

    if len(cancer_sample_ids) < 10:
        raise ValueError(f"{cancer_id}: too few samples ({len(cancer_sample_ids)})")

    usecols = [0] + cancer_sample_indices
    df = pd.read_csv(pan_cancer_path, sep="\t", usecols=usecols, index_col=0, compression="gzip")

    df.index = [eid.split(".")[0] for eid in df.index]
    df.index = [ensembl_to_symbol.get(eid, eid) for eid in df.index]
    df = df[~df.index.duplicated(keep="first")]
    df = df[~df.index.str.startswith("ENSG")]
    df = df.T

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"{cancer_id}: expression saved ({df.shape})")
    return df


def download_clinical_from_gdc(cancer_id, output_path):
    """Download clinical from GDC API."""
    if output_path.exists():
        logger.info(f"{cancer_id}: clinical cached -> {output_path}")
        return pd.read_csv(output_path, sep="\t", low_memory=False)

    logger.info(f"{cancer_id}: downloading clinical from GDC...")

    endpoint = "https://api.gdc.cancer.gov/cases"
    fields = [
        "submitter_id", "demographic.vital_status", "demographic.days_to_death",
        "demographic.days_to_last_follow_up", "demographic.gender",
        "demographic.age_at_index", "diagnoses.ajcc_pathologic_stage",
        "diagnoses.figo_stage", "diagnoses.primary_diagnosis",
    ]
    params = {
        "filters": json.dumps({"op": "in", "content": {
            "field": "cases.project.project_id", "value": [f"TCGA-{cancer_id}"]}}),
        "fields": ",".join(fields), "format": "JSON", "size": "2000",
    }
    response = requests.get(endpoint, params=params, timeout=60)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]
    logger.info(f"{cancer_id}: {len(hits)} cases from GDC")

    rows = []
    for case in hits:
        row = {"_PATIENT": case.get("submitter_id", "")}
        demo = case.get("demographic", {}) or {}
        row["vital_status"] = demo.get("vital_status", "")
        row["days_to_death"] = demo.get("days_to_death", "")
        row["days_to_last_followup"] = demo.get("days_to_last_follow_up", "")
        row["age_at_initial_pathologic_diagnosis"] = demo.get("age_at_index", "")
        diagnoses = case.get("diagnoses", []) or []
        if diagnoses:
            d = diagnoses[0]
            row["pathologic_stage"] = d.get("ajcc_pathologic_stage", "") or d.get("figo_stage", "")
        else:
            row["pathologic_stage"] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"{cancer_id}: clinical saved ({len(df)} cases)")
    return df


# ── NV Components ────────────────────────────────────────────────────────────

def compute_nv_components_all_lineages():
    """Compute NV-Score components for all unique lineages."""
    logger.info("Computing NV components for all lineages...")

    dep_df = pd.read_csv(DEPMAP_CHRONOS, index_col=0)
    dep_df.columns = [c.split(" (")[0] for c in dep_df.columns]
    model_df = pd.read_csv(DEPMAP_MODEL)

    # gnomAD pLI
    gnomad = pd.read_csv(GNOMAD_PATH, sep="\t", usecols=["gene", "lof_hc_lc.pLI"], low_memory=False)
    gnomad = gnomad.dropna(subset=["lof_hc_lc.pLI"]).drop_duplicates(subset=["gene"], keep="first")
    pli_map = dict(zip(gnomad["gene"], gnomad["lof_hc_lc.pLI"]))

    # STRING degree centrality
    try:
        info = pd.read_csv(STRING_INFO, sep="\t")
        string_name_map = dict(zip(info["#string_protein_id"], info["preferred_name"]))
        edges = pd.read_csv(STRING_PATH, sep=" ")
        edges = edges[edges["combined_score"] >= 700]
        edges["gene1"] = edges["protein1"].map(string_name_map)
        edges["gene2"] = edges["protein2"].map(string_name_map)
        edges = edges.dropna(subset=["gene1", "gene2"])
        from collections import Counter
        degree_counter = Counter()
        for _, row in edges.iterrows():
            degree_counter[row["gene1"]] += 1
            degree_counter[row["gene2"]] += 1
        max_degree = max(degree_counter.values()) if degree_counter else 1
        centrality_map = {g: d / max_degree for g, d in degree_counter.items()}
        logger.info(f"STRING: {len(centrality_map)} genes with centrality")
    except Exception as e:
        logger.warning(f"STRING loading failed: {e} -- using empty centrality")
        centrality_map = {}

    global_mean = dep_df.mean(axis=0)
    results = {}

    unique_lineages = set(CANCER_TO_LINEAGE.values())
    for lineage in sorted(unique_lineages):
        mask = model_df["OncotreeLineage"] == lineage
        line_ids = model_df.loc[mask, "ModelID"].tolist()
        subset = dep_df.loc[dep_df.index.intersection(line_ids)]

        if len(subset) < 3:
            logger.warning(f"{lineage}: only {len(subset)} cell lines, skipping")
            continue

        cancer_mean = subset.mean(axis=0)
        diff_score = cancer_mean - global_mean
        top20 = diff_score.nsmallest(20).index.tolist()
        top50 = diff_score.nsmallest(50).index.tolist()

        # Gini of top-20 dependency scores
        top20_scores = cancer_mean[top20].values
        top20_scores_abs = np.abs(top20_scores)
        n = len(top20_scores_abs)
        if n > 0 and top20_scores_abs.sum() > 0:
            sorted_scores = np.sort(top20_scores_abs)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_scores) - (n + 1) * np.sum(sorted_scores)) / (n * np.sum(sorted_scores))
        else:
            gini = 0.0

        # Selectivity: fraction of cell lines where gene is dependency IN this lineage vs all
        selectivity_scores = []
        for gene in top20:
            if gene not in dep_df.columns:
                continue
            cancer_dep = (subset[gene] < -0.5).mean() if gene in subset.columns else 0
            global_dep = (dep_df[gene] < -0.5).mean()
            sel = cancer_dep - global_dep if global_dep < cancer_dep else 0
            selectivity_scores.append(sel)
        selectivity = np.mean(selectivity_scores) if selectivity_scores else 0

        # mean_pLI
        pli_vals = [pli_map.get(g, np.nan) for g in top20]
        pli_vals = [v for v in pli_vals if not np.isnan(v)]
        mean_pli = np.mean(pli_vals) if pli_vals else 0

        # mean_centrality
        cent_vals = [centrality_map.get(g, 0) for g in top20]
        mean_centrality = np.mean(cent_vals)

        results[lineage] = {
            "lineage": lineage,
            "n_cell_lines": len(subset),
            "gini": round(gini, 4),
            "selectivity": round(selectivity, 4),
            "mean_pli": round(mean_pli, 4),
            "mean_centrality": round(mean_centrality, 4),
            "top3_genes": top20[:3],
            "top50_genes": top50,
        }
        logger.info(f"  {lineage:<25}: Gini={gini:.3f} Sel={selectivity:.3f} "
                     f"pLI={mean_pli:.3f} Cent={mean_centrality:.3f} "
                     f"top3={top20[:3]}")

    # MinMax scale and compute NV-Score
    lineages = list(results.keys())
    for comp in ["gini", "selectivity", "mean_pli", "mean_centrality"]:
        vals = [results[l][comp] for l in lineages]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax > vmin else 1
        for l in lineages:
            results[l][f"{comp}_scaled"] = round((results[l][comp] - vmin) / rng, 4)

    for l in lineages:
        r = results[l]
        nv = 0.25 * r["gini_scaled"] + 0.25 * r["selectivity_scaled"] + \
             0.25 * r["mean_pli_scaled"] + 0.25 * r["mean_centrality_scaled"]
        r["nv_score"] = round(nv, 4)
        r["sel_pli"] = round(r["selectivity"] + r["mean_pli"], 4)

        # NV-Class
        if nv >= 0.50:
            r["nv_class"] = "NV-A"
        elif nv >= 0.35:
            r["nv_class"] = "NV-B"
        else:
            r["nv_class"] = "NV-C"

    return results


# ── ssGSEA per-patient ───────────────────────────────────────────────────────

def run_ssgsea_for_cancer(cancer_id, lineage, nv_components, expression_path):
    """Run ssGSEA for a cancer type using its lineage gene set."""
    import gseapy as gp

    gene_set = nv_components[lineage]["top50_genes"]
    expr_df = pd.read_parquet(expression_path)

    available = set(expr_df.columns)
    filtered = [g for g in gene_set if g in available]
    logger.info(f"{cancer_id}: ssGSEA gene set {len(filtered)}/{len(gene_set)} overlap")

    if len(filtered) < 10:
        logger.warning(f"{cancer_id}: insufficient gene overlap ({len(filtered)})")
        return None

    expr_T = expr_df.T
    expr_T.index.name = None
    expr_T = expr_T.fillna(0)

    result = gp.ssgsea(
        data=expr_T,
        gene_sets={f"NV_proxy_{cancer_id}": filtered},
        outdir=None, no_plot=True, min_size=5, max_size=500, threads=1, verbose=False,
    )

    scores = result.res2d[["Name", "NES"]].set_index("Name")["NES"]
    logger.info(f"{cancer_id}: ssGSEA scored {len(scores)} patients")
    return scores


# ── Clinical loading ─────────────────────────────────────────────────────────

def load_clinical_for_cox(cancer_id):
    """Load clinical and prepare for Cox."""
    path = TCGA_DIR / f"{cancer_id}_clinical_raw.tsv"
    if not path.exists():
        return None

    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Detect column names (different formats)
    patient_col = "_PATIENT" if "_PATIENT" in df.columns else "sampleID" if "sampleID" in df.columns else df.columns[0]

    # OS
    if "days_to_death" in df.columns:
        df["os_days"] = pd.to_numeric(df["days_to_death"], errors="coerce")
        df["os_event"] = df["vital_status"].str.lower().str.contains("dead|deceased", na=False).astype(int)
        followup = pd.to_numeric(df["days_to_last_followup"], errors="coerce")
        df["os_days"] = df["os_days"].fillna(followup)
    elif "Days_to_date_of_Death_nature2012" in df.columns:
        df["os_days"] = pd.to_numeric(df["Days_to_date_of_Death_nature2012"], errors="coerce")
        followup = pd.to_numeric(df.get("Days_to_Date_of_Last_Contact_nature2012"), errors="coerce")
        df["os_days"] = df["os_days"].fillna(followup)
        df["os_event"] = df["os_days"].notna().astype(int)
        # Correct: use vital status if available
        if "vital_status" in df.columns:
            df["os_event"] = df["vital_status"].str.lower().str.contains("dead|deceased", na=False).astype(int)
    else:
        return None

    df["os_months"] = df["os_days"] / 30.44
    df["patient_id"] = df[patient_col]

    # Age
    age_cols = [c for c in df.columns if "age" in c.lower()]
    if age_cols:
        df["age"] = pd.to_numeric(df[age_cols[0]], errors="coerce")
    else:
        df["age"] = np.nan

    # Stage
    stage_cols = [c for c in df.columns if "stage" in c.lower() and "pathologic" in c.lower()]
    if stage_cols:
        raw_stage = df[stage_cols[0]].astype(str).str.upper()
        df["stage_numeric"] = raw_stage.map(lambda x:
            1 if "I" in x and "II" not in x and "III" not in x and "IV" not in x else
            2 if "II" in x and "III" not in x and "IV" not in x else
            3 if "III" in x and "IV" not in x else
            4 if "IV" in x else np.nan
        )
    else:
        df["stage_numeric"] = np.nan

    df = df.dropna(subset=["os_months", "os_event"])
    df = df[df["os_months"] > 0]

    return df[["patient_id", "os_months", "os_event", "age", "stage_numeric"]]


# ── Cox analysis ─────────────────────────────────────────────────────────────

def run_cox_for_cancer(cancer_id, ssgsea_scores, clinical_df, cutoffs=None):
    """Run Cox PH for a cancer type."""
    if ssgsea_scores is None or clinical_df is None:
        return {}

    # Merge
    scores_df = pd.DataFrame({"patient_id": ssgsea_scores.index, "nv_score": ssgsea_scores.values})
    # TCGA patient ID format: TCGA-XX-YYYY from TCGA-XX-YYYY-01
    scores_df["patient_id"] = scores_df["patient_id"].str[:12]
    clinical_df["patient_id"] = clinical_df["patient_id"].str[:12]

    merged = scores_df.merge(clinical_df, on="patient_id", how="inner")
    merged = merged.dropna(subset=["os_months", "os_event", "nv_score"])
    merged = merged[merged["os_months"] > 0]
    merged["nv_zscore"] = StandardScaler().fit_transform(merged[["nv_score"]])

    if len(merged) < 30 or merged["os_event"].sum() < 15:
        logger.warning(f"{cancer_id}: insufficient data (n={len(merged)}, events={merged['os_event'].sum()})")
        return {}

    results = {}

    # Full follow-up
    results["full"] = _fit_cox(merged, f"{cancer_id} full")

    # Stage-stratified
    if merged["stage_numeric"].notna().sum() > 30:
        early = merged[merged["stage_numeric"].isin([1, 2])]
        advanced = merged[merged["stage_numeric"].isin([3, 4])]
        if len(early) >= 30 and early["os_event"].sum() >= 10:
            results["early"] = _fit_cox(early, f"{cancer_id} Early (I-II)")
        if len(advanced) >= 30 and advanced["os_event"].sum() >= 10:
            results["advanced"] = _fit_cox(advanced, f"{cancer_id} Advanced (III-IV)")

    # Restricted follow-up
    if cutoffs:
        for cutoff in cutoffs:
            restricted = merged.copy()
            mask = restricted["os_months"] > cutoff
            restricted.loc[mask, "os_months"] = cutoff
            restricted.loc[mask, "os_event"] = 0
            if restricted["os_event"].sum() >= 15:
                results[f"{cutoff}mo"] = _fit_cox(restricted, f"{cancer_id} {cutoff}mo")

    return results


def _fit_cox(df, label, penalizer=0.01):
    """Fit univariate Cox: nv_zscore + age -> OS."""
    cols = ["nv_zscore", "age", "os_months", "os_event"]
    sub = df[cols].dropna()
    sub = sub[sub["os_months"] > 0]

    if len(sub) < 30 or sub["os_event"].sum() < 10:
        return {"label": label, "n": len(sub), "events": int(sub["os_event"].sum()), "error": "insufficient"}

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


# ── Dependency Architecture validation ───────────────────────────────────────

def validate_dependency_architecture(all_cox_results, nv_components):
    """Validate Sel+pLI principle across all 20 cancer types."""
    logger.info("\n" + "=" * 60)
    logger.info("DEPENDENCY ARCHITECTURE PRINCIPLE — 20 Cancer Types")
    logger.info("=" * 60)

    rows = []
    for cancer_id in ALL_CANCER_TYPES:
        lineage = CANCER_TO_LINEAGE[cancer_id]
        comp = nv_components.get(lineage, {})
        sel_pli = comp.get("sel_pli", np.nan)

        # Best HR: prioritize early, then restricted, then full
        cox = all_cox_results.get(cancer_id, {})
        best = None
        best_label = ""
        for key in ["early", "advanced", "36mo", "60mo", "84mo", "full"]:
            r = cox.get(key)
            if r and "error" not in r and r.get("p", 1) < (best.get("p", 1) if best else 1):
                best = r
                best_label = key

        if best is None:
            best = cox.get("full", {"hr": np.nan, "p": np.nan, "n": 0, "events": 0})
            best_label = "full"

        predicted = "protective" if sel_pli > 1.0 else "harmful"
        observed_hr = best.get("hr", np.nan)
        observed_p = best.get("p", np.nan)
        observed_dir = "protective" if observed_hr < 1.0 else "harmful" if observed_hr > 1.0 else "null"

        if observed_p < 0.05:
            match = "YES" if predicted == observed_dir else "NO"
        elif observed_p < 0.10:
            match = "trend" if predicted == observed_dir else "MISS"
        else:
            match = "null"

        row = {
            "cancer": cancer_id, "lineage": lineage, "sel_pli": sel_pli,
            "predicted": predicted, "best_stratum": best_label,
            "hr": observed_hr, "p": observed_p,
            "n": best.get("n", 0), "events": best.get("events", 0),
            "ci_low": best.get("ci_low", np.nan), "ci_high": best.get("ci_high", np.nan),
            "observed_dir": observed_dir, "match": match,
            "nv_score": comp.get("nv_score", np.nan),
            "nv_class": comp.get("nv_class", ""),
        }
        rows.append(row)

        sig = "***" if observed_p < 0.001 else "**" if observed_p < 0.01 else "*" if observed_p < 0.05 else "~" if observed_p < 0.10 else ""
        logger.info(f"  {cancer_id:5s} Sel+pLI={sel_pli:.3f} pred={predicted:10s} "
                     f"HR={observed_hr:.3f} p={observed_p:.4f} {sig:3s} "
                     f"dir={observed_dir:10s} match={match}")

    df = pd.DataFrame(rows)

    # Accuracy
    sig_results = df[df["p"] < 0.05]
    correct = sig_results[sig_results["match"] == "YES"]
    accuracy_005 = f"{len(correct)}/{len(sig_results)}" if len(sig_results) > 0 else "0/0"

    sig_010 = df[df["p"] < 0.10]
    correct_010 = sig_010[sig_010["match"].isin(["YES", "trend"])]
    accuracy_010 = f"{len(correct_010)}/{len(sig_010)}" if len(sig_010) > 0 else "0/0"

    logger.info(f"\n  Accuracy at p<0.05: {accuracy_005}")
    logger.info(f"  Accuracy at p<0.10: {accuracy_010}")

    return df, accuracy_005, accuracy_010


# ── LOOCV ────────────────────────────────────────────────────────────────────

def loocv_sel_pli(scorecard_df):
    """Leave-One-Cancer-Out CV of Sel+pLI threshold."""
    logger.info("\nLOOCV of Sel+pLI threshold...")
    sig = scorecard_df[scorecard_df["p"] < 0.05].copy()

    if len(sig) < 3:
        logger.warning("Too few significant results for LOOCV")
        return {}

    results = []
    for idx, row in sig.iterrows():
        left_out = row["cancer"]
        remaining = sig.drop(idx)

        # Find optimal threshold on remaining
        best_acc = 0
        best_thresh = 1.0
        for thresh in np.arange(0.5, 1.5, 0.05):
            preds = remaining["sel_pli"].apply(lambda x: "protective" if x > thresh else "harmful")
            acc = (preds == remaining["observed_dir"]).mean()
            if acc >= best_acc:
                best_acc = acc
                best_thresh = thresh

        # Predict left-out
        pred = "protective" if row["sel_pli"] > best_thresh else "harmful"
        correct = pred == row["observed_dir"]
        results.append({
            "left_out": left_out, "threshold": round(best_thresh, 2),
            "predicted": pred, "observed": row["observed_dir"],
            "correct": correct,
        })
        logger.info(f"  LOO {left_out}: thresh={best_thresh:.2f} pred={pred} obs={row['observed_dir']} {'✓' if correct else '✗'}")

    loocv_acc = sum(r["correct"] for r in results) / len(results) if results else 0
    logger.info(f"  LOOCV accuracy: {loocv_acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
    return {"results": results, "accuracy": round(loocv_acc, 4)}


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(scorecard_df, accuracy_005, accuracy_010, loocv_results, output_dir):
    """Publication-ready figure."""
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Panel A: Sel+pLI vs HR scatter ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    df = scorecard_df.copy()

    for _, row in df.iterrows():
        p = row["p"]
        if p < 0.05:
            color = "#2E7D32" if row["hr"] < 1 else "#C62828"
            size = 120
            alpha = 1.0
        elif p < 0.10:
            color = "#66BB6A" if row["hr"] < 1 else "#EF5350"
            size = 80
            alpha = 0.8
        else:
            color = "#9E9E9E"
            size = 50
            alpha = 0.5

        ax_a.scatter(row["sel_pli"], row["hr"], s=size, color=color, alpha=alpha, zorder=3, edgecolors="white", linewidth=0.5)
        ax_a.annotate(row["cancer"], (row["sel_pli"], row["hr"]),
                      fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                      textcoords="offset points")

    ax_a.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
    ax_a.axvline(1.0, color="#E91E63", ls="--", lw=1.5, alpha=0.7, label="Sel+pLI = 1.0")
    ax_a.set_xlabel("Targetability Index (Selectivity + mean_pLI)", fontsize=11)
    ax_a.set_ylabel("Hazard Ratio (NV-Score → OS)", fontsize=11)
    ax_a.set_title(f"Dependency Architecture Principle — 20 Cancer Types\n"
                    f"Accuracy: {accuracy_005} at p<0.05, {accuracy_010} at p<0.10",
                    fontweight="bold", fontsize=11)
    ax_a.legend(fontsize=9)

    # ── Panel B: Forest plot (significant results) ───────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    sig = df[df["p"] < 0.10].sort_values("hr")

    for i, (_, row) in enumerate(sig.iterrows()):
        color = "#2E7D32" if row["hr"] < 1 else "#C62828"
        marker = "o" if row["p"] < 0.05 else "D"
        match_str = "✓" if row["match"] in ["YES", "trend"] else "✗"

        ax_b.scatter([row["hr"]], [i], color=color, s=100, marker=marker, zorder=3)
        ax_b.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, lw=2)
        ax_b.text(row["ci_high"] + 0.02, i,
                  f"HR={row['hr']:.3f} p={row['p']:.4f} {match_str}",
                  va="center", fontsize=8)

    ax_b.axvline(1.0, color="black", ls="--", lw=1)
    ax_b.set_yticks(range(len(sig)))
    ax_b.set_yticklabels([f"{r['cancer']} ({r['best_stratum']})" for _, r in sig.iterrows()], fontsize=9)
    ax_b.set_xlabel("Hazard Ratio", fontsize=10)
    ax_b.set_title("Significant Results (p<0.10)\nForest Plot", fontweight="bold", fontsize=11)

    fig.suptitle("RevGate Finding 13 — Dependency Architecture Principle at 20 Cancer Types",
                 fontsize=14, fontweight="bold")

    out = output_dir / "20cancer_expansion.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Download data for new cancer types ──────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Data acquisition for 10 new cancer types")
    logger.info("=" * 60)

    for cancer_id in NEW_CANCER_TYPES:
        expr_path = TCGA_DIR / f"{cancer_id}_expression.parquet"
        clin_path = TCGA_DIR / f"{cancer_id}_clinical_raw.tsv"

        try:
            extract_expression_for_cancer(cancer_id, PAN_CANCER_GZ, ENSEMBL_MAP, expr_path)
        except Exception as e:
            logger.error(f"{cancer_id}: expression failed -- {e}")

        try:
            download_clinical_from_gdc(cancer_id, clin_path)
        except Exception as e:
            logger.error(f"{cancer_id}: clinical failed -- {e}")

    # ── Phase 2: NV components for all lineages ──────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: NV-Score components for all lineages")
    logger.info("=" * 60)

    nv_components = compute_nv_components_all_lineages()

    # ── Phase 3: ssGSEA for all 20 cancer types ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: ssGSEA per-patient scoring")
    logger.info("=" * 60)

    ssgsea_scores = {}
    for cancer_id in ALL_CANCER_TYPES:
        lineage = CANCER_TO_LINEAGE[cancer_id]
        expr_path = TCGA_DIR / f"{cancer_id}_expression.parquet"

        if not expr_path.exists():
            logger.warning(f"{cancer_id}: no expression data, skipping ssGSEA")
            continue

        # Check cache
        cache_path = TCGA_DIR / f"{cancer_id}_ssgsea_20cancer.parquet"
        if cache_path.exists():
            logger.info(f"{cancer_id}: ssGSEA cached")
            scores = pd.read_parquet(cache_path)
            ssgsea_scores[cancer_id] = scores.set_index("sample_id")["ssgsea_nv_score"]
            continue

        try:
            scores = run_ssgsea_for_cancer(cancer_id, lineage, nv_components, expr_path)
            if scores is not None:
                ssgsea_scores[cancer_id] = scores
                # Cache
                pd.DataFrame({"sample_id": scores.index, "ssgsea_nv_score": scores.values}).to_parquet(cache_path)
        except Exception as e:
            logger.error(f"{cancer_id}: ssGSEA failed -- {e}")

    # ── Phase 4: Cox PH for all 20 cancer types ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Cox PH analysis")
    logger.info("=" * 60)

    all_cox = {}
    for cancer_id in ALL_CANCER_TYPES:
        if cancer_id not in ssgsea_scores:
            logger.warning(f"{cancer_id}: no ssGSEA scores, skipping Cox")
            continue

        clinical = load_clinical_for_cox(cancer_id)
        if clinical is None:
            logger.warning(f"{cancer_id}: no clinical data, skipping Cox")
            continue

        cox = run_cox_for_cancer(cancer_id, ssgsea_scores[cancer_id], clinical,
                                 cutoffs=[36, 60, 84, 120])
        all_cox[cancer_id] = cox

        # Log best result
        for key in ["early", "advanced", "36mo", "60mo", "84mo", "full"]:
            r = cox.get(key)
            if r and "error" not in r:
                sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
                logger.info(f"  {cancer_id:5s} {key:10s}: HR={r['hr']:.3f} "
                             f"[{r['ci_low']:.3f}-{r['ci_high']:.3f}] p={r['p']:.4f} "
                             f"n={r['n']} ev={r['events']} {sig}")

    # ── Phase 5: Dependency Architecture Principle ───────────────────────────
    scorecard, acc_005, acc_010 = validate_dependency_architecture(all_cox, nv_components)

    # ── Phase 6: LOOCV ───────────────────────────────────────────────────────
    loocv = loocv_sel_pli(scorecard)

    # ── Output ───────────────────────────────────────────────────────────────
    plot_results(scorecard, acc_005, acc_010, loocv, output_dir)

    # Save
    out = {
        "timestamp": datetime.now().isoformat(),
        "finding": "Finding_13",
        "description": "Expansion to 20 cancer types + LOOCV",
        "n_cancer_types": len(ALL_CANCER_TYPES),
        "accuracy_p005": acc_005,
        "accuracy_p010": acc_010,
        "loocv": loocv,
        "scorecard": scorecard.to_dict(orient="records"),
        "nv_components": {k: {kk: vv for kk, vv in v.items() if kk != "top50_genes"} for k, v in nv_components.items()},
        "cox_results": {k: {kk: vv for kk, vv in v.items()} for k, v in all_cox.items()},
    }
    json_path = output_dir / "20cancer_expansion_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"JSON: {json_path}")

    # CSV scorecard
    scorecard.to_csv(output_dir / "20cancer_scorecard.csv", index=False)
    logger.info(f"CSV: {output_dir / '20cancer_scorecard.csv'}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RevGate Finding 13 — 20 Cancer Types")
    print(f"{'='*70}")
    print(f"  Dependency Architecture Principle accuracy:")
    print(f"    At p<0.05: {acc_005}")
    print(f"    At p<0.10: {acc_010}")
    if loocv:
        print(f"    LOOCV: {loocv.get('accuracy', 'N/A'):.1%}")
    print(f"{'='*70}\n")

    logger.info("Finding 13 complete.")


if __name__ == "__main__":
    main()
