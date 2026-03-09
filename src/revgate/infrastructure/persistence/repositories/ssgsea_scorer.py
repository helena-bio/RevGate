# ssGSEA per-patient NV-Score scorer.
# Приоритет 1 от handover документа.
#
# Метод:
#   1. За всеки cancer тип: top-50 DepMap диференциални гени (diff_score = cancer_mean - global_mean, nsmallest(50))
#   2. Дефинираме gene set
#   3. ssGSEA (gseapy) върху TCGA expression per patient
#   4. Enrichment score = per-patient NV proxy
#   5. Запазваме резултата като parquet за Cox/KM анализ

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Брой top диференциални гени за gene set
TOP_N_GENES = 50

# Минимален брой гени от gene set намерени в expression матрицата
MIN_GENES_OVERLAP = 10

# DepMap -> TCGA mapping (от depmap_repository.py)
LINEAGE_TO_TCGA: dict[str, str] = {
    "Myeloid": "LAML",
    "Skin": "SKCM",
    "Breast": "BRCA",
    "Kidney": "KIRC",
    "Pancreas": "PAAD",
    "Lung": "LUAD",
}


def _get_top_dependency_genes(
    dep_matrix: pd.DataFrame,
    lineage_cell_lines: list[str],
    top_n: int = TOP_N_GENES,
) -> list[str]:
    """Върни top-N диференциални гени за cancer тип.

    diff_score = cancer_mean - global_mean, взима nsmallest(top_n).
    По-отрицателен = по-селективна зависимост в cancer типа.

    Args:
        dep_matrix:         Хронос матрица (cell_lines x genes).
        lineage_cell_lines: Индекси на cell lines за cancer типа.
        top_n:              Брой top гени.

    Returns:
        Списък от gene symbols.
    """
    # Глобално средно за всички cell lines
    global_mean = dep_matrix.mean(axis=0)

    # Средно само за cancer-специфичните cell lines
    cancer_subset = dep_matrix.loc[
        dep_matrix.index.intersection(lineage_cell_lines)
    ]

    if cancer_subset.empty:
        logger.warning("Няма cell lines за lineage — връщам празен списък")
        return []

    cancer_mean = cancer_subset.mean(axis=0)

    # diff_score: колкото по-отрицателен, толкова по-селективен за cancer типа
    diff_score = cancer_mean - global_mean

    # nsmallest(top_n) — най-отрицателни diff scores
    top_genes = diff_score.nsmallest(top_n).index.tolist()

    return top_genes


def run_ssgsea_scoring(
    dep_matrix_path: str | Path,
    model_csv_path: str | Path,
    expression_dir: str | Path,
    output_path: str | Path,
    cancer_ids: list[str] | None = None,
    top_n: int = TOP_N_GENES,
) -> pd.DataFrame:
    """Изчисли ssGSEA per-patient NV-proxy score за всички cancer типове.

    Args:
        dep_matrix_path:  Път до CRISPRGeneEffect.csv.
        model_csv_path:   Път до Model.csv (cell line metadata).
        expression_dir:   Директория с {CANCER}_expression.parquet файлове.
        output_path:      Изходен parquet файл.
        cancer_ids:       Списък cancer типове (None = всички от LINEAGE_TO_TCGA).
        top_n:            Брой top гени за gene set.

    Returns:
        DataFrame с колони: patient_id, cancer_id, ssgsea_nv_score.
    """
    import gseapy as gp

    expression_dir = Path(expression_dir)
    output_path = Path(output_path)

    if cancer_ids is None:
        cancer_ids = list(LINEAGE_TO_TCGA.values())

    # --- Зареди DepMap матрицата веднъж ---
    logger.info("Зареждам DepMap CRISPR матрица...")
    dep_df = pd.read_csv(dep_matrix_path, index_col=0)
    # Почисти Entrez ID от имената на гените: 'MYB (4602)' -> 'MYB'
    dep_df.columns = [col.split(" (")[0] for col in dep_df.columns]

    # --- Зареди Model.csv за cell line -> lineage mapping ---
    logger.info("Зареждам DepMap Model metadata...")
    model_df = pd.read_csv(model_csv_path)
    # Обърни mapping: TCGA -> DepMap lineage
    tcga_to_lineage = {v: k for k, v in LINEAGE_TO_TCGA.items()}

    all_results: list[pd.DataFrame] = []

    for cancer_id in cancer_ids:
        logger.info(f"Обработвам {cancer_id}...")

        # --- Намери cell lines за cancer типа ---
        lineage = tcga_to_lineage.get(cancer_id)
        if lineage is None:
            logger.warning(f"Няма lineage mapping за {cancer_id} — пропускам")
            continue

        lineage_mask = model_df["OncotreeLineage"].str.contains(lineage, na=False)
        lineage_model_ids = model_df.loc[lineage_mask, "ModelID"].tolist()

        # --- Изчисли top-N диференциални гени ---
        top_genes = _get_top_dependency_genes(dep_df, lineage_model_ids, top_n)

        if len(top_genes) < MIN_GENES_OVERLAP:
            logger.warning(
                f"{cancer_id}: само {len(top_genes)} гени — пропускам"
            )
            continue

        logger.info(f"{cancer_id}: gene set = {len(top_genes)} гени, top3={top_genes[:3]}")

        # --- Зареди expression матрица ---
        expr_path = expression_dir / f"{cancer_id}_expression.parquet"
        if not expr_path.exists():
            logger.warning(f"Липсва expression файл: {expr_path} — пропускам")
            continue

        logger.info(f"Зареждам expression матрица за {cancer_id}...")
        expr_df = pd.read_parquet(expr_path)
        # expr_df: rows=patients, cols=genes — transposираме за gseapy (genes x samples)
        expr_T = expr_df.T

        # --- Намери overlap между gene set и expression гени ---
        available_genes = set(expr_T.index)
        gene_set_filtered = [g for g in top_genes if g in available_genes]

        if len(gene_set_filtered) < MIN_GENES_OVERLAP:
            logger.warning(
                f"{cancer_id}: само {len(gene_set_filtered)} гени в expression — пропускам"
            )
            continue

        logger.info(
            f"{cancer_id}: overlap {len(gene_set_filtered)}/{len(top_genes)} гени"
        )

        # --- Пусни ssGSEA ---
        # gene_sets: dict с един gene set на cancer типа
        gene_sets = {f"NV_proxy_{cancer_id}": gene_set_filtered}

        try:
            ssgsea_result = gp.ssgsea(
                data=expr_T,
                gene_sets=gene_sets,
                outdir=None,       # не записва файлове
                no_plot=True,      # без фигури
                processes=1,
                verbose=False,
            )
            # Резултатът е DataFrame с enrichment scores: rows=gene_sets, cols=samples
            scores_df = ssgsea_result.res2d

            # Pivot: искаме patient_id -> score
            # gseapy res2d формат: Term, Name, ES, NES, ...
            cancer_scores = (
                scores_df[["Name", "NES"]]
                .rename(columns={"Name": "patient_id", "NES": "ssgsea_nv_score"})
                .copy()
            )
            cancer_scores["cancer_id"] = cancer_id
            cancer_scores["n_genes_in_set"] = len(gene_set_filtered)

            all_results.append(cancer_scores)
            logger.info(
                f"{cancer_id}: {len(cancer_scores)} пациента scored, "
                f"mean NES={cancer_scores['ssgsea_nv_score'].mean():.3f}"
            )

        except Exception as exc:
            logger.error(f"{cancer_id}: ssGSEA грешка — {exc}")
            continue

    if not all_results:
        raise RuntimeError("Нито един cancer тип не беше успешно обработен")

    # --- Обедини резултатите ---
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[["patient_id", "cancer_id", "ssgsea_nv_score", "n_genes_in_set"]]

    # --- Запази като parquet ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    logger.info(
        f"Записано: {output_path} — {len(combined)} пациента, "
        f"{combined['cancer_id'].nunique()} cancer типа"
    )

    return combined
