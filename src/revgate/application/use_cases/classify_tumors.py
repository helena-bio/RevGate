# ClassifyTumorUseCase -- orchestrates 4-axis classification for all cancer types.
# Application layer -- coordinates domain services and repositories.
# No business logic here -- only orchestration.

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from revgate.application.dto.classification_dto import (
    AxisResultDTO,
    ClassificationRequestDTO,
    ClassificationResultDTO,
)
from revgate.domain.entities.cancer_type import CancerType
from revgate.domain.entities.gene import Gene
from revgate.domain.entities.tumor_classification import TumorClassification
from revgate.domain.repositories.clinical_repository import ClinicalRepository
from revgate.domain.repositories.constraint_repository import ConstraintRepository
from revgate.domain.repositories.dependency_repository import DependencyRepository
from revgate.domain.repositories.network_repository import NetworkRepository
from revgate.domain.services.cascade_analyzer import CascadeAnalyzer
from revgate.domain.services.dependency_profiler import DependencyProfiler
from revgate.domain.services.developmental_classifier import DevelopmentalClassifier
from revgate.domain.services.metastatic_classifier import MetastaticClassifier
from revgate.domain.services.vulnerability_classifier import VulnerabilityClassifier
from revgate.domain.value_objects.dependency_score import DependencyScore
from revgate.domain.value_objects.rdp_score import RDPClass, RDPScore


# RDP gene signatures -- loaded once at module level from config
_RDP_SIGNATURES: dict[str, list[str]] | None = None


def _load_rdp_signatures() -> dict[str, list[str]]:
    """Load RDP gene signatures from config/gene_signatures.yaml."""
    global _RDP_SIGNATURES
    if _RDP_SIGNATURES is not None:
        return _RDP_SIGNATURES

    import yaml

    # Locate config relative to project root
    config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "gene_signatures.yaml"
    if not config_path.exists():
        # Fallback: try relative to working directory
        config_path = Path("config/gene_signatures.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"gene_signatures.yaml not found at {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    signatures: dict[str, list[str]] = {}
    for program_key, program_data in raw.items():
        genes = program_data.get("genes", []) if isinstance(program_data, dict) else []
        signatures[program_key] = [str(g) for g in genes]

    _RDP_SIGNATURES = signatures
    return signatures


class ClassifyTumorUseCase:
    """Orchestrates 4-axis DNV-TC classification for a list of cancer types.

    Execution order per cancer type:
        1. Load cell lines + dependency scores from DepMap
        2. Profile dependencies (DependencyProfiler)
        3. Classify NV-Class (VulnerabilityClassifier)
        4. Load expression matrix from TCGA (optional -- graceful degradation)
        5. Classify RDP-Class (ssGSEA via gseapy)
        6. Classify CEP-Class (Personalized PageRank via networkx)
        7. Classify MP-Class (MetastaticClassifier)
        8. Assemble TumorClassification entity
        9. Return AxisResultDTO

    If TCGA data is not available, RDP/CEP/MP are set to UNKNOWN
    and NV classification proceeds normally.
    """

    def __init__(
        self,
        dependency_repo: DependencyRepository,
        clinical_repo: ClinicalRepository,
        network_repo: NetworkRepository,
        constraint_repo: ConstraintRepository,
    ) -> None:
        self._dependency_repo = dependency_repo
        self._clinical_repo = clinical_repo
        self._network_repo = network_repo
        self._constraint_repo = constraint_repo

        # Domain services -- stateless, instantiated once
        self._profiler = DependencyProfiler()
        self._nv_classifier = VulnerabilityClassifier()
        self._rdp_classifier = DevelopmentalClassifier()
        self._cascade_analyzer = CascadeAnalyzer()
        self._mp_classifier = MetastaticClassifier()

    def execute(self, request: ClassificationRequestDTO) -> ClassificationResultDTO:
        """Run 4-axis classification for all requested cancer types.

        Args:
            request: ClassificationRequestDTO with cancer_ids and top_n.

        Returns:
            ClassificationResultDTO with per-cancer results and any errors.
        """
        started_at = time.monotonic()
        axes: list[AxisResultDTO] = []
        errors: dict[str, str] = {}

        # Load shared data once -- used across all cancer types
        dep_matrix = self._dependency_repo.get_dependency_matrix()
        pli_scores = self._constraint_repo.get_pli_scores()
        ppi_graph = self._network_repo.get_ppi_graph()

        # Global mean scores -- used for differential (cancer-specific) scoring
        global_mean = self._dependency_repo.get_global_mean_scores()

        # Pre-compute per-cancer model IDs -- needed for selectivity
        cancer_model_ids: dict[str, set[str]] = {}
        for cid in request.cancer_ids:
            try:
                cls = self._dependency_repo.get_cell_lines_for_cancer(cid)
                cancer_model_ids[cid] = {cl.model_id for cl in cls}
            except Exception:
                cancer_model_ids[cid] = set()

        # Max degree for centrality normalization
        max_degree = max(dict(ppi_graph.degree()).values(), default=1)

        for cancer_id in request.cancer_ids:
            try:
                result = self._classify_single(
                    cancer_id=cancer_id,
                    dep_matrix=dep_matrix,
                    pli_scores=pli_scores,
                    ppi_graph=ppi_graph,
                    max_degree=max_degree,
                    top_n=request.top_n,
                    global_mean=global_mean,
                    cancer_model_ids=cancer_model_ids,
                )
                axes.append(result)
            except Exception as exc:
                errors[cancer_id] = str(exc)

        return ClassificationResultDTO(
            axes=axes,
            errors=errors,
            duration_sec=time.monotonic() - started_at,
        )

    def _classify_single(
        self,
        cancer_id: str,
        dep_matrix: pd.DataFrame,
        pli_scores: dict[str, float],
        ppi_graph: Any,
        max_degree: int,
        top_n: int,
        global_mean: "pd.Series | None" = None,
        cancer_model_ids: "dict[str, set[str]] | None" = None,
    ) -> AxisResultDTO:
        """Run full 4-axis classification for a single cancer type."""

        # Build Gene entities from dependency matrix for this cancer type
        cell_lines = self._dependency_repo.get_cell_lines_for_cancer(cancer_id)
        model_ids = {cl.model_id for cl in cell_lines}

        # Filter matrix to this cancer's cell lines
        cancer_rows = dep_matrix[dep_matrix.index.isin(model_ids)]

        genes: list[Gene] = []
        for symbol in cancer_rows.columns:
            mean_score = float(cancer_rows[symbol].mean())
            # Differential score: cancer-specific signal vs pan-cancer background
            # More negative = more essential in THIS cancer vs all others
            if global_mean is not None and symbol in global_mean.index:
                diff_score = mean_score - float(global_mean[symbol])
            else:
                diff_score = mean_score
            gene = Gene(
                gene_id=symbol,
                symbol=symbol,
                pli_score=min(max(pli_scores.get(symbol, 0.0), 0.0), 1.0),
                network_degree=ppi_graph.degree(symbol) if symbol in ppi_graph else 0,
                dependency_scores={cancer_id: DependencyScore(diff_score)},
            )
            genes.append(gene)

        # Step 1: Dependency profile + Gini
        profile = self._profiler.profile(cancer_id, genes, top_n)

        # Step 2: NV-Score -- does not require TCGA
        # Compute selectivity: how specific is each top gene to THIS cancer
        selectivity_scores = self._compute_selectivity(
            cancer_id=cancer_id,
            top_genes=profile.top_genes,
            dep_matrix=dep_matrix,
            cancer_model_ids=cancer_model_ids or {},
        )
        nv_score = self._nv_classifier.classify(
            profile, pli_scores, max_degree, selectivity_scores
        )

        # Step 3-5: TCGA-dependent axes -- graceful degradation if data missing
        rdp_class = "UNKNOWN"
        cep_class = "UNKNOWN"
        mp_class = "UNKNOWN"
        cep_value = 0.0
        invasion_zscore = 0.0

        try:
            expression_matrix = self._clinical_repo.get_expression_matrix(cancer_id)

            # Step 3: RDP-Class via ssGSEA
            enrichment_results = self._run_ssgsea(expression_matrix)
            rdp_scores = self._rdp_classifier.classify(enrichment_results)
            rdp_class = self._rdp_classifier.get_primary_class(rdp_scores).value

            # Step 4: CEP-Class via Personalized PageRank
            top_gene_symbols = [g.symbol for g in profile.top_genes]
            pagerank_vectors = self._run_pagerank(ppi_graph, top_gene_symbols)
            cep_score_obj = self._cascade_analyzer.analyze(
                profile.top_genes,
                pagerank_vectors,
                cohort_percentile=50.0,
            )
            cep_class = cep_score_obj.cep_class.value
            cep_value = cep_score_obj.value

            # Step 5: MP-Class
            rdp_emt = next(
                (s.nes for s in rdp_scores if s.program == RDPClass.RDP_III), 0.0
            )
            expression_zscores = self._compute_zscores(expression_matrix)
            mp_score_obj = self._mp_classifier.classify(rdp_emt, expression_zscores)
            mp_class = mp_score_obj.mp_class.value
            invasion_zscore = mp_score_obj.invasion_zscore

        except Exception:
            # TCGA data not available -- NV-only mode
            pass

        return AxisResultDTO(
            cancer_id=cancer_id,
            nv_class=nv_score.nv_class.value,
            rdp_class=rdp_class,
            cep_class=cep_class,
            mp_class=mp_class,
            nv_composite_score=nv_score.composite,
            gini=nv_score.gini,
            mean_pli=nv_score.mean_constraint,
            mean_centrality=nv_score.mean_centrality,
            cep_value=cep_value,
            invasion_zscore=invasion_zscore,
        )

    def _compute_selectivity(
        self,
        cancer_id: str,
        top_genes: list,
        dep_matrix: pd.DataFrame,
        cancer_model_ids: dict,
    ) -> dict[str, float]:
        """Compute selectivity score for each top gene.

        Selectivity = essentiality_in_cancer - mean_essentiality_in_other_cancers
        Clamped to [0, 1].

        Args:
            cancer_id:        Target cancer type.
            top_genes:        Top dependency gene entities.
            dep_matrix:       Full DepMap dependency matrix.
            cancer_model_ids: Dict cancer_id -> set of model IDs.

        Returns:
            Dict gene_symbol -> selectivity score in [0, 1].
        """
        if cancer_id not in cancer_model_ids:
            return {}

        target_ids = cancer_model_ids[cancer_id]
        target_rows = dep_matrix[dep_matrix.index.isin(target_ids)]

        other_cancer_ids = [c for c in cancer_model_ids if c != cancer_id]

        result: dict[str, float] = {}
        for gene in top_genes:
            symbol = gene.symbol
            if symbol not in dep_matrix.columns:
                result[symbol] = 0.0
                continue

            # Essentiality fraction in target cancer
            target_ess = float((target_rows[symbol] < -0.5).mean()) if len(target_rows) > 0 else 0.0

            # Mean essentiality across other cancers
            other_ess_list = []
            for other_id in other_cancer_ids:
                other_ids = cancer_model_ids.get(other_id, set())
                if not other_ids:
                    continue
                other_rows = dep_matrix[dep_matrix.index.isin(other_ids)]
                if len(other_rows) > 0 and symbol in other_rows.columns:
                    other_ess_list.append(float((other_rows[symbol] < -0.5).mean()))

            mean_other_ess = sum(other_ess_list) / len(other_ess_list) if other_ess_list else 0.0

            # Selectivity: how much MORE essential in this cancer vs others
            selectivity = max(0.0, target_ess - mean_other_ess)
            result[symbol] = min(1.0, selectivity)

        return result

    def _run_ssgsea(
        self,
        expression_matrix: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Run ssGSEA enrichment against RDP gene signatures.

        Uses gseapy.ssgsea on the expression matrix.
        Returns enrichment_results dict compatible with DevelopmentalClassifier.

        Args:
            expression_matrix: DataFrame (n_patients x n_genes), log2-normalized.

        Returns:
            Dict mapping RDP program key -> {'nes': float, 'fdr': float}
        """
        import gseapy as gp

        signatures = _load_rdp_signatures()

        # gseapy expects gene sets as {name: [gene_list]}
        gene_sets = {k: v for k, v in signatures.items()}

        # Expression matrix must be genes x samples for ssgsea
        # Input is samples x genes -- transpose
        expr_T = expression_matrix.T

        # Remove duplicate gene indices
        expr_T = expr_T[~expr_T.index.duplicated(keep="first")]

        # Run ssGSEA
        ss = gp.ssgsea(
            data=expr_T,
            gene_sets=gene_sets,
            outdir=None,
            sample_norm_method="rank",
            no_plot=True,
            processes=1,
        )

        results_df = ss.res2d
        enrichment: dict[str, dict[str, float]] = {}

        for _, row in results_df.iterrows():
            term = str(row.get("Term", ""))
            # gseapy NES column may vary by version
            nes = float(row.get("NES", row.get("ES", 0.0)))
            fdr = float(row.get("FDR q-val", row.get("FDR", 1.0)))
            enrichment[term] = {"nes": nes, "fdr": fdr}

        return enrichment

    def _run_pagerank(
        self,
        ppi_graph: Any,
        seed_genes: list[str],
    ) -> list[dict[str, float]]:
        """Run Personalized PageRank seeded at each top dependency gene.

        For each seed gene, runs networkx pagerank with personalization
        vector concentrated on that gene's neighbors.

        Args:
            ppi_graph:   STRING networkx Graph.
            seed_genes:  Top dependency gene symbols.

        Returns:
            List of PageRank probability dicts, one per seed gene.
        """
        import networkx as nx

        vectors: list[dict[str, float]] = []

        for seed in seed_genes:
            if seed not in ppi_graph:
                # Gene not in STRING -- return uniform over all nodes
                n = ppi_graph.number_of_nodes()
                if n > 0:
                    uniform = {node: 1.0 / n for node in ppi_graph.nodes()}
                    vectors.append(uniform)
                continue

            # Personalization: seed gene gets weight 1.0, all others 0.0
            personalization = {node: 0.0 for node in ppi_graph.nodes()}
            personalization[seed] = 1.0

            pr = nx.pagerank(
                ppi_graph,
                alpha=0.85,
                personalization=personalization,
                max_iter=100,
                tol=1.0e-6,
            )
            vectors.append(pr)

        return vectors

    def _compute_zscores(
        self,
        expression_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute mean z-score per gene across all patients.

        Args:
            expression_matrix: DataFrame (n_patients x n_genes).

        Returns:
            Dict gene_symbol -> mean z-score across patients.
        """
        # Z-score each gene across patients
        mean = expression_matrix.mean(axis=0)
        std = expression_matrix.std(axis=0)

        # Avoid division by zero
        std = std.replace(0.0, 1.0)

        zscores = (expression_matrix - mean) / std

        # Mean z-score per gene across all patients
        mean_zscores = zscores.mean(axis=0)

        return mean_zscores.to_dict()
