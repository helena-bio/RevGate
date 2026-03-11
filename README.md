# RevGate

**Empirical validation of the DNV-TC hypothesis — linking cancer gene dependencies to clinical outcomes.**

RevGate is a computational pipeline that tests the Developmental Network Vulnerability-Based Tumor Classification (DNV-TC) hypothesis proposed by Academician Draga Toncheva, DSc and Academician Vassil Sgurev, DSc (Bulgarian Academy of Sciences).

The hypothesis: tumors reactivate embryonic developmental programs, creating network dependencies whose architecture (concentrated vs. distributed) determines therapeutic response.

## Key Results

| Cancer | Stratum | HR | 95% CI | p | Direction |
|--------|---------|-----|---------|------|-----------|
| BRCA | Early (I-II) | 0.803 | [0.681–0.946] | 0.009 | Protective |
| METABRIC | 36 months | 0.810 | [0.740–0.886] | <0.0001 | Protective |
| KIRC | 84mo restricted | 0.876 | [0.794–0.965] | 0.007 | Protective |
| LUAD | Early (I-II) | 1.300 | [1.092–1.547] | 0.003 | Harmful |
| PAAD | All | 1.366 | [1.002–1.862] | 0.049 | Harmful |

**Dependency Architecture Principle:** Sel+pLI > 1.0 predicts protective (HR<1), Sel+pLI < 1.0 predicts harmful (HR>1). Accuracy: **5/5 (100%)** at p<0.05 across 10 cancer types.

## Architecture

RevGate follows Domain-Driven Design (DDD) with four layers:
```
src/revgate/
├── domain/           # Core business logic (entities, value objects, services)
├── application/      # Use case orchestration
├── infrastructure/   # External data access (DepMap, TCGA, STRING, gnomAD)
└── presentation/     # Figures, tables, reports
```

## NV-Score Formula
```
NV-Score = 0.25 × Gini + 0.25 × Selectivity + 0.25 × mean_pLI + 0.25 × mean_Centrality
```

Equal weights justified by permutation test (p=0.787) and bootstrap CV=0.40–0.50.

## Data Sources

| Source | Version | Content |
|--------|---------|---------|
| DepMap CRISPR | 24Q4 | 1,178 cell lines × 17,916 genes |
| TCGA | 2025 | RNA-seq + clinical, 10 cancer types |
| gnomAD | v4 | pLI scores for 18,204 genes |
| STRING | v12.0 | PPI network, 15,956 nodes |
| METABRIC | Nature 2012 | 1,979 patients, external validation |

## Quick Start
```bash
# Clone
git clone https://github.com/Helena-Bioinformatics/revgate.git
cd revgate

# Install
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Download data
revgate download --source all

# Run full validation
revgate validate --cancers LAML,SKCM,BRCA,KIRC,PAAD,LUAD

# Check cache status
revgate status
```

### Docker
```bash
./rebuild.sh
docker exec -it revgate-dev python3 main.py validate
```

## Analysis Scripts

Individual findings are reproduced via standalone scripts in `scripts/`:
```bash
python scripts/run_invariant_checker.py         # Finding 03: Tier 0 gate (17/17 PASS)
python scripts/run_bootstrap_pca_weights.py      # Finding 04: Weight stability
python scripts/run_ssgsea_analysis.py            # Finding 01: Per-patient ssGSEA
python scripts/run_stage_stratified_cox.py       # Finding 01: Stage-stratified Cox
python scripts/run_multivariate_cox.py           # Finding 01: Multivariate Cox
python scripts/run_mediation_brca.py             # Finding 05: BRCA suppression
python scripts/run_metabric_validation.py        # Finding 06: METABRIC (full follow-up)
python scripts/run_metabric_external_validation.py  # Finding 09: METABRIC time-dependent
python scripts/run_luad_subtype_analysis.py      # Finding 07: LUAD subtypes
python scripts/run_dependency_type_classifier.py # Finding 08: Sel+pLI principle
python scripts/run_sensitivity_analysis.py       # Sensitivity & robustness
python scripts/run_expanded_validation.py        # Finding 10: 4 new cancers + KIRC
```

## Project Structure
```
revgate/
├── src/revgate/          # Python package (DDD architecture)
├── scripts/              # Standalone analysis scripts (Findings 01–10)
├── config/               # Pipeline configuration (settings.yaml, gene_signatures.yaml)
├── tests/                # Test suite
├── docs/                 # Documentation
├── notebooks/            # Jupyter exploration
├── main.py               # CLI entry point
├── pyproject.toml        # Package definition
├── Dockerfile            # Multi-stage Docker build
└── docker-compose-development.yaml
```

## Citation

If you use RevGate in your research, please cite:

> Helena Bioinformatics (2026). RevGate: Empirical validation of the DNV-TC hypothesis.
> Biological hypothesis: Academician Draga Toncheva, DSc and Academician Vassil Sgurev, DSc,
> Bulgarian Academy of Sciences.

## License

MIT License. See [LICENSE](LICENSE).

Biological hypothesis (DNV-TC framework): Academician Draga Toncheva, DSc and Academician Vassil Sgurev, DSc, Bulgarian Academy of Sciences.
