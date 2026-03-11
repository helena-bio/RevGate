# Changelog

All notable changes to RevGate are documented here.

## [0.3.0] - 2026-03-10 (Finding 10: Expanded Validation)

### Added
- Finding 10: Expanded validation — 4 new cancer types (COAD, OV, HNSC, UCEC) + KIRC restricted follow-up
- KIRC 84mo FDR=0.042 (survives BH correction at 6 independent tests)
- COAD Advanced HR=0.640 (exploratory, FDR=0.073)
- OV restricted follow-up — flat effect across all cutoffs
- Pre-registered predictions for 4 new cancer types

## [0.2.0] - 2026-03-09 (Findings 04–09)

### Added
- Finding 04: NV-Score weight stability — equal weights (permutation p=0.787)
- Finding 05: BRCA mediation — stage is a suppressor (direct HR=0.904, p=0.036)
- Finding 06: METABRIC validation — partial replication at full follow-up
- Finding 07: LUAD subtype decomposition — HR>1 universal across all subtypes
- Finding 08: Dependency Architecture Principle — Sel+pLI predicts HR direction (3/3)
- Finding 09: METABRIC time-dependent replication — HR=0.810 at 36mo (BREAKTHROUGH)
- Sensitivity analysis: jackknife, threshold sweep, weighting comparison

### Changed
- All thresholds recalibrated to equal weights (NV-A ≥ 0.50, NV-C < 0.35)

## [0.1.0] - 2026-03-08 (Findings 01–03)

### Added
- Initial DDD architecture (domain, application, infrastructure, presentation)
- Finding 01: Per-patient ssGSEA NV-Score, stage-stratified Cox, multivariate Cox
- Finding 02: METABRIC negative replication (resolved by Finding 09)
- Finding 03: Tier 0 biological invariant checker (17/17 PASS)
- NV-Score weight derivation via PCA on 25 DepMap 24Q4 lineages
- Docker multi-stage build + docker-compose for development
- CLI interface (validate, classify, download, status)
