# NV-Score Weight Derivation via Principal Component Analysis

**Проект:** RevGate -- Computational Validation of the DNV-TC Hypothesis
**Институция:** Helena Bioinformatics
**Дата:** 2026-03-09
**Данни:** DepMap 24Q4, gnomAD v4, STRING v12.0

---

## 1. Контекст

NV-Score (Network Vulnerability Score) класифицира туморни типове по три категории:
- NV-A -- висока концентрация на зависимост, таргетна монотерапия
- NV-B -- умерена концентрация, комбинирана таргетна терапия
- NV-C -- разпределена зависимост, химиотерапия / имунотерапия

## 2. Проблем с произволните тегла

Първоначалните тегла (0.40, 0.30, 0.20, 0.10) са circular -- нагласяват се
докато класификацията изглежда правилна на 6 cancer типа.

## 3. Подход: Unsupervised PCA

PC1 на PCA матрица (25 lineages x 4 компонента) дава оптималните тегла
за максимално разграничаване между cancer типовете без supervised signal.

### Данни
- DepMap CRISPR 24Q4: 1178 cell lines x 17916 genes
- gnomAD v4: pLI за 18111 гена
- STRING v12.0: PPI мрежа, min_score=700, 15956 nodes

### PCA резултати

| PC | Explained Variance |
|----|--------------------|
| PC1 | 34.8% |
| PC2 | 31.3% |
| PC3 | 19.4% |
| PC4 | 14.4% |

PC1 loadings (normalized absolute values):

| Компонент | Loading | Normalized weight |
|-----------|---------|-------------------|
| gini | 0.5023 | 0.260 |
| selectivity | 0.3881 | 0.201 |
| mean_pLI | 0.6876 | 0.356 |
| mean_centrality | 0.3524 | 0.183 |

## 4. Сравнение: ръчни vs data-driven тегла

| Компонент | Ръчни | Data-driven | Разлика |
|-----------|-------|-------------|---------|
| gini | 0.400 | 0.260 | -0.140 |
| selectivity | 0.300 | 0.201 | -0.099 |
| mean_pLI | 0.200 | 0.356 | +0.156 |
| mean_centrality | 0.100 | 0.183 | +0.083 |

## 5. Финална формула

NV-Score = 0.26*Gini + 0.20*Selectivity + 0.36*mean_pLI + 0.18*mean_Centrality

Прагове (рекалибрирани на 25 lineages):
- NV-A: >= 0.53
- NV-C: <  0.41
- NV-B: 0.41 -- 0.53

Биологични anchor points:
- NV-A: Skin/BRAF (0.588), Kidney/HNF1B (0.558), Myeloid/MYB-CBFB (0.538)
- NV-B: Pancreas/KRAS (0.480), Lymphoid/IRF4 (0.481)
- NV-C: Lung (0.387), CNS-Brain (0.361)

## 6. Ограничения

- 20 lineages е малка извадка за PCA
- PC1 обяснява само 34.8% -- значителна вариация остава необяснена
- Thresholds изискват валидация срещу клиничен изход (Kaplan-Meier)
- Gini NaN за 5 lineages поради малък брой cell lines

## 7. Препоръки

1. Survival анализ -- Kaplan-Meier върху TCGA данни
2. Bootstrap confidence intervals за PC1 loadings
3. ROC калибрация на thresholds срещу клиничен изход

---

Helena Bioinformatics, 2026. MIT License.
Biological hypothesis: Toncheva and Sgurev, Bulgarian Academy of Sciences.
