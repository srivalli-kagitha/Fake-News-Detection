# Claim verification evaluation

Decision threshold: 0.5. Similarity-baseline threshold: 0.45.

## FEVER (dev, gold evidence)

| Method | Accuracy | Macro-F1 | F1 Support | F1 Contradiction | F1 Unknown | Claims |
|---|---|---|---|---|---|---|
| Similarity only (baseline) | 0.4000 | 0.2915 | 0.5390 | 0.0000 | 0.3356 | 900 |
| TruthLens: contradiction-aware NLI | 0.8456 | 0.8471 | 0.8954 | 0.8547 | 0.7911 | 900 |

Best threshold by macro-F1: 0.35 (macro-F1 0.8592).

## Climate-FEVER (out-of-domain)

| Method | Accuracy | Macro-F1 | F1 Support | F1 Contradiction | F1 Unknown | Claims |
|---|---|---|---|---|---|---|
| Similarity only (baseline) | 0.3400 | 0.1806 | 0.5025 | 0.0000 | 0.0392 | 450 |
| TruthLens: contradiction-aware NLI | 0.5089 | 0.4668 | 0.2556 | 0.6630 | 0.4820 | 450 |

Best threshold by macro-F1: 0.5 (macro-F1 0.4668).

## IndicXNLI Telugu (3-class NLI)

| NLI model | Accuracy | Macro-F1 |
|---|---|---|
| mDeBERTa-v3-base, multilingual (used for Telugu) | 0.7420 | 0.7446 |
| DeBERTa-v3-large, English only | 0.4000 | 0.3960 |

## Curated set, full pipeline (retrieval + NLI)

| Method | Accuracy | Macro-F1 | F1 Support | F1 Contradiction | F1 Unknown | Claims |
|---|---|---|---|---|---|---|
| Similarity only (baseline) | 0.5250 | 0.4869 | 0.6275 | 0.0000 | 0.8333 | 40 |
| TruthLens (all) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40 |
| TruthLens (English) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 30 |
| TruthLens (Telugu) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 |
