# Final segmentation-parameter decision

## Decision

Retain the original segmentation parameters for the final reported pipeline.

## Aggregate comparison across the complete final pipeline

The comparison includes five final classifiers and six evaluation protocols,
giving 30 directly matched metrics.

| Summary | Original parameters | Tuned parameters |
|---|---:|---:|
| Mean across all 30 matched metrics | **43.64%** | 40.06% |
| Mean Macro-F1 across all 30 matched metrics | **36.53%** | 32.70% |
| Number of direct wins | **14** | 5 |
| Ties | 11 | 11 |
| Mean stratified 6-fold accuracy across five models | **45.19%** | 41.48% |
| Mean none-to-paper accuracy across five models | **55.56%** | 44.44% |
| Mean none-to-hand accuracy across five models | 33.33% | **35.56%** |
| Mean front-view holdout across five models | **46.67%** | 41.11% |
| Mean 45-degree holdout across five models | **48.89%** | 43.33% |
| Mean 90-degree holdout across five models | 32.22% | **34.44%** |

## Six-fold result by final classifier

| Final classifier | Original accuracy / Macro-F1 | Tuned accuracy / Macro-F1 |
|---|---:|---:|
| Baseline SVM | 42.59% / 41.35% | **48.15% / 46.84%** |
| Geometry SVM | **44.44% / 44.83%** | 44.44% / 43.42% |
| Random Forest | **48.15% / 47.99%** | 38.89% / 37.95% |
| Gradient Boosting | **51.85% / 51.46%** | 44.44% / 44.20% |
| HOG SVM | **38.89% / 33.36%** | 31.48% / 27.62% |

## Random Forest robustness comparison

Random Forest is the most stable original final model, so its controlled
holdouts are especially important for the decision.

| Random Forest protocol | Original parameters | Tuned parameters | Change |
|---|---:|---:|---:|
| Stratified 6-fold | **48.15%** | 38.89% | -9.26 pp |
| None to paper | **66.67%** | 55.56% | -11.11 pp |
| None to hand | **50.00%** | 44.44% | -5.56 pp |
| Holdout front | **55.56%** | 44.44% | -11.11 pp |
| Holdout 45 degrees | **61.11%** | 50.00% | -11.11 pp |
| Holdout 90 degrees | **44.44%** | 38.89% | -5.56 pp |

## Interpretation

The tuned parameters were selected specifically to improve the baseline SVM,
and they did improve that one model's six-fold result. However, they reduced
the final Random Forest, Gradient Boosting, and HOG-SVM results and weakened
paper-occlusion, front-view, and 45-degree generalization. The original
parameters therefore give the stronger complete system and should remain the
reported configuration.
