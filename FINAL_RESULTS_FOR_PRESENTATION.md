# Traditional hand-pose classification: final results

## Experimental setting

- 54 images from one participant: 27 left-hand and 27 right-hand images.
- Three balanced classes: abduction, adduction, and opposition (18 images each).
- Three views: front, 45 degrees, and 90 degrees (18 images each).
- Three occlusion conditions: none, paper, and assisting hand (18 images each).
- Main pipeline uses the original fixed segmentation parameters.

## Main results

| Method | Stratified 6-fold (n=54) | None→paper (n=18) | None→hand (n=18) | Holdout front (n=18) | Holdout 45° (n=18) | Holdout 90° (n=18) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline SVM | 42.59% | **66.67%** | 27.78% | 50.00% | **61.11%** | 33.33% |
| Geometry SVM | 44.44% | 55.56% | 33.33% | 50.00% | 44.44% | 22.22% |
| Random Forest | 48.15% | **66.67%** | **50.00%** | **55.56%** | **61.11%** | **44.44%** |
| Gradient Boosting | **51.85%** | 33.33% | 33.33% | 33.33% | 33.33% | 33.33% |
| HOG SVM | 38.89% | 55.56% | 22.22% | 44.44% | 44.44% | 27.78% |

## Key findings

1. Gradient Boosting achieved the highest stratified 6-fold result: **51.85%**.
2. Random Forest was the most robust method across controlled holdouts. It
   achieved **50.00%** under hand occlusion and **44.44%** on the unseen 90° view.
3. The 90° view and assisting-hand occlusion were the most difficult conditions.
4. Adding HOG did not improve this small dataset; its 6-fold accuracy was 38.89%.
5. The random-chance baseline for three balanced classes is 33.33%.

## Exploratory parameter-sensitivity result

This result is supplementary and must not replace the main table. The parameters
were selected and measured on the same 54 images, so the result is optimistic.

| Segmentation configuration | Repeated 6-fold SVM mean | Fixed 6-fold SVM | Random Forest mean | Gradient Boosting mean |
|---|---:|---:|---:|---:|
| Original parameters | 39.63% | 42.59% | 42.59% | **49.63%** |
| SVM-selected parameters | **50.74%** | **48.15%** | **45.19%** | 46.67% |

The selected configuration improved the SVM objective but reduced Gradient
Boosting accuracy and reduced the none-to-paper result from 66.67% to 50.00%.
Therefore, the original parameters remain the primary reported configuration.

## Final full-pipeline parameter decision

After running both parameter sets through all five final classifiers and all six
evaluation protocols, the original parameters won 14 of 30 matched metrics, the
tuned parameters won 5, and 11 were tied. The mean over the 30 matched metrics
was 43.64% for the original parameters and 40.06% for the tuned parameters.
Therefore, the final pipeline retains the original segmentation parameters.

## Recommended presentation conclusion

The traditional silhouette pipeline performed above the 33.33% chance level but
remained limited for fine-grained thumb-pose recognition. Gradient Boosting gave
the best random cross-validation accuracy, while Random Forest generalized more
consistently across view and occlusion changes. Parameter sensitivity testing
showed that most original segmentation values lie in stable regions; optimizing
them for one classifier did not consistently improve other evaluation protocols.
