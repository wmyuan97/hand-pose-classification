# 54-image segmentation parameter tuning

## Question

Determine whether the fixed segmentation values are close to a good operating
region, and search for a better configuration without claiming a global optimum.

## Evaluation design

- Dataset: the original 54 images from one participant.
- Features: the 13 normalized contour and Hu-moment features used by the baseline
  SVM.
- Main tuning score: mean accuracy from repeated stratified 6-fold validation
  (5 repeats, 30 test folds) with fixed random seeds.
- Search: three coordinate-search passes. One parameter is varied at a time and
  the selected value is carried into the next comparison.
- Important limitation: there are no manually annotated ground-truth hand masks.
  Classification accuracy is therefore only a proxy for segmentation quality.
  The same 54 images are used to choose the parameters and to calculate the
  tuning score, so the best score is selection-biased.

## Original versus SVM-selected configuration

| Parameter | Original | Selected | Interpretation |
|---|---:|---:|---|
| Target height | 1280 | 1280 | Original value retained |
| Y lower bound | 0 | 0 | 0, 20, and 40 tied in the final pass |
| Y upper bound | 255 | 210 | 210, 220, 240, and 255 tied in the final pass |
| Cr lower bound | 133 | 141 | Material change; 141 was clearly better for the tuned SVM |
| Cr upper bound | 173 | 169 | Only a 0.37 percentage-point difference |
| Cb lower bound | 77 | 77 | Every tested value from 65 to 89 tied |
| Cb upper bound | 127 | 127 | Every tested value from 119 to 135 tied |
| HSV S minimum | 50 | 50 | Original value retained and better than 65 or 80 |
| Opening kernel | 5 | 5 | Original value retained |
| Closing kernel | 11 | 17 | Material change; 15-17 formed the best region |
| Minimum contour area | 1000 | 1000 | All tested values from 250 to 4000 tied in the final pass |
| Maximum component gap | 50 | 0 | Small gain only; several nonzero values were close |

## Main repeated-CV result

| Configuration | Repeated 6-fold SVM mean | Fold standard deviation | Median selected-mask area |
|---|---:|---:|---:|
| Original | 39.63% | 13.27% | 18.31% of image |
| SVM-selected | 50.74% | 15.63% | 17.96% of image |

The selected configuration improved the tuning objective by 11.11 percentage
points, but fold-to-fold variation remained large.

## Verification outside the tuning objective

| Test | Original | SVM-selected | Change |
|---|---:|---:|---:|
| Fixed stratified 6-fold SVM | 42.59% | 48.15% | +5.56 pp |
| Repeated 6-fold Random Forest | 42.59% | 45.19% | +2.60 pp |
| Repeated 6-fold gradient boosting | 49.63% | 46.67% | -2.96 pp |
| Hold out front view | 50.00% | 50.00% | 0.00 pp |
| Hold out 45-degree view | 61.11% | 61.11% | 0.00 pp |
| Hold out 90-degree view | 33.33% | 33.33% | 0.00 pp |
| Train none, test paper occlusion | 66.67% | 50.00% | -16.67 pp |
| Train none, test hand occlusion | 27.78% | 33.33% | +5.56 pp |

The selected values are therefore not universally better. They improve the
specific SVM cross-validation objective, slightly improve Random Forest, reduce
gradient-boosting accuracy, and substantially reduce the paper-occlusion result.

## Conclusion

Most original parameters are already in a stable, good-performing region:
target height, Y bounds, Cr upper bound, both Cb bounds, S threshold, opening
kernel, minimum area, and component gap can all be defended without claiming
that their exact numeric values are unique optima. The two values most worth
further investigation are the Cr lower bound (133 versus a local optimum near
141) and closing-kernel size (11 versus a local optimum around 15-17).

The original configuration should remain the primary baseline unless a separate
validation set, a second participant, or ground-truth masks confirm the changes.
For a rigorous segmentation claim, annotate masks and optimize IoU or Dice on a
training/validation split. For a rigorous classification claim, tune inside
nested cross-validation or on a validation set, then report a test set exactly
once.
