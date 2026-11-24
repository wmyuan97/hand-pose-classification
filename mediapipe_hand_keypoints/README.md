# Thumb Position Classification Using 2D Hand Keypoints

## Overview

This project classifies thumb positions in single hand images into three classes:

- **ABD** — Thumb abduction
- **ADD** — Thumb adduction
- **OPP** — Thumb opposition

The workflow uses **2D MediaPipe hand landmarks**, normalized geometric features, Principal Component Analysis (PCA), and a single **RBF Support Vector Machine (SVM)**. It does not use 3D landmarks or depth estimates.

## Dataset

The planned dataset contains **108 images**:

| Factor | Values |
| --- | --- |
| Participants | 01, 02 |
| Trials | 01, 02 |
| Thumb classes | ABD, ADD, OPP |
| Camera angles | F, 45, 90 |
| Occlusion types | NONE, HAND, PAPER |

Each class × angle × occlusion combination has four images (two participants × two trials).

Image filenames follow this pattern:

```text
<trial>_<class>_<angle>_<occlusion>_<participant>.jpg
```

Example: `01_ABD_45_HAND_01.jpeg`.

The ground-truth labels and experimental conditions are stored in `hand_ground_truth.xlsx`.

## Method

1. **Read keypoints.** Load the existing MediaPipe 2D keypoints from `mediapipe_hand_keypoints.xlsx`. MediaPipe detection is not rerun by the SVM script.
2. **Select the target hand.** When more than one hand is detected, select the valid hand whose palm center is closest to the center of the image. Unreliable detections are recorded instead of being forced into a class.
3. **Extract 2D features.** Calculate geometric features describing thumb position, distances to other fingers, relative directions, and hand overlap. Use a local hand coordinate system and normalized distances to reduce differences in image scale and hand size. No 3D coordinates are used.
4. **Preprocess and reduce dimensions.** Apply median imputation (where appropriate), standardization, and `PCA(n_components=0.95, svd_solver="full")`. PCA keeps enough components to explain at least 95% of the training data's standardized feature variance.
5. **Classify.** Train one RBF-SVM for all images (`C=1`, `gamma="scale"` as the initial settings). The model does **not** estimate the camera angle or choose separate classifiers by angle.

All imputation, scaling, and PCA steps are fitted **only on the training portion** of each fold to avoid data leakage.

## Cross-validation

The project uses **4-fold cross-validation**, jointly stratified by:

```text
final_class × camera_angle × occlusion_type
```

For each of the 27 experimental combinations, each fold uses **three images for training and one for testing**. With all 108 images included, each fold contains 81 training images and 27 test images. Every image appears in a test fold exactly once.

The true camera angle and occlusion labels are used **only for fold assignment and result analysis**, not as input features for the SVM.

**Reported classification metrics use held-out test predictions, not training predictions.** Images that cannot be classified are counted separately.

## Evaluation

The results include:

- **Overall:** accuracy, macro-F1, per-class precision/recall/F1, and a 3 × 3 confusion matrix for ABD, ADD, and OPP.
- **By camera angle:** separate results for F, 45, and 90, using the true angle only to group test predictions.
- **By occlusion:** separate results for NONE, HAND, and PAPER.
- **By class:** performance and misclassification patterns for ABD, ADD, and OPP.
- **Coverage:** number and percentage of images that could not be classified.

All subgroup results are computed from the **same combined 4-fold test predictions**. The models are not retrained for each subgroup.

## Project files

```text
MV_gloria_data/
├── data_*/                       # Source images
├── hand_ground_truth.xlsx        # Ground-truth labels and conditions
├── mediapipe_hand_keypoints/
│   └── mediapipe_hand_keypoints.xlsx
├── run_mediapipe_rbf_svm.py       # Feature extraction and SVM evaluation
└── SVM_results/
    ├── thumb_features.xlsx        # Extracted 2D features
    ├── svm_predictions.xlsx       # Held-out predictions for each image
    ├── overall_metrics.xlsx       # Overall and per-class results
    ├── group_metrics.xlsx         # Angle and occlusion results
    ├── fold_details.xlsx          # Train/test assignments by fold
    ├── pca_summary.xlsx           # PCA information (if enabled)
    ├── confusion_matrices/        # Confusion matrix images
    └── validation_log.txt         # Data checks and processing log
```

## Run

Install the required packages if they are not already available:

```bash
pip install numpy pandas scikit-learn opencv-python Pillow matplotlib openpyxl
```

Check the input and output paths in the script's configuration section, then run:

```bash
python run_mediapipe_rbf_svm.py
```

## Limitations

- This is a small dataset with only two participants. The 4-fold results do not establish performance on unseen participants because images from the same person can appear in both training and test folds.
- A 2D image may not contain enough information to distinguish some thumb positions, especially at side views or under heavy occlusion.
- Distance normalization reduces scale differences but does not remove all anatomical differences between people.
- Keeping 95% of PCA variance does not guarantee keeping 95% of the information needed for classification.

