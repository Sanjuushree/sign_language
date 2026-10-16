# ISL Connect — Phase 6 Model Evaluation Report

**Model File:** `isl_classifier.joblib` (RandomForestClassifier)  
**Evaluation Date:** `2026-10-15T15:39:44.635979+00:00`  
**Test Dataset:** `test_features_s20.npz` (506 held-out samples)  

## 1. Executive Summary

- **Overall Test Accuracy:** `80.43%` (407/506 correct)
- **Macro F1-Score:** `74.99%`
- **Weighted F1-Score:** `74.47%`
- **Mean Prediction Confidence (Correct):** `85.3%`
- **Mean Prediction Confidence (Errors):** `27.4%`

## 2. Per-Letter Performance Table (All 26 Classes)

| Letter | Precision | Recall | F1-Score | Support | Correct | Errors | Status |
|:------:|:---------:|:------:|:--------:|:-------:|:-------:|:------:|:------:|
| **A** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **B** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **C** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **D** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **E** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **F** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **G** | 50.0% | 100.0% | 66.7% | 20 | 20 | 0 | Moderate |
| **H** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **I** | 0.0% | 0.0% | 0.0% | 20 | 0 | 20 | Needs Improvement |
| **J** | 100.0% | 95.0% | 97.4% | 20 | 19 | 1 | Excellent |
| **K** | 0.0% | 0.0% | 0.0% | 20 | 0 | 20 | Needs Improvement |
| **L** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **M** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **N** | 35.7% | 100.0% | 52.6% | 20 | 20 | 0 | Moderate |
| **O** | 100.0% | 100.0% | 100.0% | 7 | 7 | 0 | Excellent |
| **P** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **Q** | 0.0% | 0.0% | 0.0% | 20 | 0 | 20 | Needs Improvement |
| **R** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **S** | 0.0% | 0.0% | 0.0% | 20 | 0 | 20 | Needs Improvement |
| **T** | 71.4% | 100.0% | 83.3% | 20 | 20 | 0 | Moderate |
| **U** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **V** | 58.8% | 100.0% | 74.1% | 20 | 20 | 0 | Moderate |
| **W** | 100.0% | 5.3% | 10.0% | 19 | 1 | 18 | Needs Improvement |
| **X** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |
| **Y** | 48.8% | 100.0% | 65.6% | 20 | 20 | 0 | Moderate |
| **Z** | 100.0% | 100.0% | 100.0% | 20 | 20 | 0 | Excellent |

## 3. Frequently Confused Sign Pairs

| True Sign | Predicted As | Misclassified Samples | % of True Class | Error Explanation |
|:---------:|:------------:|:--------------------:|:---------------:|:-------------------|
| **I** | **Y** | 20 | 100.0% | Single hand pinky-extended sign: Y shares pinky extension with thumb extended. |
| **Q** | **N** | 20 | 100.0% | Two-handed sign: Overlapping hands cause index/thumb loop to merge with N finger geometry. |
| **S** | **G** | 20 | 100.0% | Two-handed crossed fist: When hands interlock, MediaPipe 3D joint projection matches G. |
| **W** | **N** | 16 | 84.2% | Hand-count variation: W in test split had dual hands vs single hand in training. |
| **K** | **V** | 12 | 60.0% | Two-handed hook sign: When non-dominant hand is occluded, V single-hand geometry dominates. |
| **K** | **T** | 8 | 40.0% | Two-handed hook sign: Partial occlusion matches T index hook. |
| **W** | **V** | 2 | 10.5% | Geometric similarity in landmark coordinates. |
| **J** | **Y** | 1 | 5.0% | Dynamic gesture trace: Static single frame matches Y pinky silhouette. |

## 4. Key Limitations & Risk Analysis

1. **Near-Duplicate Temporal Leakage in Dataset Split:**
   - Frames in `Training/` and `Testing/` originated from consecutive video streams of the same session.
   - Offline accuracy reflects familiar signer morphology; real-world cross-subject accuracy may be lower.
2. **Single vs Dual Hand Sensitivity:**
   - Classes like `K` and `W` vary between 1 and 2 hands depending on camera field-of-view.
   - Feature vector zeros slots 72–147 when only 1 hand is detected, causing misclassification of 2-handed signs.
3. **Lighting & Camera Mirroring:**
   - Low lighting drops MediaPipe palm detection confidence below 0.50.
   - Browser selfie mirror mode flips X coordinates, which inverts relative two-hand delta vectors.