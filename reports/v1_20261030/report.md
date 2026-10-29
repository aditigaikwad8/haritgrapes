# Evaluation report: v1_20261030

## Acceptance targets (v1)

| Check | Target | Result | |
|---|---|---|---|
| Test accuracy | >= 90% | 97.3% | PASS |
| Recall Black_Rot (test) | >= 0.85 | 0.867 | PASS |
| Recall Esca (test) | >= 0.85 | 0.962 | PASS |
| Recall Healthy_Leaves (test) | >= 0.85 | 0.988 | PASS |
| Recall Leaf_Blight (test) | >= 0.85 | 0.826 | FAIL |
| Recall Not_a_Grape_Leaf (test) | >= 0.90 | 1.000 | PASS |
| Field-test accuracy | report honestly, aim >= 75% | 28.1% | BELOW AIM |
| Out-of-scope mildew check | reported | done | PASS |
| Latency (median) | <= 500 ms | 76 ms | PASS |

## test

Images: 441. Accuracy: 97.3%. Macro-F1: 0.918

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Black_Rot | 0.650 | 0.867 | 0.743 | 15 |
| Esca | 1.000 | 0.962 | 0.981 | 106 |
| Healthy_Leaves | 0.982 | 0.988 | 0.985 | 164 |
| Leaf_Blight | 0.950 | 0.826 | 0.884 | 23 |
| Not_a_Grape_Leaf | 0.993 | 1.000 | 0.996 | 133 |

Not_a_Grape_Leaf: recall 100.0%, false-accept rate 0.0% (non-leaf accepted as a grape class), false-reject rate 0.3% (grape/other class predicted as non-leaf).

Predicted class distribution: {'Black_Rot': 20, 'Esca': 102, 'Healthy_Leaves': 165, 'Leaf_Blight': 20, 'Not_a_Grape_Leaf': 134}

Figures: confusion_matrix_test.png, worst_mistakes_test.png, confidence_hist_test.png. Per-image results: predictions_test.csv. Look at the worst mistakes for mislabels and look-alikes.

## field_test

Images: 139. Accuracy: 28.1%. Macro-F1: 0.239

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Black_Rot | 0.375 | 0.100 | 0.158 | 30 |
| Esca | 0.000 | 0.000 | 0.000 | 30 |
| Healthy_Leaves | 0.294 | 0.333 | 0.312 | 30 |
| Leaf_Blight | 1.000 | 0.207 | 0.343 | 29 |
| Not_a_Grape_Leaf | 0.235 | 1.000 | 0.381 | 20 |

Not_a_Grape_Leaf: recall 100.0%, false-accept rate 0.0% (non-leaf accepted as a grape class), false-reject rate 54.6% (grape/other class predicted as non-leaf).

Predicted class distribution: {'Black_Rot': 8, 'Esca': 6, 'Healthy_Leaves': 34, 'Leaf_Blight': 6, 'Not_a_Grape_Leaf': 85}

Figures: confusion_matrix_field_test.png, worst_mistakes_field_test.png, confidence_hist_field_test.png. Per-image results: predictions_field_test.csv. Look at the worst mistakes for mislabels and look-alikes.

## Out-of-scope check (Downy Mildew / Powdery Mildew, never trained on)

Status uses the starting thresholds in src/config.py.

| Group | N | Healthy + confident (dangerous) | Healthy (any) | Uncertain/low-conf | Not a leaf | Forced into each class |
|---|---|---|---|---|---|---|
| ALL | 60 | 0% | 20% | 92% | 3% | Black_Rot 43%, Healthy_Leaves 20%, Not_a_Grape_Leaf 37% |
| Downy_Mildew | 30 | 0% | 27% | 93% | 7% | Black_Rot 17%, Healthy_Leaves 27%, Not_a_Grape_Leaf 57% |
| Powdery_Mildew | 30 | 0% | 13% | 90% | 0% | Black_Rot 70%, Healthy_Leaves 13%, Not_a_Grape_Leaf 17% |

## Latency (load + preprocess + predict, one image at a time)

Median 76 ms, p95 129 ms over 30 images (target <= 500 ms desktop).

## Notes

- GVLiD is a single source (one phone, one date): field_test accuracy is the honest measure.
- Choose thresholds and calibration on validation + field_test, never on the test set.
- If targets are missed: inspect confusions, clean labels, add varied data, retrain. Do not tune on test.
