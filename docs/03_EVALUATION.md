# 03 Evaluation

Run on: (a) test set, (b) field_test set. Save to reports/<version>/.

## Report must include
- Accuracy, per-class precision/recall/F1, macro-F1
- Confusion matrix (counts and normalised)
- 20 worst mistakes with images (look for mislabels / look-alikes)
- Confidence histograms: correct vs wrong predictions
- Not_a_Grape_Leaf: false-accept rate (random photo accepted as a grape leaf) and false-reject rate
- Latency per image (target <= 500 ms desktop)
- Per-class precision, recall and F1 for Black_Rot, Esca, Healthy_Leaves, Leaf_Blight, Not_a_Grape_Leaf
- Out-of-scope check (see below)

## Out-of-scope check (unsupported diseases)
Collect 20-30 photos of Downy Mildew and Powdery Mildew leaves into data/field_test/out_of_scope/.
Never train on them. Run them through the app's predict function and report:
- % predicted Healthy with high confidence (the dangerous case)
- % returned as uncertain or low_confidence
- % forced into each disease class
Report these numbers honestly. They define the limitation shown in the app.

## Acceptance targets (v1)
| Metric | Target |
|---|---|
| Test accuracy | >= 90% |
| Every class recall (test) | >= 0.85 |
| Not_a_Grape_Leaf recall | >= 0.90 |
| Field-test accuracy | report honestly; aim >= 75%, improve with feedback |

If targets are missed: inspect confusions, clean labels, add more varied data, then retrain. Do not touch the test set to "tune".

## Calibration
Check if "90% confidence" is right ~90% of the time. If overconfident, apply temperature scaling fit on the VALIDATION set only.
Choose thresholds on validation + field_test, never on test.
