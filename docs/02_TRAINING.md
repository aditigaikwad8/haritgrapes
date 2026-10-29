# 02 Training

## Environment
Python 3.10/3.11, virtual env, requirements.txt. Use a GPU (Colab is fine). Record versions in reports/environment.txt.

## Model A: MobileNetV2 (primary)
Phase 1 (head only): base frozen, new head (global average pooling, dropout ~0.4, softmax over 5 classes),
Adam lr 1e-3, up to ~15 epochs, early stopping on val loss.
Phase 2 (fine-tune): unfreeze top ~40 layers (keep BatchNorm frozen), Adam lr 1e-5, up to ~25 epochs, early stopping, ReduceLROnPlateau.
Loss: categorical cross-entropy. Class weights from TRAIN counts. Light label smoothing (0.05-0.1) optional.
Class weights are computed from train counts (Leaf_Blight is the smallest class).
Augmentation (in model): flips, rotation, zoom, contrast, brightness. Do NOT apply mixup at first.

## Model B (optional): EfficientNetB0
Same pipeline, same split, same seed. Compare on test + field_test. Keep the better one; put both in the report table.

## Rules that prevent past problems
1. One training script, one run. No chains of "resume" scripts.
2. Preprocessing in exactly one place (inside the model).
3. Validation/test datasets are not shuffled (otherwise reports are wrong).
4. Save at end: models/versions/v1_<date>.keras, class_names.json, training curves, config snapshot.
5. Set random seeds and log them.

## Sanity checks BEFORE trusting results
- Model predicts ~random (20%) before training; training accuracy climbs.
- Model correctly classifies 20 images taken from train/ (if not, there is a bug; stop and fix).
- Predicted class distribution is not a single class.
