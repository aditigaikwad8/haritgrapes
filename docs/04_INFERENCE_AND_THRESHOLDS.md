# 04 Inference, Not-a-Leaf, and confidence thresholds

## Prediction output
label, confidence, top-3 with probabilities, status, model version.

## Input checks (before the model)
- Accept .jpg/.jpeg/.png, max 10 MB; convert to RGB; reject unreadable files with a clear message.
- Warn on very small, very dark or blurry images (simple brightness and blur-variance checks).

## Decision rules (starting values; tune on validation + field_test)
| Condition | Status | App behaviour |
|---|---|---|
| top label = Not_a_Grape_Leaf (conf >= 0.60) | not_a_leaf | "This does not look like a grape leaf. Retake with one leaf filling the frame." No advice. |
| top conf >= 0.85 AND margin (top1-top2) >= 0.25 | confident | Show disease + advice; ask "Was this right?" |
| 0.60 <= top conf < 0.85 OR margin < 0.25 | uncertain | Show top-3, ask user to pick/confirm, advice marked "probable" |
| top conf < 0.60 | low_confidence | "Not sure. Retake photo." No advice. Not stored for training |

Rules:
- Healthy_Leaves is also gated by thresholds (avoid telling a farmer a sick plant is healthy: for Healthy require conf >= 0.90).
- Mildew-diseased leaves are not in the training classes and may be predicted as Healthy. The app must show the unsupported-disease notice on every result.
- Optional test-time augmentation (average of original + flips) to stabilise confidence.
- Thresholds live in src/config.py, not scattered in code.
- Show the model version and "AI suggestion, confirm with an agriculture officer" on every result.
