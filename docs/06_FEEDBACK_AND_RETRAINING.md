# 06 Feedback buttons and retraining

## Feedback UI (in app)
After a confident/uncertain result: "Correct" / "Wrong". If Wrong: dropdown of the 5 classes plus "Disease not listed / other" ("Actually it is ...").
Images labelled "Disease not listed / other" are logged but never used for training. This stops mildew photos being forced into a wrong class.
Optional: consent checkbox "Allow this photo to be used to improve the model".
Only store photos with consent. Strip EXIF/GPS data. Store no names or phone numbers.

## What is stored
data/feedback/pending/<uuid>.jpg + a row in logs/feedback.csv:
timestamp, image_id, image_hash, predicted_label, confidence, user_label, user_said_correct, model_version, status.
Every prediction (even without feedback) is logged in logs/predictions.csv for monitoring.
The folders under data/feedback/approved/<class>/ are created by code when needed.

## Trust rules
| Signal | Use |
|---|---|
| Correct (user confirms) | pending, then approved after light review/sampling |
| Wrong + corrected label | pending, then MUST be reviewed (a human or you) before approved |
| No feedback | never used as a label |
| Low confidence / not_a_leaf | never used |
| "Disease not listed / other" | logged only, never used for training |
Review: you (or the guide) look at pending images in a simple folder view, move to approved/<class>/ or rejected/.
Deduplicate by hash. Cap how many images one user session can add.

## Retraining trigger
Retrain when approved feedback >= 100 images in total (and >= 15 per affected class), or monthly, whichever first.
Never retrain automatically with unreviewed data.

## Retraining procedure (candidate model)
1. Dataset = original train data (replay) + approved feedback. Feedback is oversampled a little but capped at ~30% of training data.
2. Start from the current production weights; fine-tune at lr 1e-5 (do not start from scratch).
3. Validation = original val set (+ a feedback hold-out of ~15% never trained on).
4. Evaluate on the FROZEN test set and the field_test set.
5. Write reports/<new_version>/ and a comparison with production.

## Promotion rule (all must pass)
- Macro-F1 on test >= production (+0.5 pt or more on field_test)
- No class recall drops by more than 2 points
- Not_a_Grape_Leaf false-accept rate does not get worse
- Latency still within target
If any fail: reject, keep production, record the reason. If pass: copy to models/production, bump version, keep the previous version for rollback.

## Instant-learning option (later, optional)
Add an embedding-memory (nearest-neighbour) layer so corrections help immediately between retrains. Use only after the base pipeline is stable.

## Deployment note
Streamlit Community Cloud storage is not permanent. For a real deployment keep feedback in persistent storage (Hugging Face Space storage, Supabase, S3) and run retraining offline (Colab/PC).

## Report this to interviewers
A table: version, date, train size, feedback added, test macro-F1, field accuracy, promoted yes/no. This is your evidence of improvement.
