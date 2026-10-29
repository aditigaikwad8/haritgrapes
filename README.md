# HaritGrapes – Grape Leaf Disease Model (clean restart)

Goal: classify a grape leaf photo as **Bacterial Rot, Downey Mildew, Healthy Leaves, Powdery Mildew**
(or **Not a Grape Leaf**), then link to a pesticide/treatment recommendation, and improve over time
from **human-confirmed** user feedback.

## Read in this order
1. docs/00_OVERVIEW.md – decisions and architecture
2. docs/01_DATA.md → 02_TRAINING.md → 03_EVALUATION.md
3. docs/04_INFERENCE_AND_THRESHOLDS.md → 05_RECOMMENDATION.md
4. docs/06_FEEDBACK_AND_RETRAINING.md → 07_APP.md
5. docs/08_TESTING_AND_DONE.md (checklist: you are finished when every box is ticked)

## Folder map
- data/raw: original images, one folder per class (folder name = label)
- data/split: generated train/val/test (never edit by hand)
- data/field_test: your own real phone photos, used ONLY for testing
- data/feedback: pending → approved / rejected user feedback
- models/versions: every trained model; models/production: the one the app loads
- reports: metrics, confusion matrices, version comparison
- src: pipeline scripts (empty placeholders for now); app: Streamlit UI; logs: prediction/feedback logs
