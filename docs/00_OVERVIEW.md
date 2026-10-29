# 00 Overview and decisions

## What we build
Photo -> quality check -> classifier -> confidence rules -> (advice | retake | "not a grape leaf") -> feedback -> scheduled retraining.

## Why a clean restart
Old runs mixed input scaling (0-255 vs 0-1), used different splits, leaked augmented copies across
train/val, had a wrong class order in the app, and reset the classifier head. Retrain once, cleanly.
The dataset was later replaced with GVLiD (Black Rot, Esca, Leaf Blight, Healthy), so classes and counts changed.

## Fixed decisions (do not change mid-project)
| Topic | Decision |
|---|---|
| Classes (model output) | Black_Rot, Esca, Healthy_Leaves, Leaf_Blight, Not_a_Grape_Leaf |
| Data source | GVLiD (4 grape classes, one phone, one date) + separate Not_a_Grape_Leaf images. Metadata file not used. |
| Base model | MobileNetV2 (ImageNet); optional EfficientNetB0 for comparison |
| Input | 224x224 RGB |
| Preprocessing | INSIDE the model (one place only). The app sends raw 0-255 pixels. |
| Class order | Saved to class_names.json at training; the app ALWAYS loads it. Never hard-code. |
| Split | 70/15/15, done once on ORIGINAL images, seeded, before any augmentation |
| Augmentation | Only during training, inside the model |
| Test set | Frozen forever. Feedback images never enter it. |
| Learning | Human-in-the-loop, scheduled retraining. NOT live self-training on its own predictions. |

## Why no live self-learning
A model cannot know when it is wrong. Training on its own predictions amplifies mistakes
(confirmation bias). Only human-confirmed labels are used, and a new model goes live only if it beats the old one.
