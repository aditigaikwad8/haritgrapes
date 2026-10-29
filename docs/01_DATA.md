# 01 Data

## 1. Organise raw data
data/raw/<Class>/ images. Folder name is the label; no label files.
Use underscores in folder names (Healthy_Leaves). Display names ("Healthy Leaves") live in the app, not in folders.

## 2. Count and record
Write counts per class in reports/data_summary.md (before cleaning and after). Target >= 800 originals per class
(you have ~1000 each). If a class is far smaller, say so; it will need more data or class weights.

## 3. Clean
- Remove corrupted/unreadable files, non-RGB oddities, tiny images (<100 px).
- Remove exact duplicates (file hash) and near-duplicates (same leaf photographed repeatedly: keep a few).
- Look through ~50 random images per class. Remove mislabeled ones. This step matters more than model choice.
- Do NOT use an already-augmented folder. Use originals only.

## 4. Not_a_Grape_Leaf class (new)
Purpose: stop random photos being forced into a disease.
Target 800-1000 images, a mix of:
- leaves of other plants (other crops/trees), including diseased ones
- non-leaf scenes: soil, hands, sky, fruit, grape bunches, vine stems, trellis wires, tools
- poor quality: blurry, dark, overexposed, partially covered leaves
- documents/screens/people/objects (everyday photos)
Sources: other PlantVillage crops, open image datasets, your own phone photos. Never reuse images from the 4 disease classes.

## 5. Split (once)
Stratified 70/15/15 into data/split/{train,val,test}, seed fixed (42). Record the file list so the split can be reproduced.
Keep leaves from the same plant/photo session in ONE split where you can tell (prevents leakage).

## 6. Field test set (very important)
data/field_test: 20-30 real phone photos per class from real vines (or from sources you did not train on).
Used only to measure real-world performance; never trained on. Update it as you collect more.

## 7. Known limitation to document
Diseases outside the 4 classes (e.g. Anthracnose, Black Rot) will be forced to the nearest class. State this in the app and report.
