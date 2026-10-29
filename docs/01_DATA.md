# 01 Data

## 1. Organise raw data
data/raw/<Class>/ images. Folder name is the label; no label files.
Use underscores in folder names (Healthy_Leaves, Black_Rot, Esca, Leaf_Blight, Not_a_Grape_Leaf). Display names ("Healthy Leaves") live in the app, not in folders.

## 2. Count and record
Write counts per class in reports/data_summary.md (before cleaning and after). Counts before cleaning: Black_Rot 808, Esca 888, Healthy_Leaves 1109, Leaf_Blight 672, Not_a_Grape_Leaf 928.
Target >= 800 originals per class. Cleaning removes duplicates, so Black_Rot (808) may fall below 800 and
Leaf_Blight (672) is already below it. Say so in the report; these classes use class weights.

## 3. Clean
- Remove corrupted/unreadable files, non-RGB oddities, tiny images (<100 px).
- Remove exact duplicates (file hash) and near-duplicates (same leaf photographed repeatedly: keep a few).
- Look through ~50 random images per class. Remove mislabeled ones. This step matters more than model choice.
- Do NOT use an already-augmented folder. Use originals only.
- GVLiD folders were built from filename prefixes and some labels may be wrong. Spot-check about 50 random images
  per class by eye and remove or move wrong ones before the split.

## 4. Not_a_Grape_Leaf class (new)
Purpose: stop random photos being forced into a disease.
Not_a_Grape_Leaf must NOT contain any GVLiD images or any grape-disease images.
Target 800-1000 images, a mix of:
- leaves of other plants (other crops/trees), including diseased ones
- non-leaf scenes: soil, hands, sky, fruit, grape bunches, vine stems, trellis wires, tools
- poor quality: blurry, dark, overexposed, partially covered leaves
- documents/screens/people/objects (everyday photos)
Sources: other PlantVillage crops, open image datasets, your own phone photos. Never reuse images from the 4 disease classes.

## 5. Split (once)
Stratified 70/15/15 into data/split/{train,val,test}, seed fixed (42). Record the file list so the split can be reproduced.
Keep leaves from the same plant/photo session in ONE split where you can tell (prevents leakage).
Split randomly per class (stratified, seed 42) after removing duplicates and near-duplicates.
Photos from the same session look alike, so test scores can look better than real-world performance.
Do not trust the test score alone; the field test (section 6) is the honest measure.

## 6. Field test set (very important)
data/field_test: 20-30 real phone photos per class from real vines (or from sources you did not train on).
Used only to measure real-world performance; never trained on. Update it as you collect more.
GVLiD is a single source (one phone, one date), so field photos are the real measure of performance.

## 7. Known limitation to document
Downy Mildew and Powdery Mildew are not in the training classes. A leaf with these diseases may be misclassified,
including as Healthy. State this in the app and report.