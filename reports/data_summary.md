# Data summary

Generated 2026-10-30 01:19. Seed 42. Source: GVLiD (4 grape classes) + own Not_a_Grape_Leaf images. GVLiD metadata file not used.

## Counts before cleaning, removals, after cleaning

| Class | Before | unreadable | empty_file | too_small | grayscale | manual_review | exact_duplicate | exact_duplicate_conflicting_labels | near_duplicate | near_duplicate_conflicting_labels | After |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Black_Rot | 808 | 0 | 0 | 0 | 0 | 0 | 695 | 16 | 0 | 1 | 96 |
| Esca | 888 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 179 | 707 |
| Healthy_Leaves | 1109 | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | 1 | 1092 |
| Leaf_Blight | 672 | 0 | 0 | 0 | 0 | 0 | 341 | 0 | 0 | 179 | 152 |
| Not_a_Grape_Leaf | 928 | 0 | 0 | 0 | 0 | 30 | 10 | 0 | 0 | 0 | 888 |
| **Total** | 4405 | 0 | 0 | 0 | 0 | 30 | 1062 | 18 | 0 | 360 | 2935 |

## Split (stratified 70/15/15, seed 42, near-duplicate groups kept together)

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Black_Rot | 66 | 15 | 15 | 96 |
| Esca | 495 | 106 | 106 | 707 |
| Healthy_Leaves | 764 | 164 | 164 | 1092 |
| Leaf_Blight | 106 | 23 | 23 | 152 |
| Not_a_Grape_Leaf | 621 | 134 | 133 | 888 |
| **Total** | 2052 | 442 | 441 | 2935 |

## Notes

- Classes below the 800-original target after cleaning: Black_Rot, Esca, Leaf_Blight. Class weights (computed from train counts) compensate.
- Manual review: 30 files listed in reports/review/remove_list.txt were left out.
- Non-RGB images converted to RGB: none.
- Near-duplicates: dhash distance <= 3/64, at most 3 kept per cluster. See reports/near_duplicate_clusters.csv.
- Every removed file and its reason: reports/removed_files.csv. Exact split list: reports/split_manifest.csv. data/raw is never modified.
- Images in data/split are re-saved as RGB JPEG (EXIF removed, longest side <= 512px).
- Limitation: GVLiD comes from one phone on one date and no metadata is used for grouping, so photos from one session can still sit in different splits. Test scores can therefore look better than real use; the field test is the honest measure.
