# 05 Linking recommendation

Flow: prediction status == confident/uncertain -> look up advice for the label -> show.
status not_a_leaf / low_confidence -> NO advice, show retake guidance.

## Label mapping
Model labels (folder names) -> advice keys, in one dictionary in config: Bacterial_Rot, Downey_Mildew (advice key "Downy_Mildew"), Healthy_Leaves ("Healthy"), Powdery_Mildew.

## Content review (mandatory before demo/publishing)
- The old map recommends Mancozeb (a fungicide) for Bacterial Rot; this must be checked.
- Have an agronomist or your state agriculture department/KVK review every product, dosage and interval.
- Cite the source for each recommendation (label, ICAR/NRCG advisory, university extension).
- Show: product, dosage per acre/L, interval, safety (PHI, protective gear), and "follow the product label".
- Add non-chemical advice (canopy management, sanitation, irrigation) and biocontrol.
- Keep wording as guidance, not a prescription.
