# 05 Linking recommendation

Flow: prediction status == confident/uncertain -> look up advice for the label -> show.
status not_a_leaf / low_confidence -> NO advice, show retake guidance.

## Label mapping
Model labels (folder names) -> advice keys, in one dictionary in config: Black_Rot, Esca, Healthy_Leaves ("Healthy"), Leaf_Blight.

## Content review (mandatory before demo/publishing)
- Have an agronomist or your state agriculture department/KVK review every product, dosage and interval.
- Esca is a trunk disease. Advice must focus on pruning, sanitation and wound protection, not a leaf spray as a cure. Have it reviewed by an expert.
- Cite the source for each recommendation (label, ICAR/NRCG advisory, university extension).
- Show: product, dosage per acre/L, interval, safety (PHI, protective gear), and "follow the product label".
- Add non-chemical advice (canopy management, sanitation, irrigation) and biocontrol.
- Keep wording as guidance, not a prescription.
