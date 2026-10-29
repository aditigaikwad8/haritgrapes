# 08 Testing and Definition of Done

## Automated tests (tests/)
- Preprocessing: output shape (224,224,3) and range as expected
- Class order: class_names.json matches model output size and the app mapping
- 20 training images -> >= 95% correct through the same predict function the app uses
- Thresholds: each status returned for crafted probability vectors
- Feedback: file saved, row logged, EXIF removed, duplicates ignored
- Promotion: candidate worse than production is rejected
- Label mapping covers all 5 classes (model label -> display name -> advice key)

## Manual tests
- 5 images per class from train, 5 from test, 5 from field_test through the actual app
- Random non-leaf photos (shoe, face, screenshot) -> not_a_leaf or low_confidence
- Blurry/dark photos, wrong file type, huge file
- Feedback Wrong -> appears in pending; retrain on a small batch -> new version evaluated

## Definition of Done (tick all)
- [ ] Data cleaned, counts recorded, Not_a_Grape_Leaf collected
- [ ] One clean training run, v1 saved with class_names.json
- [ ] Test targets met (docs/03) and field-test results reported honestly
- [ ] Thresholds tuned on validation/field set and stored in config
- [ ] App uses the same predict function and loads class order from file
- [ ] Recommendations reviewed by an expert and sources listed
- [ ] Feedback saved with consent, review step working
- [ ] One full retrain cycle completed with a promote/reject decision and a version table
- [ ] README with screenshots, limitations and ethics notes
- [ ] Labels spot-checked by eye (about 50 per class) and wrong ones removed
- [ ] Split done once (seed 42, stratified) after removing duplicates; every class in every split
- [ ] Unsupported-disease notice shown on every result
- [ ] Out-of-scope (mildew) check reported

## Suggested schedule (about 2 weeks)
Days 1-2 data | 3-4 training + evaluation | 5 thresholds + Not-a-leaf | 6-7 app + recommendation | 8-9 feedback system | 10-11 retrain cycle + report | 12-14 tests, polish, website.
