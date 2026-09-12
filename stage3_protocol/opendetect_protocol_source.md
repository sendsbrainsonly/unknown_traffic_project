# Open-Detect Protocol Source

## Recovered official USTC scenarios

Source: read-only upstream `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/Open-Detect/code/data/splits.py:1-12`.

| Scenario | Known / Unknown | Unknown class names |
|---|---:|---|
| A-1 | 19 / 1 | Tinba |
| A-2 | 17 / 3 | Geodo; Htbot; Tinba |
| A-3 | 15 / 5 | Geodo; Htbot; Tinba; Miuref; Neris |

The translation is name-based because official and current-project numeric IDs differ. See `class_inventory.csv`.

## Paper requirements

- USTC contains ten normal and ten abnormal classes: extracted paper lines 551-556.
- Dataset split is 8:1:1 train/validation/test: lines 525-528.
- Open-world USTC settings are 19/1, 17/3, 15/5: lines 659-664.
- Tables report avg.±std. over five folds: lines 586-589 and 640-644.
- Threshold accepts 95% of Known Validation: lines 429-451.
- Open-world tables report Accuracy and F1; discussion also uses TPR/FPR: lines 640-667.
- Rendered Table III confirmation: `.artifacts/pdf/opendetect_protocol_page_10.png`.

Local extracted text: `.artifacts/pdf/opendetect_stage3_protocol_audit.txt`.

## Released-code behavior

- `train.py:51-60`: filters train/validation by known class and constructs an unknown test loader.
- `train.py:70-81`: checkpoint selection uses known “validation,” which is actually the released test NPZ.
- `data/dataset.py:78-87,124-133`: only train/test USTC NPZs exist; there is no standalone validation input.
- `test.py:46-58`: threshold is selected from labeled known+unknown final-test scores by maximizing TPR-FPR.
- `test.py:118-144`: evaluation seed is 2021 and both known/unknown are drawn from the test NPZ.
- `train.py:46`: training seed is 2022.
- `model.py:21-28,37-48`: eval representation is deterministic `mu_x`; native score is prototype KL.

## Frozen resolution

The paper rule, not the released oracle-test threshold, governs Stage 3. Unknown is not loaded until Final Test. The class scenarios are official; the sample split is the project's already-frozen `compatible_min1` split, not a claimed reconstruction of the unpublished five folds.
