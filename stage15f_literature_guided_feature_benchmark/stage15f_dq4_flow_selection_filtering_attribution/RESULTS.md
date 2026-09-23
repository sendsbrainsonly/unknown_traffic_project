# RESULTS — Stage 15F-DQ-4

## Conclusion and next step

Gates: `EVALUATION_POPULATION_EFFECT, FILTERING_EFFECT, CLASS_CONDITIONAL_SELECTION_EFFECT`  
Persistent limitations: `WEAK_CAPTURE_LABEL`, `CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`.

The experiment separates evaluation-population selection from training-selection effects on the frozen 3,065-flow DQ-3F mother population. No Known Test or Unknown Test feature was read.

Stop at DQ-4. DQ-5--DQ-7, TrafficFormer/TFE-GNN training, Byte--Behavior and Open-Set evaluation require separate authorization.

## Data and split

- A: 3065 flows.
- B (unique packet-consistent CATE parent match): 2101 flows.
- C_PARENT/C_FINAL Service: 1551/1551 flows.
- D = B intersection C: 1440 flows.
- C_PARENT equals C_FINAL for Service inclusion, but Native sessions and TrafficFormer parent flows are not one-to-one.

## Configuration and execution

- M-A reuses three frozen DQ-3F F3 checkpoints after exact data-array, prediction-order and checkpoint-hash parity.
- M-B/M-C/M-D use the unchanged DQ-3F F3 ResNet18/Open-Detect training configuration on B/C_FINAL/D Train respectively.
- Seeds: 2022, 2023 and 2024; checkpoint selection uses each model's own Known Validation Accuracy.
- Nine new runs executed on physical GPUs 0/2/3 in three model queues; all raw logs, predictions, configs and checkpoints are retained.

## Core results

## Within-population results

| Model | Accuracy mean | Macro-F1 mean | Weighted-F1 mean |
|---|---:|---:|---:|
| M-A | 0.890547 | 0.856050 | 0.888036 |
| M-B | 0.773558 | 0.650267 | 0.745753 |
| M-C | 0.896739 | 0.887715 | 0.894373 |
| M-D | 0.789474 | 0.675056 | 0.757698 |

These values are not a fair cross-model ranking because both Train and Validation populations differ.

## Common D Validation comparison

| Model | Accuracy mean | Macro-F1 mean | Weighted-F1 mean |
|---|---:|---:|---:|
| M-A | 0.892788 | 0.865662 | 0.890030 |
| M-B | 0.805068 | 0.673803 | 0.769943 |
| M-C | 0.910331 | 0.898537 | 0.907816 |
| M-D | 0.789474 | 0.675056 | 0.757698 |

Mean matched-D Macro-F1 deltas versus M-A: `{"M-B": -0.19185937290000213, "M-C": 0.03287505337935792, "M-D": -0.19060606271090022}`. Positive seeds out of 3: `{"M-B": 1, "M-C": 2, "M-D": 1}`.

## Paired-seed stability

| Seed | M-A | M-B | M-C | M-D | M-C − M-A |
|---:|---:|---:|---:|---:|---:|
| 2022 | 0.893865 | 0.905074 | 0.939618 | 0.894200 | +0.045753 |
| 2023 | 0.770481 | 0.220551 | 0.848157 | 0.221315 | +0.077677 |
| 2024 | 0.932641 | 0.895784 | 0.907837 | 0.909653 | −0.024805 |

The `FILTERING_EFFECT` gate is therefore partial and seed-sensitive, not a three-seed universal gain. M-B and M-D suffer an optimization collapse at seed 2023 (own-population Macro-F1 `0.200651/0.221315`), while M-C remains stable enough to rescue that seed. No failed run was deleted or rerun. This instability is part of the result, not evidence for retuning.

## Full A Validation stress test

| Model | Accuracy mean | Macro-F1 mean | Weighted-F1 mean |
|---|---:|---:|---:|
| M-A | 0.890547 | 0.856050 | 0.888036 |
| M-B | 0.588060 | 0.431240 | 0.545049 |
| M-C | 0.698507 | 0.622356 | 0.701865 |
| M-D | 0.530348 | 0.400851 | 0.497350 |

This is distribution-shift stress evidence only; A Validation was not used for checkpoint selection in M-B/M-C/M-D.

## Limitations

- Evaluation-population-only Macro-F1 deltas for the unchanged M-A model versus A: `{"B": -0.019417204683047973, "C_FINAL": 0.003781883955520926, "D": 0.009612288665265756}`.
- The unchanged-model deltas are small and mixed; the large M-B/M-D degradation cannot be explained by simply making Validation easier and is dominated by training instability plus changed class/support composition.
- Any apparent gain when only changing A_val to B/C/D_val is therefore a population-composition effect, not a learned model improvement.
- Training-selection claims use only the common D_val matched comparison. The combined D model is not automatically preferred unless its paired delta is consistently positive.
- CATE B removes flows absent from the target-flow five-tuple list; this status is not authoritative evidence that removed flows are mislabeled.
- TrafficFormer C removes parents below 2,048 captured bytes (then checks >=3 packets). It strongly changes Service composition and observational support.
- On common D_val, M-C improves Chat, File-Transfer, Streaming and VoIP mean F1, is near M-A on P2P, and is slightly lower on Email. The effect is class-conditional.
- B/D VoIP Validation support is only four flows and C's minimum class support is six; per-Service conclusions for these cells are fragile.
- P2P remains one-capture in the shared31 cohort, so capture-generalization cannot be identified.

## Cross-paper gap boundary

Flow selection and filtering can explain a measurable part of reported evaluation differences when the unchanged M-A model changes score across A/B/C/D. They do not alone prove that TrafficFormer or TFE-GNN model mechanisms are better, because input representation, architecture, supervision and split construction also differ. Conversely, within-population model differences cannot be interpreted without the matched-D control.

## Preserved evidence

- M-A: reused 3/3 frozen DQ-3F F3 checkpoints after exact array/prediction/hash parity.
- M-B/M-C/M-D: 9/9 frozen-config runs completed.
- Known Test feature values used: 0.
- Unknown Test feature values used: 0.
- TrafficFormer success-parent to final-Service multiset parity: PASS.
