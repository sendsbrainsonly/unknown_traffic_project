# RESULTS — Stage 15F-1B

Status: complete
Conclusions: `STATISTICAL_FEATURE_BENEFIT, BURST_COMPLEMENTARITY_CONFIRMED, CLASS_CONDITIONAL_BEHAVIOR_BENEFIT`

## Core results

S-ABCD-S12 mean validation Macro-F1 delta: +0.019473. S-Burst-S-ABCD: +0.006094. S-Burst-S12: +0.025567.

## Configuration and execution

Five frozen pilots; exact Stage 15R LightGBM configuration; FULL_FLOW six sets and EARLY_16 three sets; seed 2022; Known-Validation logloss checkpoint selection.

## Data and split

Known Train/Validation only. Known Test and Unknown Test feature values used: 0/0.

## Preserved evidence

All run models, predictions, confusion matrices, per-class metrics, histories, feature importance, input hashes, cache audits, source audits, failures, and final tables are retained in this bundle.

## Limitations

Five-pilot diagnostic with one fixed seed. Full-flow and Early-16 are different observation budgets. Gain importance is not causal evidence.

## Conclusion and next step

Gate labels: `STATISTICAL_FEATURE_BENEFIT, BURST_COMPLEMENTARITY_CONFIRMED, CLASS_CONDITIONAL_BEHAVIOR_BENEFIT`. Stop after Stage 15F-1B; full15, Stage 15F-1C, open-set evaluation, and Byte-Behavior model development remain NOT_RUN.
