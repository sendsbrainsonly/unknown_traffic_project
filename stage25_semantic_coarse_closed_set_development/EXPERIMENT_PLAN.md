# Stage 25 — Semantic-coarse closed-set development, retaining fine-class ability

Recorded before Stage 25 metric computation. This is a **development** comparison: Stage 20 Known Test has already been exposed in prior stages, so it is not independent confirmation.

## Frozen inputs and label rule

- Read Stage 20 `closed_service_manifest.csv` without modifying its flow IDs, roles or 6/7-class Service labels. Reuse Stage 21 cached `packet_counts` and Stage 22/23/23B frozen predictions/checkpoints.
- Semantic map fixed in advance: `Chat`, `Email`, `VoIP` → `Communication`; all other Services remain separate. ISCX-VPN has 4 coarse classes (`Communication`, `File-Transfer`, `P2P`, `Streaming`); ISCXTor2016 has those four plus `Browsing` (5). Keep every original Service and sample in the main scope.
- Secondary scope: packet count ≥2, selected solely by frozen input metadata. The threshold was chosen from Known Train/Validation coverage before this experiment. No class removal or per-class cap. Report retained counts by dataset, role and original Service. Do not substitute this conditional scope for the full-flow main result.
- Original 6/7-class fine heads/checkpoints and predictions are immutable. Coarse scores are additional outputs; report fine Accuracy/Macro-F1/Weighted-F1 as a capability guardrail.

## Sequential experiments

1. **A — No-training hard-prediction remap.** On exactly matched Known Validation/Test flow IDs, remap both true and predicted fine Service labels. Evaluate all seven existing Stage 23/23B methods. This remaps the hard decision; it is **not** probability pooling or a newly trained coarse classifier. Replay original fine metrics and report both full-flow and ≥2-packet conditional metrics. This establishes how much apparent gain comes purely from label semantics.
2. **B — Frozen-encoder coarse-head training.** For E1 and E3 separately, each dataset × seeds 2022/2023, fit one fixed multinomial logistic-regression head on Stage 22 **Known Train** representations only. Fit `StandardScaler` on Known Train only; `LogisticRegression(C=1, solver=lbfgs, max_iter=200, random_state=0)` with no hyperparameter search. Known Validation is diagnostic; Known Test is final development evaluation, never used for fitting. Save scaler and coefficients as numeric arrays, per-flow predictions, convergence and metrics. Original encoders/fine heads stay frozen; this head is additive.
3. **C — Conditional-scope report.** Use the same A/B predictors, with no retraining, on the fixed ≥2-packet subset. Compare each method only against the same subset and the same coarse labels.

Run A, then B, then finalize C. Do not select labels, thresholds, sample caps or model variants from A/B/Test outcomes. No Unknown data, open-set detector or new encoder training. No Stage 20–24 file modification.

## Reporting and checks

- Accuracy, Macro-F1, Weighted-F1, per-class precision/recall/F1/support, paired seed-level results and mean ± population std. Include original fine task and explicit coverage fractions.
- Validate exact flow-ID/label/role alignment, source SHA256 before/after, no duplicate sample IDs, no Unknown usage, and no Test fitting. Compare no-training fine scores with historical Stage 23 metrics. Preserve errors and partial outputs.
- Main interpretation: a higher coarse score is a different task, not evidence of improved original 6/7-class ability. Auxiliary-head results are not method-equivalent to hard-remap baselines; present separate tables.
