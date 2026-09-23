# RESULTS — Stage 15F-DQ DQ-0 to DQ-3 Interim

## Scope and decision

This interim report covers **DQ-0 through DQ-3 only** on the shared 31 ISCX-VPN VPN PCAPs. ISCXTor, DQ-4--DQ-7, new model training, Known Test, and Unknown Test are all `NOT_RUN`.

The interim gate is **STOP_RELATED_BRANCH_MAPPING_AMBIGUITY**. Flow reconstruction is reliable, but a full same-population Fine→Coarse evaluation is not identifiable because an application label can represent multiple services.

## DQ-0: the datasets and flow populations are not the same

- Native uses 137 captures and 22142 selected sessions.
- TFE-GNN and TrafficFormer each use the same 31 VPN captures, not Native's complete capture population.
- Within those 31 captures, Native has 4224 sessions, TFE-GNN has 1902 CATE target flows (1662 paper-text-valid), and TrafficFormer has 17769 capture-wide parent flows of which 1526 are eligible.
- TrafficFormer reconstruction matched its stored processing audit exactly on all 31 captures.

These counts are not a method leaderboard: the unit of flow and filtering rules differ.

## DQ-1: what the other pipelines remove/select

| Set | Flows | Captures | 1-packet | 2-packet | <=2 packets |
|---|---:|---:|---:|---:|---:|
| A Native shared31 | 4224 | 31 | 35.20% | 36.65% | 71.85% |
| B CATE-matched | 2720 | 31 | 50.48% | 13.24% | 63.71% |
| C TrafficFormer-parent eligible | 1984 | 28 | 43.55% | 12.30% | 55.85% |
| D B intersect C | 1804 | 28 | 47.67% | 12.53% | 60.20% |

CATE matching selects 64.39% of Native shared31 sessions. This is a selection difference, not evidence that the unmatched 1504 sessions are mislabeled.

## DQ-2: frozen Native Known Validation diagnosis

- Shared31 subset: N=335, Accuracy=0.802985, Macro-F1=0.770803, Weighted-F1=0.817823.
- Packet-group Accuracy: 1=0.8397, 2=0.8462, 3--7=0.6400, 8--15=0.6667, >=16=0.7460.
- <=2-packet flows account for 37/66=56.06% of errors because they comprise 235/335=70.15% of samples. Their error rate is 15.74%, lower than 29.00% for >=3-packet samples. Thus this frozen subset does not support the claim that short flows are disproportionately difficult.
- MATCHED error rate=19.41%; UNMATCHED error rate=20.41%. The 1.00 percentage-point difference is descriptive and composition-confounded.
- Fine errors becoming Coarse-correct: 22/66=33.33% overall, 22/61=36.07% among mappable Fine errors.

The CSVs include packet group crossed with application, service, capture, and CATE status. Small cells remain descriptive; no causal label-noise claim is made.

## DQ-3: feasibility, not a successful coarse-task experiment

Service support is non-zero for all six services, but Facebook maps to both Chat and VoIP in the active Known classes. Across the full label inventory, Hangouts and Skype are also multi-service. Only 321/335 frozen validation rows have both true and predicted applications deterministically mappable.

The partial mapped subset improves from Fine Macro-F1 0.770803 to Coarse Macro-F1 0.840611, but this is neither complete-population F2 nor a trained F3. No claim that Service is the better formal task is permitted yet.

## Interim answers to the research questions

1. **Why can prior papers look better?** Already-identifiable protocol differences include fewer PCAPs, VPN-only scope, service rather than application labels, CATE target-flow selection, TrafficFormer short/byte filtering, different flow units, and different splits/preprocessing. Their reported scores cannot be directly ranked against Native.
2. **How much Fine error is within service?** 33.33% of all Fine errors and 36.07% of mappable Fine errors in the frozen shared31 validation subset.
3. **Does retrained coarse classification improve?** `NOT_RUN`; the semantic mapping gate failed before training.
4. **Can Service become the main task?** `INSUFFICIENT_EVIDENCE`; it needs a service-native label space plus capture-group feasibility.
5--9. Filtering causal effect, domain mixing, split sensitivity, and matched-model comparison are `NOT_RUN`.
10. **Current proven limitation:** protocol/task non-equivalence and mapping identifiability. A representation/model gap has not been isolated.
11. **Next task definition:** first freeze service-native labels that preserve multi-service applications; do not collapse application predictions post hoc.
12. **Ready to restart Byte--Behavior research?** No. The task/protocol definition must pass DQ-3 and DQ-6 before model conclusions are interpretable.

## Integrity

- Known Test values used: 0.
- Unknown Test values used: 0.
- New encoders trained: 0.
- Frozen input hash comparison: PASS (208 files; changed=0).
