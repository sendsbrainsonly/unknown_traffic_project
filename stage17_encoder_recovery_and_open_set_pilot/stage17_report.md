# Stage 17 report

## Outcome

Historical Model A/B code and USTC checkpoints were recovered; Model C code was recovered but its historical checkpoint was not found. The strict pilot retrained E1/E2/E3 for each frozen LOSO run and used the same DES-v1 rule.

## DES-v1 pilot means

| Unknown | Encoder | Known Macro-F1 | AUROC | AUPRC | UFAR | Known FRR |
|---|---|---:|---:|---:|---:|---:|
| Email | E0 | 0.698090 | 0.718684 | 0.653641 | 0.948413 | 0.031847 |
| Email | E1 | 0.099522 | 0.605146 | 0.554318 | 1.000000 | 0.070064 |
| Email | E2 | 0.485736 | 0.617569 | 0.613006 | 0.912698 | 0.063694 |
| Email | E3 | 0.907480 | 0.801853 | 0.763249 | 0.805556 | 0.055202 |
| Streaming | E0 | 0.694651 | 0.674952 | 0.936397 | 0.883090 | 0.039886 |
| Streaming | E1 | 0.119760 | 0.504437 | 0.865952 | 0.990257 | 0.028490 |
| Streaming | E2 | 0.483253 | 0.565708 | 0.938445 | 0.713640 | 0.025641 |
| Streaming | E3 | 0.795084 | 0.720970 | 0.948430 | 0.783925 | 0.051282 |

## Decision

- E3 AUROC exceeds E1 in 6/6 runs.
- Mean E3 - E0 DES-v1 AUROC: +0.064594.
- Four-dataset promotion: **NOT_YET_SUPPORTED**. No promotion gate was preregistered, only two Services were tested, and strict E1 did not learn a usable Known classifier.
- This two-Service development pilot is not a six-Service or four-dataset conclusion.

## Answers to the 17 requested questions

1. **Found:** the original TrafficFormer and FIG/TAGCN code were found; the concat fusion code was found, but no historical Model C checkpoint was found.
2. **Locations:** Model A is under `tf_runtime/code`, `scripts/task03*`, `task08*`, and `outputs/stage1/modelA`; Model B is under `src/preprocessing/fig_graph.py`, `src/stage1/`, `task07*`, and `outputs/stage1/modelB`; Model C is `scripts/task09_model_c_fusion.py`.
3. **Inputs:** E1 uses first 5 packets, 64 bytes after the 14-byte Ethernet header, overlapping byte bigrams, SEP boundaries and length-320 padding. E2 uses a per-flow packet/burst graph. E3 uses no new raw feature.
4. **Dimensions/fusion:** `z_t=768`, `z_g=128`, and train-standardized concat `z_f=896`; the recovered fusion head is linear.
5. **Graph information:** direction, captured length, relative timestamp, current-burst packet/byte counts, and ratios to the previous burst; edges encode within- and adjacent-burst structure.
6. **Historical checkpoint reproduction:** A/B checkpoints load and match their expected 20-class heads; Model B metadata retains its historical best epoch 47. Full historical metric re-execution was not used as a current LOSO result.
7. **Current Unknown-Free assets:** only frozen E0 is directly reusable. Historical A/B/C weights are not LOSO-compatible; official TrafficFormer pretraining exposure is unverified and excluded.
8. **Retraining:** E1, E2 and E3 all required protocol-specific retraining. E0 was not retrained.
9. **TrafficFormer E1:** strict random-init E1 was weak/collapsed: Email mean Macro-F1 `0.099522`, AUROC `0.605146`; Streaming `0.119760` / `0.504437`.
10. **FIG/TAGCN E2:** Email mean Macro-F1/AUROC `0.485736/0.617569`; Streaming `0.483253/0.565708`. It has signal but trails E0 in mean AUROC.
11. **Fusion E3:** Email mean Macro-F1/AUROC/AUPRC `0.907480/0.801853/0.763249`; Streaming `0.795084/0.720970/0.948430`.
12. **Against E0+DES-v1:** E3 mean AUROC gains are `+0.083169` for Email and `+0.046018` for Streaming, positive in all 6 paired runs.
13. **UFAR:** E3 reduces mean UFAR by `-0.142857` on Email and `-0.099165` on Streaming; absolute UFAR remains high (`0.805556` / `0.783925`).
14. **Streaming:** the prior negative mechanism is improved in this pilot: E3 AUROC `0.720970` vs E0 `0.674952`, and UFAR `0.783925` vs `0.883090`.
15. **Independent graph information:** paired decisions are non-identical and E3 rescues E1/E2 errors, but the experiment does not isolate pure graph information from the fusion probe and Known-classifier quality. It is evidence of complementarity, not causal proof.
16. **Multimodal Known Support:** geometry and rescue patterns justify a later preregistered study, but this stage does not support selecting component counts or introducing a new support model.
17. **Four-dataset promotion:** not yet. First resolve the strict TrafficFormer pretraining/exposure or training-collapse problem and validate beyond the two development Services without changing the frozen detector.
