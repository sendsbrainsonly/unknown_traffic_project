# DQ-3 Granularity Feasibility and Preregistration

## Gate outcome

**BLOCKED_MAPPING_AMBIGUITY.** DQ-3 was executed as a feasibility gate; no new F1/F3 model was trained.

The capture-level service label is available, and all six services have non-zero Known Train/Validation support. However, application predictions cannot be deterministically mapped to service for the complete population: `{"Facebook": ["Chat", "VoIP"], "Hangouts": ["Chat", "VoIP"], "Skype": ["Chat", "File-Transfer", "VoIP"]}`. In the active `medium_seed2022` Known set, Facebook is present and maps to both Chat and VoIP. A majority rule, a capture-conditioned prediction map, or use of the true capture to map a predicted Facebook label would each add an unregistered decision rule.

## Same-flow support

| Task label | Train support | Validation support |
|---|---|---|
| Fine applications | `{"AIM": 46, "BitTorrent": 804, "Email": 145, "FTPS": 301, "Facebook": 79, "ICQ": 54, "Netflix": 371, "SFTP": 31, "Spotify": 234, "Vimeo": 249, "VoIPBuster": 416}` | `{"AIM": 9, "BitTorrent": 100, "Email": 23, "FTPS": 36, "Facebook": 11, "ICQ": 2, "Netflix": 48, "SFTP": 7, "Spotify": 30, "Vimeo": 26, "VoIPBuster": 43}` |
| Coarse services | `{"Chat": 132, "Email": 145, "File-Transfer": 332, "P2P": 804, "Streaming": 854, "VoIP": 463}` | `{"Chat": 17, "Email": 23, "File-Transfer": 43, "P2P": 100, "Streaming": 104, "VoIP": 48}` |

Support count is not the blocking issue. Mapping identifiability is.

## Frozen-prediction diagnostics (not F3)

- F1-like existing frozen shared31 subset: N=335, Accuracy=0.802985, Macro-F1=0.770803, Weighted-F1=0.817823.
- Partial F2 on the N=321 deterministically mappable rows: Accuracy=0.878505, Macro-F1=0.840611, Weighted-F1=0.880973.
- Complete F2 coverage: 95.82%; 14 rows are outside the deterministic mapping domain.
- Of 66 Fine errors, 22 become Coarse-correct, 39 remain Coarse-wrong, and 5 are unmappable. Thus 33.33% of all Fine errors, or 36.07% of mappable Fine errors, are within-service errors.

These numbers are hierarchical evaluation of an existing Fine model. They are not evidence for the performance of a separately trained Service classifier.

## Preregistered protocol that would be required after resolving the semantic definition

1. Freeze a service-native label definition at capture level before training.
2. Use exactly the same shared31 flow IDs and existing Known Train/Validation membership for Fine and Service tasks.
3. Do not use Known Test or Unknown Test for mapping, normalization, checkpoint selection, or model choice.
4. Specify how multi-service applications are represented. A valid option must preserve service information in the prediction label space; a single application label such as Facebook is insufficient for full F2.
5. Re-run F1 and F3 with identical architecture/training budget, then compare complete-population F2 and F3.
6. Before promoting Service to a formal task, pass DQ-6 capture-group feasibility and redefine open-set semantic isolation at service level.

## Decision

- `COARSE_TASK_SUPPORTED`: **not established**.
- `FINE_TASK_SUPPORTED`: **not adjudicated in DQ-3**.
- New training started: **NO**.
- DQ-4 through DQ-7 started: **NO**.
