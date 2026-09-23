# Experiment results: stage16-coarse-service-opendetect-benchmark-20260920-v1

- Status: `partial / BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`
- Experiment type: `method-identity-and-closed-set-benchmark`
- Claim scope: `diagnostic`

## Data and split

- Protocol A: 2,730 Known Train + 335 Known Validation = 3,065 flows.
- Protocol C: 1,367 Known Train + 184 Known Validation = 1,551 flows.
- Classes: Chat, Email, File-Transfer, P2P, Streaming, VoIP.
- Labels: 3,065/3,065 `WEAK_CAPTURE_LABEL`.
- Known Test / Unknown Test feature usage: `0 / 0`.
- A/C flow membership and Service labels pass hash parity; see `dataset_parity_audit.csv`.

## Configuration and execution

- Stage 16 first applied the preregistered method-identity Gate.
- DQ-3F directly imports and instantiates Open-Detect's
  `CorrectedOpenDetectNet`, `run_epoch`, `reset_prototypes_in_place`, and
  `weight_init`.
- The DQ-3F frozen configuration records `training_protocol=corrected-paper`.
- Therefore the candidate called `OURS-Service6` is a corrected Open-Detect
  reproduction, not an independently identified method.
- New Open-Detect training runs: `0`; GPU selected/used: `NO`.

## Core results

The following frozen DQ-3F F3 results are preserved with their actual identity
`CorrectedOpenDetectNet_DQ3F`; they are not presented as an independent method.

| Seed | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 2022 | 0.910448 | 0.898074 | 0.909877 |
| 2023 | 0.832836 | 0.765339 | 0.825692 |
| 2024 | 0.928358 | 0.904738 | 0.928541 |

Mean ± population standard deviation:

- Accuracy: `0.890547 ± 0.041458`
- Macro-F1: `0.856050 ± 0.064200`
- Weighted-F1: `0.888036 ± 0.044738`

The new baseline and all two-method deltas are
`NOT_RUN_IDENTITY_GATE / NOT_IDENTIFIABLE`; no values were fabricated.

## Preserved evidence

- `method_identity_audit.md`: direct code provenance and source hashes.
- `matched_service_manifest.csv`: 3,065-row frozen A manifest plus C flags.
- `dataset_parity_audit.md` and `.csv`: A/C counts and canonical hashes.
- `ours_service_results.csv`: historical DQ-3F results with corrected identity.
- `opendetect_service_results.csv`: explicit `NOT_RUN_IDENTITY_GATE` rows.
- `per_service_results.csv` and `confusion_matrices/`: historical DQ-3F evidence.
- `protected_asset_hashes_before.json` / `after.json`: 19/19 unchanged.
- `completion_verification.json`: fail-closed completion state.

## Limitations

- The requested independent OURS-versus-Open-Detect comparison cannot be
  identified from DQ-3F because both are Open-Detect implementations.
- All Service labels are capture-derived weak labels.
- P2P has one independent capture, so capture-disjoint generalization is not
  identifiable.
- No independent Service-level Test protocol is frozen.

## Conclusion and next step

Final Gate: `BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`.

The scientifically valid next action requires a new preregistration selecting
one of two explicit questions:

1. compare **corrected Open-Detect versus released-code Open-Detect** as an
   implementation-variant study; or
2. select the genuinely separate Stage 14C-6 F2 own-method, rebuild it on the
   frozen Service A/C flows, and compare it with Open-Detect.

Stage 16 did not start Open-Set evaluation, DES/H1 changes, Byte–Behavior
training, or any new encoder run.
