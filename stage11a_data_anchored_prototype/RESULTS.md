# Experiment results: stage11a-data-anchored-prototype-20260914-v1

- Status: `success`
- Experiment type: `method-development`
- Claim scope: `diagnostic`
- Objective: Paired USTC-only feasibility study of every-epoch data-anchored Open-Detect prototypes.

## Data and split

- Frozen v6 USTC A1/A2/A3, five paired seeds 2022–2026.
- Repeated grouped-image-disjoint 8:1:1 local splits; not author-exact folds.
- Known Train only for anchoring and density fitting; Known Validation only for checkpoint selection and P95 thresholds.

## Configuration and execution

- B0 reused 15 frozen checkpoints; DAP trained 15 paired runs.
- DAP prototypes are optimizer-free registered buffers and are anchored before epoch 1 and after every epoch.
- Native and fixed K1/K2 diagnostics only; no threshold or K search.

## Core results

- Final gate: **NO_GO**; mechanism: **P3**.
- Scenario mean DAP-Native minus B0-Native AUROC: A1=-0.167800, A2=-0.068712, A3=-0.036409.
- DAP-better directions: A1=0/5, A2=0/5, A3=1/5.
- Covariance verdict: **COVARIANCE_REQUIREMENT_NOT_STABLE**.
- Multi-component verdict: **MULTICOMPONENT_SECONDARY**.

## Preserved evidence

- `outputs/summary/*.csv`, `mechanism_case.md`, `final_gate.md`, and `provenance_verification.json`.
- Per-run configs, logs, drift history, frozen best checkpoints/hashes, results, density models, RESULTS, and manifests.

## Limitations

- `DEVELOPMENT_RESULT`; `NOT_INDEPENDENT_EXTERNAL_VALIDATION`.
- CipherSpectrum Stage 9 sample-level Test was not used. USTC scenarios are development scenarios.

## Conclusion and next step

Stage 11A terminates at **NO_GO**. No subsequent stage or final covariance-aware method was started.
