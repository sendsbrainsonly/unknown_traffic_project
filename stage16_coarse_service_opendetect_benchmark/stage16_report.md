# Stage 16 — Coarse Service Open-Detect benchmark

## Terminal status

`BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`

The data parity Gate passed, but the method identity Gate failed. DQ-3F's F3
trainer directly instantiates Open-Detect's `CorrectedOpenDetectNet`, imports
Open-Detect's training loop and prototype reset, and uses its VAE/prototype loss.
Consequently, the requested independent `OURS-Service6` versus
`OpenDetect-Service6` experiment is not scientifically identifiable.

## Data audit

- Protocol A: 2,730 Known Train + 335 Known Validation = 3,065 flows.
- Protocol C: 1,367 Train + 184 Validation = 1,551 flows.
- Classes: Chat, Email, File-Transfer, P2P, Streaming, VoIP.
- Labels: 3,065/3,065 `WEAK_CAPTURE_LABEL`.
- Known Test / Unknown Test feature usage: 0 / 0.

## Existing DQ-3F F3 evidence (correctly identified)

These are real historical results but belong to the corrected Open-Detect
reproduction, not an independent new method:

| Seed | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 2022 | 0.910448 | 0.898074 | 0.909877 |
| 2023 | 0.832836 | 0.765339 | 0.825692 |
| 2024 | 0.928358 | 0.904738 | 0.928541 |


Mean ± population std:

- Accuracy: `0.890547 ± 0.041458`
- Macro-F1: `0.856050 ± 0.064200`
- Weighted-F1: `0.888036 ± 0.044738`

## Required questions

1. **Our method result:** no independently identified method result exists in
   DQ-3F. The reported 0.856050 Macro-F1 is corrected
   Open-Detect evidence.
2. **New Open-Detect result:** `NOT_RUN_IDENTITY_GATE`; running it would compare
   two Open-Detect variants, not the requested two method entities.
3. **Metric difference:** not identifiable; no deltas are fabricated.
4. **Seed stability:** the historical corrected variant is seed-sensitive
   (Macro-F1 std 0.064200; seed 2023 is weakest).
5. **Per-Service winner:** not identifiable without a second distinct method.
6. **Open-Detect rescue:** not identifiable.
7. **Error complementarity:** not run because paired independent predictions do
   not exist.
8. **C protocol:** historical DQ-4 A/C selection evidence is retained, but no
   two-method C comparison was run.
9. **Limitations:** weak labels and the single P2P capture remain material.
10. **Cause attribution:** current evidence only establishes a corrected-versus-
    released implementation distinction, not a representation/supervision win
    for a new method.
11. **Service open set:** blocked pending a newly frozen Service-level test and
    capture-aware protocol.
12. **Byte–Behavior:** its possible value cannot be decided from this invalid
    two-method premise; the existing Stage 14C-6 F2 own-method is the proper
    independent starting point if explicitly selected in a new preregistration.

## Gates

- `COARSE_SERVICE_BENCHMARK_COMPLETED`: **NO**
- `METHOD_IDENTITY_NOT_DISTINCT`: **YES**
- `OURS_SERVICE_ADVANTAGE_OBSERVED`: **NOT_IDENTIFIABLE**
- `OPENDETECT_SERVICE_ADVANTAGE_OBSERVED`: **NOT_IDENTIFIABLE**
- `SERVICE_OPEN_SET_PROTOCOL_BLOCKED`: **YES**

No GPU was selected or used, no encoder was trained, no legacy checkpoint or
prediction was modified, and no open-set experiment was started.
