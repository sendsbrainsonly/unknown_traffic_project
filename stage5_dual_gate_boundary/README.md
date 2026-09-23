# Stage 5 — Dual-Gate Support Boundary Freeze

## Motivation

Stage 3 showed that a K2 density model can improve Known fit without reliably
improving Unknown rejection. Stage 4 then showed that one global threshold has
heterogeneous Known coverage, while pure Component-P05 can remove the absolute
density constraint. Stage 5 freezes a simple AND rule that retains both signals.

## Stage3/4 Evidence

- Stage 3 Fixed Multi-Full-K2: Gate D — MIXED.
- Failure diagnosis: Diagnosis A — DENSITY-UTILITY MISMATCH.
- Stage 4: class/component coverage under one global threshold is heterogeneous.
- Stage 4: pure local calibration improves component coverage consistency but
  increased USTC post-hoc UFAR in A-1 and A-3.

USTC Unknown has already been observed and is not used anywhere in Stage 5.

## DGSB Definition

For frozen PCA representation `z` and each frozen per-class K2 full-covariance
GMM:

`s_yk(z) = log(w_yk) + log N(z | mu_yk, Sigma_yk)`

`G(z) = max_y logsumexp_k s_yk(z)`

`(y*, k*) = argmax_y,k s_yk(z)` and `L(z) = s_y*k*(z)`.

DGSB accepts Known if and only if:

`G(z) >= tau_global_dual AND L(z) >= tau_local_y*k*`.

The predicted class is `y*`. There is no weighted fusion, learned boundary,
temperature, or Adaptive K.

## Known-Only Calibration

Local thresholds are Known Validation P05 values after true-class posterior
component assignment. After the Local Gate is fixed, the Global Gate threshold
is selected only from Known Validation absolute scores to make final AND
acceptance closest to 95%; equal-error thresholds use the higher value.

The frozen USTC Known-only sanity result is:

| Setting | `tau_global_dual` | Validation acceptance | Known Test FRR | Validation mean component gap |
|---|---:|---:|---:|---:|
| A-1 | -22.984130 | 0.949999 | 0.049021 | 0.058899 |
| A-2 | -71.731738 | 0.949990 | 0.050517 | 0.060669 |
| A-3 | -96.222612 | 0.949461 | 0.050071 | 0.057567 |

These are implementation/coverage checks, not Unknown Detection results.

## Rule Freeze

The frozen constants are K=2, full covariance, `reg_covar=1e-3`, PCA-64,
local quantile 0.05, minimum local validation count 30, class-P05 fallback, and
strict AND logic. The machine-readable rule and its SHA-256 are under
`outputs/rule_freeze/`.

## Why USTC Is Not Independent Validation

USTC Unknown results informed the Stage 3/4 research question. Stage 5 therefore
uses only USTC Known Validation and Known Test as an implementation sanity check.
No USTC Unknown flow, label, score, prediction, transition, or UFAR file may be
read by the Stage 5 calibration program.

## CSTNET Dataset

CSTNET-TLS1.3 supplies 46,372 readable per-flow PCAPs in 120 domain/service
classes. Stage 5 reads the completed suitability audit and only the fixed-size
PCAP header required to construct a capture-time grouping proxy.

## Eligible Classes

Eligibility is decided only by raw sample count. Counts for thresholds 50, 100,
200, and 500 are reported. The default engineering rule is at least 100 PCAPs
per class, selected before any model is trained or evaluated.

## Group-Aware Split

No authoritative session manifest is present. The protocol freezes the UTC
one-minute window of the first packet as a deterministic capture-group proxy.
Known samples sharing a group with a held-out Unknown class are excluded, then
remaining Known groups are split approximately 80/10/10 using a deterministic
10-fold StratifiedGroupKFold procedure.

This control is stricter than random-PCAP splitting but does not prove true
session independence. Domain labels and unresolved endpoint metadata remain a
declared `PROTOCOL_RISK`.

## Open-Set Protocol

Held-out Unknown classes are forbidden from encoder training/validation,
checkpoint selection, scaler/PCA/GMM fitting, boundary calibration, and tuning.
They are reserved for a future final test after all rules and splits are frozen.

## Frozen External Folds

Low, Medium, and High class-held-out folds use seeds 42, 43, and 44. Candidate
folds may be rejected only when group-overlap purging leaves a Known class below
the predeclared sample/group minimum; every rejection is recorded.

The final eligibility rule retains 119 of 120 classes and excludes only
`chia.net` (16 PCAPs). The Low/Medium/High folds contain 6/18/30 Unknown
classes. Their active Train/Validation/Test/Unknown capture-group overlap is
zero and all split hashes pass. The High fold uses the fourth deterministic
candidate from seed 44 after three count-only data-quality rejections.

## Next Experiment

The next task may execute the frozen CSTNET protocol. Stage 5 itself stops after
method/protocol freeze and does not train, infer, or inspect Unknown scores.
