# Stage 4 Local Boundary Diagnosis

## Validity boundary

This is POST-HOC USTC MECHANISM DIAGNOSIS. A-1/A-2/A-3 Unknown Test has already been observed repeatedly. USTC results below are not independent performance validation.

## Prediction reproduction

- Native, Single, and Multi scores/classes/binary decisions: `100% parity` over 146,733 setting-sample rows; maximum replay score errors are recorded in `run_metadata.json`.

## Global coverage heterogeneity

- Largest Known-Validation component acceptance gap: A-3/Virut = 0.889119.
- Mean component gap, Global/Class/Local on validation: 0.169174/0.234170/0.072024.
- Mean component gap, Global/Class/Local on Known Test: 0.171906/0.228309/0.070406.

- Class-P05 makes per-class coverage much more uniform but does not equalize the two component coverages; Component-P05 materially reduces the mean component gap on both Known Validation and Known Test.
- Component-P05 validation coverage is partly calibration-by-construction and must be judged together with Known Test generalization and the fixed `<30` fallback rows.

## Overall Known FRR

- A-1 Validation Global/Class/Local: 0.050001/0.047795/0.038682; Test: 0.047419/0.046025/0.036974.
- A-2 Validation Global/Class/Local: 0.050010/0.049110/0.041402; Test: 0.049386/0.048186/0.041794.
- A-3 Validation Global/Class/Local: 0.050021/0.049865/0.035385; Test: 0.048595/0.047973/0.035177.

## Htbot–Tinba case

- Focus rows: 86. Median global/class/local margins: 7.608549/42.333714/59.266630.
- Class-P05 rejects 0/86; Component-P05 rejects 0/86.
- Htbot component counts: {"1": 86}; median assigned-component validation percentile: 0.292899.
- The 86 Tinba are inside Htbot component 1 rather than merely crossing a badly scaled global threshold; both calibrated rules make their margins more positive. Local calibration does not explain or repair this failure.

## Virut–Neris case

- Focus rows: 236. Median global/class/local margins: -5.401150/16.895042/45.602908.
- Class-P05 rejects 10/236; Component-P05 rejects 0/236.
- Virut raw component assignments: {"0": 179, "1": 57}; median assigned-component validation percentile: 0.616511.
- Component-P05 normalized-margin predictions: {"Virut/component-0": 236}. The rule reassigns all focus rows to Virut component 0 and removes the Stage 3 rejection mechanism.
- Thus the Stage 3 Virut gain primarily comes from the stringent global threshold against its lower-score-scale support, not from a clean between-component low-density gap that Component-P05 preserves.

## Threshold-factor association

- Strongest pooled association with local-minus-class threshold: `model_weight`, rho=0.681106, p=3.34884e-15, n=102. Association is Known-only and diagnostic.

## Post-hoc Unknown results

- A-1 Global/Class/Local UFAR: 0.623529/0.784706/0.858824 (`POST_HOC_DIAGNOSTIC_ONLY`).
- A-2 Global/Class/Local UFAR: 0.620900/0.560315/0.544721 (`POST_HOC_DIAGNOSTIC_ONLY`).
- A-3 Global/Class/Local UFAR: 0.093538/0.218222/0.285368 (`POST_HOC_DIAGNOSTIC_ONLY`).

## Final diagnosis

**Diagnosis C — LOCAL BOUNDARY NECESSARY.**

This diagnosis is about Known component-coverage consistency, not demonstrated Unknown-detection benefit. The observed USTC post-hoc results show substantial utility risk and are not used to tune or reselect the frozen P05 formula.

Future candidate: **Component-P05**. This candidate was not independently validated here. Its first formal test must be a pre-frozen CSTNET-TLS1.3 strict Unknown-Free protocol.
