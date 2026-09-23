# RESULTS — Stage 15F-1A

Status: complete
Final Gate: `CLASS_CONDITIONAL_WINDOW_BENEFIT`

## Core results

T16 is the strongest longer window: mean validation Macro-F1 delta versus T8 is +0.015363, with 5/5 positive protocols and worst delta +0.005718.
T32 mean delta versus T8 is -0.002620; its worst protocol delta is -0.056010.
Across 52 class-protocol units, T16 produces 23 positive, 23 negative, and 6 unchanged deltas; median delta is +0.000000.

The detailed evidence is in `window_pilot_results.csv`, `window_paired_comparison.csv`, `window_per_class_results.csv`, `window_confusion_analysis.csv`, `window_error_complementarity.csv`, and `window_coverage_vs_performance.csv`.

## Configuration and execution

Formal T8/T16/T32 runs use the same signed log frame length, log IAT, valid-packet mask, mask-safe CNN, cross-entropy objective, optimizer, seed, checkpoint rule, and 100-epoch budget. Only maximum packet count changes. All 15 runs completed and retained histories, predictions, per-class metrics, confusion matrices, and hashed best checkpoints.

## Data and split

The five frozen pilots are ISCX-VPN medium-2022, ISCXTor medium-2022, VNAT medium-2025, VNAT medium-2026, and USTC A-2. Normalization is fit on Known Train only. Known Test and Unknown Test feature usage are both zero.

## Preserved evidence

The bundle preserves the first-32-packet caches and audits, formal runs, legacy T8 parity evidence, pre/post frozen-asset hashes, failed direction-reconstruction and CPU-parity attempts, aggregation outputs, tests, logs, and `completion_verification.json`.

## Limitations

This is a five-protocol Known-Validation pilot with one fixed seed, not a final Test or open-set claim. Historical Stage 15R E2 used padding-sensitive BatchNorm, whereas the formal controlled comparison uses one mask-safe implementation; historical-to-formal T8 changes are not window effects.

## Conclusion and next step

The final Gate is `CLASS_CONDITIONAL_WINDOW_BENEFIT`. Retain T16 only as a class-conditional temporal candidate; do not promote T32 or enter T64. Stop here. A later preregistered experiment may compare richer statistics, burst structure, or structured byte inputs, but no such experiment was started.

No Open-Detect, DES, H1, T64, or multiview training was run.
