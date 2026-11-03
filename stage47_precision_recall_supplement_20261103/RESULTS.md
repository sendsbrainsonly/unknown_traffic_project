# Experiment results: stage47-iscx-known-test-precision-recall-20261103

- Status: `success` / `PREDICTION_AND_METRIC_REPLAY_PASS`
- Experiment type: `posthoc-metric-supplement`
- Claim scope: `diagnostic`
- Objective: Supplement frozen ISCX-VPN/Tor closed-set Precision and Recall.

## Data and split

VPN6: 8,764 / 1,098 / 1,093; Tor7: 8,946 / 1,118 / 1,117 (Train/Val/Test). Seeds2022/2023; unchanged Stage20 membership.

## Configuration and execution

CPU metric recomputation from existing Known-Test predictions; no model execution, fitting, threshold or checkpoint selection. Fixed workspace Conda, tmux. Macro/Weighted/Micro; zero_division=0; population SD ddof=0.

## Core results

32/32 runs, 16 summaries, 208 per-class rows. All original 96 run-level Accuracy/F1 values and 96 summary cells reproduced within 1e-12. See [full comparison table](table_precision_recall.md).

## Preserved evidence

recompute.py; run_metrics.csv; method_summary.csv; per_class_metrics.csv; confusion_matrices.csv; source_audit.csv; source_hashes_before/after.json; completion_verification.json; manifest.json; project-local tmux logs referenced by manifest. Original source files retained in place and SHA256 registered.

## Limitations

Two-seed development evaluation with exposed Test; baseline reconstruction/adaptation identities preserved. VPN6/Tor7 cannot be substituted for later coarse4/5 tasks. Averaged Macro-F1 is not the harmonic mean of averaged Precision/Recall.

## Conclusion and next step

Precision/Recall supplemental computation complete; no new training required. No further experiment started.

## Manuscript integration (2026-11-04, Asia/Shanghai)

The supplemental metrics are integrated into current chapter4 table4-5 and README links. The full table is unified to the same two-seed mean ± population SD and includes the early E3 configuration. The original chapter/README/manifest snapshots, source checks and80-cell table verification are retained in `publication_20261104/`. See the chapter revision record for exact corrections. No original prediction or frozen experiment file is changed.
