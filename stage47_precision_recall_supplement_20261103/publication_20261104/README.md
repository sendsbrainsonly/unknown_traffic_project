# Precision/Recall manuscript integration and GitHub handoff

Date: 2026-11-04, Asia/Shanghai. Agent: [agent:codex].

The existing `method_summary.csv` is integrated into chapter4 table4-5. All eight configurations use seeds2022/2023, VPN6/Tor7, mean ± population standard deviation. Macro Precision/Recall are added alongside the original Accuracy and F1 metrics. Weighted P/R remain available in the complete supplemental table/CSV.

`chapter4.before.md`, `project_readme.before.md`, and `manifest.before-publication.json` preserve the prior state. `snapshot.json` records original hashes. `check_integration.py verify` checks all80 table cells, unchanged source metrics and unrelated manuscript sections. `verification.json` is the verification receipt. The root manifest inventories this directory with the metric supplement.

Publication includes code, small metric/summary/per-class tables, documentation and verification evidence. Model weights, packet/feature caches, raw data, per-sample predictions, runtime logs and unrelated Stage48 work are excluded from this release.
