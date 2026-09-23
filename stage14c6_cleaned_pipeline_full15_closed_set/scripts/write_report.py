#!/usr/bin/env python3
"""Write the formal Stage 14C-6 closed-set validation report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONCLUSION = "CLOSED_SET_PASS_WITH_HARD_PROTOCOLS"


def md(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    output = frame[columns].copy()
    for column in output.select_dtypes(include="number").columns:
        output[column] = output[column].map(
            lambda value: f"{value:.{digits}f}" if pd.notna(value) else ""
        )
    return output.to_markdown(index=False)


def main() -> None:
    runs = pd.read_csv(ROOT / "run_level_results.csv")
    settings = pd.read_csv(ROOT / "setting_summary.csv")
    paired = pd.read_csv(ROOT / "paired_vs_native.csv")
    paired_summary = pd.read_csv(ROOT / "paired_vs_native_summary.csv")
    classes = pd.read_csv(ROOT / "class_summary.csv")
    hard = pd.read_csv(ROOT / "hard_class_vs_native_summary.csv")
    parity = pd.read_csv(ROOT / "stage14c4_parity.csv")

    run_view = runs[[
        "protocol_id", "best_epoch", "train_loss", "val_loss", "val_accuracy",
        "val_macro_f1", "val_weighted_f1", "weak_run", "checkpoint_sha256",
    ]]
    setting_view = settings[[
        "setting", "runs", "val_accuracy_mean", "val_accuracy_std",
        "val_macro_f1_mean", "val_macro_f1_std", "val_weighted_f1_mean",
        "val_weighted_f1_std", "weak_runs",
    ]]
    pair_view = paired[[
        "protocol_id", "native_accuracy", "val_accuracy", "delta_accuracy_our_minus_native",
        "native_macro_f1", "val_macro_f1", "delta_macro_f1_our_minus_native",
        "native_weighted_f1", "val_weighted_f1", "delta_weighted_f1_our_minus_native",
    ]]
    pair_summary_view = paired_summary[[
        "setting", "runs", "delta_accuracy_mean", "our_wins_accuracy",
        "delta_macro_f1_mean", "our_wins_macro_f1", "delta_weighted_f1_mean",
        "our_wins_weighted_f1",
    ]]
    class_view = classes[[
        "class", "runs", "recall_mean", "recall_std", "f1_mean", "f1_std",
    ]]

    report = f"""# Stage 14C-6 — Cleaned Pipeline Full 15-Run Closed-Set Validation

## Final conclusion

```text
{CONCLUSION}
```

The frozen cleaned F2 pipeline completed all 15 Stage 14B protocols without a crash, NaN, early stop, Test/Unknown access, or DES execution. Its overall Known-Validation performance is effectively tied with Native Open-Detect on Accuracy and Weighted-F1 and is lower by `0.009875` Macro-F1 on average. The remaining variance is concentrated in frozen class compositions rather than an observed training implementation failure.

## Integrity

- Runs completed: `15/15`; every run completed `100/100` epochs.
- Verified best checkpoints and SHA256 values: `15/15`.
- Stage 14B freeze hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Known Test samples used: `0`.
- Unknown Test samples used: `0`.
- DES executed: `false`.
- Weak runs under the frozen rule (`Macro-F1 < 0.50` or `Accuracy < 0.60` or NaN/crash): `0`.
- Medium-2025/2026 exactly reproduce the Stage 14C-4 selected metrics to `1e-12`, confirming wrapper/configuration reproducibility.

## All 15 formal runs

{md(run_view, list(run_view.columns))}

## Low / Medium / High summary

Values are mean and sample standard deviation across five frozen protocols.

{md(setting_view, list(setting_view.columns))}

Overall cleaned-pipeline mean ± std:

- Accuracy: `0.868512 ± 0.096451`
- Macro-F1: `0.821029 ± 0.109748`
- Weighted-F1: `0.858092 ± 0.105427`

## Paired comparison with existing Native Open-Detect

Native models were not retrained. Each row compares the same frozen protocol and Known-Validation samples. Each method retains its previously frozen training-seed policy, so the comparison is paired by protocol/data but is not a single-seed architecture ablation.

{md(pair_view, list(pair_view.columns))}

### Paired summary

{md(pair_summary_view, list(pair_summary_view.columns))}

Overall Our-Cleaned − Native:

- Δ Accuracy: `+0.000451`; Our-Cleaned wins `7/15`, ties `1/15`.
- Δ Macro-F1: `−0.009875`; Our-Cleaned wins `8/15`.
- Δ Weighted-F1: `−0.001386`; Our-Cleaned wins `8/15`.

This is best described as **comparable**, not a consistent superiority claim. Medium-2024 (`Δ Macro-F1 = −0.175685`) and Low-2025 (`−0.100601`) account for most of the aggregate Macro-F1 deficit. Positive examples include Low-2024 (`+0.053280`), Low-2026 (`+0.043697`) and Medium-2023 (`+0.037837`).

## Per-class result

{md(class_view, list(class_view.columns))}

### Difficult classes versus Native

{md(hard, ['class', 'runs', 'our_recall_mean', 'native_recall_mean', 'delta_recall_mean', 'our_f1_mean', 'native_f1_mean', 'delta_f1_mean'])}

- `sftp` remains the lowest-recall class (`0.451807`), but is only `−0.019277` below Native Recall.
- `rsync` remains highly composition-sensitive (Recall std `0.401645`) and is nearly tied with Native on its cross-protocol mean.
- `scp` and `rdp` improve over Native on mean Recall by `+0.034061` and `+0.113636` respectively.
- High-support stable classes remain `skype`, `ssh`, `vimeo`, and `zoiper`.

## Hard protocols and composition

- **Medium-2025 remains a hard composition:** Macro-F1 `0.727247`; rsync Recall `0.031414` because rsync and scp are simultaneously Known. It exactly reproduces Stage 14C-4 and is only `−0.012508` below Native, so this is reproducible composition difficulty rather than a new run failure.
- **Low-2025:** Macro-F1 `0.581333`; it contains rsync, scp, and sftp simultaneously. The largest losses versus Native occur on netflix and rsync, partly offset by higher scp/rdp performance.
- **Medium-2024:** Macro-F1 `0.677427`; its deficit versus Native is concentrated in netflix, youtube, rdp, and sftp.
- Across the 15 frozen protocols, simultaneous `rsync/scp` presence is descriptively associated with mean Macro-F1 `0.738908` versus `0.875776` when they are not both Known. This association is confounded by other class-composition changes and is not treated as a causal estimate.

## Required answers

1. **Stable across 15/15?** Operationally yes: 15/15 completed, no crash/NaN, all hashes and frozen-data gates passed. Performance variance remains substantial across protocols.
2. **Overall performance?** Accuracy `0.868512`, Macro-F1 `0.821029`, Weighted-F1 `0.858092`.
3. **Versus Native Open-Detect?** Comparable overall: Accuracy `+0.000451`, Macro-F1 `−0.009875`, Weighted-F1 `−0.001386`; Macro-F1 wins on 8/15 protocols.
4. **Composition effect?** Yes, the largest repeatable weaknesses are concentrated in protocols containing difficult combinations such as rsync/scp/sftp and small-support rdp, although pairwise composition statistics are descriptive rather than causal.
5. **Remaining implementation issue?** None observed under the requested checks. Medium-2025/2026 exact metric parity with Stage 14C-4 provides a direct reproducibility check.
6. **Can the cleaned pipeline be frozen?** Yes, with hard-protocol behavior explicitly documented.
7. **Can it enter open-set evaluation?** Yes from the closed-set and integrity gates. This stage did not read Unknown Test or start open-set evaluation.

## Stage 14C-4 reproducibility check

{md(parity, list(parity.columns))}
"""
    (ROOT / "stage14c6_report.md").write_text(report, encoding="utf-8")
    print(ROOT / "stage14c6_report.md")


if __name__ == "__main__":
    main()
