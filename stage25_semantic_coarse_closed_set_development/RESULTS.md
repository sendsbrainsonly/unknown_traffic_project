# Experiment results: stage25-semantic-coarse-closed-set-20260924-v1

- Status: `success / SEMANTIC_COARSE_BENCHMARK_COMPLETE`
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-24T15:24:00Z`
- Objective: Evaluate semantically coarser labels on matched frozen Stage20 flows while preserving original fine-class model capability; separately audit >=2-packet subset and frozen-encoder coarse heads

## Data and split

- Stage20 frozen ISCX-VPN 6-Service (10,955 flows; Train/Val/Test 8,764/1,098/1,093) and ISCXTor 7-Service (11,181; 8,946/1,118/1,117). Flow IDs, roles and original fine labels were not changed.
- Fixed semantic map: Chat+Email+VoIP → Communication, with all other Services unchanged. This yields VPN 4 and Tor 5 coarse classes on **all original flows**. The ≥2-packet scope is a separately labeled conditional analysis, retaining 807/1,093 VPN and 595/1,117 Tor Known Test flows.

## Configuration and execution

- Phase A: hard-remap frozen predictions from seven methods on identical Validation/Test flow IDs; no model fitting. Phase B: fit 8 additive multinomial logistic coarse heads (E1/E3 × 2 datasets × seeds 2022/2023) with a Known-Train-only StandardScaler, `C=1`, `lbfgs`, `max_iter=200`, no search; original encoder and fine head stay frozen. Phase C: report the same predictions on the ≥2-packet subset without retraining.
- Named tmux sessions: `codex_stage25_phase_a_retry1_20260924`, `codex_stage25_phase_b_20260924`, `codex_stage25_finalize_20260924`, `codex_stage25_independent_verify_20260924`; all exit 0. The first A attempt failed on a TFE validation filename assumption; its log is preserved in `.tmux-task/codex_stage25_phase_a_20260924/`. Retry used the previously frozen TFE `selection.json` and did not reselect checkpoints. No GPU was needed.

## Core results

- Full-flow Known Test Macro-F1, two-seed mean:

| Dataset | Method | Original fine | Coarse hard remap | Additive coarse head |
| --- | --- | ---: | ---: | ---: |
| VPN | Our E3 | 0.859493 | 0.896004 | 0.897301 |
| VPN | TrafficFormer E1 | 0.860607 | 0.901497 | 0.896161 |
| VPN | YaTC | 0.887085 | 0.923689 | — |
| Tor | Our E3 | 0.822178 | 0.852783 | 0.854317 |
| Tor | TrafficFormer E1 | 0.821259 | 0.852637 | 0.853024 |
| Tor | YaTC | 0.827393 | 0.876259 | — |

- E3 coarse-head Accuracy/Macro-F1/Weighted-F1: VPN `0.890668/0.897301/0.890396`; Tor `0.851388/0.854317/0.851888`. All seven methods and both scopes are in `stage25_report.md` and `stage25_summary.csv` (including per-seed rows and population standard deviations).
- ≥2-packet conditional E3 coarse hard-remap Macro-F1 is VPN `0.878272` (**lower** than all-flow 0.896004) and Tor `0.880108` (**higher**, but only 53.27% of Test retained). Thus sample filtering is not a universal improvement.
- Integrity: 28/28 historical fine Test metric rows replayed; 48/48 protected source SHA256 unchanged. Independent replay passed all 61,964 remapped predictions, 224 metric rows, 8 head numeric artifacts and 17,704 head decisions. Unknown use=0; Test fit/selection use=0; no convergence warnings.

## Preserved evidence

- `EXPERIMENT_PLAN.md`, `run_stage25.py`, `verify_stage25.py`, `stage25_report.md`, `remap_*`, `coarse_head_*`, `heads/`, `scope_coverage.csv`, `stage25_summary.csv`, protected hashes, all phase and independent verification JSON, original tmux logs/status files, and `manifest.json`.

## Limitations

- The Stage20 Known Test was already exposed before this stage, so these are **development** results, not independent validation. Coarse labels define a different task; their higher numbers do not improve the original 6/7-class classifier. Hard remapping of argmax labels is not probability pooling. The new E1/E3 coarse heads are additional supervised training and should not be ranked against hard-remap-only methods as if training were identical.
- Service labels are capture-derived weak labels; Stage20 is not uniformly capture-disjoint. Two seeds are insufficient for strong stability claims. The ≥2-packet subset changes class composition and covers only 53.27% of Tor Test.

## Conclusion and next step

- Coarser semantic labels raise absolute closed-set scores, but E3 still does not overtake the matched YaTC baseline. The additive coarse head preserves the original fine head and gives only a small E3 gain over remapping (`+0.001298` VPN, `+0.001535` Tor Macro-F1). Keep original fine-task results alongside the new coarse task; do not use ≥2-packet filtering as the main result or claim encoder improvement. No further training is automatically launched.
