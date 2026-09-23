# Experiment results: stage3-failure-diagnosis-20260912

- Status: `success`
- Experiment type: `post-hoc-diagnosis`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-12T23:55:56Z`
- Objective: Explain class-dependent Fixed-K2 open-set utility using frozen Stage 3 outputs and Known-only diagnostics

## Data and split

- Stage 3 A-1/A-2/A-3 frozen outputs and frozen `mu_x` artifacts.
- Known Train/Validation were used for geometry and candidate-signal diagnostics only.
- Already-observed Unknown Final Test predictions were used for post-hoc attribution; frozen Unknown `mu_x` was used only to recover component assignments with exact score replay parity.

## Configuration and execution

- Command: `python stage3_failure_diagnosis/scripts/analyze_failure.py --output-dir stage3_failure_diagnosis/outputs`
- tmux: `stage3_failure_diagnosis_formal_v2`; exit code `0`; unit tests `4 passed`.
- Fixed Conda prefix: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`.
- No fitting, threshold/K change, random sampling, GPU workload, or Final Test rerun.

## Core results

- A-1: Htbot alone changes absorption by `+86` (Tinba 444→530); all 86 regressions map to Htbot K2 component 1.
- A-2: Miuref `+16` and Shifu `−1`, net `+15/5,579`; 63 SM and 48 MS transitions nearly cancel.
- A-3: Virut alone changes absorption by `−236` (Neris 1,198→962); no SM regressions.
- A-1/Htbot improves Known-validation NLL by `14.075196` despite worse Unknown absorption; A-2/Miuref shows the same direction (`15.435538`, `+16`).
- Pooled DeltaNLL2 vs DeltaAbsorb Spearman is `rho=−0.021545`, `p=0.880711`, `n=51`: no useful monotonic relationship.
- All 51 setting-class cells have no `<1%` tiny or validation-empty K2 component; Htbot TV is `0.003103`.
- Final: **Diagnosis A — DENSITY-UTILITY MISMATCH**.

## Preserved evidence

- `manifest.json`
- `provenance_snapshot.md`, `audit_summary.md`, and `candidate_hypotheses.md`
- Nine independently reviewable CSV/Parquet diagnosis tables, including 16,735 per-Unknown transitions.
- Formal tmux logs: `.tmux-task/stage3_failure_diagnosis_formal_v2/` (project-local external reference).

## Limitations

- The A-1/A-2/A-3 Final Test was already observed; all correlations and candidate rules are post-hoc and exploratory.
- DeltaAbsorb is sparse and zero-inflated (only four non-zero setting-class cells), so correlations have low inferential power.
- Top-3 lists contain deterministic zero-delta ties; only Htbot in A-1 and Virut in A-3 are non-zero causal contributors.
- Simple-statistics tests show association, not semantics or causal effect on Unknown utility.

## Conclusion and next step

- Fixed K2 density fit quality is not a reliable surrogate for open-set utility in these frozen settings.
- Candidate R1/R2/R3 were documented but not executed. Any future rule must be frozen before evaluation on an untouched benchmark, prioritizing CSTNET-TLS1.3 and then CipherSpectrum.
