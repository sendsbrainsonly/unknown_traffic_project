# Experiment results: stage4-local-boundary-diagnosis-20260913

- Status: `success`
- Experiment type: `post-hoc-mechanism-diagnosis`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-13T00:37:01Z`
- Objective: Diagnose global/class/component boundary coverage using frozen Stage 3 assets and freeze a CSTNET external-validation candidate

## Data and split

- Frozen Stage 3 A-1/A-2/A-3 `mu_x`, scaler, PCA64, K1/K2 density models, thresholds, and final predictions.
- Calibration used Known Validation only; Known Test was read after threshold freeze; USTC Unknown Test was read last for post-hoc mechanism diagnosis only.
- Replayed 146,733 setting-sample rows across Known/Unknown test populations.

## Configuration and execution

- Fixed rules: Stage 3 Global, true-class Class-P05, and posterior-assigned Component-P05.
- Fixed fallback: component validation count `<30` uses class P05; one WorldOfWarcraft component per setting triggered fallback.
- Command: `python stage4_local_boundary_diagnosis/scripts/run_boundary_diagnosis.py --output-dir stage4_local_boundary_diagnosis/outputs`.
- Formal tmux session: `stage4_boundary_formal_v3`; exit code `0`; unit tests `4 passed`.
- Environment: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`; CPU only.

## Core results

- Native/Single/Multi score, predicted class, and binary decision replay: 100% parity; maximum score error `0`.
- Largest Global Known-Validation component acceptance gap: A-3/Virut `0.889119` (`0.097204` vs `0.986323`).
- Mean component gap, Validation Global/Class/Local: `0.169174/0.234170/0.072024`; Known Test: `0.171906/0.228309/0.070406`.
- Known Test overall FRR, Global/Class/Local: A-1 `0.047419/0.046025/0.036974`; A-2 `0.049386/0.048186/0.041794`; A-3 `0.048595/0.047973/0.035177`.
- Htbot–Tinba focus: all 86 belong to Htbot component 1; median validation percentile `0.292899`; Class/Local reject `0/86`.
- Virut–Neris focus: raw component counts `179/57`; Class rejects 10/236, Local rejects 0/236 and maps all to Virut component 0.
- Strongest pooled local-minus-class threshold association: component weight, Spearman `rho=0.681106`, `p=3.35e-15`, `n=102`.
- USTC post-hoc UFAR Global/Class/Local: A-1 `0.623529/0.784706/0.858824`; A-2 `0.620900/0.560315/0.544721`; A-3 `0.093538/0.218222/0.285368`.
- Final: **Diagnosis C — LOCAL BOUNDARY NECESSARY**, restricted to Known component-coverage consistency.

## Preserved evidence

- `manifest.json`
- Seventeen required Stage 4 diagnostic outputs, including coverage, thresholds, case studies, leakage audit, and external protocol candidate.
- Project-local formal logs under `.tmux-task/stage4_boundary_formal_v3/`.

## Limitations

- USTC Unknown Test was already repeatedly observed; all Unknown-side results are `POST_HOC_DIAGNOSTIC_ONLY`.
- Component-P05 validation equalization is partly calibration-by-construction; Known Test supports coverage generalization but not Unknown utility.
- The fixed `<30` fallback leaves one highly imbalanced WorldOfWarcraft component per setting.
- USTC post-hoc results expose substantial open-set utility risk and are not used to tune/reselect P05.

## Conclusion and next step

- Component-specific calibration is warranted as a Known-coverage hypothesis, but has no independent Unknown-detection evidence.
- Component-P05, Class-P05, and Global formulas are frozen in `external_validation_protocol_candidate.md`; the first independent test must use untouched CSTNET-TLS1.3. No external training was started.
