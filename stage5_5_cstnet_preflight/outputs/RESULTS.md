# Experiment results: stage5_5_cstnet_preflight_20260913_v1

- Status: `complete`
- Experiment type: `preflight-diagnostic`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-13T02:51:04Z`
- Objective: Audit frozen DGSB gate contribution, CSTNET Open-Detect input leakage, and calibration sample sufficiency without training or Unknown inference.

## Data and split

- DGSB replay: frozen USTC Known Validation and Known Test latent files for A-1/A-2/A-3 only.
- CSTNET audit: seed `20260913`, 30 eligible classes, at most 20 PCAP/class, 600 PCAP total, with Known/Unknown audit-role coverage in Low/Medium/High.
- Frozen group split remains disjoint with zero active overlap in all three folds.

## Configuration and execution

- Environment: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`.
- DGSB: frozen Stage 5 `K=2`, full covariance, PCA64, `reg_covar=1e-3`, unchanged thresholds.
- Actual Open-Detect input: first 8 packets; 80 header + 48 payload bytes each; IP source/destination zeroed; 32×32 uint8.
- Successful run: tmux `s55_run_preflight_v2`, exit 0, `2026-09-13T02:59:51Z`–`03:01:09Z`.
- First attempt `s55_run_preflight` failed before scoring/PCAP work because the protocol fold key was named `class_holdout_folds`; its failure log and exit status remain preserved.

## Core results

- Integrity: 26/26 rule, protocol, and frozen Known-asset SHA-256 checks PASS.
- Global-exclusive rejection, Validation/Test: A-1 `0.0624%/0.0395%`; A-2 `0.0739%/0.0554%`; A-3 `0%/0%`.
- Local-exclusive rejection, Validation/Test: A-1 `4.0388%/4.0449%`; A-2 `4.2348%/4.3086%`; A-3 `4.9010%/4.8724%`.
- Global-vs-Local best-class mismatch: 0/259,992 (0%).
- CSTNET actual inputs: 600/600 valid; full domain 0%, SNI 0%, domain/token union 0%; raw PCAP domain/SNI 52/600 (8.67%).
- Endpoint fields: IP masked 4,800/4,800; transport ports visible 4,800/4,800.
- Validation P05 counts, Low/Medium/High: 11.6/11.0/12.0. Classes n<20: 18/19/16; n<30: 32/29/25; n<50: 86/81/78.
- Expected small-component n<30 fallback at 50/50: 109/113, 99/101, 88/89; all Known classes at 80/20, 90/10, and 95/5.
- Final Preflight Gate: `NOT_READY`.

## Preserved evidence

- `dgsb_gate_audit/`: transitions, gate effectiveness, consistency, confusion, summary.
- `cstnet_input_audit/`: sample/input manifest, byte-source mapping, saved 32×32 tensors, domain/SNI/token audit, endpoint audit.
- `calibration_sample_audit/`: split counts, reliability categories, component-weight stress, fallback risk, summary.
- `final_preflight_summary.md`, `run_metadata.json`, and `manifest.json`.
- External execution evidence: `.tmux-task/s55_run_preflight*/`, `.tmux-task/s55_independent_verify/`.

## Limitations

- The CSTNET sample is a fixed bounded input audit, not a full-dataset prevalence estimate.
- TLS parsing is deterministic and only reports complete parsable ClientHello records; parse failures are retained, not guessed.
- Component counts are hypothetical weight stresses because no CSTNET encoder/GMM exists yet.
- Port visibility is not proof that a trained model uses endpoint shortcuts.
- No CSTNET training, formal latent extraction, estimator fit, threshold calibration, Unknown scoring, UFAR, AUROC, or AUPRC occurred.

## Conclusion and next step

The sole blocking gate is absent Global-exclusive rejection in A-3 Known Validation and Known Test. Stop here. Separately decide whether to freeze a protocol v2 for class eligibility or a larger Known-only calibration pool before authorizing any training; Stage 5.5 does not choose or apply either change.
