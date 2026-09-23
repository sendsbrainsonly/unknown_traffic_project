# Experiment results: stage5_dual_gate_boundary_freeze_v1

- Status: `success`; final gate: `READY`
- Experiment type: `method_and_protocol_freeze`
- Claim scope: `METHOD_FREEZE_PLUS_PROTOCOL_FREEZE_NO_UNKNOWN_EVALUATION`
- Formal run: `2026-09-13T02:11:47Z`–`2026-09-13T02:12:36Z`
- Objective: freeze DGSB from USTC Known-only assets and freeze a strict
  CSTNET-TLS1.3 external protocol without training or Unknown score access.

## Data and split

- USTC inputs: frozen Stage 3 K2/scaler/PCA and Known Validation/Test only.
- USTC Unknown inputs accessed: `0`.
- CSTNET: 46,372 readable PCAPs, 120 original domain classes.
- Eligibility: `sample_count >= 100`; 119 eligible, one excluded (`chia.net`, 16).
- Group proxy: first-packet UTC timestamp floored to one minute.
- Known rows sharing an Unknown-class group are excluded before deterministic
  group-aware 80/10/10 splitting.

## Configuration and execution

- K=2, full covariance, `reg_covar=1e-3`, PCA-64.
- Local threshold: Known Validation P05; `n<30` falls back to class P05.
- DGSB: strict Global AND winning-component Local gate; no weighted fusion.
- Global threshold: selected from Known Validation after Local freeze to make
  final acceptance closest to 0.95; target-error ties use the higher threshold.
- Formal tmux session: `stage5_formal_freeze_20260913_v1`, exit code `0`.
- CPU-only; no GPU used.

## Core results

| Setting | Dual global threshold | Val acceptance | Test FRR | Val mean component gap |
|---|---:|---:|---:|---:|
| A-1 | -22.984130 | 0.949999 | 0.049021 | 0.058899 |
| A-2 | -71.731738 | 0.949990 | 0.050517 | 0.060669 |
| A-3 | -96.222612 | 0.949461 | 0.050071 | 0.057567 |

- DGSB rule SHA-256:
  `959f0a0dba96bb3c8b0519cc2c15f3ca3a9520df4de9d08636924415fe70c6a0`.
- Eligibility counts for minimum 50/100/200/500: `119/119/98/3`.
- Low/Medium/High folds: Known/Unknown classes `113/6`, `101/18`, `89/30`;
  seeds `42/43/44`.
- High selected the fourth deterministic candidate after three count-only
  post-group-purge rejections.
- Active capture-group overlap across Train/Validation/Test/Unknown: `0`.
- All rule and split hashes: `PASS`.
- CSTNET training executed: `false`; CSTNET Unknown scores accessed: `false`.

## Preserved evidence

- `rule_freeze/`: frozen formulas, USTC Known-only sanity, input access ledger,
  thresholds, metadata and SHA-256.
- `cstnet_protocol/`: class inventory, eligibility table, grouping audit, fold
  selection audit, three fold JSONs, full split manifest, protocol and hashes.
- Formal command log:
  `../../.tmux-task/stage5_formal_freeze_20260913_v1/output.log`.
- Formal exit status:
  `../../.tmux-task/stage5_formal_freeze_20260913_v1/exit.status`.

## Limitations

- No USTC Unknown evaluation is permitted in Stage 5; Known-only sanity does not
  establish Unknown Detection utility.
- No authoritative CSTNET capture/session manifest exists. The one-minute group
  is a deterministic proxy, not proof of session identity.
- The domain directory is the class label and endpoint metadata is not
  flow-parsed. Domain/endpoint shortcut risk remains `PROTOCOL_RISK`.
- The external fold has been frozen but not executed; no baseline or DGSB may be
  called superior before the first untouched CSTNET Unknown evaluation.

## Conclusion and next step

Final gate: **READY** for a separate, future CSTNET execution task under the
frozen hashes. This task stops at method/protocol freeze and does not start the
next stage.
