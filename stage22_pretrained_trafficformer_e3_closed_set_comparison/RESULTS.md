# Experiment results: stage22-pretrained-trafficformer-e3-closed-set-20260923-v1

- Status: `running`
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-23T02:49:05Z`
- Objective: On the frozen Stage20 ISCX-VPN and ISCXTor closed-Service splits, compare official-pretrained TrafficFormer fine-tuned for 20 epochs with the same pretrained TrafficFormer plus the frozen TAGCN and linear E3 fusion pipeline.

## Data and split

- Frozen Stage20 ISCX-VPN and ISCXTor2016 coarse-Service closed-set manifest.
- ISCX-VPN: 10,955 flows, Train/Validation/Test = 8,764/1,098/1,093.
- ISCXTor2016: 11,181 flows, Train/Validation/Test = 8,946/1,118/1,117.
- Seeds: 2022 and 2023; Test is not used for checkpoint selection.

## Configuration and execution

- Official pretrained TrafficFormer SHA256: `be9dcc1e0c6b68db005e506adf13a2d937a2f3982539a2c8a58f78c55af0d5ce`.
- TrafficFormer: 20 epochs, batch 64, LR 6e-5, sequence length 320, First pooling.
- TAGCN and fusion retain Stage21 settings.
- Static preflight: PASS (16 checks).
- Official-weight forward smoke: PASS.
- Batch64 forward/backward/optimizer smoke: PASS; peak allocated memory 31.72 GB.
- Formal tmux session: `stage22-pretrained-e3-grid-20260923`.

## Core results

- Not recorded yet.

## Preserved evidence

- `manifest.json`, `config.json`, `preflight.json`, `smoke_result.json`, `batch64_smoke_result.json`.
- Failed pre-context memory-stat smoke is preserved under `failed_attempts/`.

## Limitations

- Not recorded yet.

## Conclusion and next step

- Not recorded yet.
