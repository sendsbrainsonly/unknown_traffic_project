# Experiment results: stage14c-vnat-encoder-training-20260917-v1

- Status: `success`
- Completed (UTC): `2026-09-17T14:38:28Z`
- Objective: Train 15 independent Known-only Open-Detect encoders on the frozen Stage 14B VNAT protocols without Unknown/Test evaluation or Stage 14D.

## Data and split

- Frozen VNAT clean pool: 23,449 flows across 15 pre-registered protocols.
- Model-visible inputs per run: frozen Known Train and Known Validation only.
- Unknown Train/Validation samples used: 0/0; Known Test samples used: 0.
- Stage 14B freeze hash before/after: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` / `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Flow-cache audit: 23,449 flows, 162 captures, 0 packet-count mismatches, 0 zero-packet flows, 16 released-code-compatible zero-filled non-IPv4 packet encodings.

## Configuration and execution

- 15 independent released Open-Detect initializations; latent dimension 128; batch 512; Adam LR 1e-3; MultiStepLR milestones 50/80; lambda 0.005.
- Maximum 100 epochs; Known Validation Accuracy/Macro-F1 harmonic-mean selection; early-stopping patience 5.
- Physical GPUs selected immediately before campaign: 5, 4, 3, 0; physical GPUs 1 and 2 were not used.

## Core results

| Low | 0.817794 ± 0.085747 | 0.594534 ± 0.133691 | 0.658260–0.904673 | 0.445140–0.804239 |
| Medium | 0.759880 ± 0.165323 | 0.559701 ± 0.240777 | 0.542959–0.986918 | 0.284444–0.923316 |
| High | 0.821455 ± 0.140071 | 0.690267 ± 0.143272 | 0.590674–0.966964 | 0.429056–0.849572 |

- Checkpoints saved/hash-verified/unique: 15/15/15.
- NaN/crashes: none; weak Macro-F1 flags below 0.35: 2/15.
- Detailed run table: `stage14c_training_summary.csv` and `stage14c_encoder_training.md`.

## Preserved evidence

- `checkpoints/`: 15 best encoder/model checkpoints, including optimizer state and frozen training config.
- `runs/`: Known-only input audits, per-epoch metrics, configs, selected-checkpoint records, latest checkpoints, and stdout logs.
- `flow_image_cache/`: flow-aligned preprocessing cache and audit ledger.
- `stage14c_training_summary.csv`, `stage14c_run_manifest.csv`, `final_verification.json`, and `manifest.json`.
- `failed_attempts/`: preserved strict-IPv4 cache attempt and interruption evidence.

## Limitations

- No Known Test or Unknown Test evaluation was performed; no Open-Detect unknown metrics or detection threshold was computed.
- Medium-2024 and Medium-2025 have weak Known Validation Macro-F1 and must remain visible in downstream interpretation.
- The 16 non-IPv4 packet encodings follow the released Open-Detect exception behavior (zero-filled packet block); no flow was deleted.

## Conclusion and next step

- Stage 14C is complete with quality flags. It is technically ready for Stage 14D, but Stage 14D was not started.
