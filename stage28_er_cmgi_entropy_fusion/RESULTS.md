# Stage28 ER-CMGI-inspired entropy fusion

- Status: success (all 20 Known-only diagnostic units complete).
- Claim scope: diagnostic mechanism adaptation; not original ER-CMGI reproduction.

## Data and split

- Stage20 frozen ISCX-VPN and ISCXTor Known Train/Validation only; no Test/Unknown usage.

## Configuration and execution

- 4 encoder pairs × 5 training seeds; A0/A1 adapters and P0–P3 heads, 30 epochs each; ≤3 physical GPUs in queue.
- Full configuration, hashes, logs and checkpoints retained in each run bundle.

## Core results

- Preregistered P3 gate: **FAIL**.
- P3 minus forced-equal mean Macro-F1: +0.000382.
- See `stage28_report.md` and the CSV tables for all dataset, method, seed, class and paired values.

## Preserved evidence

- 20 validated run bundles, source hashes, training histories, checkpoints, predictions, logits, counterfactuals and 120 independent CPU checkpoint replays.

## Limitations

- Previously exposed Test not touched; Known-Val development evidence only.
- Repeated head seeds share flow pools; not independent external trials.

## Conclusion and next step

- P3 does not clear the preregistered Known-Val gate; no Stage28B diffusion or Test-based tuning.
