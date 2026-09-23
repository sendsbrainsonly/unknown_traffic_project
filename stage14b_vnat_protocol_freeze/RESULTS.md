# Experiment results: stage14b-vnat-open-set-protocol-freeze-20260917-v1

- Status: `success`
- Experiment type: `protocol-freeze`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-17T10:11:41Z`
- Completed (UTC): `2026-09-17T10:19:43Z`
- Objective: Freeze a model-independent VNAT application-held-out open-set protocol with permanent data-integrity exclusions and deterministic flow-level 8:1:1 Known splits

## Data and split

- Frozen source: Stage 14A PCAP flow records and data-integrity evidence.
- Ten semantic classes: `netflix`, `rdp`, `rsync`, `scp`, `sftp`, `skype`, `ssh`, `vimeo`, `youtube`, `zoiper`.
- All ten applications are retained under the requested flow-level rule; minimum clean class size is RDP with 44 flows.
- Permanent exclusions: four cross-application Rsync/SFTP duplicate flow rows; both failed PCAPs as whole captures; all flow contribution from duplicate `vpn_skype-chat_capture4` while retaining `vpn_skype-chat_capture3`.
- Clean population: 23,449 unique-contribution flows.
- Known data: deterministic per-application flow-random Train/Validation/Test = 8:1:1. Capture/group fields are retained as metadata but are not hard split boundaries.
- Unknown data: complete application-held-out flow population assigned only to `unknown_test`.

## Configuration and execution

- Settings: Low = 2 Unknown / 8 Known; Medium = 3/7; High = 4/6.
- Protocol seeds: `2022, 2023, 2024, 2025, 2026`.
- Unknown schedule: model-independent balanced nested cyclic schedule from master seed 2022. Low covers every application exactly once across five seeds; High covers every application exactly twice.
- Integer split rule for every Known class: Validation = Test = `max(1, floor(N/10))`; Train receives the remainder.
- Same class/seed flow assignment is reused across Low/Medium/High whenever that class remains Known.
- Command: `python stage14b_vnat_protocol_freeze/scripts/freeze_vnat_protocol.py`.
- CPU only; no model, embedding, Open-Detect, DES, score, or test metric was produced or consulted.
- Checkpoint-selection rule: not applicable at this stage. Frozen future data roles permit checkpoint selection from Known Validation only.

## Core results

- Freeze status: `FROZEN_READY_FOR_STAGE_14C`.
- Freeze hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Protocols: 15 setting × seed combinations.
- Flow manifest rows: 351,735 = 23,449 clean flows × 15 protocols.
- Strict Unknown-Free verifier: PASS for all 15 protocols.
- Empty retained classes: 0; Known classes with an empty Train/Validation/Test split: 0.
- RDP's Known split is 36/4/4; it is usable but statistically small.
- Class-size range is 44–13,563 flows (308.2×).
- VPN/non-VPN imbalance is severe: only Zoiper has a substantial VPN fraction; several classes have below 1% VPN flows.
- Because the user explicitly requested flow-random Known splits, 18–74 groups and 21–77 captures span multiple Known splits depending on protocol. Strict Unknown-Free passes, but capture-disjoint generalization is not claimed.

## Preserved evidence

- Required outputs: `vnat_final_class_audit.csv`, `vnat_open_set_protocol.json`, `vnat_split_manifest.csv`, `stage14b_protocol_freeze.md`.
- Integrity exclusions: `outputs/integrity_exclusions.csv`.
- Freeze and artifact hashes: `outputs/freeze_hashes.json`.
- Generator and independent verifier: `scripts/`; targeted tests: `tests/`.
- Machine-readable experiment inventory: `manifest.json`.

## Limitations

- Known Train/Validation/Test is flow-disjoint but not capture/group-disjoint; this is an explicit protocol choice, not an undetected failure.
- Vimeo has only one conservative group, so all of its Known splits share that group.
- RDP Validation/Test each contain only four flows.
- Application labels remain filename-derived capture labels rather than authoritative per-flow labels.
- Flow-count balance does not imply independent samples or balanced VPN/non-VPN domains.

## Conclusion and next step

Stage 14C may start only after verifying the freeze hash. Unknown applications, protocol seeds, all flow assignments, integrity exclusions, the 8:1:1 rule, and global Known-Validation-P95 threshold rule are immutable and may not be changed in response to VNAT Test results. No Stage 14C training was started here.
