# Experiment results: stage20-dual-coarse-service-protocol-20260921-v1

- Status: `success / PROTOCOL_FROZEN_NOT_TRAINED`
- Experiment type: `protocol-freeze`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-21T14:37:31Z`
- Objective: Freeze a coarse Service-level closed/open-set task for ISCX-VPN and ISCXTor2016 without overwriting fine-grained experiments

## Data and split

- Reused Stage 12 flow-level 32×32 byte-image pools and provenance; raw PCAPs were not reparsed.
- ISCX-VPN: six Services — Chat, Email, File-Transfer, P2P, Streaming, VoIP.
- ISCXTor2016: seven Services — Browsing, Chat, Email, File-Transfer, P2P, Streaming, VoIP. Official Audio and Video categories are merged into Streaming.
- Each Service is capped at 2,000 flows after excluding cross-Service duplicate images, then deterministically split 80/10/10 by exact-image group.
- Selected unique flows: VPN `10,955`; TOR `11,181`; total `22,136`.
- Closed roles: VPN `8,764/1,098/1,093`; TOR `8,946/1,118/1,117` for Train/Validation/Test.

## Configuration and execution

- Seed: `20260921`; split unit: exact 32×32 image hash group.
- Open-set protocol: Leave-One-Service-Out (LOSO), producing 6 VPN and 7 TOR protocols.
- Held-out Service is Final Unknown Test only; it contributes zero samples to training, validation, normalization, support fitting, or threshold calibration.
- No GPU, model training, checkpoint selection, or Test-driven tuning was performed.
- Successful build/verification session: `codex-coarse-protocol-build-retry-20260921`, exit code `0`.
- First attempt was rejected because one VPN image hash appeared under both File-Transfer and VoIP. That failure is preserved under `failed_attempts/cross_service_duplicate_gate_v1/`; the successful protocol excludes the complete conflicting image group before selection.

## Core results

- Task decision: coarse Service classification becomes the primary VPN/TOR task; fine Application classification remains historical diagnostic evidence.
- Frozen LOSO protocols: `13`; manifest rows: `143,997`.
- Strict Unknown-Free audit: `PASS`.
- Cross-role flow-ID overlap: `0`; cross-role exact-image overlap: `0` in every protocol.
- VPN cross-Service conflict removal: one image group, three rows (`File-Transfer=2`, `VoIP=1`). TOR conflicts: zero.
- This stage freezes data and task semantics only; it does not report new closed-set or open-set model performance.

## Preserved evidence

- `config.json`: frozen task definition.
- `closed_service_manifest.csv`: one row per selected flow and closed-set role.
- `loso_service_protocol_manifest.csv`: all 13 strict Unknown-Free LOSO protocols.
- `protocol_summary.json`, `protocol_audit.json`, and `completion_verification.json`.
- `failed_attempts/cross_service_duplicate_gate_v1/`: preserved rejected first attempt.
- `manifest.json`: machine-readable artifact inventory.

## Limitations

- Service labels are capture-derived weak labels rather than independent per-flow annotations.
- Splits are flow- and exact-image-disjoint but not uniformly capture-disjoint; VPN P2P has one capture.
- The source pool was historically capped per fine Application before this Service-level reselection.
- Switching to a coarser task changes the scientific question. Its scores must not be presented as an improvement over the fine task on an identical label space.

## Conclusion and next step

The coarse Service task is frozen and ready for training. The next authorized experiment should train the selected encoder/detector on the 13 frozen LOSO protocols and report closed-set Accuracy/Macro-F1 together with AUROC, AUPRC, UFAR, and Known FRR. Historical fine-grained artifacts must remain unchanged.
