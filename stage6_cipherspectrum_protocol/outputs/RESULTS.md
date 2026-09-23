# Experiment results: stage6-cipherspectrum-protocol-freeze-20260913

- Status: `success`
- Experiment type: `protocol-freeze`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-13T04:25:26Z`
- Completed (UTC): `2026-09-13T04:37:20Z`
- Objective: Freeze the canonical CipherSpectrum 120k corpus and three deterministic Strict Unknown-Free group-aware open-set protocols without training or Unknown inference.

## Data and split

- Source manifest SHA-256: `3ec053adcdf3fb39693129bdc53ab45fccd9f5563a1ed47791f01b98267cd5be`.
- Canonical manifest SHA-256: `6dbef11e4818de00d70cfaf86187d8d6749db7a16a6eacd9fa0c4cbeef1d5020`.
- Canonical corpus: 120,000 PCAPs, 40 classes, 3,000/class, three cipher sources at 1,000/class/source.
- MIX/getpocket samples: 0/0; `split_group_id` coverage: 100%.
- Known split target: 70/15/15 by indivisible `split_group_id` after Unknown-group purge.

## Configuration and execution

- Unknown candidate seeds: Low=42, Medium=43, High=44.
- Candidate RNG: NumPy PCG64 over sorted class names; complete candidate-set rejection only.
- Group objective: total, per-class, and per-class×cipher-source ratio errors only.
- Fixed environment: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`.
- Freeze session/exit: `stage6_protocol_freeze_20260913` / 0.
- Independent verification session/exit: `stage6_protocol_verify_20260913` / 0.
- GPU used: no.

## Core results

| Setting | Accepted candidate | Known/Unknown classes | Train | Validation | Test | Unknown Test | Purged Known |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 3 | 38/2 | 75,692 | 16,242 | 16,246 | 6,000 | 5,820 |
| Medium | 15 | 34/6 | 56,641 | 12,168 | 12,162 | 18,000 | 21,029 |
| High | 15 | 30/10 | 49,976 | 10,740 | 10,746 | 30,000 | 18,538 |

- Active Train/Validation/Test group overlap: 0 for all settings.
- Active Known/Unknown group overlap: 0 for all settings.
- Minimum per-class Validation/Test: Low 249/250, Medium 130/130, High 186/186.
- Input leakage: domain 4.25%, SNI 0%, token/combined 4.50%; `LOW_DIRECT_DOMAIN_LEAKAGE`.
- Port visibility: 3,200/3,200; `PORT_SHORTCUT_RISK=PRESENT`.
- Strict Unknown-Free audit: PASS; Unknown inference executed: no.
- Protocol hashes: 16/16 PASS; corpus hashes: 2/2 PASS.
- Final Gate: `READY`.

## Preserved evidence

- Canonical manifest and per-class summary under `corpus/`.
- Low/Medium/High fold JSON and the 360,000-row role manifest under `splits/`.
- Candidate rejections, calibration, component stress, leakage, group integrity, and Unknown-Free records under `audits/`.
- Frozen protocol, preprocessing identity, run metadata, and hashes under `protocol/`.
- `manifest.json` contains the complete artifact inventory.

## Limitations

- This is a metadata-only protocol freeze, not an external performance result.
- Directory/SNI-oriented class semantics and retained transport ports remain potential shortcuts.
- DGSB-v1 remains excluded from formal external evaluation.
- No training, embedding extraction, scaler/PCA/GMM fitting, calibration, prediction, UFAR, AUROC, AUPRC, or Unknown inference was run.

## Conclusion and next step

The Stage 6 protocol is `READY` and was created before CipherSpectrum Unknown evaluation. Stop at this boundary. A future explicitly authorized task may train three independent Known-only encoders under the frozen splits.
