# Experiment results: four_dataset_final_readiness_audit_20260913_v1

- Status: `complete`
- Experiment type: `data-readiness-audit`
- Claim scope: `diagnostic`
- Updated (UTC): `2026-09-13T04:13:13.816937+00:00`

## Core results

- USTC-TFC2016: **READY**, samples=489,101, usable_classes=20.
- CipherSpectrum: **READY**, samples=120,000, usable_classes=40.
- CSTNET-TLS1.3: **CONDITIONALLY_READY**, samples=46,372, usable_classes=119.
- CIC-IDS-2017: **CONDITIONALLY_READY**, samples=2,087,440, usable_classes=3.
- CipherSpectrum MIX: **PARTIALLY_OVERLAPPING**, exact=0, normalized=0.
- CICIDS strict matched: **2,087,440**, labels=15, primary=BENIGN, DoS Slowhttptest, PortScan.

## Boundaries and limitations

- No model training, embedding extraction, GMM, threshold fitting, Unknown inference, or formal metric evaluation.
- Directory and endpoint labels remain dataset-semantic risks; readiness does not imply paper-equivalent validity.
- Original dataset files were read only.

## Preserved evidence

- `manifest.json`
- Per-dataset CSV/Markdown files under `ustc/`, `cipherspectrum/`, `cstnet/`, and `cicids2017/`.
- Unified matrix under `summary/`.
