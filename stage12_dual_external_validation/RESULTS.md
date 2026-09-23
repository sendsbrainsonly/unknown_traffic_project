# Stage 12 — Dual Independent External Validation

## Core results

- Status: `success`.
- Final gate: **EXTERNAL_CONFIRMED**.
- ISCX-VPN mean paired DES-v0 minus Open-Detect Native AUROC: `+0.0234515062`.
- ISCXTor2016 mean paired DES-v0 minus Open-Detect Native AUROC: `+0.0697335048`.
- Final verification: 30/30 successful encoder runs and 60 method-result rows.
- Methods: Open-Detect Native and DES-v0 only, paired on the same 30 encoders.

## Data and split

- ISCX-VPN: 16 eligible classes; all three settings use `FLOW_DISJOINT_ONLY` because BitTorrent cannot satisfy the source-PCAP group split.
- ISCXTor2016: 10 eligible classes; all three settings use `FLOW_DISJOINT_ONLY` because FTP cannot satisfy the source-PCAP group split.
- Train, validation, checkpoint selection, prototypes, and thresholds are strictly unknown-free.
- The test set was opened once, only after all 30 runs reached `READY_FOR_ONE_SHOT_TEST`.

## Configuration and execution

- Five frozen training seeds: 2022, 2023, 2024, 2025, and 2026.
- One shared encoder per dataset, openness setting, and seed; both methods use the same checkpoint.
- Preprocessing: eight packets per flow, 80 header bytes plus 48 payload bytes per packet, reshaped to 32×32 grayscale.
- Threshold: Known Validation 0.95 quantile only.
- Final evaluation selected physical GPU 0 from the allowed set 0,3,4,5,6,7; GPUs 1 and 2 were excluded.
- Post-Test method modification: `NO`.

## Preserved evidence

- Detailed dataset results: `outputs/iscx_vpn/` and `outputs/iscx_tor/`.
- Cross-dataset summary and gate: `outputs/summary/`.
- Per-run checkpoints, histories, metrics, bootstrap results, and manifests: `runs/`.
- Preflight and protocol decisions: `PROTOCOL_PREFLIGHT.md` and `protocol/`.
- Test-opening record: `outputs/summary/FINAL_TEST_OPENING.json`.

## Limitations

- Both datasets required the preregistered `FLOW_DISJOINT_ONLY` fallback, so the result is an independent external confirmation under flow-level disjointness, not a strict unseen-capture-domain generalization claim.
- Application labels inherit the controlled-capture filename/directory semantics documented by the provenance audit.
- Classes below the preregistered eligibility minimum were excluded rather than relabeled or augmented.

## Conclusion and next step

Stage 12 supports the preregistered `EXTERNAL_CONFIRMED` conclusion for DES-v0 versus Open-Detect Native on both external datasets. No Stage 13 or additional method was started; the next step remains intentionally unset.
