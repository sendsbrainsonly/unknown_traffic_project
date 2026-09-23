# Stage 12 Protocol Preflight

## Frozen before training

- Primary comparison: `M1_DES_V0 - M0_OPEN_DETECT_NATIVE`.
- Datasets: ISCX-VPN and ISCXTor2016, audited independently of frozen Stage 6–11C material.
- Canonical class: application/service identity; VPN/non-VPN and Tor/non-Tor remain metadata.
- Eligibility: at least one readable source capture and at least 200 deterministically selected flows; cap 2,000 flows/class.
- Preferred split: duplicate-linked source-PCAP group-aware 80/10/10.
- Required fallback: `FLOW_DISJOINT_ONLY` with flow-ID and exact-image-hash disjointness when fewer than three duplicate-linked source groups remain. The affected class and claim reduction are recorded.
- Unknown selection: canonical sort, `numpy.random.default_rng(42)`, nested 5%/15%/25% prefixes.
- Training seeds: 2022–2026.
- Thresholds: Known Validation P95 with NumPy `method="higher"` only.
- Test boundary: all expected runs must pass `verify_stage12.py --phase pretest` before `FINAL_TEST_OPENING.json` can be created.

## Evidence available before preprocessing

- Dataset/provenance audit: `PASS`.
- Open-Detect corrected-v6 frozen runs: 15/15 successful.
- Frozen Stage 10a–11c artifacts: present and hashed.
- ISCX references in frozen Stage 10a–11c README/RESULTS: none.
- ISCX-VPN: 140/140 PCAPs readable; 16 mapped canonical classes before flow eligibility.
- ISCXTor2016: 85/95 PCAPs readable; 10 unreadable/cut-short captures retained in the inventory; 15 mapped canonical classes before flow eligibility.
- Readable `(canonical class, capture basename)` keys are unique: VPN 140/140, Tor 85/85.
- Unit tests: 11 passed, covering mappings, nested openness, validation quantile, source-group split, cross-class duplicate linking, flow fallback, corrected Open-Detect model/readout API, DES scoring, metrics, paired bootstrap, and the final Gate.

## Preserved non-experimental command failures

- The first compile command used project-root-relative `scripts/...` paths instead of the Stage 12 subdirectory; the corrected command passed.
- The first inventory-key one-liner omitted the Stage 12 `scripts/` import path; the corrected read-only check passed.
- The first model API test constructed the released model on CUDA but evaluated it on CPU without the formal `.to(device)` step; the corrected CPU-only test patches the released constructor's hard-coded `Tensor.cuda()` call and passes.

Both failures are retained under their project-local `.tmux-task/` directories and did not read Test data or alter source datasets.

## Preserved interrupted preprocessing attempt

The first ISCX-VPN preprocessing launch used the upstream sequential all-class driver. It completed AIM, BitTorrent, and Email, then was interrupted while scanning FTPS after the 5.84 GB capture exposed a single-core bottleneck. Its partial directory and tmux log are preserved. The replacement invokes the same upstream driver independently per class with identical class-local inputs and seeds, then deterministically merges the pools; only execution scheduling changes.

The first Tor class-parallel launch exposed Google and Twitter with fewer than 10 selected flows. The upstream writer correctly refused their empty validation splits. Because both are already far below the frozen 200-flow eligibility minimum, v2 preserves their counts/logs as explicit sample-insufficient exclusions and does not manufacture split rows for them.
