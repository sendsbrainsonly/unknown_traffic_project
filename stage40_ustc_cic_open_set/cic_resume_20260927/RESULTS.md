# Experiment results: stage40-cic-independent-resume-20260927

- Status: `failed_orchestrator_worker_continued`
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-27T09:49:00Z`
- Objective: Run both pre-frozen CIC-IDS-2017 leave-one-attack-out open-set pilots with Known-only training and validation P95 calibration

## Data and split

- Frozen Stage 40 CIC-IDS-2017, seed 2022, two group-disjoint leave-one-attack-out roles; no split regeneration.
- Unknown PortScan: Known Train/Val/Test 7,818/2,110/264; Unknown Test 1,136. Known classes: BENIGN and DoS Slowhttptest.
- Unknown Slowhttptest: Known Train/Val/Test 27,698/1,802/2,272; Unknown Test 132. Known classes: BENIGN and PortScan.
- Source and role-manifest SHA256 preflight PASS on 2026-09-27. Raw dataset and frozen protocol remain read-only.

## Configuration and execution

- Reuse the Stage 40 three-view recipe: TrafficFormer 20 epochs, TAGCN 50, YaTC 200, equal-feature-fusion adapters/head 30+30. Existing completed PortScan graph branch is hash-verified and reused.
- Fit encoder/support only on Known Train, select checkpoint/calibrate on Known Validation. Frozen scores: MSP, Energy, centroid and DES-v1 (k=10, equal global/local); threshold: Known-Val P95. Unknown and Known Test are evaluation-only.
- Independent runner `../run_cic_independent.py` calls the original Stage 40 branch, calibration, evaluation and replay logic but does not wait on unrelated Stage 39. It permits at most one GPU workload, with live capacity selection before each phase.
- Original `cic_queue_failure.json` and old tmux sessions are preserved; the present attempt uses `progress.json` and distinct `stage40cic_*_0927` worker sessions.

## Core results

- This coordinator attempt failed with tmux helper exit 124 after its 60-second synchronous wait, while its child training session continued. It produced no CIC open-set metric. The child worker is being awaited by the v2 coordinator; this attempt's failure evidence is retained.

## Preserved evidence

- `manifest.json`, `progress.json`, and `failure.json` for this failed coordinator attempt.
- Frozen protocol, per-role model checkpoints, caches, calibration, score CSVs and independent verification remain under the parent Stage 40 directory; worker logs and exit codes remain in project-local `.tmux-task/`.

## Limitations

- Single-seed diagnostic pilot; each role has only two Known classes and attack labels may correlate with capture date/endpoints. Unknown Slowhttptest has only 132 Test flows.

## Conclusion and next step

- Await both frozen pilots, then inspect per-method results and verify all source/role/checkpoint hashes before drawing conclusions.
