# Experiment results: stage40-ustc-cic-open-set-20260926

- Status: `partial / USTC verified; CIC queue failed before open-set evaluation` (queue snapshot 2026-09-26 16:16 UTC)
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-26T15:54:06Z`
- Objective: Evaluate frozen three-view model on USTC A-2 and train Known-only two-class CIC leave-one-attack-out pilots

The later `cic_queue_progress.json` records `FAILED` because the upstream Stage39 training queue failed. No CIC Unknown score or open-set result was produced. The waiting-queue description below is retained as historical execution context.

## Data and split

- USTC A-2: reuse the frozen Stage34 three-view Known-only model. Known Train/Validation/Test have 34,665/4,333/4,333 flows; Geodo/Htbot/Tinba are entire Unknown classes, with 410/63/85 sampled Test flows (558 total). This is the original Stage34 10%-per-class Test sampling rule, not a new split selected by detector performance.
- CIC-IDS-2017: two pre-frozen, group-disjoint two-class Known-only pilots. `unknown_portscan` has 7,818/2,110/264 Known Train/Validation/Test and 1,136 Unknown PortScan Test flows. `unknown_slowhttptest` has 27,698/1,802/2,272 Known Train/Validation/Test and 132 Unknown Slowhttptest Test flows. The attack-specific Test rows are held out completely from fitting and calibration.
- Role manifests and source SHA256 values are frozen in `protocols.json`; raw datasets and historical Stage34/Stage34B artifacts are unchanged.

## Configuration and execution

- Three-view Stage31 recipe (TrafficFormer, packet graph, YaTC feature branches, equal feature fusion); USTC uses existing Known-only checkpoints; CIC requires fresh Known-only two-class checkpoints because the old three-class CIC checkpoints saw both attacks.
- Scores fixed before Test evaluation: MSP, Energy, empirical centroid distance, and DES-v1 (centroid plus kNN-10, each Known-Val median/MAD normalized, equal weights). Threshold is each method's Known Validation P95. No Unknown or Test data is used to fit representation, support, normalization, or threshold.
- USTC input build, CPU calibration, GPU3 frozen evaluation, and independent score replay completed successfully. The `stage40_cic_queue_v2_0926` named tmux queue waits for the existing Stage39 three-GPU training queue to finish, then completes the two CIC pilots on at most one GPU3 workload at a time. The first CIC graph branch was already completed using Known Train/Validation only. The earlier GPU4 graph launch failed at live capacity selection before model training; its log is preserved. The initial waiting-only queue session was closed and replaced by v2 to add bounded GPU3-capacity retries; it had not launched any worker.

## Core results

USTC A-2, Known Test 4,333, Unknown Test 558, independent metric replay PASS:

| Unknown score | AUROC | AUPRC | UFAR @ Known-Val P95 | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.916286 | 0.629202 | 0.508961 | 0.053312 | 0.515522 |
| Energy | 0.871426 | 0.584590 | 0.526882 | 0.050312 | 0.507692 |
| Centroid | 0.677617 | 0.439162 | 0.537634 | 0.048004 | 0.503906 |
| DES-v1 | 0.945727 | 0.711960 | 0.460573 | 0.054927 | 0.548769 |

DES-v1 per Unknown family: Geodo AUROC 0.926979 / UFAR 0.626829 (410 flows), Htbot 0.996535 / 0 (63), Tinba 0.998499 / 0 (85). Geodo dominates the unresolved misses despite high aggregate ranking performance.

CIC open-set results are **not yet available**; do not infer them from its high closed-set F1 or the completed graph branch.

## Preserved evidence

- `protocols.json`, `ustc_a2_roles.csv`, `unknown_portscan_roles.csv`, `unknown_slowhttptest_roles.csv`
- `ustc_a2/detection/calibration.json`, `open_set_results.csv`, `per_unknown_class.csv`, `sample_scores.csv`, `independent_verification.json`
- CIC queue status: `cic_queue_progress.json`; worker logs and exit codes: project-local `.tmux-task/stage40*`
- `manifest.json` (artifact hashes to refresh after both CIC pilots complete)

## Limitations

- This is a single-seed diagnostic pilot, not a multi-seed estimate. USTC is not capture-disjoint; the historical Stage34 protocol was retained for paired consistency.
- CIC attacks are recorded on different days/captures. Group-disjoint splitting prevents within-capture leakage but cannot remove attack-versus-day confounding. Slowhttptest has only 132 Unknown Test flows.
- A strong closed-set Macro-F1 does **not** imply effective open-set rejection: USTC DES-v1 still accepts 46.06% of Unknown flows at the fixed P95 operating point.

## Conclusion and next step

USTC demonstrates strong ranking but incomplete practical rejection, especially for Geodo. The pre-frozen CIC pilots are queued and must be reported from their own independently verified Test scores before making a USTC-versus-CIC conclusion.
