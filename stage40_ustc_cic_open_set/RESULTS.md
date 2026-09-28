# Experiment results: stage40-ustc-cic-open-set-20260926

- Status: `success / USTC and both CIC diagnostic roles independently verified` (updated 2026-09-27 14:32 UTC)
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-26T15:54:06Z`
- Objective: Evaluate frozen three-view model on USTC A-2 and train Known-only two-class CIC leave-one-attack-out pilots

## Current completion correction (2026-09-27)

The older queue failure and “CIC results not yet available” statements below describe the historical 2026-09-26 attempt and are superseded by the completed v2 continuation at `cic_resume_20260927_v2/`. Both frozen CIC roles completed, passed independent score replay, and retained their frozen role/source hashes. The v2 coordinator exited 0. At Known-Val P95, the PortScan role is very weak for all scores, while Slowhttptest is strongly ranked; these are distinct single-seed diagnostic roles, not a general CIC performance estimate.

Historical execution note (superseded): the first independent queue attempt exposed the helper's 60-second wait timeout while its TrafficFormer child continued. The v2 replacement coordinator polled child status, completed the Slowhttptest role, and exited successfully. The initial attempt and Stage39-gated queue failure remain preserved in their original locations; both final role results are now reported below and in `cic_resume_20260927_v2/`.

The later `cic_queue_progress.json` records `FAILED` because the upstream Stage39 training queue failed. No CIC Unknown score or open-set result was produced. The waiting-queue description below is retained as historical execution context.

## Data and split

- USTC A-2: reuse the frozen Stage34 three-view Known-only model. Known Train/Validation/Test have 34,665/4,333/4,333 flows; Geodo/Htbot/Tinba are entire Unknown classes, with 410/63/85 sampled Test flows (558 total). This is the original Stage34 10%-per-class Test sampling rule, not a new split selected by detector performance.
- CIC-IDS-2017: two pre-frozen, group-disjoint two-class Known-only pilots. `unknown_portscan` has 7,818/2,110/264 Known Train/Validation/Test and 1,136 Unknown PortScan Test flows. `unknown_slowhttptest` has 27,698/1,802/2,272 Known Train/Validation/Test and 132 Unknown Slowhttptest Test flows. The attack-specific Test rows are held out completely from fitting and calibration.
- Role manifests and source SHA256 values are frozen in `protocols.json`; raw datasets and historical Stage34/Stage34B artifacts are unchanged.

Explicit class-role and sample distribution for the completed CIC pilots:

| Protocol | Known classes | Unknown class | Known Train | Known Validation | Known Test | Unknown Test |
|---|---|---|---:|---:|---:|---:|
| `unknown_portscan` | BENIGN, Slowhttptest | PortScan | 7,818 | 2,110 | 264 | 1,136 |
| `unknown_slowhttptest` | BENIGN, PortScan | Slowhttptest | 27,698 | 1,802 | 2,272 | 132 |

For `unknown_slowhttptest`, only BENIGN and PortScan participate in encoder training, Known Validation calibration, support construction, and the Known Test evaluation. Slowhttptest is absent from all fitting/calibration stages and appears only as the 132-flow Unknown Test set.

## Configuration and execution

- Three-view Stage31 recipe (TrafficFormer, packet graph, YaTC feature branches, equal feature fusion); USTC uses existing Known-only checkpoints; CIC requires fresh Known-only two-class checkpoints because the old three-class CIC checkpoints saw both attacks.
- Scores fixed before Test evaluation: MSP, Energy, empirical centroid distance, and DES-v1 (centroid plus kNN-10, each Known-Val median/MAD normalized, equal weights). Threshold is each method's Known Validation P95. No Unknown or Test data is used to fit representation, support, normalization, or threshold.
- USTC input build, CPU calibration, GPU3 frozen evaluation, and independent score replay completed successfully. Historical CIC attempts included a Stage39-gated queue, a GPU4 capacity-selection refusal before training, and a waiting-only session that launched no worker; all evidence remains preserved. The successful v2 CIC continuation ran at most one GPU workload at a time and is complete.

## Core results

USTC A-2, Known Test 4,333, Unknown Test 558, independent metric replay PASS:

| Unknown score | AUROC | AUPRC | UFAR @ Known-Val P95 | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.916286 | 0.629202 | 0.508961 | 0.053312 | 0.515522 |
| Energy | 0.871426 | 0.584590 | 0.526882 | 0.050312 | 0.507692 |
| Centroid | 0.677617 | 0.439162 | 0.537634 | 0.048004 | 0.503906 |
| DES-v1 | 0.945727 | 0.711960 | 0.460573 | 0.054927 | 0.548769 |

DES-v1 per Unknown family: Geodo AUROC 0.926979 / UFAR 0.626829 (410 flows), Htbot 0.996535 / 0 (63), Tinba 0.998499 / 0 (85). Geodo dominates the unresolved misses despite high aggregate ranking performance.

CIC results, Known-Val P95, independent replay PASS:

| Unknown role | Known Test / Unknown Test | Score | AUROC | AUPRC | UFAR | Known FRR | Binary F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| PortScan | 264 / 1,136 | MSP | 0.847244 | 0.921874 | 0.951585 | 0.041667 | 0.091514 |
| PortScan | 264 / 1,136 | Energy | 0.719677 | 0.867625 | 0.963028 | 0.018939 | 0.071006 |
| PortScan | 264 / 1,136 | Centroid | 0.285411 | 0.714805 | 0.956866 | 0.162879 | 0.079805 |
| PortScan | 264 / 1,136 | DES-v1 | 0.463178 | 0.761719 | 0.948063 | 0.166667 | 0.095238 |
| Slowhttptest | 2,272 / 132 | MSP | 0.954846 | 0.646372 | 0.045455 | 0.049296 | 0.681081 |
| Slowhttptest | 2,272 / 132 | Energy | 0.950461 | 0.619815 | 0.075758 | 0.048856 | 0.668493 |
| Slowhttptest | 2,272 / 132 | Centroid | 0.962588 | 0.630503 | 0.045455 | 0.054577 | 0.659686 |
| Slowhttptest | 2,272 / 132 | DES-v1 | 0.997753 | 0.955759 | 0.000000 | 0.071303 | 0.619718 |

The independent verifiers replayed all four score methods on 1,400 PortScan-role rows and 2,404 Slowhttptest-role rows; both report `PASS`, unchanged role/source hashes, and zero Unknown/Test fitting. The full sample scores and calibration/audit evidence remain under each role's `detection/` directory. The reproducible attempt-level summary is `cic_resume_20260927_v2/RESULTS.md`.

## Preserved evidence

- `protocols.json`, `ustc_a2_roles.csv`, `unknown_portscan_roles.csv`, `unknown_slowhttptest_roles.csv`
- `ustc_a2/detection/calibration.json`, `open_set_results.csv`, `per_unknown_class.csv`, `sample_scores.csv`, `independent_verification.json`
- CIC queue status: `cic_queue_progress.json`; worker logs and exit codes: project-local `.tmux-task/stage40*`
- `manifest.json` (artifact inventory and hashes refreshed after both CIC pilots completed)

## Limitations

- This is a single-seed diagnostic pilot, not a multi-seed estimate. USTC is not capture-disjoint; the historical Stage34 protocol was retained for paired consistency.
- CIC attacks are recorded on different days/captures. Group-disjoint splitting prevents within-capture leakage but cannot remove attack-versus-day confounding. Slowhttptest has only 132 Unknown Test flows.
- A strong closed-set Macro-F1 does **not** imply effective open-set rejection: USTC DES-v1 still accepts 46.06% of Unknown flows at the fixed P95 operating point.

## Conclusion and next step

USTC demonstrates strong ranking but incomplete practical rejection, especially for Geodo. CIC is highly role-dependent: PortScan is not effectively rejected at the fixed operating point (DES-v1 AUROC 0.463178, UFAR 0.948063), while Slowhttptest is ranked strongly (AUROC 0.997753, UFAR 0.000000, Known FRR 0.071303). Because CIC attacks are confounded with capture/day and the Slowhttptest Unknown Test has only 132 flows, treat these as diagnostic pilot observations rather than broad generalization. Both CIC roles are complete; no additional tuning or experiment was started.
