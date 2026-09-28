# Experiment results: stage42y-cic-four-prelisted-candidates-20260928

- Status: `complete / independent replay PASS`
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T08:39:58Z`
- Objective: Sequential frozen Known-only open-set diagnosis for SSH-Patator, Web Attack Brute Force, Web Attack XSS, and full DoS Slowhttptest

## Data and split

- Frozen Stage40 Known classes: `BENIGN + PortScan`; unchanged Known Train, Validation, and 2,272 Known Test flows.
- Fixed sequential Unknown candidates: SSH-Patator (2,987 flows, 14 groups); Web Attack - Brute Force (1,364, 10); Web Attack - XSS (629, 5); DoS Slowhttptest (5,096, 5).
- All matched flows are included. Brute Force and XSS additionally have score-blind, hash-frozen 1:1 Known Test subsets.
- Full Slowhttptest contains 132 flow IDs already evaluated in Stage40; this is a post-hoc diagnostic, not untouched validation.

## Configuration and execution

- Named tmux session: `stage42y-sequential-20260928`.
- Command: `python run_series.py sequential`. The queue extracts and evaluates one candidate at a time, with live GPU capacity selection before each frozen-model inference.
- Exit code: `0`; all four inference jobs selected physical GPU 3 at launch. No encoder training or threshold refitting was performed.
- Protocol and input hashes: `series_protocol.json` and per-candidate `candidate_protocol.json`.
- Progress: `queue_progress.json` and `<candidate>/progress.json`.
- Frozen Known-Validation P95 threshold was reused for each score; Unknown and Test fitting counts were zero in all four independent verifications.

## Core results

Natural-prevalence Test results (same 2,272 Known Test flows; Unknown is positive):

| Unknown class | Unknown flows | MSP AUROC | DES-v1 AUROC | DES-v1 AUPRC | DES-v1 UFAR | DES-v1 Known FRR |
|---|---:|---:|---:|---:|---:|---:|
| SSH-Patator | 2,987 | 0.869745 | 0.991275 | 0.989281 | 0.000000 | 0.071303 |
| Web Attack - Brute Force | 1,364 | 0.916091 | 0.995652 | 0.991421 | 0.000000 | 0.071303 |
| Web Attack - XSS | 629 | 0.977971 | 0.997489 | 0.988430 | 0.000000 | 0.071303 |
| DoS Slowhttptest | 5,096 | 0.859683 | 0.993400 | 0.996257 | 0.001374 | 0.071303 |

- DES-v1 wrongly accepted 0/2,987 SSH-Patator, 0/1,364 Brute Force, 0/629 XSS, and 7/5,096 Slowhttptest flows at the unchanged threshold; it wrongly rejected 162/2,272 Known Test flows in each comparison.
- The 1:1 views for Brute Force and XSS used pre-frozen Known subsets of 1,364 and 629 flows, respectively; natural scores and thresholds were not changed. All four candidates have eight method-by-view metric rows independently replayed. Full MSP, Energy, centroid, and DES-v1 results are in the four `detection/open_set_results.csv` files.
- The higher DES-v1 ranking metrics do not imply uniformly better binary F1. For XSS, DES-v1 natural binary F1 is 0.885915 versus MSP 0.900593 because Known rejection is higher.

## Preserved evidence

- `manifest.json`, `series_protocol.json`, four candidate manifests/protocols, score-blind balanced Known subsets for the two small classes, and named tmux logs.
- Each candidate keeps packet-input caches, `detection/sample_scores.csv`, `detection/open_set_results.csv`, `detection/evaluation_audit.json`, and `detection/independent_verification.json`.

## Limitations

- Candidate order was selected after viewing earlier CIC diagnostic results. Attack capture/date/endpoint shortcuts may contribute to any high score. Slowhttptest reuses 132 historical Unknown Test flows.

## Conclusion and next step

- Four fixed-order diagnostics completed; all independent replays PASS. This is favorable evidence for the frozen DES-v1 score on these CIC roles, but not an independent generalization claim. No further candidate is started.
