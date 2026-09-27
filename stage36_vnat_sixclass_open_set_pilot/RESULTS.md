# Experiment results: stage36-vnat-sixclass-open-set-20260925-v1

- Status: `in_progress / Known-only gates PASS; conditional queue waiting for Stage34/35`
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-25T14:47:53Z`
- Objective: Evaluate frozen Stage33 three-view six-class VNAT representation for application-level Unknown detection under frozen Stage14B medium_seed2025 Unknown-Free protocol

## Data and split

- Stage 14B `medium_seed2025`: Known Train/Val/Test 15,704/1,960/1,960; frozen Unknown Test 3,825 (sftp/vimeo/zoiper). Stage 33 six-class head; no new split or training.

## Configuration and execution

- `EXPERIMENT_PLAN.md` freezes four scores, Known-Val-only P95 thresholds and all role boundaries before Unknown features are opened. `run_pilot.py preflight` passed exact Stage14B freeze/manifest, Stage31 branch and Stage33 head checkpoint hashes and 15,704/1,960/1,960/3,825 role counts. Named `stage36_knownval_parity` passed exact raw-packet parity on 11 Known Validation flows × six arrays = 66 comparisons. A 30-minute-bounded GPU2 Known-only smoke passed on 1,960 Known Validation flows: frozen 128-D fused features and historical logits matched exactly (maximum absolute difference 0). The first smoke launcher failed only because of a relative selector path; the failure log was preserved and the corrected retry passed. GPU2 was released. `stage36_conditional_queue` waits for Stage34 and Stage35 independent verification PASS, then uses only GPUs0/1 for Unknown evaluation.

## Core results

- Pending. Unknown packet values and detection outcomes have not yet been read.

## Preserved evidence

- `manifest.json`, `EXPERIMENT_PLAN.md`, `preflight.json`, `parity.json`, `known_smoke.json`, Known-Val parity cache, implementation scripts, named project-local tmux logs. The failed selector-path attempt and successful retry are both retained.

## Limitations

- Application-level Unknown, not Service-level LOSO; Vimeo shares Streaming semantics with Known classes. Single seed; Stage 33 taxonomy is post-hoc development and VNAT is not capture-disjoint.

## Conclusion and next step

- The conditional queue is waiting for Stage34/35 terminal verification. It will then run Unknown input recovery, frozen score evaluation, independent score replay and artifact validation; do not infer Unknown detection quality from closed-set Macro-F1.
