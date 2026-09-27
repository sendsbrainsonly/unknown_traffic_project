# Experiment results: stage33-vnat-six-class-single-seed-20260925-v1

- Status: `success / SIX_CLASS_EXPLORATORY_DIAGNOSTIC_COMPLETE`
- Experiment type: `closed-set-label-granularity-benchmark`
- Claim scope: `diagnostic; post-hoc after four-class Test exposure`
- Created (UTC): `2026-09-25T11:34:02Z`
- Objective: VNAT Medium-2025 frozen known-flow six-class equal-feature fusion closed-set evaluation

## Data and split

- Stage 14B VNAT `medium_seed2025`, seven Known applications and 15,704/1,960/1,960 frozen Known Train/Val/Test flows. Unknown applications `zoiper/vimeo/sftp` excluded.
- Six classes: `streaming={netflix,youtube}`; `rdp`, `rsync`, `scp`, `skype`, `ssh` remain separate. RDP support is only 36/4/4 Train/Val/Test.

## Configuration and execution

- Reuse frozen Stage 31 three-view embeddings/encoders and the exact Stage 32 T0 adapter/head training implementation, seed 2022. Train-only scalers, 30+30 epochs, Known-Val Macro-F1 checkpoint selection. See `run_six.py` and `README.md`.

## Core results

- Known Validation (n=1,960): Accuracy `0.972449`, Macro-F1 `0.956710`, Weighted-F1 `0.972422`.
- Known Test (n=1,960): Accuracy `0.963776`, Macro-F1 `0.943109`, Weighted-F1 `0.963750`.
- Known Test F1/support: streaming `1.0000/54`, rdp `1.0000/4`, rsync `0.8127/191`, scp `0.8460/229`, skype `1.0000/126`, ssh `1.0000/1,356`.
- All `71` Test errors are `rsync→scp` (`37`) or `scp→rsync` (`34`). See `runs/vnat/known_test_evaluation/confusion_matrix.csv`.
- The original four-class Stage 32 Known Test was 1.0000 on the same frozen flow membership. This is a **different label task**, so the numerical gap is evidence of increased label difficulty, not an estimate of model improvement/regression.

## Preserved evidence

- `manifest.json`, `README.md`, `run_six.py`, `verify_six.py`, `preflight.json`, frozen-source hashes, both selected checkpoints and their SHA256, complete 30+30 epoch history, Known Validation/Test predictions and logits, per-class results, confusion matrix and `completion_verification.json`. Named tmux logs/status are project-local under `.tmux-task/codex_stage33_vnat6_train_20260925/` and `.tmux-task/codex_stage33_vnat6_eval_20260925/`.

## Limitations

- The six-class choice followed observation of Stage 32 four-class Test accuracy 1.0; this is exploratory development, not untouched validation. VNAT is flow-random rather than capture-disjoint, and the RDP Test support is four. The perfect streaming/skype/ssh/rdp scores do not establish capture-independent generalization. One training seed cannot establish stability. No open-set detection was run.

## Conclusion and next step

- Six-class refinement exposed a concentrated rsync/scp difficulty that the four-class mapping hid. Independent 1,960-row saved-logit replay, confusion-matrix parity, selected-checkpoint hashes, Stage 32 source/cache hashes, and unchanged Stage 14B freeze hash `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` all passed. Stop here; a capture-disjoint protocol would require a separately frozen sensitivity study.
