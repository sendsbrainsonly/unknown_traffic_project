# Formal Checkpoint Selection

- best_epoch: 32
- best_val_accuracy: 0.986158249847
- best_val_macro_f1: 0.987371195898
- best_combined_score: 0.986764350130
- model_selection_rule: harmonic mean of validation Accuracy and Macro-F1; strict improvement
- checkpoint_path: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project/opendetect_ustc_encoder_audit/artifacts/latent_gaussian_audit/best_checkpoint.pt`
- checkpoint_sha256: `f8e53d36bd6dc8333451b2f159300bc5a2cd875fccaff01a2c6dc43356951b36`
- source_checkpoint_path: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project/opendetect_ustc_encoder_audit/artifacts/best_checkpoint.pt`
- source_checkpoint_sha256: `f8e53d36bd6dc8333451b2f159300bc5a2cd875fccaff01a2c6dc43356951b36`
- training_stop_epoch: 37
- early_stop_reason: validation combined score did not improve for 5 consecutive epochs
- early_stopping_patience: 5
- consecutive_non_improving_epochs_at_stop: 5

The checkpoint was frozen before latent extraction. No Gaussian result was used
for checkpoint selection. All downstream latent extraction must use the frozen
path and SHA-256 above.

## Known strict-reproduction limitation

At the user-requested GPU migration after epoch 5, the then-available checkpoint
did not contain optimizer or RNG state. Training resumed from model weights with
a reconstructed optimizer/scheduler state and a restarted shuffle/augmentation
stream. Later resumes restored model, optimizer, scheduler, and DataLoader RNG
state, but they cannot remove the epoch-5 discontinuity. The frozen checkpoint
is not modified to compensate for this limitation.
