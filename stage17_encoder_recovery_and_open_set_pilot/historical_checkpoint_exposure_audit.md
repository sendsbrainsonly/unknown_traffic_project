# Historical checkpoint exposure audit

- `pretrained_model.bin`: official downloadable TrafficFormer pretraining artifact, loadable, but the public README does not identify the pretraining corpus. Status: `PRETRAINING_EXPOSURE_UNVERIFIED`. It is excluded from the strict Stage17 comparison.
- `finetuned_model.bin`: trained on the project USTC-20 closed-set split; wrong labels and dataset for Service-LOSO. Engineering smoke only.
- `modelB_best.pt`: trained on USTC-20; wrong labels and dataset for Service-LOSO. Engineering smoke only.
- Model C: no historical checkpoint found.
- E0: Stage16S checkpoint is protocol-specific and already passed strict Unknown-Free checks.

No historical A/B/C checkpoint is inserted into support fitting, normalization, classifier fitting or threshold calibration in the strict pilot.
