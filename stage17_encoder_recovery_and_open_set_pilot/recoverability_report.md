# Recoverability report

- Model A: `CODE_AND_HISTORICAL_CHECKPOINT_RECOVERED`; strict LOSO requires retraining.
- Model B: `CODE_AND_HISTORICAL_CHECKPOINT_RECOVERED`; it is an independent reconstruction and strict LOSO requires retraining.
- Model C: `CODE_RECOVERED_CHECKPOINT_NOT_AVAILABLE`; execution can reproduce its documented concat probe without inventing a new model.
- E0: `FROZEN_STAGE16S_REUSABLE`.

Historical A/B weights are loadable but not compatible with current label space. The official TrafficFormer pretraining exposure is unverified, so strict pilot uses random initialization.
