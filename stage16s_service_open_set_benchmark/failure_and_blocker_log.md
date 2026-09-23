# Failure and blocker log

- Training completed and produced a best checkpoint for 18/18 runs.
- Original end-to-end script: 17 SUCCESS; Streaming-2023 preserved a post-training parity assertion failure.
- Streaming-2023 was not retrained. Canonical inference used its frozen epoch-79 checkpoint after diagnosing a one-ULP PIL/ToTensor versus manual float conversion difference (1/114 validation predictions).
- Canonical frozen-checkpoint inference: 18/18 successful; original failure evidence remains untouched under `runs/loso_streaming/seed2023/`.
- P2P LOSO is `PROTOCOL_LIMITED_SINGLE_CAPTURE`; its three runs are retained but cannot support cross-capture claims.
- Every protocol is flow-level and uses weak capture labels; capture generalization is not identifiable.
- No Unknown sample was used for training, support, normalization, percentile fitting, checkpoint selection, or threshold fitting.
- Optimization collapses, if any, are visible in `known_classifier_results.csv` and preserved training logs; no seed was removed.
