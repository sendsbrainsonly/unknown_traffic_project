# Authorized TrafficFormer epoch1 selection

2026-11-17: the user explicitly requested using epoch1. At inspection, the original20-epoch job had completed17epochs and retained epoch1 as its in-memory best. It has not written any model checkpoint. Stopping that process now would discard the only held best state.

Reproduce one epoch in an isolated directory with the exact same official pretraining, frozen Train/Val arrays and ordering, seed2022, optimizer, batch size, dropout and original20-epoch scheduler horizon. Change only the outer loop endpoint; setting the configured horizon to one would incorrectly change the scheduler and is prohibited. Compare every stored epoch1 training/validation metric against the original log before accepting the replay. This is a deterministic replay with metric parity, not a direct extraction from the original process or proof of parameter-by-parameter equality.

Stop only the Stage49 parent orchestrator before the replay so it cannot advance using a later epoch. Keep the original TrafficFormer worker running during replay as a recoverability precaution; total Stage49 training concurrency remains at most two GPUs. After the first-epoch checkpoint/features pass verification, stop that exact original worker if still running, preserve its directory and complete available epoch history in an original-attempt directory, and promote the verified replay artifacts as the TrafficFormer branch.

Continue unchanged adapters/head, frozen Test packet extraction/evaluation and independent replay. Full-data protocol, all flow IDs, labels and group memberships remain unchanged. No Test information is used for this decision. Preserve all original source hashes. The finalizer wrapper changes only the expected TrafficFormer completed budget from20to1, records the original scheduler horizon20, and enforces best_epoch=1. Existing source files are not edited.

Expected selected epoch1: training Accuracy99.255419%, training Macro-F199.262119%, Validation Accuracy98.980801%, Validation Macro-F196.811379%. Accuracy and F1 parity tolerance1e-12; mean training loss tolerance1e-7. On mismatch, do not promote replay output or stop the original worker.
