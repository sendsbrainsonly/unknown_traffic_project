# Stage 44B — Open-Detect Native on the Frozen VNAT Service LOSO Protocol

## Fixed comparison

- Source pool: the exact 23,449 Stage 44 VNAT flow UIDs; four unchanged Service LOSO role manifests, seed 2022.
- Known Train/Validation/Test and Unknown Test are copied by flow ID from Stage 44, including its group/capture-disjoint assignment.
- The released Open-Detect F0 image encoding comes from the previously audited protocol-neutral 32×32 VNAT image cache. Only Known Train/Validation images and service labels are materialized for native training.
- Each fold trains its own Open-Detect ResNet-18 VAE, latent 128, with the unmodified Stage 14C native training loop: Adam 0.001, batch 128/64, 100 epochs, lambda 0.005, MultiStepLR 50/80, prototype resets after zero-based epochs 50/80, seed 2022, and maximum Known-Validation Accuracy checkpoint selection.
- Open-set score is the native minimum learned-prototype KL. Thresholds are Known-Validation P90/P95/P99 with NumPy `method="higher"`; the paired Stage 44 operating-point convention is `score > threshold`, with P95 primary.
- Known Test and Unknown Test are opened only after the fold's checkpoint is selected. Natural and the identical Stage 44 deterministic 1:1 evaluation membership are compared.
- Primary comparison: Open-Detect Native versus Stage 44 three-view DES-v1 on the same flow IDs and service roles. Scores, checkpoint selection, normalization, and threshold fitting never use Unknown or Test data.
- Single-GPU serial execution; no Stage 44 or Stage 14B frozen asset is overwritten. This remains a single-seed diagnostic on a previously developed dataset.
