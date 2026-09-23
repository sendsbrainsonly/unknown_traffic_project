# Stage 14C.5 — VNAT Feature Representation Audit

This is a **Known-only feature ablation**. It reuses the 15 frozen Stage 14B
protocols and the exact Stage 14C Open-Detect architecture/training policy.
Known Test and Unknown Test are outside the model-visible boundary.

## Frozen comparison

- F0: existing Stage 14C byte image (first 8 packets, 80 header bytes + 48
  payload bytes per packet).
- F1: F0 plus per-packet inter-arrival time (IAT).
- F2: F0 plus 8 flow statistics: packet count, total bytes, duration, packet
  size mean/std, IAT mean/std, and forward/reverse packet-count ratio.
- F3: F0 plus explicit first-8 packet length, direction, IAT, valid-packet
  mask, and all 8 flow statistics.

The input remains one `32 x 32` uint8 channel for every feature version. F1,
F2, and F3 write only into the 8 IPv4 source/destination address bytes that F0
already zeroes in every 128-byte packet block. Therefore no non-zero F0 byte is
discarded and the encoder architecture is unchanged.

Numeric transforms are frozen before validation evaluation:

1. non-negative timing, length, count, byte and duration values use `log1p`;
2. forward/reverse ratio uses `log((forward+1)/(reverse+1))`;
3. median and IQR are fitted on that protocol's Known Train only;
4. robust z values are clipped to `[-4, 4]` and mapped to uint8 `[0, 255]`.

F0 checkpoints and aggregate validation metrics are reused from Stage 14C;
they are not retrained. They are evaluated again on Known Validation only to
produce per-class metrics and confusion matrices. F1–F3 are trained from
scratch using the same seed, architecture, augmentation, loss, optimizer,
scheduler, maximum epochs, and early-stopping policy as Stage 14C.

## Pre-registered interpretation

- Every run and every feature/setting summary reports both **Macro-F1** and
  **Weighted-F1**. Macro-F1 remains the primary decision metric because it gives
  each application equal weight; Weighted-F1 is reconstructed from per-class
  validation F1 using `val_support` and is reported to expose majority-class
  dominance. Adding Weighted-F1 does not change checkpoint selection, early
  stopping, or any pre-registered decision threshold.
- A run analogous to the two weak Stage 14C Medium runs is defined as
  `Known Val Macro-F1 < 0.35`.
- A per-run minority class has Known-Train support below that run's median
  class support. Minority recall is macro-averaged over those classes.
- Feature representation is called the **primary bottleneck** only if the best
  non-F0 version has overall mean Macro-F1 gain at least `+0.05`, wins at least
  `12/15` paired runs, improves minority recall at least `+0.05`, and reduces
  within-setting seed standard deviation in at least `2/3` settings.
- A new feature is recommended for freezing only if its overall mean Macro-F1
  gain is at least `+0.03`, it wins at least `10/15` paired runs, and no setting
  mean falls by more than `0.01` versus F0.
- Stability is reported as the mean of the three within-setting seed standard
  deviations; lower is more stable. Mean performance and worst-run performance
  are always reported beside it.

No Unknown score, Unknown inference, Known Test metric, or test-tuned decision
is produced by this experiment.
