# Stage 35 — Three-View Joint vs Staged Training

## Question and scope

On the **same frozen USTC A-2 17-class 10% flows** as Stage 34, does training
TrafficFormer, TAGCN, YaTC, the three view adapters, and the final fusion
classifier together improve closed-set classification relative to the current
staged recipe? Both variants classify from **three fused feature vectors**;
neither combines branch probabilities at inference.

This is a single-seed development ablation, not evidence that either training
strategy wins on all traffic datasets. CIC-IDS-2017 is reserved for a later
same-code extension after the USTC comparison is verified; no CIC result will be
imputed from USTC.

## Frozen inputs

- Stage 34 `ustc_a2_10pct_manifest.csv`, seed 2022: Known Train 34,665,
  Validation 4,333, Test 4,333; 17 classes.
- Stage 34 `input_caches/ustc/A-2/tf_fig` and `yatc_mfr` for Train/Validation.
- Stage 34's gated Test caches, opened only after the joint checkpoint is frozen.
- Official TrafficFormer and YaTC pretrained weights at their source-verified
  SHA256; TAGCN and fusion modules use seed 2022 initialization.
- Flow IDs, labels, class order, source and cache hashes are checked before
  training. No Unknown flow is used.

## Models

**S — current staged reference:** use Stage 34's existing three branch
checkpoints (TrafficFormer 20 epochs, TAGCN 50, YaTC 200) and its 30-epoch
per-view adapter plus 30-epoch `T0_equal` fusion head. Do not rerun or edit S.
Its branch validation metrics are training diagnostics; only its fused Test
predictions count as S's closed-set result.

**J — joint end-to-end:** instantiate the same raw encoders and final inference
path from their original initializations. TrafficFormer emits 768-D, TAGCN
128-D, and YaTC 192-D features. Fit each feature mean/std **once on Known Train
at initialization**, then hold those scalers fixed. Map each standardized view
through the Stage 34 two-layer 128-unit adapter to 64-D deterministic `mu`.
Multiply each `mu` by 1/3, concatenate the three 64-D vectors, and apply the
unchanged 192→128→128 fusion MLP and one final class head. Use final fused
cross-entropy only; gradients pass through all three original encoders.
Unused branch classification and adapter-logvar heads do not make predictions.

J trains for a fixed 20 epochs with effective batch 64; a memory-only smoke
selects microbatch 32 or 16 with accumulation 2 or 4. No validation metric is
used to choose batch size. TrafficFormer uses its 6e-5 optimizer/scheduler;
TAGCN and fusion modules use Adam 1e-3; YaTC uses AdamW 5e-4, weight decay
0.05 and layer decay 0.75. YaTC warmup is 2/20 epochs, preserving the 10%
fraction in the staged YaTC 20/200 warmup. All randomness uses seed 2022.
Run all 20 epochs with no early stopping; select the J checkpoint by fused
Known-Validation Macro-F1 only.

## Comparison and limits

Record Validation and one-shot matched Known Test Accuracy, Macro-F1,
Weighted-F1, per-class F1, paired sample decisions, elapsed time, GPU peak
memory and GPU-hours. Verify identical Test flow IDs and truths before any
paired difference. The Test is read only after both S and J checkpoints are
frozen. Stage 34 stays read-only.

The practical comparison changes both gradient flow and allocation of training
epochs: S gives the three original encoders 20/50/200 epochs and then trains
the fusion modules, whereas J gives every component 20 simultaneous epochs.
Thus a win or loss answers which **fixed recipe** works better on this USTC
protocol; it cannot isolate gradient coupling alone. We report compute cost
beside metrics and do not call a near-tie a universal training-law result.

## Execution gates

1. Preflight: frozen manifest/cache/weight hashes, ID alignment, class coverage,
   Train-only scaler fit and no Test/Unknown values loaded.
2. Memory smoke: one training step with microbatch 32; use 16 only if 32 fails
   for memory. Preserve each attempt and peak memory.
3. Train J on a live-selected GPU only when the project's owned GPU count stays
   at or below three; do not interrupt Stage 34 or other users.
4. Freeze J checkpoint and hashes; then evaluate the identical Known Test once.
5. Compare J with S by flow ID, preserve all predictions and failures, refresh
   and validate the experiment bundle, and update the project index.

No Unknown detection, new label split, probability ensemble, model selection on
Test, or changes to Stage 34 are part of this experiment.
