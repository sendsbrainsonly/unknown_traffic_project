# Stage49 finalize recovery attempt 1

This metadata-only recovery preserves the original
`stage49_selected_finalize_20261117` exit-1 evidence. The numerical finalizer had
already generated and independently replayed the saved Test predictions, logits,
metrics, per-class table, and confusion matrix. It failed only because 21 tmux
log artifacts were recorded as absolute `path` values, while the project bundle
validator accepts relative artifact paths.

`recover.py` performs no training, feature extraction, model loading, GPU
inference, sample selection, or metric recomputation. It:

1. verifies the existing completion receipt and frozen source/checkpoint hashes;
2. verifies that all completed Stage49 worker sessions exited successfully and
   that no Stage49 worker/controller remains alive;
3. records hashes of the original failed finalizer/controller evidence;
4. converts external artifact paths to bundle-relative references, deduplicates
   them, and adds the original failure evidence;
5. refreshes and validates the experiment bundle twice before marking the
   metadata-only recovery successful.
