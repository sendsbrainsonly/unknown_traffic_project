# Stage 14C — Released Open-Detect on frozen VNAT

This experiment measures **closed-set Known Validation representation quality**
for the released Open-Detect implementation on the 15 frozen VNAT Stage 14B
protocols. It is an external-dataset baseline, not a claim that the paper's
original dataset result has been reproduced.

## Frozen boundary

- Stage 14B freeze hash:
  `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`
- Model-visible data: each protocol's existing `Known Train` and
  `Known Validation` arrays only.
- `Known Test` and `Unknown Test` arrays are not loaded or evaluated.
- Unknown application names are read only from the frozen protocol document to
  prove class disjointness; no Unknown sample bytes or scores are accessed.
- No Open-Detect source file and no Stage 14B file is modified.

## Meaning of "Open-Detect original"

The baseline follows the released author code behavior, including its published
implementation quirks; it is not the locally corrected-paper variant. The thin
VNAT adapter imports the existing read-only `Open-Detect/code/model.py` and the
reproduction helpers that mirror the official training loop.

Frozen training configuration:

- input: first 8 packets, 80 header bytes + 48 payload bytes per packet,
  IPv4 addresses masked, zero-padded to a 32x32 grayscale image;
- train augmentation: `RandomCrop(32, padding=4)` and
  `RandomHorizontalFlip()`; validation uses `ToTensor()` only;
- model: released `OpenDetectNet`, ResNet-18 VAE, latent dimension 128;
- loss: released generative/discriminative objective, `lambda=0.005`,
  temperatures 1/1;
- Adam, learning rate 0.001, betas 0.9/0.999;
- MultiStepLR milestones 50 and 80, gamma 0.1;
- train batch 128, validation batch 64, workers 4;
- DataLoader temporary IPC remains physically under this experiment directory,
  but is addressed through a short `/proc/self/fd/<dirfd>` alias to avoid the
  host's `AF_UNIX` path-length limit; default tensor sharing, worker count, and
  batches are unchanged;
- exactly 100 epochs, no early stopping;
- official fixed training seed 2022 for every frozen protocol;
- official prototype replacement after zero-based epochs 50 and 80. The
  optimizer is deliberately not rebuilt, matching released behavior;
- checkpoint selected only by maximum Known Validation Accuracy.

Macro-F1, Weighted-F1, and per-class recall are post-hoc metrics of the selected
checkpoint and never influence training or checkpoint selection.

The first launch attempt is retained under `.tmux-task/stage14c_native_gpu*/`
and its empty run scaffolds under `failed_attempts/afunix_initial/`. A subsequent
one-batch `file_system` smoke failure is retained in
`.tmux-task/stage14c_native_ipc_smoke/`. Neither reached a completed epoch; these
are environment-failure evidence, not scientific runs.
