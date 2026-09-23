# Protocol Differences

## Validation early stopping added during the formal run

- The formal run now stops when the harmonic mean of validation accuracy and
  validation macro-F1 fails to strictly improve for 5 consecutive completed
  epochs (`patience=5`, `min_delta=0`). This balanced score penalizes either
  metric becoming a bottleneck; validation loss remains a separately reported
  diagnostic and is not mixed across incompatible scales.
- Validation remains evaluation-only: it selects the best checkpoint and the
  stopping epoch, but it is never used for gradient updates.
- The maximum remains 100 epochs. Early stopping is a local robustness-audit
  adaptation and is not claimed to be part of the released Open-Detect
  training protocol.
- The first early-stopping run resumed from the latest completed epoch 16
  checkpoint; model, optimizer, scheduler, and DataLoader generator states were
  all restored. Composite-score monitoring was introduced afterward and is
  therefore tracked from the selected checkpoint available at that transition.

## Fixed controls

- Dataset/sample universe: the existing 489,101 Stage-0 USTC flows.
- Label mapping: the historical 20-class map; SMB-1 and SMB-2 remain class SMB.
- Split: the existing Stage-1 391,280/48,910/48,911 train/validation/test
  flow-ID membership.
- Training context: all 20 classes.
- Gaussian diagnosis: FTP, Cridex, Miuref and Outlook only; train fit and
  held-out validation evaluation.
- TrafficFormer Stage 2.5 values are read-only inputs and are not recomputed.

## Necessary adaptations from released Open-Detect code

| Area | Released behavior | Audit behavior | Reason |
|---|---|---|---|
| USTC data | author NPZ arrays with no flow IDs and a different class order/sample universe | byte images reconstructed from existing Stage-0 raw frames, keyed by historical flow IDs | preserve the controlled sample universe and split |
| labels | author USTC order in `data/splits.py` | immutable main-project 20-class mapping | required control; prevents SMB 21-class drift |
| device | hard-coded `.cuda()` | explicit PyTorch device | portability and auditable GPU selection |
| model outputs | `forward` hides `mu/logvar` | adapter exposes unchanged `mu/logvar` computation | requested deterministic `mu_x` export |
| checkpoint | pickle whole object on best validation accuracy | state dict plus config/metrics on best validation accuracy; macro-F1 is reported | robust, auditable artifact; validation only |
| formal validation | author “test” NPZ is used as validation | fixed historical validation flow IDs | prevent test use |
| test | author evaluates open-set test | no test evaluation in the Gaussian audit | outside task scope |
| GMM | not part of author training code | train-only StandardScaler, train-only PCA64, full GMM K=1/2/3, seeds 0/1/2 | predeclared encoder robustness audit |

## Deliberately preserved released-code semantics

- custom ResNet18 encoder and lateral CVAE decoder;
- 128-dimensional `mu/logvar`;
- one learnable mean prototype per class with identity covariance;
- stochastic reparameterized `z` during training and deterministic `mu` in
  evaluation;
- MSE reconstruction, target-prior KL, released Euclidean discriminative term,
  and `lambda=0.005` total weighting;
- author training augmentation (random crop with padding 4 and horizontal
  flip), unless the smoke test shows it invalidates byte-image training;
- Adam learning rate 1e-3 and milestones 50/80;
- released decoder final-stage omission and single-element intra-softmax
  behavior, because changing them would produce a corrected-paper model rather
  than the actual released representation learner;
- prototype reset at epochs 50 and 80 with the released parameter-replacement
  semantics. This is explicitly recorded as an upstream limitation.

## Predeclared engineering/resource differences

- Formal batch size is 512 rather than the author default 128. This is a
  pre-training engineering decision for the fixed 391,280-row training universe
  (about 12.6 times the released USTC train array). It still yields about 38,200
  optimizer updates over 50 epochs and about 76,500 over 100 epochs, versus
  about 24,300 for the released
  31,126-row array at batch 128. The loss remains a per-batch mean and its
  definition is unchanged. A batch-1024 attempt was rejected by the finite-loss
  gate: fixed-seed replay found `logvar.exp()` overflow at update 4
  (`logvar_max=105.71`). Batch 512 stayed finite for a 100-update replay and
  showed a stabilizing log-variance trajectory; no numerical clamp or loss
  change was introduced.
- At the user's request, the first formal process was moved off physical GPU 1
  and forbidden from selecting GPUs 1 or 2. CUDA state cannot migrate between
  devices, so training resumed from the validation-best epoch-5 model weights.
  That pre-migration checkpoint did not yet contain Adam, scheduler or data-RNG
  state; these states were therefore reconstructed (`optimizer_state_restored=false`)
  and the shuffle/augmentation stream restarted. From the resumed run onward,
  a latest checkpoint stores model, optimizer, scheduler and data-generator
  state every epoch. This is a real reproducibility limitation introduced by
  the requested device migration, not an unreported exact continuation.
- DataLoader workers are fixed to 0. With project-local temporary directories,
  PyTorch multiprocessing exceeded the Unix-domain socket path limit. The
  memory-mapped 1 KB examples make single-process loading practical; this
  changes throughput only.
- Images are stored as project-local NumPy arrays rather than individual PNGs;
  pixel values are identical and this avoids nearly half a million filesystem
  objects.
- Formal epoch count initially targets the author default 100. Any bounded
  resource-driven reduction must be recorded before interpreting the result;
  smoke output is never promoted to a formal result.

## Comparison protocol difference already declared in the task

TrafficFormer Stage 2.5 used GMM seed 0 and `max_iter=200`. Open-Detect uses
seeds 0/1/2 and `max_iter>=300`. The immutable seed-0 TrafficFormer result is
the historical comparator; Open-Detect seed dispersion is reported separately.
