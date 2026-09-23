# Stage 14C-3 — VNAT Closed-Set Failure Diagnosis

## Scope and verdict

**Final diagnosis: `ROOT_CAUSE_IDENTIFIED`.**

The F0/F2 deficit is not caused by a different frozen split, label mismatch, a
different byte representation, or a different optimizer/loss definition. The
dominant causes are the changed training policy: batch size 512 reduces the
number of optimizer updates per epoch by about four times, patience-5 early
stopping terminates every F0/F2 run at epoch 9–30, and protocol-dependent
training seeds expose a highly unstable optimization path. Consequently, all
F0/F2 runs stop before the epoch-50 scheduler milestone and the first prototype
reset. Class imbalance is real and explains which classes are difficult, but it
is shared by native Open-Detect and therefore does not explain the method gap.

This is a Known-Train/Known-Validation-only diagnostic. Known Test, Unknown
Test, DES and open-set evaluation were not used.

## 1. Aggregate gap

| Method | Val Accuracy | Val Macro-F1 | Val Weighted-F1 | Macro-F1 vs F0 |
|---|---:|---:|---:|---:|
| F0 | 0.799710 | 0.614834 | 0.770825 | — |
| F2 | 0.820070 | 0.652969 | 0.796212 | +0.038136 |
| Native Open-Detect | 0.868061 | 0.830904 | 0.859478 | +0.216070 |

The native gap is `0.216070` over F0 and `0.177935` over F2. F2 therefore
provides a modest independent average gain, but does not repair the training
failure.

| Setting | F0 Macro-F1 | F2 Macro-F1 | Native OD Macro-F1 |
|---|---:|---:|---:|
| Low | 0.594534 | 0.647453 | 0.790635 |
| Medium | 0.559701 | 0.621443 | 0.829624 |
| High | 0.690267 | 0.690012 | 0.872453 |

## 2. Configuration diff

The exhaustive parameter table is in `training_config_diff.md`. Material
differences are:

| Factor | Native OD | F0/F2 | Finding |
|---|---|---|---|
| Batch size | 128 | 512 | F0/F2 receive about one quarter of the updates per epoch |
| Epoch policy | fixed 100 | max 100, patience 5 | F0/F2 actually stop at epoch 9–30 |
| Training seed | fixed 2022 | protocol seed | creates strong run-to-run optimization sensitivity |
| Checkpoint rule | max Val Accuracy | max harmonic Accuracy/Macro-F1 | not a positive explanation in ablation |
| Logvar | raw | upper clamp at 20 | guard activated once in 30 F0/F2 runs |
| Workers | 4 | 0 | runtime/RNG implementation difference; not isolated as a loss change |

Architecture, F0 input bytes, loss, Adam parameters, learning rate,
weight decay, scheduler definition, initialization, dropout, sampler and label
encoding are otherwise equal. The PIL-versus-tensor augmentation audit checked
100 samples and found 0 mismatches with maximum absolute difference 0.

## 3. Data alignment

All 30 OD-versus-F0/F2 comparisons passed. For every protocol, the following
are exactly equal and independently hashed:

- Known class list;
- ordered Train and Validation sample IDs;
- ordered labels;
- per-class counts;
- combined data manifest.

The Stage 14B freeze hash remains
`c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
Thus the gap is not a split, label, or sample-order problem.

## 4. Feature and normalization audit

- F0 arrays are byte-identical to the native OD arrays and are divided by 255
  exactly once.
- Every F2 scaler was recomputed from Known Train only; all 15 matched the
  frozen scaler, and the same scaler is used for Train and Validation.
- No NaN/Inf, duplicate normalization, or unbounded raw flow statistic was
  found.
- Mean standardized Train/Val distribution shift is similar: F0/OD `0.284574`,
  F2 `0.286150`.
- F2 places statistics in only 64/1024 positions (`6.25%`), but those slots
  contain `35.45%–36.41%` of input squared energy and `31.80%–42.32%` of
  gradient-times-input attribution in the two audited protocols. The feature is
  bounded but can dominate the input locally.

Therefore normalization is valid, while F2's injection strength is a plausible
source of its seed-dependent behavior. It is not the cause of the F0/native gap.

## 5. Training dynamics

| Method | Mean actual stop epoch | Range | Mean optimizer updates | Diagnosis counts |
|---|---:|---:|---:|---|
| F0 | 16.13 | 9–23 | 415.47 | 11 underfitting, 4 normal |
| F2 | 18.93 | 10–30 | 471.00 | 12 underfitting, 3 normal |
| Native OD | 100 | 100 | 9986.67 | 1 underfitting, 14 normal |

All F0/F2 runs stop before the scheduler milestone and prototype reset at epoch
50. Training and validation metrics remain finite; there is no systematic NaN
failure. The large nominal `epochs=100` similarity is therefore misleading:
the effective update budgets differ by roughly an order of magnitude or more.

The two highlighted failures do not reproduce under native training:

| Protocol | F0 Macro-F1 | F2 Macro-F1 | Native OD Macro-F1 |
|---|---:|---:|---:|
| Medium-2024 | 0.334134 | 0.373376 | 0.853112 |
| Medium-2025 | 0.284444 | 0.653313 | 0.739755 |

## 6. Controlled one-factor ablation

The normal protocol is `Medium-2026`; the weak protocol is `Medium-2025`.
D2 and D3 were not rerun because optimizer/LR and scheduler definitions are
already exactly identical to D0. D8 is the frozen native reference.

| Variant | Only changed factor | Weak Macro-F1 | Δ vs D0 | Normal Macro-F1 | Δ vs D0 |
|---|---|---:|---:|---:|---:|
| D0 | Current F0 | 0.284444 | — | 0.923316 | — |
| D1 | Batch 512→128 | 0.487150 | +0.202706 | 0.930687 | +0.007372 |
| D4 | Checkpoint: Accuracy | 0.284444 | +0.000000 | 0.886568 | -0.036748 |
| D5 | Remove logvar clamp | 0.321001 | +0.036557 | 0.923316 | +0.000000 |
| D6 | Training seed→2022 | 0.666589 | +0.382146 | 0.816054 | -0.107261 |
| D7 | Disable early stop; 100 epochs | 0.553769 | +0.269325 | 0.976402 | +0.053086 |
| D8 | Full native OD reference | 0.739755 | +0.455311 | 0.983318 | +0.060002 |

Interpretation:

- D1 explains `44.5%` of the weak D0→D8 gap by changing batch size alone.
- D7 explains `59.2%` of the weak gap and `88.5%` of the normal gap by removing
  early stopping alone.
- D6 explains `83.9%` of the weak gap but damages the normal run, directly
  demonstrating high seed sensitivity rather than a universally superior seed.
- D4 and D5 do not explain the aggregate deficit.
- No one factor fully reproduces native OD; the gap is the interaction of update
  budget, premature stopping, and stochastic optimization policy.

## 7. Class-level and imbalance diagnosis

Mean recall across protocols for the requested classes:

| Class | F0 | F2 | Native OD |
|---|---:|---:|---:|
| netflix | 0.295455 | 0.413636 | 0.836364 |
| rdp | 0.204545 | 0.204545 | 0.704545 |
| rsync | 0.390290 | 0.374108 | 0.545455 |
| scp | 0.707424 | 0.751528 | 0.744541 |
| sftp | 0.374096 | 0.433735 | 0.471084 |
| skype | 0.839683 | 0.949206 | 0.997619 |
| youtube | 0.596257 | 0.692513 | 0.978610 |

The largest F0 deficits are netflix, rdp and youtube. The persistent semantic
confusion cluster is rsync/scp/sftp, especially `rsync→scp`, `sftp→scp`, and
their reverse directions. F0/F2 additionally intensify netflix/youtube
confusion.

There are no class weights or weighted samplers in any method. Across the
frozen protocols, ssh can contribute about 67% of Train samples while rdp can
contribute around 0.4%; hence the unweighted objective is dominated by large
classes. This affects minority recall, but because native OD uses the same
unweighted sampler and still performs much better, imbalance is a secondary
difficulty rather than the root of the implementation gap.

## 8. Required answers

1. **Data problem?** No. IDs, labels, class lists, counts and hashes match.
2. **Feature preprocessing/normalization?** Not for F0. F2 scaling is valid but
   its 6.25% injected slots receive disproportionate attribution.
3. **Optimizer/LR/batch/scheduler?** Optimizer/LR/scheduler definitions match.
   Batch 512 materially reduces update count; early stopping prevents the
   scheduler and prototype resets from ever executing.
4. **Checkpoint/early stopping?** Checkpoint rule is not the cause; patience-5
   early stopping is a major cause.
5. **Class imbalance?** Real and harmful, but secondary and shared by native OD.
6. **Code bug/data错位?** No data错位 or arithmetic bug was found. The failure
   is a training-policy implementation mismatch relative to native OD.
7. **What explains the 0.18–0.22 gap?** Primarily the combined effective update
   budget (`batch=512` plus premature stopping) and seed-sensitive optimization.
8. **Does F2 provide independent gain?** Yes, average Macro-F1 `+0.038136` and
   Weighted-F1 `+0.025387` over F0, but the gain is not consistent enough to
   replace the corrected native training policy.
9. **Recommended next action:** use the native Open-Detect training pipeline as
   the authoritative baseline (batch 128, fixed seed policy, full 100 epochs,
   native checkpoint rule). Retain F0 only as the diagnosed legacy baseline;
   keep F2 as a separate feature ablation, not as the formal encoder until it is
   reevaluated under the corrected native training policy using Known Val only.

## 9. Integrity statement

- Known Test samples read: `0`
- Unknown Test samples read: `0`
- DES executed: `false`
- Stage 14B protocol modified: `false`
- Formal new ablation runs: `10/10 success`
- First four-way launch: stopped by user, preserved under
  `interrupted_attempts/`, excluded from formal tables
- Formal GPU policy: physical GPU 4 and 5 only, one serial queue per GPU
- Freeze hash after all audits: unchanged

**Final conclusion: `ROOT_CAUSE_IDENTIFIED`.**
