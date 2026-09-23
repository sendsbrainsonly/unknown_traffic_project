# Experiment results: stage14c-native-opendetect-vnat-20260918-v1

- Status: `success`
- Experiment type: `closed-set-baseline`
- Claim scope: `approximate`
- Created (UTC): `2026-09-18T00:25:49Z`
- Completed (UTC): `2026-09-18T01:49:42Z`
- Objective: evaluate the released Open-Detect implementation's closed-set
  representation quality on the 15 frozen VNAT Stage 14B protocols.

## Data and split

- Stage 14B freeze hash remained
  `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Frozen protocol and split SHA256 values remained
  `5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced`
  and `66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e`.
- Model-visible roles were only Known Train and Known Validation.
- Unknown samples used in training/validation: `0/0` in all 15 runs.
- Known Test samples used: `0` in all 15 runs.
- No DES and no open-set evaluation ran.

## Configuration and execution

- Read-only Open-Detect code commit:
  `b50a18515a01799468c10f6c9b60c01f8a6a4e7c`.
- Released `OpenDetectNet`, ResNet-18 VAE, latent dimension 128.
- Original F0 32x32 byte image and official train augmentation.
- Adam, learning rate 0.001, batch 128, lambda 0.005, 100 epochs.
- MultiStepLR at epochs 50/80 and official prototype replacement after
  zero-based epochs 50/80.
- Official fixed training seed 2022 for every protocol.
- Checkpoint selection: maximum Known Validation Accuracy only.
- Macro-F1, Weighted-F1 and per-class Recall are post-hoc diagnostics and did
  not influence checkpoint selection.
- Physical GPUs: 0, 4, 5, 6 and 7; GPUs 1, 2 and 3 were not used.
- A project-local `/proc/self/fd/<dirfd>` temporary-directory alias preserved
  official four-worker DataLoaders while avoiding the host Unix-socket path
  limit. It did not alter model/data/optimization semantics.
- The user's later early-stop suggestion was not mixed into the in-flight grid:
  all 15 formal runs retained one uniform official 100-epoch policy.

## Core results

| Setting | Runs | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Low | 5 | 0.878106 +/- 0.040043 | 0.790635 +/- 0.073635 | 0.868515 +/- 0.039255 |
| Medium | 5 | 0.838928 +/- 0.160893 | 0.829624 +/- 0.111139 | 0.829615 +/- 0.158438 |
| High | 5 | 0.887148 +/- 0.091585 | 0.872453 +/- 0.078039 | 0.880303 +/- 0.093890 |
| Overall | 15 | 0.868061 +/- 0.103538 | 0.830904 +/- 0.089524 | 0.859478 +/- 0.103120 |

- Completed runs/checkpoints/hashes: `15/15/15`; failures and NaNs: `0`.
- No run has Macro-F1 below 0.35.
- `medium_seed2023` is a meaningful weak run by Accuracy `0.587112` and
  Weighted-F1 `0.587748`, although its Macro-F1 is `0.704279`.
- In that run, rsync/scp/sftp recalls are
  `0.424084/0.427948/0.403614`; vimeo/youtube/zoiper are near one.
- Across all appearances, mean Recall is lowest for sftp `0.471084` and rsync
  `0.545455`; rdp is unstable with range `0.25–1.00`.

## Comparison with prior Known-Validation experiments

- Native minus prior F0 mean Accuracy/Macro-F1/Weighted-F1:
  `+0.068351/+0.216070/+0.088653`.
- Native Macro-F1 wins over prior F0 in `14/15` matched protocols.
- Native minus prior F2 overall Accuracy/Macro-F1/Weighted-F1:
  `+0.047991/+0.177934/+0.063266`.
- This is not a pure one-factor ablation: the arrays and Known Validation split
  match for F0, but training seed policy, batch size, early stopping, numerical
  guard and checkpoint rule differ.

## Preserved evidence

- `outputs/stage14c_native_run_results.csv`: all 15 single-run results.
- `outputs/stage14c_native_setting_summary.csv`: setting and overall summaries.
- `outputs/stage14c_native_per_class_recall.csv`: per-run class metrics.
- `outputs/stage14c_native_application_recall_summary.csv`: application Recall.
- `outputs/stage14c_native_vs_prior_f0.csv`: matched Known-Val comparison.
- `outputs/stage14c_native_report.md`: full audit report.
- `outputs/input_audit.json`, `outputs/source_code_hashes.json`, and
  `outputs/native_config.json`: frozen inputs, source evidence and config.
- `runs/*/history.csv`, `result.json`, `per_class_metrics.json`, confusion
  matrices and 15 SHA256-addressed best checkpoints.
- First-launch AF_UNIX failure evidence is preserved under
  `failed_attempts/afunix_initial/` and `.tmux-task/stage14c_native_gpu*/`.

## Limitations

- This is an external VNAT benchmark using released code, not an exact
  reproduction of the paper's original dataset result.
- Only Known Validation was evaluated. There is no Known Test or Unknown Test
  performance claim and no open-set conclusion.
- The prior-F0 comparison changes multiple training-policy details, so it shows
  an implementation/policy effect but does not identify one causal detail.
- Validation support is very small for some class/protocol pairs (for example,
  rdp has support 4 in Medium-2023), making per-class estimates fragile.

## Conclusion and next step

The released Open-Detect implementation reaches overall Known-Validation
Accuracy `0.868061`, Macro-F1 `0.830904`, and Weighted-F1 `0.859478` on frozen
VNAT. The much larger improvement over prior F0/F2 results indicates that the
earlier implementation/training policy was the more prominent source of the
low closed-set result, although VNAT retains genuinely difficult and unstable
classes (especially sftp, rsync and rdp). Do not infer Unknown-detection quality
from this stage and do not automatically start open-set evaluation.
