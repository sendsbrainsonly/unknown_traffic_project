# Stage 14C — Released Open-Detect on VNAT

## Scope and integrity

- Stage 14B freeze hash before/after: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` (PASS).
- Runs completed: 15/15.
- Model-visible data: Known Train and Known Validation only.
- Unknown samples used in training/validation: 0/0.
- Known Test samples used: 0.
- DES and open-set evaluation: not run.
- Checkpoint selection: maximum Known Validation Accuracy only.

## Aggregate Known Validation results

| Setting | Runs | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Low | 5 | 0.878106 +/- 0.040043 | 0.790635 +/- 0.073635 | 0.868515 +/- 0.039255 |
| Medium | 5 | 0.838928 +/- 0.160893 | 0.829624 +/- 0.111139 | 0.829615 +/- 0.158438 |
| High | 5 | 0.887148 +/- 0.091585 | 0.872453 +/- 0.078039 | 0.880303 +/- 0.093890 |
| Overall | 15 | 0.868061 +/- 0.103538 | 0.830904 +/- 0.089524 | 0.859478 +/- 0.103120 |

## Lowest Macro-F1 runs

| Protocol | Accuracy | Macro-F1 | Weighted-F1 | Best epoch | Anomaly |
|---|---:|---:|---:|---:|---|
| low_seed2025 | 0.841957 | 0.681934 | 0.836487 | 68 | none |
| medium_seed2023 | 0.587112 | 0.704279 | 0.587748 | 57 | none |
| medium_seed2025 | 0.896939 | 0.739755 | 0.862802 | 53 | none |
| high_seed2025 | 0.903427 | 0.765535 | 0.885933 | 85 | none |
| low_seed2026 | 0.904206 | 0.780423 | 0.871694 | 39 | none |

## Known-Validation comparison with the earlier F0 implementation

- Mean native-minus-prior F0 Accuracy: `+0.068351`.
- Mean native-minus-prior F0 Macro-F1: `+0.216070`.
- Mean native-minus-prior F0 Weighted-F1: `+0.088653`.
- Native Macro-F1 wins: `14/15` protocols.
- Native-minus-prior F2 overall Accuracy/Macro-F1/Weighted-F1: `+0.047991` / `+0.177934` / `+0.063266`.
- This is not a pure feature ablation: arrays and Known Validation splits match, but training seed policy, batch size, early stopping, numerical guard, and checkpoint rule differ.

## Weak-run and class-level diagnosis

- No run has Macro-F1 below 0.35 and no run crashed or produced non-finite metrics.
- `medium_seed2023` is nevertheless a clear weak run by Accuracy (0.587112) and Weighted-F1 (0.587748). Its rsync/scp/sftp recalls are 0.424084/0.427948/0.403614, while vimeo/youtube/zoiper are 0.966942/1.000000/0.989247. The 4-sample rdp validation support also makes its class estimate fragile.
- Across protocol appearances, mean recall is lowest for sftp (0.471084) and rsync (0.545455); rdp is unstable (mean 0.704545, range 0.25-1.00).

## Answer to the bottleneck question

The evidence favors the earlier implementation/training policy as the larger source of the previously low closed-set result: with the same F0 arrays and Known Validation splits, released-code training improves mean Macro-F1 by +0.216070 and wins 14/15 protocols. Its Macro-F1 is also +0.177934 above the earlier F2 mean. This does not prove that VNAT is easy: sftp, rsync, rdp, and the Medium-2023 composition remain genuinely difficult or unstable. Because the comparison changes multiple training details at once, it identifies an implementation/policy effect but cannot isolate which single detail caused it.

## Interpretation boundary

This is the released-code Open-Detect baseline on frozen VNAT protocols. It does not use Unknown Test and therefore makes no unknown-detection claim. The comparison with earlier F0/F1/F2/F3 Known Validation runs is performed separately and cannot attribute causality from a single confounded comparison.
