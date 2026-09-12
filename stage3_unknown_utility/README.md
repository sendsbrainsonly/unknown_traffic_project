# Stage 3 Strict Unknown-Free Utility Test

Core question: Does multi-local-support modeling reduce unknown false acceptance compared with a single global Gaussian, under strict unknown-free evaluation?

## Frozen Execution Plan

A-1, A-2, and A-3 are all primary benchmarks and were executed in that order. The execution supplement did not change any frozen class list or flow split.

## A-1

Known (19): Gmail, FTP, Nsis-ay, Facetime, Weibo, Cridex, Zeus, SMB, BitTorrent, WorldOfWarcraft, Shifu, Outlook, Virut, Geodo, MySQL, Htbot, Skype, Miuref, Neris
Unknown (1): Tinba
Single/Multi UFAR: 0.522353 / 0.623529; delta +0.101176.

## A-2

Known (17): Gmail, FTP, Nsis-ay, Facetime, Weibo, Cridex, Zeus, SMB, BitTorrent, WorldOfWarcraft, Shifu, Outlook, Virut, MySQL, Skype, Miuref, Neris
Unknown (3): Geodo, Htbot, Tinba
Single/Multi UFAR: 0.618211 / 0.620900; delta +0.002689.

## A-3

Known (15): Gmail, FTP, Nsis-ay, Facetime, Weibo, Cridex, Zeus, SMB, BitTorrent, WorldOfWarcraft, Shifu, Outlook, Virut, MySQL, Skype
Unknown (5): Geodo, Htbot, Tinba, Miuref, Neris
Single/Multi UFAR: 0.116437 / 0.093538; delta -0.022899.

## Strict Unknown-Free Audit

All three manifests passed the zero-Unknown assertions for encoder train/validation, scaler/PCA/density fitting, threshold calibration, and checkpoint selection. Unknown rows were evaluated only from the original test split.

## Single vs Multi Results

See `outputs/summary/cross_setting_results.csv` and each setting's `detector_metrics.csv`.

## Per-Unknown-Class Results

See each setting's `per_unknown_class_results.csv`, `decision_transition_matrix.csv`, and `unknown_absorption_by_known_class.csv`.

## Openness Trend

See `outputs/summary/openness_trend.md`; the three points are descriptive and no trend is forced.

## Final Gate

Gate D — MIXED. Multi local support utility proven: no.

## Limitations

The official Open-Detect sample-fold identities are unavailable. Results use official class-holdout lists with the frozen local 80/10/10 flow split and are not author-fold equivalent. Fixed K2 components are support-model parameters, not semantic subgroups.

A-2 exposed a finite-forward/non-finite-gradient overflow in `exp(logvar)`. The
two failed formal attempts are preserved under `outputs/A-2/failed_attempts/`.
The successful run uses the audited one-sided upper guard `logvar <= 20` and
recorded one guard activation; this numerical stabilization is part of the
local execution semantics and must be disclosed when comparing results.

## Next Step

This task stops here. Adaptive K, prototype-specific boundaries, Unknown discovery, CSTNET, and CipherSpectrum are explicitly outside this execution.
