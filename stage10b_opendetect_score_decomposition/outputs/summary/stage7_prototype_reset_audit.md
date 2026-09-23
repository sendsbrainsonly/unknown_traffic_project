# Stage 7 prototype reset static audit

- Scope: `POST_HOC_DIAGNOSTIC_ONLY`
- Verdict: **PROTOTYPE_OPTIMIZER_RISK (latent; not triggered in the frozen Low/Medium/High runs)**
- Scheduled reset indices: `50, 80` (zero-based), corresponding to logged epochs `51, 81`.
- Reset operation: `model.prototypes = released_reset_prototypes(...)`.
- Helper return: a new `nn.Parameter`; this is Parameter replacement, not an in-place copy.
- Optimizer behavior: the optimizer is created before the training loop and is not rebuilt or given the replacement Parameter. If a reset executes, subsequent optimizer steps retain the old prototype object and do not track the replacement prototype.
- Local USTC reproduction consistency: Stage 7 imports the same `released_reset_prototypes` helper from the audited/corrected USTC reproduction. That helper deliberately preserves the released author's Parameter-replacement behavior.
- Frozen-run impact: **none observed**, because early stopping ended every formal run before the first scheduled reset.

| Setting | Best epoch | Stop epoch | Actual resets |
|---|---|---|---|
| low | 26 | 31 | 0 |
| medium | 22 | 27 | 0 |
| high | 22 | 27 | 0 |

This is a static implementation risk marker only. Stage 7 was not changed or retrained.

Source hashes:

- `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project/stage3_unknown_utility/scripts/train_known_only.py`: `5e22b0d5804b227e7e58ba4a69ee71f41c4a8c1061ceff15308783d6863e28c8`
- `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project/opendetect_ustc_encoder_audit/scripts/train_opendetect.py`: `ea09f228256bbb447328a4e57b64e65a32028e0ead1d3d57df29c531449f2e92`
