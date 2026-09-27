# Stage 26 — Frozen YaTC + E3 probability fusion

This is a matched Stage20 development experiment; Test was previously exposed. No encoder training, weight search or Unknown use.

## Full-flow Known Test Macro-F1 (mean ± population std, two seeds)

| Dataset | Label task | E3 | YaTC | Equal-weight fusion | ΔFusion−YaTC |
| --- | --- | ---: | ---: | ---: | ---: |
| iscx_vpn | fine | 0.859493 ± 0.007015 | 0.887085 ± 0.011593 | 0.878475 ± 0.000945 | -0.008610 |
| iscx_vpn | coarse | 0.895838 ± 0.004075 | 0.923689 ± 0.005804 | 0.906884 ± 0.000635 | -0.016805 |
| iscx_tor | fine | 0.822178 ± 0.007643 | 0.827393 ± 0.000774 | 0.848813 ± 0.003996 | +0.021420 |
| iscx_tor | coarse | 0.852381 ± 0.001372 | 0.875886 ± 0.004282 | 0.877392 ± 0.005296 | +0.001506 |

## Primary gate and interpretation

- Gate: `FUSION_NOT_CONFIRMED`. Coarse ΔFusion−YaTC by dataset: VPN -0.016805, Tor +0.001506; positive paired cells 1/4.
- Fine task is retained as a guardrail; both original model checkpoints are unchanged. A two-model ensemble adds inference and storage cost; it is not a single improved E3 encoder.
- Coarse probabilities are obtained by summing the three Communication Service probabilities before argmax. This differs from Stage25's hard-prediction remap; compare within this Stage26 table only.
- Individual results, per-class metrics, conditional ≥2-packet results and sample-level rescue/degradation are preserved in the CSVs. Any isolated Test gain is development evidence, not a new weight-selection signal.
- YaTC and E3 use different representations, pretraining and training budgets; this comparison cannot attribute a gain or failure to one architectural module.

## Integrity

- Four/four frozen checkpoint pairs passed exact historical Validation/Test argmax parity; all per-run source SHA256 values unchanged.
- No Unknown data, Test fitting, encoder training, weight tuning or threshold calibration.

## Paired error and cost diagnosis

- VPN coarse Test: Fusion rescues 27 YaTC errors in Communication but loses 42 YaTC-correct decisions; for File-Transfer it rescues 20 and loses 44. P2P is unchanged. The degradation is concentrated in exactly the two classes where YaTC previously led E3.
- Tor coarse Test: Communication gains 68 Fusion-only versus 38 YaTC-only decisions, but File-Transfer loses 21 versus 7 rescued; Browsing and Streaming are nearly balanced. This explains the small mean gain and the seed inconsistency.
- The frozen YaTC checkpoint is about 7.46 MB; E3's required TrafficFormer E1 checkpoint is about 528.59 MB, plus small E2/E3 heads. Recorded 3.4–4.0 s per run covers cached E3 embeddings and YaTC inference, **not** end-to-end E3 feature extraction or a deployment latency benchmark.
- `independent_verification.json` replays 26,556 model decisions and all 96 metric rows from saved logits; it also checks 60 source-hash entries.
