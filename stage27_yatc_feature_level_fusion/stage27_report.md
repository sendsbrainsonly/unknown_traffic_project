# Stage 27 — Frozen YaTC feature-level fusion

Claim scope: development diagnostic; Stage20 Test was previously exposed.

All four YaTC and E3 encoders/checkpoints remained frozen. Three newly trained Known-only feature heads were compared on exactly matched Stage20 flows. Stage26 probability averaging is a separate control, not this feature fusion.

## Known Test Macro-F1 (mean of seeds 2022/2023)

| Dataset | Labels | E3 | YaTC | F1 concat | F2 equal-projected | F3 feature gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iscx_vpn | fine | 0.859493 | 0.887085 | 0.888900 | 0.884258 | 0.880846 |
| iscx_vpn | coarse | 0.896004 | 0.923689 | 0.916859 | 0.913219 | 0.913114 |
| iscx_tor | fine | 0.822178 | 0.827393 | 0.837946 | 0.842260 | 0.821191 |
| iscx_tor | coarse | 0.852783 | 0.876259 | 0.870674 | 0.871719 | 0.854625 |

## Paired interpretation

- F1_LinearConcat − YaTC, iscx_vpn fine: mean +0.001815; positive seeds 1/2.
- F1_LinearConcat − YaTC, iscx_tor fine: mean +0.010553; positive seeds 2/2.
- F2_EqualProjected − YaTC, iscx_vpn fine: mean -0.002827; positive seeds 1/2.
- F2_EqualProjected − YaTC, iscx_tor fine: mean +0.014866; positive seeds 2/2.
- F3_FeatureGate − YaTC, iscx_vpn fine: mean -0.006239; positive seeds 1/2.
- F3_FeatureGate − YaTC, iscx_tor fine: mean -0.006202; positive seeds 0/2.

Feature concatenation and feature gating are distinct from Stage26 output-score averaging. The gate is learned from Known Train features only; no ER-CMGI latent entropy, diffusion, or Unknown data was used.
The prior Stage20 Test exposure precludes independent confirmation and must not be used to revise the formula or choose a new width/weight.

## Class-level fine-label change

Known Test class F1 averaged over the two fixed seeds; deltas are relative to frozen YaTC.

| Dataset | Service | E3 | YaTC | F1 concat−YaTC | F2 projected−YaTC | F3 gate−YaTC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iscx_vpn | Chat | 0.9092 | 0.9077 | +0.0318 | +0.0281 | +0.0244 |
| iscx_vpn | Email | 0.9258 | 0.9304 | +0.0167 | +0.0155 | +0.0034 |
| iscx_vpn | File-Transfer | 0.7228 | 0.7967 | -0.0223 | -0.0274 | -0.0241 |
| iscx_vpn | P2P | 0.9975 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| iscx_vpn | Streaming | 0.9631 | 0.9667 | +0.0061 | +0.0013 | -0.0026 |
| iscx_vpn | VoIP | 0.6385 | 0.7210 | -0.0214 | -0.0344 | -0.0386 |
| iscx_tor | Browsing | 0.8224 | 0.8733 | -0.0278 | -0.0243 | -0.0504 |
| iscx_tor | Chat | 0.5851 | 0.5818 | +0.0178 | +0.0408 | -0.0076 |
| iscx_tor | Email | 0.9537 | 0.9322 | +0.0264 | +0.0308 | +0.0259 |
| iscx_tor | File-Transfer | 0.9424 | 0.9723 | -0.0218 | -0.0207 | -0.0285 |
| iscx_tor | P2P | 0.9027 | 0.9171 | +0.0012 | -0.0003 | -0.0118 |
| iscx_tor | Streaming | 0.7661 | 0.7841 | +0.0126 | +0.0098 | -0.0152 |
| iscx_tor | VoIP | 0.7828 | 0.7310 | +0.0656 | +0.0681 | +0.0443 |

## Gate behavior and interpretation

- iscx_vpn/2022 mean gate TrafficFormer/FIG/YaTC: 0.999998/0.000001/0.000001.
- iscx_vpn/2023 mean gate TrafficFormer/FIG/YaTC: 0.543411/0.000084/0.456505.
- iscx_tor/2022 mean gate TrafficFormer/FIG/YaTC: 0.999948/0.000007/0.000045.
- iscx_tor/2023 mean gate TrafficFormer/FIG/YaTC: 0.999982/0.000002/0.000015.

The gate often collapses toward TrafficFormer, so the adaptive mechanism did not consistently exploit YaTC or FIG. This is an observation, not a reason to retune it on the exposed Test set.
No open-set or Unknown detection result is produced by Stage 27.
