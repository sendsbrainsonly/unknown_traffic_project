# Stage 28 — ER-CMGI-inspired entropy fusion: Known-only development results

This is a mechanism adaptation, not an author-equivalent ER-CMGI reproduction.
The Stage20 Test was previously exposed and was not read in this experiment; Unknown usage was 0.

## Frozen inputs and execution

- Exact Stage20 Known Train/Validation flow IDs; frozen E3 and YaTC representations; 4 encoder pairs × 5 new training seeds = 20 units.
- Two 30-epoch Gaussian adapters (A0/A1); four 30-epoch fusion heads P0–P3. Matched F1/F2 heads were retrained at each new seed.
- All scaler/PCA/entropy calibration fit Known Train; checkpoints selected only by Known Validation Macro-F1.
- All 20 unit bundles passed saved-logit metric replay and independent CPU checkpoint replay (6 models per unit); frozen input hashes unchanged.

## Closed-set Known Validation results

| Dataset | Method | Macro-F1 mean ± SD | Accuracy mean | Weighted-F1 mean |
|---|---|---:|---:|---:|
| iscx_vpn | YaTC | 0.889929 ± 0.003184 | 0.881148 | 0.880159 |
| iscx_vpn | F1_LinearConcat | 0.884469 ± 0.013729 | 0.874590 | 0.874036 |
| iscx_vpn | F2_EqualProjected | 0.885249 ± 0.015605 | 0.875228 | 0.874890 |
| iscx_vpn | P0_A0_equal | 0.885486 ± 0.014346 | 0.875956 | 0.875530 |
| iscx_vpn | P1_A0_entropy | 0.885821 ± 0.013947 | 0.876321 | 0.875895 |
| iscx_vpn | P2_A1_equal | 0.885742 ± 0.012955 | 0.875956 | 0.875597 |
| iscx_vpn | P3_A1_entropy | 0.885803 ± 0.012915 | 0.876047 | 0.875663 |
| iscx_tor | YaTC | 0.864355 ± 0.002026 | 0.872540 | 0.871826 |
| iscx_tor | F1_LinearConcat | 0.867963 ± 0.002851 | 0.871914 | 0.873295 |
| iscx_tor | F2_EqualProjected | 0.867495 ± 0.004937 | 0.870394 | 0.871139 |
| iscx_tor | P0_A0_equal | 0.871728 ± 0.001966 | 0.874329 | 0.875573 |
| iscx_tor | P1_A0_entropy | 0.871566 ± 0.001772 | 0.874240 | 0.875491 |
| iscx_tor | P2_A1_equal | 0.870925 ± 0.003367 | 0.873166 | 0.873788 |
| iscx_tor | P3_A1_entropy | 0.871088 ± 0.003394 | 0.873435 | 0.874048 |

## Preregistered paired P3 comparisons

| Dataset | Control | Mean ΔMacro-F1 | Positive / 10 | Worst Δ | Pass |
|---|---|---:|---:|---:|---|
| iscx_vpn | YaTC | -0.004126 | 5/10 | -0.021740 | False |
| iscx_vpn | F1_LinearConcat | +0.001335 | 6/10 | -0.004684 | False |
| iscx_vpn | F2_EqualProjected | +0.000555 | 6/10 | -0.005023 | False |
| iscx_vpn | P2_A1_equal | +0.000062 | 3/10 | -0.000927 | False |
| iscx_tor | YaTC | +0.006733 | 9/10 | -0.002085 | True |
| iscx_tor | F1_LinearConcat | +0.003124 | 9/10 | -0.005716 | False |
| iscx_tor | F2_EqualProjected | +0.003593 | 8/10 | -0.009229 | False |
| iscx_tor | P2_A1_equal | +0.000163 | 2/10 | -0.000702 | False |

## Mechanism contrasts

| Dataset | Contrast | Mean ΔMacro-F1 | Positive / 10 |
|---|---|---:|---:|
| iscx_vpn | P1_A0_entropy − P0_A0_equal | +0.000335 | 4/10 |
| iscx_vpn | P2_A1_equal − P0_A0_equal | +0.000256 | 4/10 |
| iscx_vpn | P3_A1_entropy − P2_A1_equal | +0.000062 | 3/10 |
| iscx_vpn | P3_A1_entropy − P1_A0_entropy | -0.000018 | 4/10 |
| iscx_tor | P1_A0_entropy − P0_A0_equal | -0.000162 | 0/10 |
| iscx_tor | P2_A1_equal − P0_A0_equal | -0.000803 | 6/10 |
| iscx_tor | P3_A1_entropy − P2_A1_equal | +0.000163 | 2/10 |
| iscx_tor | P3_A1_entropy − P1_A0_entropy | -0.000478 | 6/10 |

## Class-conditional P3 versus YaTC

| Dataset | Class | Δ mean F1 | P3 mean recall | YaTC mean recall |
|---|---|---:|---:|---:|
| iscx_vpn | Chat | +0.018515 | 0.907692 | 0.892308 |
| iscx_vpn | Email | +0.018226 | 0.917000 | 0.927500 |
| iscx_vpn | File-Transfer | -0.033514 | 0.768500 | 0.792500 |
| iscx_vpn | P2P | -0.001478 | 1.000000 | 1.000000 |
| iscx_vpn | Streaming | -0.019710 | 0.972772 | 0.992574 |
| iscx_vpn | VoIP | -0.006794 | 0.751500 | 0.740000 |
| iscx_tor | Browsing | -0.025671 | 0.836000 | 0.877500 |
| iscx_tor | Chat | +0.037873 | 0.740625 | 0.593750 |
| iscx_tor | Email | +0.012272 | 1.000000 | 0.972222 |
| iscx_tor | File-Transfer | -0.016277 | 0.977000 | 0.980000 |
| iscx_tor | P2P | +0.022585 | 0.896500 | 0.900000 |
| iscx_tor | Streaming | -0.018066 | 0.820500 | 0.850000 |
| iscx_tor | VoIP | +0.034413 | 0.845500 | 0.817500 |

## Entropy mechanism and counterfactual

- P3 minus forced-equal Macro-F1 mean: +0.000382; positive units: 7/20.
- Batch-invariance maximum logit absolute difference: 6.6757202e-06.
- With exactly two views, if λ_b+λ_s=1 (including the 0.5/0.5 initialization), the proposed high/low blend is algebraically 0.5/0.5 regardless of per-sample entropy. Learned coefficients must jointly leave this cancellation line before dynamic weights can matter. The mechanism test and equal/shuffle controls assess this explicitly.
- This is a structural limitation of the preregistered two-view formula, not a post-hoc hyperparameter invitation.

## Limitations

- Only two datasets and two frozen encoder seeds; five head seeds share the same validation samples and are not independent dataset replicates.
- Content view mixes TrafficFormer and YaTC; behavior view is FIG, unlike the paper's raw byte/packet-length pair.
- No Known Test or Unknown outcome is included. No open-set or external-generalization claim follows.

## Gate and conclusion

- Gate: **FAIL**. P3 does not clear the preregistered Known-Val gate; no Stage28B diffusion or Test-based tuning.
