# USTC Cross-Encoder Component Correspondence & Shortcut Audit

## Protocol

- Main analysis: K=2, full covariance, PCA64, seed=0 for FTP, Cridex, Miuref and Outlook.
- Both GMMs were fit on train; correspondence metrics use held-out validation as the primary result.
- TrafficFormer K=2 was refit from the immutable Stage 2.5 z_f inputs and passed the 1e-6 train/validation NLL reproduction gate.
- Open-Detect reused the frozen scaler, PCA64 and K=2 GMM artifacts; validation NLL reproduced within 1e-6 and train component counts matched the formal record exactly.
- No encoder training, split change, K search, test use, BIC, Adaptive K or Unknown Detection was performed.

## Per-class answers

### FTP

- Question: Do TF and OD partition the same validation flows, and is OD still driven by packet_count?
- Validation correspondence: NMI=0.216783, AMI=0.216726, ARI=0.262295, Hungarian accuracy=0.756112; MODERATE_CORRESPONDENCE.
- OD strongest validation continuous feature: total_bytes, epsilon-squared=0.251088.
- OD metadata Model S/S+ validation Macro-F1: 0.966178/0.991876.
- Train/validation component TV: TF=0.004739, OD=0.003207.
- Classification: Case 4.

### Cridex

- Question: Are component assignments consistent across encoders?
- Validation correspondence: NMI=0.987642, AMI=0.987636, ARI=0.995119, Hungarian accuracy=0.998779; HIGH_CORRESPONDENCE.
- OD strongest validation continuous feature: backward_packet_count, epsilon-squared=0.895206.
- OD metadata Model S/S+ validation Macro-F1: 1.000000/1.000000.
- Train/validation component TV: TF=0.004427, OD=0.006868.
- Classification: Case 4.

### Miuref

- Question: Does backward_packet_count still dominate the OD components?
- Validation correspondence: NMI=0.472249, AMI=0.471955, ARI=0.431592, Hungarian accuracy=0.828635; MODERATE_CORRESPONDENCE.
- OD strongest validation continuous feature: packet_count, epsilon-squared=0.904430.
- OD metadata Model S/S+ validation Macro-F1: 0.998348/0.999175.
- Train/validation component TV: TF=0.028012, OD=0.021581.
- Classification: Case 4.

### Outlook

- Question: Does total_bytes still dominate the OD components?
- Validation correspondence: NMI=0.224510, AMI=0.221326, ARI=0.395169, Hungarian accuracy=0.946879; MODERATE_CORRESPONDENCE.
- OD strongest validation continuous feature: total_bytes, epsilon-squared=0.310925.
- OD metadata Model S/S+ validation Macro-F1: 0.608637/0.775269.
- Train/validation component TV: TF=0.004127, OD=0.006094.
- Classification: Case 3.

## Overall Gate

**C. SIMPLE-STATISTICS PERSISTENCE**

The audit supports only stable or encoder-dependent representation-level local structure as indicated by the reported metrics. It does not establish semantic multimodality, natural Gaussian modes, attack stages, or an Unknown Detection result.

## Interpretation safeguards

- High metadata predictability means component assignment is strongly associated with simple flow statistics; it does not by itself make the structure false.
- High cross-encoder correspondence means the two learned representations partition many of the same flows after label permutation; it does not identify a semantic subtype.
- Source PCAP/source file are constant within each audited class and are explicitly marked NOT_IDENTIFIABLE_CONSTANT_WITHIN_CLASS.

## K=3 Correspondence Sensitivity

This is an appendix-only robustness check. K=2 remains the fixed main analysis and the main Gate is not reselected.

| class | NMI | AMI | ARI | Hungarian accuracy | label |
|---|---:|---:|---:|---:|---|
| FTP | 0.336017 | 0.335871 | 0.329931 | 0.715431 | MODERATE_CORRESPONDENCE |
| Cridex | 0.790774 | 0.790468 | 0.777831 | 0.832112 | MODERATE_CORRESPONDENCE |
| Miuref | 0.668105 | 0.667576 | 0.641185 | 0.811573 | MODERATE_CORRESPONDENCE |
| Outlook | 0.220940 | 0.213484 | 0.338466 | 0.895086 | MODERATE_CORRESPONDENCE |

K=3 does not establish semantic modes and is not used to alter the K=2 SIMPLE-STATISTICS PERSISTENCE Gate.
