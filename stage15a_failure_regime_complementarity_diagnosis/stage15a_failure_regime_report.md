# Stage 15A — Open-Detect vs DES Failure-Regime & Complementarity Diagnosis

## Scope and controls

This is a read-only diagnostic over frozen scores and latent arrays. No encoder or detector was trained, no score was refit, and no threshold, k, or fusion weight was searched. VNAT is the primary dataset and is not described as untouched. USTC is development evidence. ISCX-VPN and ISCXTor2016 provide frozen M0/M1-only supplementary evidence; they cannot support a DES-v1 cross-dataset claim.

- Protocols analyzed: 60 (VNAT 15, USTC 15, ISCX-VPN 15, ISCXTor2016 15).
- Unknown-class protocol units: 160.
- Sample/comparison rows: 481190.
- Frozen source files hashed: 255.

## Dataset-level results

| Dataset | M0 AUROC | M1 AUROC | M2 AUROC | M1-M0 | M2-M0 |
|---|---:|---:|---:|---:|---:|
| VNAT | 0.8375 | 0.8517 | 0.8531 | 0.0142 | 0.0155 |
| USTC-TFC2016 | 0.9395 | 0.9710 | 0.9800 | 0.0315 | 0.0404 |
| ISCXTor2016 | 0.5246 | 0.5943 | NA | 0.0697 | NA |
| ISCX-VPN | 0.5530 | 0.5765 | NA | 0.0235 | NA |

## VNAT unknown-class failure regimes

| Unknown class | units | M0 AUROC | M1 AUROC | M2 AUROC | M1-M0 | overlap | margin | V_post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| netflix | 4 | 0.8300 | 0.9176 | 0.9443 | 0.0876 | 0.6695 | 4.4559 | 9.9078 |
| rdp | 4 | 0.9181 | 0.9660 | 0.9775 | 0.0480 | 0.2045 | 1.9112 | 17.8584 |
| rsync | 4 | 0.5450 | 0.3838 | 0.3531 | -0.1612 | 0.9814 | 0.4941 | 2.2771 |
| scp | 5 | 0.8164 | 0.7710 | 0.7511 | -0.0453 | 0.9589 | 2.3925 | 4.6366 |
| sftp | 5 | 0.5337 | 0.7367 | 0.7207 | 0.2029 | 0.9627 | 2.2285 | 1.0406 |
| skype | 5 | 0.9513 | 0.9665 | 0.9771 | 0.0152 | 0.0736 | 1.6932 | 14.8053 |
| ssh | 5 | 0.9395 | 0.9319 | 0.9495 | -0.0077 | 0.2791 | 1.7142 | 17.5805 |
| vimeo | 4 | 0.8961 | 0.9023 | 0.9403 | 0.0062 | 0.7808 | 2.8532 | 4.5593 |
| youtube | 4 | 0.9622 | 0.9234 | 0.9403 | -0.0389 | 0.6708 | 4.5060 | 7.3637 |
| zoiper | 5 | 0.9247 | 0.9638 | 0.9764 | 0.0391 | 0.2735 | 2.2526 | 7.4716 |

## Hypothesis checks

- H1 prototype mismatch: Spearman rho=0.2007, p=0.1863, n=45. This is exploratory because repeated protocols/classes are not independent.
- H2 nearest support distance: Spearman rho=0.3007, p=0.0448, n=45. This is exploratory because repeated protocols/classes are not independent.
- H2 ambiguity ratio: Spearman rho=0.2125, p=0.1610, n=45. This is exploratory because repeated protocols/classes are not independent.
- H2 support overlap: Spearman rho=-0.3436, p=0.0209, n=45. This is exploratory because repeated protocols/classes are not independent.
- closed-set confusion: Spearman rho=-0.1278, p=0.4028, n=45. This is exploratory because repeated protocols/classes are not independent.
- H3 posterior uncertainty: Spearman rho=0.1007, p=0.5106, n=45. This is exploratory because repeated protocols/classes are not independent.

- In VNAT DES-v0-loss units, V_post improves M0 over prototype-distance-only in 19/21 units.

### VNAT M1_vs_M0

- Operating point: OD-only rescue=15129, DES-only rescue=11585, both correct=72154, both wrong=38427.
- Among discordant operating decisions: OD share=0.5663, DES share=0.4337.
- Exact ranking pairs: OD-only=6203271, DES-only=7741572; shares among discordant pairs=0.4448/0.5552.

### VNAT M2_vs_M0

- Operating point: OD-only rescue=12818, DES-only rescue=13141, both correct=74465, both wrong=36871.
- Among discordant operating decisions: OD share=0.4938, DES share=0.5062.
- Exact ranking pairs: OD-only=5852971, DES-only=7827690; shares among discordant pairs=0.4278/0.5722.

## Known-Validation hard-pair confusion

These are native learned-prototype closed-set confusion rates, averaged only over VNAT protocols where both classes are Known.

| Direction | mean confusion | valid protocols |
|---|---:|---:|
| rsync -> scp | 0.6597 | 6 |
| scp -> rsync | 0.2205 | 6 |
| rsync -> sftp | 0.1501 | 6 |
| sftp -> rsync | 0.3715 | 6 |
| scp -> sftp | 0.1653 | 7 |
| sftp -> scp | 0.4243 | 7 |
## rsync / scp / sftp

| Unknown | units | M1-M0 | overlap | margin | M0-Dproto AUROC | modal nearest Known |
|---|---:|---:|---:|---:|---:|---|
| rsync | 4 | -0.1612 | 0.9814 | 0.4941 | 0.1580 | scp |
| scp | 5 | -0.0453 | 0.9589 | 2.3925 | 0.3335 | rsync |
| sftp | 5 | 0.2029 | 0.9627 | 2.2285 | -0.0952 | rsync |

## Required answers

1. **Real complementarity:** yes. At P95, OD alone rescues 15129 sample decisions and DES-v0 alone rescues 11585; exact ranking has 6203271 OD-only and 7741572 DES-only correct pairs. VNAT class-protocol deltas split 24 positive / 21 negative.
2. **DES gain regime:** lower empirical-support overlap is the repeatable VNAT signal (overlap-vs-gain rho below); by class, the clearest mean gains are sftp (+0.2029), netflix (+0.0876), rdp (+0.0480), and zoiper (+0.0391). sftp is an important mismatch-dominated exception: it has high overlap but also the largest mean normalized prototype gap (2.461), so empirical support repairs a poor native prototype.
3. **DES failure regime:** rsync (-0.1612) and scp (-0.0453) are the clearest failures and have support-overlap fractions 0.9814 and 0.9589. rsync is most often nearest scp; scp is most often nearest rsync. youtube is a smaller failure (-0.0389). Contrary to the initial concern, sftp does not fail on average: it improves in 4/5 VNAT units.
4. **Prototype mismatch:** VNAT rho=0.2007, p=0.1863. This does not establish H1.
5. **Support overlap/ambiguity:** overlap rho=-0.3436 (p=0.0209) and d1 rho=0.3007 (p=0.0448) support the overlap regime. The ratio/margin ambiguity tests are not significant, so centroid-boundary ambiguity itself is not established.
6. **Posterior uncertainty rescue:** 19/21 VNAT DES-loss units have M0 AUROC above D_proto-only AUROC. Mean rescue is +0.1580 for rsync and +0.3335 for scp, but -0.0952 for sftp; V_post specifically preserves signal in the two principal DES failure classes.
7. **Hybrid-design evidence:** Gate = **COMPLEMENTARITY_CONFIRMED**. A next-stage design is justified only if this gate is CONFIRMED or a clearly mechanism-backed PARTIAL result; this stage does not design it.

## Gate

**COMPLEMENTARITY_CONFIRMED**

Gate evidence:

- `vnat_positive_class_protocol_fraction`: 0.5333333333333333
- `vnat_negative_class_protocol_fraction`: 0.4666666666666667
- `vnat_operating_od_share_of_discordant`: 0.5663322602380774
- `vnat_operating_des_share_of_discordant`: 0.4336677397619226
- `vnat_ranking_od_share_of_discordant`: 0.4448433732814346
- `vnat_ranking_des_share_of_discordant`: 0.5551566267185655
- `predicted_direction_significant_mechanism_count`: 2
- `supplementary_gain_and_loss_dataset_present`: True
- `gate`: COMPLEMENTARITY_CONFIRMED

## Limitations

- Correlations are exploratory: class-protocol observations share data, encoders, and class identities.
- AUROC is threshold-free; operating complementarity uses each frozen method's Known-Validation P95.
- Exact ranking complementarity is pairwise and uses strict `unknown_score > known_score`; ties count as incorrect for that method.
- USTC uses its existing frozen balanced 1:1 evaluation, while VNAT and ISCX use their existing natural test composition.
- ISCX datasets have no frozen DES-v1 scores, so no M2 cross-dataset claim is made.
