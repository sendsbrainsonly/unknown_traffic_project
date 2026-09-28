# Stage 43 — CIC Mixed-Unknown Composition and Prevalence Stress Test

- Status: complete; frozen-score diagnostic
- Final Gate: **COMPOSITION_SENSITIVE**
- Encoder training / threshold fitting / PCAP reads: 0 / 0 / 0
- Mixtures: 300; membership rows: 600,000

## DES-v1 prevalence results

| Setting | AUROC mean | AUPRC mean | UFAR mean | Known FRR mean | Binary Acc | Binary F1 |
|---|---:|---:|---:|---:|---:|---:|
| known1_unknown1 | 0.981011 | 0.977296 | 0.068350 | 0.071250 | 0.930200 | 0.930291 |
| known1_unknown3 | 0.980384 | 0.991864 | 0.068867 | 0.072300 | 0.930275 | 0.952450 |
| known1_unknown9 | 0.980961 | 0.997304 | 0.069500 | 0.068250 | 0.930625 | 0.960223 |
| known3_unknown1 | 0.981316 | 0.939144 | 0.069100 | 0.070367 | 0.929950 | 0.869191 |
| known9_unknown1 | 0.981042 | 0.852588 | 0.067250 | 0.071056 | 0.929325 | 0.725275 |

## DES-v1 composition results

| Setting | AUROC mean | AUPRC mean | UFAR mean | Known FRR mean | Binary Acc | Binary F1 |
|---|---:|---:|---:|---:|---:|---:|
| all_balanced | 0.980976 | 0.976932 | 0.067100 | 0.070650 | 0.931125 | 0.931247 |
| authentication_web | 0.995600 | 0.994210 | 0.000000 | 0.070850 | 0.964575 | 0.965790 |
| dos | 0.975739 | 0.966079 | 0.033950 | 0.073250 | 0.946400 | 0.947435 |
| easy_hard_mixed | 0.968182 | 0.960974 | 0.169700 | 0.070350 | 0.879975 | 0.873679 |
| natural_frequency | 0.967780 | 0.937234 | 0.052300 | 0.070850 | 0.938425 | 0.938999 |
| previously_easy | 0.989289 | 0.986469 | 0.000300 | 0.070850 | 0.964425 | 0.965645 |
| previously_hard | 0.947600 | 0.921027 | 0.338350 | 0.071000 | 0.795325 | 0.763697 |

## Class-conditional diagnosis

In the all-balanced mixture (100 samples per Unknown class per repeat), DES-v1 had:

| Unknown class | AUROC mean | UFAR mean |
|---|---:|---:|
| Bot | 0.948393 | 0.495000 |
| DDoS | 0.949185 | 0.169500 |
| DoS GoldenEye | 0.970371 | 0.002000 |
| DoS Hulk | 0.973789 | 0.001500 |
| DoS Slowhttptest | 0.993088 | 0.002000 |
| DoS slowloris | 0.992581 | 0.001000 |
| FTP-Patator | 0.998241 | 0.000000 |
| SSH-Patator | 0.991031 | 0.000000 |
| Web Attack - Brute Force | 0.995741 | 0.000000 |
| Web Attack - XSS | 0.997344 | 0.000000 |

The aggregate all-balanced UFAR of 0.0671 therefore hides a 49.5% Bot false-acceptance rate and a 16.95% DDoS false-acceptance rate. In the Bot+DDoS-only mixture, aggregate UFAR rose to 0.33835 and AUROC fell to 0.94760.

## Known composition shift

| BENIGN:PortScan | AUROC mean | UFAR mean | Known FRR mean | Binary Acc | Binary F1 |
|---|---:|---:|---:|---:|---:|
| 1:1 | 0.980723 | 0.069350 | 0.070500 | 0.930075 | 0.930116 |
| 3:1 | 0.978469 | 0.067850 | 0.091250 | 0.920450 | 0.921367 |
| 1:3 | 0.983714 | 0.067250 | 0.050250 | 0.941250 | 0.940746 |

The difference is explained by unequal class-conditional rejection: at the 1:1 setting, BENIGN FRR was 0.1075 versus PortScan FRR 0.0335. Thus the aggregate Known FRR changes when the Known class mixture changes even though the detector and threshold do not.

## Paired method comparison

Across all 20 all-balanced repeats, DES-v1 exceeded MSP, Energy and centroid in AUROC and reduced UFAR. Relative to MSP, the all-balanced mean deltas were +0.326430 AUROC, +0.205209 AUPRC and -0.377150 UFAR. These large deltas are specific to the frozen CIC roles and do not remove the capture/date confounding.

## Interpretation boundary

- AUROC, UFAR and Known FRR are primary across prevalence; AUPRC, Accuracy and F1 change mechanically with class prevalence.
- previously_easy and previously_hard are retrospective stress sets defined from exposed Stage42 outcomes.
- Per-class results must be read alongside aggregate mixtures so large/easy attack classes cannot hide Bot/DDoS failures.
- CIC day/capture/endpoint confounding is not removed by score-level mixing.

## Gate checks

- Prevalence cells pass: True
- Composition cells pass: False
- All-balanced cell pass: True
- Rule: mean AUROC>=0.95, mean UFAR<=0.10, mean Known FRR<=0.10
