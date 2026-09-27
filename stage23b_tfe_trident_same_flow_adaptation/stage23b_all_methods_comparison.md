# Stage 23B: all-method matched comparison

All rows use the frozen Stage20 ISCX-VPN and ISCXTor Service splits,
seeds 2022/2023. Stage23B rows are adapted methods; Stage23 rows are
the previously completed matched baselines. TFE and Trident Stage23B
results are not author-native reproductions.

## ISCX-VPN: per-seed and mean ± population SD

| Method | Acc 2022 | Acc 2023 | Acc mean±SD | Macro-F1 2022 | Macro-F1 2023 | Macro-F1 mean±SD | Weighted-F1 mean±SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| OURS-E3-T8-pretrained | 0.8582 | 0.8417 | 0.8500 ± 0.0082 | 0.8665 | 0.8525 | 0.8595 ± 0.0070 | 0.8464 ± 0.0074 |
| Open-Detect corrected-paper | 0.7429 | 0.7145 | 0.7287 ± 0.0142 | 0.7585 | 0.7300 | 0.7443 ± 0.0142 | 0.7254 ± 0.0135 |
| RoNeTC-runnable-reconstruction | 0.7136 | 0.6999 | 0.7068 ± 0.0069 | 0.7238 | 0.7077 | 0.7157 ± 0.0081 | 0.7013 ± 0.0051 |
| TFE-GNN-8-UDP-short | 0.6295 | 0.6679 | 0.6487 ± 0.0192 | 0.6439 | 0.6732 | 0.6585 ± 0.0146 | 0.6445 ± 0.0163 |
| TrafficFormer-pretrained | 0.8573 | 0.8490 | 0.8532 ± 0.0041 | 0.8646 | 0.8566 | 0.8606 ± 0.0040 | 0.8474 ± 0.0043 |
| Trident-early8-86D | 0.6157 | 0.6112 | 0.6134 ± 0.0023 | 0.6316 | 0.6328 | 0.6322 ± 0.0006 | 0.6065 ± 0.0005 |
| YaTC-official-pretrained-Stage20 | 0.8902 | 0.8664 | 0.8783 ± 0.0119 | 0.8987 | 0.8755 | 0.8871 ± 0.0116 | 0.8765 ± 0.0126 |

## ISCXTor2016: per-seed and mean ± population SD

| Method | Acc 2022 | Acc 2023 | Acc mean±SD | Macro-F1 2022 | Macro-F1 2023 | Macro-F1 mean±SD | Weighted-F1 mean±SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| OURS-E3-T8-pretrained | 0.8389 | 0.8254 | 0.8321 ± 0.0067 | 0.8298 | 0.8145 | 0.8222 ± 0.0076 | 0.8339 ± 0.0060 |
| Open-Detect corrected-paper | 0.7117 | 0.5452 | 0.6285 ± 0.0833 | 0.6763 | 0.4769 | 0.5766 ± 0.0997 | 0.6148 ± 0.0913 |
| RoNeTC-runnable-reconstruction | 0.6150 | 0.5936 | 0.6043 ± 0.0107 | 0.5539 | 0.5142 | 0.5340 ± 0.0198 | 0.5934 ± 0.0174 |
| TFE-GNN-8-UDP-short | 0.5998 | 0.6016 | 0.6007 ± 0.0009 | 0.5398 | 0.5674 | 0.5536 ± 0.0138 | 0.5881 ± 0.0087 |
| TrafficFormer-pretrained | 0.8371 | 0.8254 | 0.8312 ± 0.0058 | 0.8276 | 0.8149 | 0.8213 ± 0.0064 | 0.8324 ± 0.0051 |
| Trident-early8-86D | 0.6356 | 0.6303 | 0.6329 ± 0.0027 | 0.6071 | 0.6034 | 0.6053 ± 0.0019 | 0.6324 ± 0.0033 |
| YaTC-official-pretrained-Stage20 | 0.8424 | 0.8451 | 0.8438 ± 0.0013 | 0.8282 | 0.8266 | 0.8274 ± 0.0008 | 0.8436 ± 0.0015 |

## Per-service F1 averaged across seeds

### ISCX-VPN

| Service | E3-T8 | OpenDetect | RoNeTC | TFE-GNN-adapt | TrafficFormer | Trident-adapt | YaTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chat | 0.9092 | 0.7803 | 0.7412 | 0.7297 | 0.9077 | 0.6150 | 0.9077 |
| Email | 0.9258 | 0.7961 | 0.7928 | 0.6072 | 0.9281 | 0.5686 | 0.9304 |
| File-Transfer | 0.7228 | 0.5452 | 0.5507 | 0.4556 | 0.7410 | 0.4717 | 0.7967 |
| P2P | 0.9975 | 0.9468 | 0.8695 | 0.8047 | 1.0000 | 0.9107 | 1.0000 |
| Streaming | 0.9631 | 0.8145 | 0.8429 | 0.8325 | 0.9642 | 0.8072 | 0.9667 |
| VoIP | 0.6385 | 0.5828 | 0.4973 | 0.5215 | 0.6226 | 0.4199 | 0.7210 |
### ISCXTor2016

| Service | E3-T8 | OpenDetect | RoNeTC | TFE-GNN-adapt | TrafficFormer | Trident-adapt | YaTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Browsing | 0.8224 | 0.6166 | 0.6971 | 0.5708 | 0.8180 | 0.5948 | 0.8733 |
| Chat | 0.5851 | 0.1511 | 0.0885 | 0.1187 | 0.5834 | 0.2786 | 0.5818 |
| Email | 0.9537 | 0.6811 | 0.4943 | 0.6956 | 0.9581 | 0.7022 | 0.9322 |
| File-Transfer | 0.9424 | 0.7552 | 0.7299 | 0.7925 | 0.9438 | 0.8038 | 0.9723 |
| P2P | 0.9027 | 0.7995 | 0.6685 | 0.7078 | 0.9045 | 0.7129 | 0.9171 |
| Streaming | 0.7661 | 0.5126 | 0.5514 | 0.4403 | 0.7661 | 0.6122 | 0.7841 |
| VoIP | 0.7828 | 0.5199 | 0.5083 | 0.5496 | 0.7749 | 0.5323 | 0.7310 |

## Paired mean differences: adaptation minus baseline

Positive values favor the Stage23B adaptation. Values are means of the two seed-level paired differences.

| Dataset | Candidate | Baseline | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 |
|---|---|---|---:|---:|---:|
| iscx_tor | TFE-GNN-8-UDP-short | OURS-E3-T8-pretrained | -0.2314 | -0.2686 | -0.2458 |
| iscx_tor | TFE-GNN-8-UDP-short | Open-Detect corrected-paper | -0.0278 | -0.0230 | -0.0267 |
| iscx_tor | TFE-GNN-8-UDP-short | RoNeTC-runnable-reconstruction | -0.0036 | +0.0196 | -0.0053 |
| iscx_tor | TFE-GNN-8-UDP-short | TrafficFormer-pretrained | -0.2305 | -0.2676 | -0.2443 |
| iscx_tor | TFE-GNN-8-UDP-short | YaTC-official-pretrained-Stage20 | -0.2431 | -0.2738 | -0.2555 |
| iscx_tor | Trident-early8-86D | OURS-E3-T8-pretrained | -0.1992 | -0.2169 | -0.2015 |
| iscx_tor | Trident-early8-86D | Open-Detect corrected-paper | +0.0045 | +0.0287 | +0.0176 |
| iscx_tor | Trident-early8-86D | RoNeTC-runnable-reconstruction | +0.0286 | +0.0712 | +0.0389 |
| iscx_tor | Trident-early8-86D | TrafficFormer-pretrained | -0.1983 | -0.2160 | -0.2000 |
| iscx_tor | Trident-early8-86D | YaTC-official-pretrained-Stage20 | -0.2108 | -0.2221 | -0.2113 |
| iscx_vpn | TFE-GNN-8-UDP-short | OURS-E3-T8-pretrained | -0.2013 | -0.2009 | -0.2019 |
| iscx_vpn | TFE-GNN-8-UDP-short | Open-Detect corrected-paper | -0.0801 | -0.0857 | -0.0809 |
| iscx_vpn | TFE-GNN-8-UDP-short | RoNeTC-runnable-reconstruction | -0.0581 | -0.0572 | -0.0568 |
| iscx_vpn | TFE-GNN-8-UDP-short | TrafficFormer-pretrained | -0.2045 | -0.2021 | -0.2029 |
| iscx_vpn | TFE-GNN-8-UDP-short | YaTC-official-pretrained-Stage20 | -0.2296 | -0.2285 | -0.2320 |
| iscx_vpn | Trident-early8-86D | OURS-E3-T8-pretrained | -0.2365 | -0.2273 | -0.2400 |
| iscx_vpn | Trident-early8-86D | Open-Detect corrected-paper | -0.1153 | -0.1121 | -0.1190 |
| iscx_vpn | Trident-early8-86D | RoNeTC-runnable-reconstruction | -0.0933 | -0.0835 | -0.0948 |
| iscx_vpn | Trident-early8-86D | TrafficFormer-pretrained | -0.2397 | -0.2284 | -0.2410 |
| iscx_vpn | Trident-early8-86D | YaTC-official-pretrained-Stage20 | -0.2649 | -0.2549 | -0.2700 |

## Detailed source tables

- `stage23b_all_methods_run_level.csv`: 28 run rows × Accuracy/Macro-F1/Weighted-F1.
- `stage23b_all_methods_paired.csv`: 40 seed-paired rows, both adapted methods against all five baselines.
- `stage23b_all_methods_paired_summary.csv`: 20 mean paired comparisons.
- `stage23b_all_methods_per_class.csv`: 182 Known-Test service rows with precision, recall, F1 and support.
- `stage23b_subgroup_analysis.csv`: TCP/UDP and 1–2, 3–4, 5–8 packet bins for Stage23B adaptations.

The five existing baselines are TrafficFormer-pretrained, OURS-E3-T8-pretrained,
Open-Detect corrected-paper, RoNeTC-runnable-reconstruction, and YaTC-official-pretrained-Stage20.
All seven methods have matching Train/Validation/Test counts per dataset and PASS sample parity.
