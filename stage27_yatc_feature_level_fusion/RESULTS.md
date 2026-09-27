# Stage 27 — YaTC feature-level fusion with E3

- Status: success; diagnostic development benchmark.

## Data and split

- Exact frozen Stage20 ISCX-VPN/ISCXTor Known Train/Val/Test flow IDs, seeds 2022/2023.
- Original fine Service labels primary; fixed Communication hard-remap secondary. Unknown usage: 0.

## Configuration and execution

- Frozen E3 896-D and YaTC 192-D penultimate features; Known-Train-only YaTC z-score. F1 linear concat, F2 equal projected, F3 learned feature gate.
- Three new heads per run; Adam 1e-3, batch 256, 30 full epochs; select on Known Val fine Macro-F1.
- Four named tmux GPU0 runs exit 0; source hashes unchanged; no encoder weight updates.

## Core results

Known Test fine Macro-F1, mean of two seeds:

| Dataset | E3 | YaTC | F1 concat | F2 projection | F3 gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| iscx_vpn | 0.859493 | 0.887085 | 0.888900 | 0.884258 | 0.880846 |
| iscx_tor | 0.822178 | 0.827393 | 0.837946 | 0.842260 | 0.821191 |

- 120/120 metrics independently replayed from saved logits and predictions; all checkpoint hashes PASS.
- F1/F2 sometimes add fine-label signal, especially Tor; F3 did not show stable improvement and often collapses to TrafficFormer.

## Preserved evidence

- `stage27_report.md`, `run_metrics.csv`, `per_class.csv`, `paired_comparison.csv`, `setting_summary.csv`, `gate_weight_summary.csv`, `independent_verification.json`, all four full run bundles and tmux logs/status.

## Limitations

- Stage20 Test was previously exposed, so the figures are development evidence only.
- No Unknown detection/open-set result; cannot infer open-set benefit from closed-set Macro-F1.
- Learned gate is a feature-level adaptation, not a reproduction of ER-CMGI.

## Conclusion and next step

- The intended feature-layer question is now tested. Do not tune another formula on this exposed Test or silently replace Stage26.
