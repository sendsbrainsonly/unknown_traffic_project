# Stage 24 — 24A pilot / 24B context-feasibility results

Status: `PILOT_NO_GLOBAL_GAIN`. This is a Known Train/Validation pilot, not a Test or Unknown Detection result.

## Frozen evidence and integrity

- Stage20/21/22 protected files: 19/19 before/after SHA256 unchanged.
- 16/16 preregistered pilot run bundles passed independent artifact validation.
- Known Test feature usage = 0; Unknown Test feature usage = 0.

## Known Validation Macro-F1

| Dataset | Candidate | Macro-F1 mean ± population std | Δ vs E1 | Δ vs E3 | Wins vs E1/E3 |
| --- | --- | ---: | ---: | ---: | ---: |
| iscx_vpn | S1 | 0.850469 ± 0.000915 | +0.001335 | -0.001408 | 1/0 of 2 |
| iscx_vpn | G1 | 0.850401 ± 0.000983 | +0.001268 | -0.001476 | 1/0 of 2 |
| iscx_vpn | G2 | 0.849936 ± 0.001280 | +0.000802 | -0.001941 | 1/0 of 2 |
| iscx_vpn | G2-shuffle | 0.850997 ± 0.001502 | +0.001864 | -0.000880 | 1/1 of 2 |
| iscx_tor | S1 | 0.846332 ± 0.006558 | -0.004337 | +0.002225 | 0/2 of 2 |
| iscx_tor | G1 | 0.846216 ± 0.006765 | -0.004453 | +0.002109 | 0/2 of 2 |
| iscx_tor | G2 | 0.846903 ± 0.007075 | -0.003766 | +0.002796 | 0/2 of 2 |
| iscx_tor | G2-shuffle | 0.846500 ± 0.006350 | -0.004169 | +0.002393 | 0/2 of 2 |

The 2022/2023 E1 checkpoints and representations are shared exactly within each paired cell.
S1 and G1 test early-eight behavior information; G2 uses packet-to-burst-to-flow hierarchy; G2-shuffle is a fixed-seed burst-order negative control.

The frozen E1/E3 means are respectively 0.849134/0.851877 on VPN and 0.850669/0.844107 on Tor. Thus the small gains against the weaker baseline do not demonstrate an improvement over the stronger baseline. G2 minus G2-shuffle is −0.001061 on VPN and +0.000403 on Tor; two seeds do not support a burst-order contribution. S1/G1/G2 have respectively 3,334/4,358/11,654 trainable branch parameters on VPN and 3,463/4,487/11,783 on Tor, so architectural comparisons also confound capacity. Their mean run times are 21.4/23.9/35.1 s on VPN and 20.8/24.4/36.0 s on Tor.

Class-level effects are mixed, not a hidden broad gain: against the stronger per-class E1/E3 baseline, Tor G2 gains about +0.0062 F1 on Browsing but loses about −0.0129 on VoIP and −0.0113 on Chat; VPN G2 loses about −0.0128 on File-Transfer. The per-class baseline here is chosen separately for each class and is only a diagnostic reference, not a new model. See `pilot_per_class.csv` and `pilot_error_rescue.csv` for all classes, supports and paired decisions.

## Short-flow and cross-flow context constraints

- iscx_vpn: 7805/9862 Known Train/Val flows have 1–2 packets (79.14%); 0 provenance flows have non-monotone packet timestamps.
- iscx_tor: 5887/10064 Known Train/Val flows have 1–2 packets (58.50%); 5 provenance flows have non-monotone packet timestamps.
- Full-class capture-disjoint cross-flow graph: **NOT FEASIBLE on the frozen Stage20 VPN task**, because P2P has only one capture and current flow-random roles share captures.
- Frozen provenance lacks an endpoint/session identifier; causal previous-flow counts are only a read-only capture-time inventory, not an authorized edge set.
- No cross-flow GNN was trained. Inter-flow work requires a separate feasible context protocol, not a retrofit to this single-flow table.

## Decision

- Pilot continuation gate: `PILOT_NO_GLOBAL_GAIN`.
- The two-seed continuation rule was specified after one negative S1 result was visible; see `DECISION_RULE_AMENDMENT.md`. It is not fully pre-result preregistered, and no confirmatory significance claim is made.
- Chosen candidate for any five-seed confirmation: `none`.
- A two-seed pilot cannot establish stable five-seed gains. Existing Stage20 Test was already exposed, so it is not untouched validation.
- No open-set score or Unknown-Free claim is produced; official TrafficFormer pretraining exposure remains unresolved.

## Evidence

- `pilot_run_results.csv`, `pilot_dataset_summary.csv`, `pilot_per_class.csv`, `pilot_error_rescue.csv`, `pilot_packet_count_results.csv`, individual run directories, protected hashes, and `completion_verification.json`.
