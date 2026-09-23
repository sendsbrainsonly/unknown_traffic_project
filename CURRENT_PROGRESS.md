# Current Project Status

Updated: 2026-09-23 UTC

## Executive summary

The project has moved from USTC-only Gaussian/multi-prototype diagnosis to a multi-dataset, evidence-gated study of representation quality and open-set support modeling. The current primary task is coarse Service classification and Leave-One-Service-Out unknown detection on ISCX-VPN and ISCXTor2016.

The latest completed experiment is Stage 22. There is no active training stage.

## Current stage

| Stage | Status | What is established |
|---|---|---|
| Stage 19 RoNeTC comparison | `ABORTED_BY_USER_WITH_FOUR_COMPLETED_PROTOCOLS` | Four completed matched protocols may be compared; partial USTC training is excluded |
| Stage 20 coarse Service protocol | `PROTOCOL_FROZEN_NOT_TRAINED` | VPN-6, TOR-7, 22,136 flows and 13 strict Unknown-Free LOSO protocols are frozen |
| Stage 21 OURS-E3-T8 closed set | `CLOSED_SET_DIAGNOSTIC_COMPLETE` | Four formal runs completed and 98/98 completion checks passed |
| Stage 22 pretrained TrafficFormer/E3 | `CLOSED_SET_DIAGNOSTIC_COMPLETE` | Four formal runs completed and 99 completion checks passed |

## Stage 21 results

Mean and population standard deviation over seeds 2022 and 2023:

| Dataset | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| ISCX-VPN | 0.5201±0.0389 | 0.5249±0.0479 | 0.5129±0.0451 |
| ISCXTor2016 | 0.6222±0.0134 | 0.5911±0.0146 | 0.6176±0.0127 |

E3 outperformed E2 on all four paired runs, with mean Macro-F1 gains of `+0.1149` on VPN and `+0.2128` on TOR. This establishes an internal fusion benefit, not competitive absolute performance. The strict random-initialized TrafficFormer branch was severely underfit.

## Stage 22 results

Stage 22 tested whether official TrafficFormer pretraining and a longer 20-epoch budget explain the Stage 21 gap. It used the same frozen Stage 20 membership, seeds 2022/2023, validation-only checkpoint selection and a three-GPU queue.

Mean and population standard deviation over seeds 2022 and 2023:

| Dataset | Encoder | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| ISCX-VPN | TrafficFormer E1 | 0.8532±0.0041 | 0.8606±0.0040 | 0.8474±0.0043 |
| ISCX-VPN | E3 fusion | 0.8500±0.0082 | 0.8595±0.0070 | 0.8464±0.0074 |
| ISCXTor2016 | TrafficFormer E1 | 0.8312±0.0058 | 0.8213±0.0064 | 0.8324±0.0051 |
| ISCXTor2016 | E3 fusion | 0.8321±0.0067 | 0.8222±0.0076 | 0.8339±0.0060 |

E3−E1 mean Macro-F1 is `-0.001115` on VPN and `+0.000918` on TOR, with only `1/2` positive seeds on each dataset. This is statistical parity at the tested two-seed resolution, not evidence that E3 consistently improves TrafficFormer. Stage 22 completed `99/99` checks; Test samples used for checkpoint selection=`0`.

## Current scientific boundary

- Fixed K2 multi-local support did not show stable unknown-detection utility across the original USTC A-1/A-2/A-3 settings.
- DES and hybrid variants show dataset/protocol-conditional improvements, while absolute UFAR often remains high.
- Coarse Service labels materially improve some closed-set settings, but they change the task and cannot be reported as gains on the earlier fine-grained label space.
- Stage 22 does not support claiming that E3 consistently outperforms the same-run pretrained TrafficFormer: VPN is slightly lower and TOR slightly higher, with mixed seed signs.
- Adaptive K, Unknown Discovery and Test-driven threshold/model selection remain unsupported.

## Data and evaluation limitations

- Service labels are capture-derived weak labels rather than independently annotated flow labels.
- Splits are flow- and exact-image-disjoint but not uniformly capture-disjoint; VPN P2P has one capture.
- Only two seeds were requested for Stage 21/22, so variability estimates are descriptive rather than confidence intervals.
- Stage 21-to-Stage 22 differences combine pretrained initialization and a 3-to-20 epoch budget change, so they are not a pure pretraining ablation.

## Evidence entry points

- Full experiment index: [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)
- Durable execution log: [`EXECUTION_PROGRESS.md`](EXECUTION_PROGRESS.md)
- Stage 20 protocol: [`stage20_dual_coarse_service_protocol/RESULTS.md`](stage20_dual_coarse_service_protocol/RESULTS.md)
- Stage 21 result: [`stage21_coarse_service_ours_e3_benchmark/RESULTS.md`](stage21_coarse_service_ours_e3_benchmark/RESULTS.md)
- Stage 22 result: [`stage22_pretrained_trafficformer_e3_closed_set_comparison/RESULTS.md`](stage22_pretrained_trafficformer_e3_closed_set_comparison/RESULTS.md)
- Publication boundary: [`PUBLIC_REPOSITORY_CONTENTS.md`](PUBLIC_REPOSITORY_CONTENTS.md)

## Safe next action

Stop after this closed-set comparison. Before any open-set use of the external pretrained checkpoint, separately freeze and audit the pretraining-exposure policy; do not infer Unknown-Free validity from the Stage 22 closed-set result.
