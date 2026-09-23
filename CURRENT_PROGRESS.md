# Current Project Status

Updated: 2026-09-23 UTC

## Executive summary

The project has moved from USTC-only Gaussian/multi-prototype diagnosis to a multi-dataset, evidence-gated study of representation quality and open-set support modeling. The current primary task is coarse Service classification and Leave-One-Service-Out unknown detection on ISCX-VPN and ISCXTor2016.

The latest completed experiment is Stage 21. The active experiment is Stage 22 and is not yet a result.

## Current stage

| Stage | Status | What is established |
|---|---|---|
| Stage 19 RoNeTC comparison | `ABORTED_BY_USER_WITH_FOUR_COMPLETED_PROTOCOLS` | Four completed matched protocols may be compared; partial USTC training is excluded |
| Stage 20 coarse Service protocol | `PROTOCOL_FROZEN_NOT_TRAINED` | VPN-6, TOR-7, 22,136 flows and 13 strict Unknown-Free LOSO protocols are frozen |
| Stage 21 OURS-E3-T8 closed set | `CLOSED_SET_DIAGNOSTIC_COMPLETE` | Four formal runs completed and 98/98 completion checks passed |
| Stage 22 pretrained TrafficFormer/E3 | `running` | Preflight and GPU smoke passed; final metrics do not yet exist |

## Stage 21 results

Mean and population standard deviation over seeds 2022 and 2023:

| Dataset | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| ISCX-VPN | 0.5201±0.0389 | 0.5249±0.0479 | 0.5129±0.0451 |
| ISCXTor2016 | 0.6222±0.0134 | 0.5911±0.0146 | 0.6176±0.0127 |

E3 outperformed E2 on all four paired runs, with mean Macro-F1 gains of `+0.1149` on VPN and `+0.2128` on TOR. This establishes an internal fusion benefit, not competitive absolute performance. The strict random-initialized TrafficFormer branch was severely underfit.

## Stage 22 live scope

Stage 22 tests whether official TrafficFormer pretraining and a longer 20-epoch budget explain the Stage 21 gap. It uses the same frozen Stage 20 membership, seeds 2022/2023, validation-only checkpoint selection and a three-GPU queue.

At this snapshot:

- static preflight: PASS;
- official-weight forward smoke: PASS;
- batch-64 forward/backward/optimizer smoke: PASS;
- formal training: running;
- final Accuracy/Macro-F1 and E3 comparison: not yet available.

Partial checkpoints and live logs are not scientific results and are excluded from the public Git snapshot.

## Current scientific boundary

- Fixed K2 multi-local support did not show stable unknown-detection utility across the original USTC A-1/A-2/A-3 settings.
- DES and hybrid variants show dataset/protocol-conditional improvements, while absolute UFAR often remains high.
- Coarse Service labels materially improve some closed-set settings, but they change the task and cannot be reported as gains on the earlier fine-grained label space.
- Stage 21 does not support claiming that the current project method outperforms TrafficFormer or the strongest historical baseline.
- Adaptive K, Unknown Discovery and Test-driven threshold/model selection remain unsupported.

## Data and evaluation limitations

- Service labels are capture-derived weak labels rather than independently annotated flow labels.
- Splits are flow- and exact-image-disjoint but not uniformly capture-disjoint; VPN P2P has one capture.
- Only two seeds were requested for Stage 21/22, so variability estimates are descriptive rather than confidence intervals.
- Stage 22 must finish and pass completion verification before its metrics are added here.

## Evidence entry points

- Full experiment index: [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)
- Durable execution log: [`EXECUTION_PROGRESS.md`](EXECUTION_PROGRESS.md)
- Stage 20 protocol: [`stage20_dual_coarse_service_protocol/RESULTS.md`](stage20_dual_coarse_service_protocol/RESULTS.md)
- Stage 21 result: [`stage21_coarse_service_ours_e3_benchmark/RESULTS.md`](stage21_coarse_service_ours_e3_benchmark/RESULTS.md)
- Stage 22 live record: [`stage22_pretrained_trafficformer_e3_closed_set_comparison/RESULTS.md`](stage22_pretrained_trafficformer_e3_closed_set_comparison/RESULTS.md)
- Publication boundary: [`PUBLIC_REPOSITORY_CONTENTS.md`](PUBLIC_REPOSITORY_CONTENTS.md)

## Safe next action

Allow Stage 22 to finish without changing its data, model, seeds or selection rule. Then aggregate the four formal runs, verify artifacts and hashes, update Stage 22 `RESULTS.md`, this status document and `EXPERIMENT_RESULTS.md`, and publish a separate completion commit.
