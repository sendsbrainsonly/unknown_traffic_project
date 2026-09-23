# Experiment results: stage18-pp-opennet-inspired-e4-20260920-v1

- Status: `completed`
- Experiment type: `controlled_representation_benchmark`
- Claim scope: `approximate_pp_opennet_inspired_independent_reconstruction`
- Created (UTC): `2026-09-20T14:15:17Z`
- Completed (UTC): `2026-09-20T14:42:31Z`
- Final gate: `E4_FAIL_NO_FUSION`

## Objective and status

在 Stage 17 冻结的 Email/Streaming service open-set 协议上，从随机初始化训练独立 E4 包级时间—长度—方向编码器，验证多尺度与循环融合能否同时改善闭集分类和 Known/Unknown 检测。

6/6 protocol-seed runs 与 42/42 受控变体均完成。预注册晋级条件未全部满足，因此未启动 E3+E4 融合。

## Data and split

- 冻结缓存：`stage17_encoder_recovery_and_open_set_pilot/feature_cache/service3065_historical_inputs.npz`
- 缓存 SHA256：`aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff`
- Email：Known Train/Val/Test = `2585/155/157`，Unknown Test = `168`
- Streaming：Known Train/Val/Test = `1876/114/117`，Unknown Test = `958`
- Unknown training/validation/normalization/support/threshold usage：全部为 `0`
- Known Test checkpoint-selection usage：`0`

## Configuration and execution

- 输入：前 30 包 `log1p(length) + log1p(IAT-us) + direction`，显式 mask。
- 归一化：只用 Known Train median/IQR；长度/IAT 裁剪 `[-8,8]`。
- E4_FULL：多尺度 Conv1d `k=1/3/5` + masked statistics + packet/fusion GRU，128 维 embedding。
- 训练：100 epochs、batch 128、AdamW、LR `1e-3`、weight decay `1e-4`、MultiStepLR `[50,80]`。
- checkpoint：只按 Known Validation Macro-F1/Accuracy 选择。
- 变体：Full、Time-only、Length-only、Time+Length、No-MS、No-RNN、Base。
- 正式训练使用物理 GPU 1–4；独立重放按实时容量选择物理 GPU 1。
- Git snapshot：`ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc`，工作树在实验前已存在未提交改动。

## Core results

### E4_FULL closed set（mean ± population std, 3 seeds）

| Unknown | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Email | 0.743100 ± 0.007944 | 0.749697 ± 0.012937 | 0.741966 ± 0.008488 |
| Streaming | 0.809117 ± 0.022433 | 0.798280 ± 0.019562 | 0.809736 ± 0.023025 |

### E4_FULL open set（feature distance, Known-Val P95）

| Unknown | AUROC | AUPRC | UFAR | Known Test acceptance |
|---|---:|---:|---:|---:|
| Email | 0.627692 ± 0.019838 | 0.646520 ± 0.005339 | 0.882937 ± 0.026767 | 0.972399 ± 0.006005 |
| Streaming | 0.560439 ± 0.007646 | 0.924559 ± 0.004368 | 0.838900 ± 0.029082 | 0.920228 ± 0.004029 |

总体 E4−E0：ΔMacro-F1 `+0.077618`，ΔAUROC `-0.056970`，ΔAUPRC `+0.011305`，ΔUFAR@95 `-0.031723`，AUROC 胜出 `3/6`。总体 E4−E3：ΔMacro-F1 `-0.077294`，ΔAUROC `-0.167346`，ΔAUPRC `-0.070301`，ΔUFAR@95 `+0.066178`，AUROC 胜出 `0/6`。

严格单因素消融显示：Time 加到 Length 后带来 `+0.055788 Macro-F1 / -0.035383 AUROC`，Length 加到 Time 后带来 `+0.176222 Macro-F1 / -0.031306 AUROC`；Direction 几乎不改闭集均值，但带来 `+0.035479 AUROC`。Multi-scale 为 `+0.011391 Macro-F1 / +0.009839 AUROC`；RNN 为 `+0.016056 Macro-F1 / -0.007519 AUROC`。

结论不是“序列无效”，而是“当前 30 包 E4 单分支不能同时满足闭集、AUROC 与绝对 UFAR”。

## Preserved evidence

- `stage18_report.md`：完整方法、结果与结论。
- `e4_closed_set_results.csv`：42 行逐 run/variant 闭集结果。
- `e4_open_set_results.csv`：378 行逐 score/threshold 开集结果。
- `e4_per_class_results.csv`、`confusion_matrices.csv`：类别级结果。
- `e4_predictions.csv`：35,049 行逐样本分数与预测。
- `e4_training_dynamics.csv`：4,200 行 epoch 动态。
- `paired_comparison.csv`、`component_ablation.csv`：配对基线与消融。
- `runs/`：42 个 best checkpoint、embedding、history、逐 run SHA256 manifest。
- `replay_verification.json`：42/42 checkpoint 重放通过，最大差 `0.0`。
- `protected_asset_hashes_before.json`、`protected_asset_hashes_after.json`：19 个冻结资产哈希一致。
- `.tmux-task/stage18_*`：训练、失败 smoke、修复后 smoke、汇总及验证日志。

## Limitations

- 独立近似实现，不是 PP-OpenNet 作者代码或论文等价复现。
- 当前缓存最多 30 包，明显短于论文 1,000-packet/two-second slice。
- 只覆盖 Email/Streaming 两个 Stage 17 可执行 Unknown 服务设置。
- 服务标签为弱 capture-activity 标签；不是新的 untouched external validation。
- 未实现论文 background class、ARPL、PVRP，避免改变本实验 Strict Unknown-Free 边界。

## Conclusion and next step

`E4_FAIL_NO_FUSION`。保留 E4 作为负结果和类别/模块证据，不启动多分支融合。若未来继续研究，应先解决长窗口数据、support overlap 与 Known-Val/Test 阈值迁移问题，并形成新的预注册阶段，不能根据本次 Test 结果回调 Stage 18。
