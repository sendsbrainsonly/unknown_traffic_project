# Experiment results: stage41-a123-matched-open-set-20260927

- Status: `success`
- Experiment type: `open-set-comparison`
- Claim scope: `diagnostic`
- Completed (UTC): `2026-09-27T08:58:53.365262+00:00`

## Data and split

The same USTC flow IDs and original Train/Validation/Test memberships were used by both methods. A-1/A-2/A-3 hold out 1/3/5 complete Unknown classes, respectively. The deterministic 10%-per-class/source-split pool exactly reproduces the old Stage34/40 A-2 IDs. The balanced 1:1 Test subset was frozen before any score was read.

## Configuration and execution

Seed 2022. Open-Detect released-code ResNet18/prototype-KL is trained from scratch for each scenario with 100 epochs, Adam, batch 128, Known-Validation checkpoint selection. The three-view method uses official TrafficFormer/YaTC initialization, FIG graph branch, separate Known-only branch training, fixed equal feature fusion, and DES-v1 k=10 0.5/0.5 support score. A-2 reuses byte-identical frozen Stage34/40 weights/scores; A-1/A-3 are newly trained. Primary threshold: each method's Known-Validation P95; the Test-Youden row is labeled retrospective oracle. The interrupted A-1 attempt used physical GPU0; after the two-GPU cap, completed jobs run through named tmux sessions on physical GPUs 6/7, with live capacity selection.

## Core results

# Stage41 matched-flow Open-Detect vs three-view DES-v1

Single seed 2022, same USTC flow IDs and source Train/Val/Test partitions for both methods.
The 1:1 Test subset was frozen by flow ID before score inspection. Primary operating point
is Known Validation P95; Test-Youden is explicitly retrospective/oracle, not deployable.

| Scenario | Known/Unknown classes | Balanced Test Known/Unknown | OD AUROC | Ours AUROC | ΔAUROC | OD F1@P95 | Ours F1@P95 | ΔF1 | OD UFAR | Ours UFAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-1 | 19/1 | 85/85 | 0.9907 | 0.9769 | -0.0138 | 0.9659 | 0.9605 | -0.0055 | 0.0000 | 0.0000 |
| A-2 | 17/3 | 558/558 | 0.9290 | 0.9406 | +0.0116 | 0.5038 | 0.6741 | +0.1703 | 0.6434 | 0.4606 |
| A-3 | 15/5 | 1031/1031 | 0.9782 | 0.9609 | -0.0173 | 0.9191 | 0.9231 | +0.0039 | 0.1125 | 0.0921 |

The full natural Test prevalence and retrospective oracle-threshold
metrics are in `comparison_run_results.csv`. This local 10% flow subset is
not the authors' exact five-fold dataset/protocol. Our method includes
external pretrained TrafficFormer/YaTC branches; OD is trained from scratch.
A-1 has only 85 Unknown Test flows, so its single-seed estimate is fragile.

## 闭集与开集配对结果

闭集仅评价各场景的全部 Known Test 流，不进行未知拒识。两种方法使用同一场景、同一 Known Test flow ID 和类别定义。数值依次为 Accuracy / Macro-F1 / Weighted-F1：

| 场景 | Known Test 流数 | Open-Detect Native | 三路等权特征融合 |
|---|---:|---:|---:|
| A-1 | 4,806 | 0.977112 / 0.977538 / 0.976780 | **0.985643 / 0.988065 / 0.985611** |
| A-2 | 4,333 | 0.975537 / 0.978454 / 0.975034 | **0.984306 / 0.987649 / 0.984315** |
| A-3 | 3,860 | 0.998705 / 0.996973 / 0.998719 | **1.000000 / 1.000000 / 1.000000** |

OD 的 Macro-F1 由各场景 `od_sample_scores.csv` 中保存的 Known Test 真实标签和预测类别重算；重算的 Accuracy、Weighted-F1 与对应 `od_run*/test_metrics.json` 完全一致。A-1/A-3 三路闭集值来自本实验的 `detection/known_test_closed_set.json`；A-2 使用已冻结的 `stage40_ustc_cic_open_set/ustc_a2/detection/known_test_closed_set.json`，其角色与本实验 A-2 manifest 已核对为完全相同。重算命令及输出保存在项目根目录 `.tmux-task/stage41_closed_metrics_0927/output.log`。

开集评价比较 Open-Detect Native 与三路等权特征融合后的 DES-v1。上节给出平衡 1:1 Test 的 AUROC、Unknown F1@Known-Val P95 和 UFAR；补充 AUPRC 如下：

| 场景 | OD AUPRC | 三路 DES-v1 AUPRC | ΔAUPRC |
|---|---:|---:|---:|
| A-1 | 0.987177 | 0.930846 | -0.056331 |
| A-2 | 0.896795 | 0.925018 | +0.028223 |
| A-3 | 0.975703 | 0.967920 | -0.007783 |

**配对结论：** 三路方法在三个场景的闭集 Accuracy、Macro-F1、Weighted-F1 均较高；开集 AUROC/AUPRC 仅在 A-2 同时较高。A-2 的 P95 未知 F1 从 0.5038 升至 0.6741，UFAR 从 0.6434 降至 0.4606；A-1/A-3 的 AUROC 下降。闭集分类提升不能直接推断未知检测排序稳定提升。原始比例 Unknown F1 与类别级漏检见下一节及 `comparison_run_results.csv`、`per_unknown_class.csv`。

## 中文记录：三场景复现与 F1 差异

本次为 USTC-TFC2016 的本地同流配对实验，固定 seed 2022、每类及原始 split 各抽取 10% 流。两种方法各自使用 Known Validation P95 阈值，Unknown 为二分类正类。下表的 F1 均为 **Unknown binary F1**；`balanced_1to1` 在查看分数前按 flow ID 冻结，并非原始测试集类别比例。

| 场景 | Unknown 类数 | OD / 三路 AUROC（平衡） | OD / 三路 F1（平衡） | OD / 三路 F1（原始比例） | OD / 三路 UFAR |
|---|---:|---:|---:|---:|---:|
| A-1 | 1 | 0.9907 / 0.9769 | 0.9659 / 0.9605 | 0.4136 / 0.4106 | 0.0000 / 0.0000 |
| A-2 | 3 | 0.9290 / 0.9406 | 0.5038 / 0.6741 | 0.4028 / 0.5488 | 0.6434 / 0.4606 |
| A-3 | 5 | 0.9782 / 0.9609 | 0.9191 / 0.9231 | 0.8571 / 0.8651 | 0.1125 / 0.0921 |

F1 的直接计数依据是 `F1 = 2TP / (2TP + FP + FN)`。在平衡测试集上，A-1 的 OD/三路 `TP,FP,FN` 分别为 `85,6,0` / `85,7,0`；A-2 为 `199,33,359` / `301,34,257`；A-3 为 `915,45,116` / `936,61,95`。A-2 的已知误拒仍约 6%，F1 下降主要由 Unknown 漏检造成。

- A-1 仅留出 85 条 Tinba，两个方法都全部检出。原始测试比例为 85 Unknown 对 4,806 Known；OD/三路在原始 Known Test 上分别产生 241/244 个假阳性，因此原始比例 F1 只有 0.4136/0.4106。平衡子集的高 F1 不能代表原始流量比例下的精度。
- A-2 的 558 条 Unknown 中有 410 条 Geodo。OD 漏检其中 337 条，三路方法漏检 257 条；OD 对 Htbot/Tinba 另漏检 19/3 条，三路方法在这两类均无漏检。因此 A-2 的 F1 明显低于另两场景，三路方法虽救回部分 Geodo，仍有 257 条漏检。
- A-3 的 Geodo 与 A-2 是同一批 410 个 flow ID（`a2_roles.csv`/`a3_roles.csv` 集合核对），OD/三路只漏检 1/0 条；主要漏检转为 Neris 的 111/95 条。A-3 将 Miuref、Neris 从 A-2 的 Known 移入 Unknown，两个场景分别训练模型并校准阈值。现有结果说明任务组成影响很大，尚不能单独证明是哪一个 Known 类或哪一项训练变化导致 Geodo 差异。

A-2 在平衡测试上的 AUROC 仍为 0.9290/0.9406，而 P95 工作点的 Unknown Recall 只有 0.3566/0.5394，说明排序效果与固定工作点效果需要分别报告。测试集 Youden 阈值仅保留为事后上界：它将 A-2 平衡 F1 提高到 0.8715/0.8927，同时使 Known FRR 升至 0.1254/0.2025；它使用测试标签，不是正式部署阈值。

**结论范围：** 三路方法在 A-2 的 AUROC、F1 和 UFAR 上均改善；A-1、A-3 的 AUROC 低于 Open-Detect。单 seed、10% 本地样本和不同的预训练来源不足以声称三场景稳定优越，也不等同 Open-Detect 论文的原始五折结果。原始数值见 `comparison_run_results.csv`、`per_unknown_class.csv`；样本集合核对日志见项目根目录 `.tmux-task/stage41_f1_overlap_0927/output.log`。

## Preserved evidence

`matched_protocol.json`, `common_flow_pool.csv`, per-scenario role manifests, `balanced_test_ids.csv`, three Open-Detect image pools/checkpoints/logs/sample scores, A-1/A-3 three-view caches/checkpoints/validation histories/Test scores, `comparison_run_results.csv`, `comparison_paired.csv`, `per_unknown_class.csv`, `comparison_report.md`, `completion_verification.json`, `manifest.json`, and project-local named `.tmux-task/stage41_*` execution logs. The initial A-1 GPU0 attempt was interrupted at 52/100 epochs after the user reduced Stage41 to two GPUs; its partial checkpoint and history are retained, while reported A-1 metrics use only the new complete `od_run_2gpu` checkpoint.

## Limitations

This is one local 10% seed, not the paper's exact five folds. A-1 has only 85 Unknown Test flows. The three-view method uses external pretrained weights while Open-Detect is randomly initialized; this compares complete methods on equal flows, not isolated architecture components. Balanced Test composition differs from natural prevalence. Test-oracle thresholds use Test labels and are not deployable validation results.

## Conclusion and next step

The same-flow paired result is in the table above. Stop after Stage41; do not modify historical frozen protocols or automatically start tuning.
