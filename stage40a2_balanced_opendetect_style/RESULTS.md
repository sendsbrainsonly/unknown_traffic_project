# Experiment results: stage40a2-balanced-opendetect-style-20260927

- Status: `success`（单次、非同流配对的回顾性诊断）
- Experiment type: `retrospective-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-27T03:10:46Z`
- Objective: Score the frozen Stage40 USTC A-2 three-view model on deterministic balanced Known/Unknown subsets with validation-P95 and released-code Test-oracle threshold rules

## Data and split

沿用冻结的 Stage40 USTC-TFC2016 A-2（三路 TrafficFormer＋包图＋YaTC 特征融合，17 Known／Geodo、Htbot、Tinba 整类 Unknown）。原有逐样本分数覆盖 Known Test 4,333、Unknown Test 558。从中不看分数，以种子 `2022` 和 `SHA256("stage40-a2-balanced-2022:" + flow_id)` 排序，选最低的 558 条 Known；558 条 Unknown 全部保留。新测试组成严格为 558 Known＋558 Unknown。所选 Known 17 类均有样本；Unknown 类计数 Geodo 410、Htbot 63、Tinba 85。选择列表及其 SHA256 写入 `selection_manifest.csv` 与核验文件。

这是对原 Stage40 测试样本的平衡子集分析。独立 Open-Detect v6 的 A-2 是另一套 34,585 图像本地 PCAP 数据、五个独立 8:1:1 划分，每次 600 Known＋600 Unknown。两者类别设置和正负比例相近，但没有可证明一致的流 ID、原始样本或训练划分，因此**禁止当作配对/同流/论文等价对比**。

## Configuration and execution

在固定 Conda 环境、项目本地 tmux `stage40a2_balanced_eval_0927` 中运行 `python stage40a2_balanced_opendetect_style/run_balanced.py`，随后在独立会话 `stage40a2_replay_0927` 中运行 `replay_and_per_class.py`。仅使用原 `stage40_ustc_cic_open_set/ustc_a2/detection/sample_scores.csv` 的冻结预测，不训练模型、不重新推理、不拟合 support，也不修改原 Stage40 与兄弟 Open-Detect 项目。四种预先存在的分数：MSP、Energy、Centroid、DES-v1。

并列报告两种工作点：`Known-Val P95` 原冻结阈值（`score > threshold`）；Open-Detect 发布代码式 `Test-oracle Youden`，在该平衡子集带标签的 Test ROC 上取首个 `argmax(TPR−FPR)`（`score >= threshold`）。后者是用户允许的阈值尝试，但用到了 Test 真值，仅作回顾性分析。四种分数在 P95 阈值上均无 Test 分数恰好等于阈值，故 `>` 与复现代码的 `>=` 此处不改变 P95 决策。Unknown=positive，F1 为二分类 Unknown-positive F1。

## Core results

平衡子集 Known 闭集 Accuracy/Weighted-F1/Macro-F1：`0.989247/0.989247/0.990166`，但只覆盖 558 条 Known，不替代原 4,333 条 Known 的正式闭集结果。

| 方法 | AUROC | AUPRC | P95 Accuracy | P95 Unknown F1 | P95 UFAR | P95 Known FRR | Test-oracle Accuracy | Test-oracle Unknown F1 | Oracle UFAR | Oracle Known FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSP | 0.920042 | 0.907464 | 0.716846 | 0.634259 | 0.508961 | 0.057348 | 0.902330 | 0.910729 | 0.003584 | 0.191756 |
| Energy | 0.869901 | 0.864438 | 0.709677 | 0.619718 | 0.526882 | 0.053763 | 0.846774 | 0.866511 | 0.005376 | 0.301075 |
| Centroid | 0.687372 | 0.762526 | 0.706989 | 0.612100 | 0.537634 | 0.048387 | 0.710573 | 0.620447 | 0.526882 | 0.051971 |
| **DES-v1** | **0.949962** | **0.944431** | **0.749104** | **0.682540** | **0.460573** | **0.041219** | **0.894265** | **0.901830** | **0.028674** | 0.182796 |

DES-v1 P95 的 TP/FP/TN/FN=`301/23/535/257`，Unknown Precision/Recall=`0.929012/0.539427`；Test-oracle 阈值 `2.506504` 时为 `542/102/456/16`，Unknown Precision/Recall=`0.841615/0.971326`。P95 阈值为 `40.231499`。ROC 排序强，但 P95 漏接的 257 条 Unknown 全部是 Geodo；Test-oracle 仅漏接 Geodo 16 条，代价是 Known 误拒从 23 增至 102。

描述性、**非配对**参照：Open-Detect v6 本地 A-2 五次均值在其 600＋600 数据上为 P95 Accuracy/F1=`0.825333/0.797951`、AUROC=`0.853424`，Test-oracle Accuracy/F1=`0.832500/0.807674`。论文 Table V 为 Accuracy/F1=`0.8922/0.8910`，但作者折、F1 细节及训练样本不同。当前三路模型的 Test-oracle `0.894265/0.901830` 数值接近论文，**不能因此宣称公平超过论文或 Open-Detect**；当前 P95 `0.749104/0.682540` 更明确暴露了工作点校准问题。

## Preserved evidence

- `selection_manifest.csv`、`class_support.csv`：score-blind 固定样本列表与类别计数。
- `balanced_results.csv`、`balanced_closed_set.json`、`per_unknown_class.csv`：完整指标、阈值、混淆计数和逐 Unknown 类结果。
- `comparison_context.json`：论文/本地 OD v6 参照及不可配对边界。
- `completion_verification.json`、`independent_replay.json`：来源 SHA256、独立混淆计数/阈值/排名指标重放。
- `run_balanced.py`、`replay_and_per_class.py`、`manifest.json`；tmux 原始日志见项目 `.tmux-task/stage40a2_*_0927/`。

## Limitations

- 只有一个已冻结的三路模型与一组固定平衡子集；不是五模型/五折平均。Subset Known 各类支持数不等，Unknown 也明显偏重 Geodo。
- 原 Stage40 测试集和方法分数已在前序阶段暴露；本分析是回顾性敏感性检查，不是新的 untouched external validation。
- Test-oracle 阈值由同一 Test 的 Unknown 标签优化，可展示性能上限/取舍，但不能当作独立测试或部署效果；换一批数据阈值可能失效。
- OD v6 和当前 Stage40 使用不同流构造、图像池、训练量、折划分、模型；表面分数差异不能归因于模型本身。
- 论文表格只给 Accuracy/F1；其 F1 平均定义没有在 PDF 中完全说明，当前按发布代码的二分类 Unknown-positive F1 计算。

## Conclusion and next step

已完成用户要求的快速 A-2 平衡样本试验：当前三路模型 DES-v1 的 AUROC/AUPRC 为 `0.949962/0.944431`；在允许回顾性 Test 选阈值时，二分类 Accuracy/Unknown F1 达 `0.894265/0.901830`，接近论文 A-2 表格数字。但固定 Known-Val P95 仅 `0.749104/0.682540`、Geodo 漏检显著；当前证据证明的是**排序与阈值工作点分离**，不是严格复现或同样本优于 Open-Detect。若需可发表的公平模型比较，下一步必须先在同一冻结 A-2 流 ID 和相同五折训练/测试协议上重新构建三视图输入并训练/评价两个模型；不应把本单次回顾性结果升级为正式结论。
