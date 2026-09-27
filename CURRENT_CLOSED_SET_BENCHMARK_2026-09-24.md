# 当前闭集任务结果记录：Stage 23 + Stage 23B

记录日期：2026-09-24。状态：**完成**。本文件汇总当前 ISCX-VPN / ISCXTor2016 的 Service 闭集 Known Test 对照；它不包含 Unknown Detection 指标。

## 任务与数据

| 数据集 | 类别数 | 冻结 flow 数 | Known Train | Known Validation | Known Test |
|---|---:|---:|---:|---:|---:|
| ISCX-VPN | 6 Service | 10,955 | 8,764 | 1,098 | 1,093 |
| ISCXTor2016 | 7 Service | 11,181 | 8,946 | 1,118 | 1,117 |

两个数据集均使用 Stage 20 冻结的 flow ID、Service 标签、Train/Validation/Test membership，重复 seed 为 2022 和 2023。Stage 20 manifest SHA256：`6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb`。Known Validation 用于选模，Known Test 用于最终闭集评价。不同方法的输入表示、预训练和训练预算并不完全相同，表中的可比性是**同 flow、标签、划分和评价指标**，不能将分数差单独归因于网络结构。

## Known Test 总结果

下表为两个 seed 的均值 ± **总体标准差**。E3 是本项目的 OURS-E3-T8 预训练模型。TFE-GNN、Trident 为 Stage 23B 适配版，方法身份见下文。

| 数据集 | 方法 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| ISCX-VPN | YaTC 官方预训练 | **0.8783 ± 0.0119** | **0.8871 ± 0.0116** | **0.8765 ± 0.0126** |
| ISCX-VPN | TrafficFormer 官方预训练 | 0.8532 ± 0.0041 | 0.8606 ± 0.0040 | 0.8474 ± 0.0043 |
| ISCX-VPN | OURS-E3-T8 预训练 | 0.8500 ± 0.0082 | 0.8595 ± 0.0070 | 0.8464 ± 0.0074 |
| ISCX-VPN | Open-Detect corrected-paper | 0.7287 ± 0.0142 | 0.7443 ± 0.0142 | 0.7254 ± 0.0135 |
| ISCX-VPN | RoNeTC 可运行重构 | 0.7068 ± 0.0069 | 0.7157 ± 0.0081 | 0.7013 ± 0.0051 |
| ISCX-VPN | TFE-GNN 前 8 包全协议适配 | 0.6487 ± 0.0192 | 0.6585 ± 0.0146 | 0.6445 ± 0.0163 |
| ISCX-VPN | Trident 前 8 包 86 维适配 | 0.6134 ± 0.0023 | 0.6322 ± 0.0006 | 0.6065 ± 0.0005 |
| ISCXTor2016 | YaTC 官方预训练 | **0.8438 ± 0.0013** | **0.8274 ± 0.0008** | **0.8436 ± 0.0015** |
| ISCXTor2016 | OURS-E3-T8 预训练 | 0.8321 ± 0.0067 | 0.8222 ± 0.0076 | 0.8339 ± 0.0060 |
| ISCXTor2016 | TrafficFormer 官方预训练 | 0.8312 ± 0.0058 | 0.8213 ± 0.0064 | 0.8324 ± 0.0051 |
| ISCXTor2016 | Trident 前 8 包 86 维适配 | 0.6329 ± 0.0027 | 0.6053 ± 0.0019 | 0.6324 ± 0.0033 |
| ISCXTor2016 | Open-Detect corrected-paper | 0.6285 ± 0.0833 | 0.5766 ± 0.0997 | 0.6148 ± 0.0913 |
| ISCXTor2016 | TFE-GNN 前 8 包全协议适配 | 0.6007 ± 0.0009 | 0.5536 ± 0.0138 | 0.5881 ± 0.0087 |
| ISCXTor2016 | RoNeTC 可运行重构 | 0.6043 ± 0.0107 | 0.5340 ± 0.0198 | 0.5934 ± 0.0174 |

## 当前结论

1. YaTC 在两套闭集任务上取得最高的均值。相对 TrafficFormer，配对 Macro-F1 平均差为 VPN `+0.02648`、Tor `+0.00613`；Tor 的优势较小。
2. E3 与 TrafficFormer 基本持平：E3 − TrafficFormer 的配对 Macro-F1 均值为 VPN `−0.00111`、Tor `+0.00092`，每个数据集仅 `1/2` seed 为正。当前闭集结果不支持声称 E3 稳定优于 TrafficFormer。
3. Stage 23B 的 TFE-GNN/Trident 适配版在 VPN 上低于五个既有方法；在 Tor 上 Trident 的平均 Macro-F1 高于 Open-Detect `+0.02870`、RoNeTC `+0.07125`，但相对 TrafficFormer/E3/YaTC 仍低约 `0.216–0.222`。Trident 对 Open-Detect 的提升仅 `1/2` seed 为正，不构成稳定领先证据。
4. 类别级困难仍突出：VPN 的 File-Transfer、VoIP，以及 Tor 的 Chat。Tor Chat 的两 seed 平均 F1：E3 `0.5851`、TrafficFormer `0.5834`、YaTC `0.5818`；TFE-GNN 适配版 `0.1187`，Trident 适配版 `0.2786`。完整类别 Precision/Recall/F1/support 见下方 CSV。

## 方法身份与解释边界

- Stage 23 的五方法正式表包括 TrafficFormer、E3、Open-Detect corrected-paper、RoNeTC 可运行重构、YaTC。Open-Detect 行是本地 corrected-paper 实现；RoNeTC 行是可运行重构。Stage 23 有 `20/20` 结果行、`22,441` 项独立重放检查通过。
- Stage 23B 将 TFE-GNN 和 Trident **重新运行在相同的冻结 flow** 上，而非引用别的样本上的历史分数。TFE-GNN 采用前 8 包、UDP/短流可用的图适配；Trident 的 86 维统计仅由前 8 包计算。这两行不代表作者原生预处理下的论文结果。Stage 23B 有 `8/8` 结果行、`8,904` 项独立检查通过；原 Stage 23 表 SHA256 仍为 `b10cbf5ccead40312768301c5379027ce983f4bcb8ec3cb31338f5b80ced0bdf`。
- Service 标签主要来源于 capture 活动，缺乏逐 flow 权威真值；划分也不是统一 capture-disjoint。Stage 20 Test 已在先前开发实验中暴露，因此这里是**本协议内的开发性闭集比较**。每个均值只有两个 seed，标准差不能代替充分的不确定性评估。本表不能推导任何开集 Unknown Detection 性能。

## 可追溯数据

- [Stage 23 正式报告](stage23_closed_set_method_table/STAGE23_FINAL_REPORT.md)：五方法协议、逐 seed 与类别现象、重放审计。
- [Stage 23B 七方法对比](stage23b_tfe_trident_same_flow_adaptation/stage23b_all_methods_comparison.md)：逐 seed Accuracy/Macro-F1、全部类别平均 F1 与配对差值。
- [28 行逐 seed 原始结果](stage23b_tfe_trident_same_flow_adaptation/stage23b_all_methods_run_level.csv)：七方法 × 两数据集 × 两 seed，含 Accuracy/Macro-F1/Weighted-F1。
- [40 行逐 seed 配对差值](stage23b_tfe_trident_same_flow_adaptation/stage23b_all_methods_paired.csv)与[20 行配对均值](stage23b_tfe_trident_same_flow_adaptation/stage23b_all_methods_paired_summary.csv)。
- [182 行类别级原始指标](stage23b_tfe_trident_same_flow_adaptation/stage23b_all_methods_per_class.csv)：每类 Precision/Recall/F1/support；[TCP/UDP 与包数分组](stage23b_tfe_trident_same_flow_adaptation/stage23b_subgroup_analysis.csv)为适配版事后诊断。
- [Stage 23B 实验结果包](stage23b_tfe_trident_same_flow_adaptation/RESULTS.md)、[独立校验](stage23b_tfe_trident_same_flow_adaptation/completion_verification.json)和[产物清单](stage23b_tfe_trident_same_flow_adaptation/manifest.json)。最后一次结果包校验为 `success`，`178/178` 个产物通过校验。
