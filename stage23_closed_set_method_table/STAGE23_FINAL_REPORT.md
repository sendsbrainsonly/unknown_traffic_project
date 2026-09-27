# Stage 23 — Frozen-Flow Matched Closed-Set Traffic Benchmark

状态：**COMPLETE / FIVE_METHOD_MATCHED_TABLE_VERIFIED**（2026-09-23）。
本轮是已暴露 Stage20 Service 任务上的开发性闭集诊断，不是 untouched external validation，也不包含 Unknown Detection。

## 冻结任务与验收

ISCX-VPN 为 Service-6：10,955 flows，Train/Val/Test 为 8,764/1,098/1,093。ISCXTor2016 为 Service-7：11,181 flows，8,946/1,118/1,117。总计 22,136 条冻结 flow；seeds 为 2022、2023。Stage20 manifest SHA256 在运行前后均为：

    6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb

五方法 × 两数据集 × 两 seed = **20/20** 正式结果行。独立重放检查 **22,441/22,441 PASS**：每个新 run 的最佳权重 SHA256、Known Validation 选模规则、完整 epoch 历史、逐流 ID/标签/角色、Test 预测、Accuracy/Macro-F1/Weighted-F1、原始样本覆盖均核对通过。八条 E1/E3 行来自已验证 Stage22；Open-Detect 四条先经额外 40,025 项独立检查。Test 用于最终评价，不用于 checkpoint 或超参数选择；Unknown 使用为零。

## 闭集结果

下表为 Known Test，两 seed 均值 ± **population std**；同时提供原始单 run 与 sample std，不能把 n=2 视为稳定的统计显著性证据。

| 数据集 | 方法 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| ISCX-VPN | TrafficFormer E1（官方预训练） | 0.8532 ± 0.0041 | 0.8606 ± 0.0040 | 0.8474 ± 0.0043 |
| ISCX-VPN | OURS-E3-T8（同源预训练） | 0.8500 ± 0.0082 | 0.8595 ± 0.0070 | 0.8464 ± 0.0074 |
| ISCX-VPN | Open-Detect corrected-paper | 0.7287 ± 0.0142 | 0.7443 ± 0.0142 | 0.7254 ± 0.0135 |
| ISCX-VPN | RoNeTC runnable reconstruction | 0.7068 ± 0.0069 | 0.7157 ± 0.0081 | 0.7013 ± 0.0051 |
| ISCX-VPN | YaTC（官方预训练） | **0.8783 ± 0.0119** | **0.8871 ± 0.0116** | **0.8765 ± 0.0126** |
| ISCXTor | TrafficFormer E1（官方预训练） | 0.8312 ± 0.0058 | 0.8213 ± 0.0064 | 0.8324 ± 0.0051 |
| ISCXTor | OURS-E3-T8（同源预训练） | 0.8321 ± 0.0067 | 0.8222 ± 0.0076 | 0.8339 ± 0.0060 |
| ISCXTor | Open-Detect corrected-paper | 0.6285 ± 0.0833 | 0.5766 ± 0.0997 | 0.6148 ± 0.0913 |
| ISCXTor | RoNeTC runnable reconstruction | 0.6043 ± 0.0107 | 0.5340 ± 0.0198 | 0.5934 ± 0.0174 |
| ISCXTor | YaTC（官方预训练） | **0.8438 ± 0.0013** | **0.8274 ± 0.0008** | **0.8436 ± 0.0015** |

YaTC − TrafficFormer 的配对 Macro-F1：VPN 平均 **+0.02648**（2022 +0.03411、2023 +0.01885；2/2 正），Tor **+0.00613**（+0.00053、+0.01174；2/2 正）。YaTC − E3：VPN **+0.02759**（2/2 正），Tor **+0.00522**（1/2 正）。E3 − E1：VPN **−0.00111**、Tor **+0.00092**，均为 1/2 seeds 正；因此“我们的融合分支显著优于 TrafficFormer”在这两个闭集任务上**没有得到支持**。YaTC 在两个数据集上最高，但 Tor 对 E1 的收益很小，不能据此宣称大幅或跨数据集普遍优势。

逐 seed 关键数据：YaTC VPN Macro-F1 为 0.898678/0.875492，Tor 0.828168/0.826619；TrafficFormer VPN 为 0.864568/0.856647，Tor 0.827635/0.814883。Open-Detect Tor 为 0.676264/0.476862；低质量 2023 run 原样保留，没有剔除或重训。

## 类别级现象

两 seed 平均每类 F1 中，YaTC 相对 TrafficFormer 的 VPN File-Transfer 从 0.7410 提至 0.7967，VoIP 从 0.6226 提至 0.7210；VPN P2P 两者均约 1.0。Tor Browsing 从 0.8180 提至 0.8733，File-Transfer 从 0.9438 提至 0.9723；但 Tor VoIP 从 0.7749 降至 0.7310、Email 从 0.9581 降至 0.9322。Tor Chat 仍困难：TrafficFormer/E3/YaTC 分别约 0.5834/0.5851/0.5818，YaTC 的总体优势没有解决该类。全部类别 Precision/Recall/F1/support 与混淆矩阵见完整 CSV，不能只依据总体均值描述类别改善。

## 方法身份与计算预算

TrafficFormer E1 与 E3 复用 Stage22 官方预训练权重微调结果，不重训。Open-Detect 行是本地 corrected-paper 实现，不冒充 released-code Native。RoNeTC 行是可运行重构：三种 8 包 view，100 epochs，batch 128，Known Validation Accuracy 选模；Tor 原 view 缓存缺的 137 条流按冻结 packet refs 从原始 PCAP 补全。YaTC 使用官方预训练权重、作者首 5 包 40×40 MFR 字节规则、200 epochs、batch 64、作者训练循环，Known Validation weighted-F1 选模；短于 5 包按作者规则零填充。方法输入/预训练/训练预算不同，**只对齐 flow、标签、角色、seed 与最终指标**，分数差不能单独归因于网络结构。

新训练单 run 平均耗时与峰值 GPU allocated：Open-Detect 约 8.1–8.2 分钟、1.98 GiB；RoNeTC 约 57–58 分钟、32.57 GiB；YaTC 约 19.8–20.2 分钟、1.98 GiB。这里的耗时是并行运行条件下的记录值，不是受控单卡吞吐基准。

## 准入排除和数值审计

ET-BERT 原生 flow 规则会因少于 3 包先排除至少 **15,195/22,136** 条；TFE-GNN 原生 TCP 图规则仅 Train/Val 已不能纳入 **8,316** 条 UDP flow。Trident 现有 86-D 缓存对应另一批完整双向流且官方预处理不完整；UnDiff 原生为 one-class anomaly 任务。它们不进入本轮 100% 冻结样本的正式表，也没有借历史高分替代。详见 METHOD_ADMISSION_AUDIT.md。

首次 YaTC Tor CPU 推理的 seed2023 无法精确重放训练时的 Known Validation 指标，故**未接受其 CPU Test 结果**；seed2022 CPU 诊断结果和失败日志均保留。之后两 seed 使用训练一致的 CUDA autocast 路径，以新文件名重放，Known Validation 先通过严格一致性检查，再接受 Test，checkpoint 未变。Tor 测试 MFR 缓存是在 seed2023 的 checkpoint 选定后建立，seed2022 训练进程未读取它；VPN 测试 MFR 在两 seed 均完成后建立。任何 Test 值均未参与训练或选模。

RoNeTC 首次 Tor smoke 曾因 DataLoader 的 AF_UNIX 临时路径过长而中断；该尝试的日志与 INTERRUPTED.md 被保留。随后仅将临时 socket 指向**项目内**短路径，模型、数据、损失、训练配置均未变化；重试 smoke 通过，四个正式 run 从头完整训练并通过重放。首次失败不算作正式结果。

## 解释边界与结论

同一批冻结样本与标签上的横向对照说明，先前较大的“我们与其他项目高分差距”并非只由项目之间使用了不同 flow 引起：当严格对齐之后，YaTC 仍略/明显领先，而 E3 与 E1 基本持平，Open-Detect/RoNeTC 则低很多。但原因仍可能是输入字节表示、预训练语料/初始化、训练目标与预算的组合；**不能由本表单独判定是某一层代码 bug 或某一个特征的因果效应**。Service 标签主要来自 capture 活动，非逐流权威真值；划分并非统一 capture-disjoint，且 Stage20 Test 已在 Stage22 暴露。因此结果只支持本协议内的开发性比较，不支持独立外部泛化声明，也不推出任何开集性能结论。

完整重放产物：stage23_full_run_results.csv、stage23_full_summary.csv、stage23_paired_vs_e1_e3.csv、stage23_all_method_per_class.csv、stage23_all_method_confusion.csv、completion_verification.json；逐 run checkpoint、配置、history、预测及 SHA 保存在 runs/ 下。原始 Stage20–22 文件和其他方法项目未修改。
