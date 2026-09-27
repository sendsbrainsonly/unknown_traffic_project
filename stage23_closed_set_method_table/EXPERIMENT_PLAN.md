# Stage 23 — 冻结流样本的多方法闭集公平对照实验

英文名：**Frozen-Flow Matched Closed-Set Traffic Benchmark**。简称 **Stage 23 Matched Closed-Set Benchmark**。

版本：v1.0（2026-09-23）。性质：正在进行的诊断性实验计划，不是其他论文的作者等价复现承诺，也不是未接触测试集的独立验证。本计划不修改 Stage 20–22 的协议或结果。

## 1. 研究问题

在**完全相同的流、Service 标签和 Train/Validation/Test 成员关系**下，我们的 OURS-E3-T8 与 TrafficFormer、Open-Detect 及其他输入可适配方法的闭集分类差距有多大？差距可能来自输入表示、预训练、优化配置，还是原本不可比的样本/任务定义？

本阶段只做闭集。主要指标为 Known Test Macro-F1；同时报告 Accuracy、Weighted-F1、每类 Precision/Recall/F1/support、混淆矩阵及逐 seed 配对差值。不运行 Unknown detection；不把闭集涨点推论为开集增益。

正式同样本表只接纳冻结样本覆盖率 100%、标签/角色一致且可逐样本重放的方法。历史项目使用不同数据、标签或划分得到的高分只作背景，绝不直接并入配对排名。

## 2. 冻结数据和任务

唯一闭集成员来源：`../stage20_dual_coarse_service_protocol/closed_service_manifest.csv`，当前 SHA256 `6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb`；正式运行前后复核不变。

| 数据集 | 目标 | Service 数 | 总流数 | Train | Validation | Test |
|---|---|---:|---:|---:|---:|---:|
| ISCX-VPN | Service-6 | 6 | 10,955 | 8,764 | 1,098 | 1,093 |
| ISCXTor2016 | Service-7 | 7 | 11,181 | 8,946 | 1,118 | 1,117 |

VPN 类别为 Chat、Email、File-Transfer、P2P、Streaming、VoIP；Tor 额外包含 Browsing。Tor 的 Audio/Video 已由 Stage 20 冻结为 Streaming，本阶段不得重新映射。共 22,136 条流。采用现有配对 seeds 2022/2023；不得移动样本、删除难类、重建标签或调整角色。

科学边界：Service 真值是 capture 级弱标签，不是逐流人工真值；虽按 exact-image group 隔离，仍非统一 capture-disjoint。公共预训练数据与目标流量是否重叠未完全查清。这是一项同样本闭集**开发性诊断**，不能称为 unseen-capture 或 untouched external validation。

## 3. 方法清单和身份

| 方法 | 身份 | 当前准入状态/动作 |
|---|---|---|
| TrafficFormer / E1 | Stage 22 官方预训练权重初始化的闭集分支 | 样本/预测已核验；复用，不重训 |
| OURS-E3-T8 / E3 | 预训练 TrafficFormer + FIG/TAGCN 融合 | Stage 22 同样本结果；复用，不冒充 Open-Detect |
| Open-Detect corrected-paper | Stage 16S corrected-paper 训练实现 | Stage 20 输入适配通过；完成 4 个 dataset×seed run 并独立复核 |
| RoNeTC | 原有 view 缓存基本覆盖 Stage 20 | 核查 Tor 缺口及每流三种 8 包 view parity |
| YaTC | 原项目 MFR 图像来自不同样本 | 逐条 Stage 20 flow 重建输入、核查覆盖 |
| ET-BERT | 原 TSV 样本/标签不同 | 审计 packet/TSV 重建和短流策略 |
| TFE-GNN | 原 graph/segment population 不同 | 同 flow 建图并核验角色/标签 |
| Trident | 原 86 维统计来自其他任务 | 审计是否可对全部 Stage 20 flow 提取同定义特征 |
| UnDiff | 原生 one-class anomaly 方法 | 不纳入 6/7 类闭集正式表；独立异常检测问题另评 |

其他项目源码、权重、已有结果均只读。不能重建相同 flow 输入的模型标为 `NOT_COMPARABLE`，不以静默丢流换取可跑性。Open-Detect 本轮明确标为 corrected-paper，不冒充未改动的 released-code Native。

## 4. 公平性、调参和泄漏控制

完全对齐：flow ID、Service 标签、Train/Validation/Test membership、seeds 2022/2023、最终评价脚本与 F1 定义。每个新方法训练前输出输入清单，逐角色核查 ID 集合、样本数、标签、来源哈希和跨角色交叉；任何单流缺失或标签冲突都阻断正式表准入。若需研究共同交集，另立协议，不修改 Stage 23 正式表。

方法本身的输入表示、架构、损失、初始化/预训练及合理训练预算允许不同，但必须逐项记录；强行统一 epoch、batch 或 LR 可能改变方法身份。预训练与随机初始化分列标记，不能把分数差直接归因于架构。

先运行各方法当前可复现的固定配置。用户允许有限调参：每个通过输入门禁的方法，须**在运行候选前**登记小范围候选配置、次数及预算；仅用 Known Validation 选候选和 checkpoint，不根据 Test 差值增删配置。保留默认配置、所有候选及失败证据，而不只报告最高分。若无法预先限定搜索，就先不启动调参。

scaler、特征选择、采样权重及其他拟合变换只在 Known Train 拟合；Validation 仅用于预登记的选择；Known Test 仅在模型冻结后加载评估。Unknown/LOSO Test 不进入本闭集实验。

## 5. 执行顺序和门禁

### A. 预检及既有基线（已完成）

`scripts/build_preflight.py` 已通过 120/120 项：Stage 20 membership、来源索引/标签、Stage 21 缓存和 Stage 22 E1/E3 逐样本重放。E1/E3 共 8 条 dataset×seed×method 行已核验后复用。VPN E1/E3 Macro-F1 为 `0.860607±0.003961` / `0.859493±0.007015`；Tor 为 `0.821259±0.006376` / `0.822178±0.007643`。这是现有基线，不是完整横向排名。

### B. 完成 Open-Detect 4 个配对 run

两数据集各 seeds 2022/2023。沿用锁定的 Stage 16S corrected-paper 配置：最多 100 epochs、batch 128、Adam LR 0.001、prototype reset epoch 51/81，Known Validation Accuracy 选 checkpoint。保存每 run 配置、历史、最佳权重、预测和哈希。

本计划落地时进度脚本显示 VPN-2022、VPN-2023、Tor-2022 为 `SUCCESS`，Tor-2023 为 `NOT_STARTED`。前三项仍须独立核查 checkpoint SHA、Test ID/标签和指标重放，不能仅凭状态文件进正式表。启动第四项前重新检查 tmux/GPU 现场，不重复训练已完成项。

### C. 其他方法逐项输入审计

为每个方法登记 method card：源码/权重来源及 SHA、作者原任务、样本单位、原输入定义、Stage 20 转换方式、逐角色可用 flow 数、缺失原因、预训练语料和预算。优先检查已有资产能否以最小转换得到同样本输入；**CPU/只读覆盖审计通过后才考虑 GPU 训练**。准入要求 Train/Validation/Test 各 100% 覆盖、逐流标签和角色一一对应、无额外样本或静默过滤。

### D. 配对训练和汇总

通过准入的方法分别对两数据集和两 seeds 训练/评估。保存初始化、预算、最佳 epoch、耗时、峰值显存、checkpoint SHA、逐样本预测及失败原因。独立从预测重算 Accuracy/Macro-F1/Weighted-F1、每类指标与混淆矩阵，要求与记录指标一致。

逐 dataset×method 报告两 seed 的 mean±population std（与 Stage 22 既有汇总约定一致），同时保留 sample std 列和所有原始单 run；在相同 dataset×seed 上计算各方法减 E1、减 E3 的配对差值与胜出次数。两 seed 不足以作强统计推断。主表并列展示输入覆盖、预训练状态、计算预算和最差类别。

## 6. GPU 与运行约束

所有命令使用项目规定的命名 tmux 会话和固定 Conda 环境；GPU 工作启动前按实时显存和进程选卡。最多同时占用**6 张本用户可用的物理 GPU**，一进程一卡，不接触其他用户进程；6 卡是上限，不是必须填满的目标。

按用户当前要求，**2026-09-24 08:00 北京时间（00:00 UTC）前至少释放其中 4 张卡**。预计跨越该截止时间的任务，若没有可靠、可核验的停机安排，不应启动。记录会话、PID、物理 GPU、起止时间和退出状态。本次落地计划不启动或停止任何训练。

所有输出仅写入本项目 Stage 23 目录；Stage 20–22、其他方法项目和原始数据保持不变。失败、中断和完成运行的证据都保留。

## 7. 交付物与验收

已有：`PROTOCOL.md`、`method_readiness.csv`、`sample_inventory.csv`、`sample_parity.csv`、`preflight_verification.json`、`closed_set_run_results.csv`、`RESULTS.md`、`manifest.json` 及脚本。后续按实际准入方法补充 method cards、逐 run 配置/历史/checkpoint/预测、逐类结果、混淆矩阵、配对对照表、聚合表和 `completion_verification.json`；若实际文件名调整，在 `RESULTS.md` 标出路径。项目级实验索引与执行交接记录终态。

验收条件：Stage 20 哈希不变；正式行 100% 样本 parity；checkpoint 哈希和指标重放通过；Test 选模/调参为 0；未准入方法原因明确；四个 Open-Detect run 或失败证据完整；报告单 run、均值/标准差、逐类错误、资源与预训练差异。**分数是否漂亮不是验收条件。** 输入不可重建的方法诚实报告 `NOT_COMPARABLE`，不补造可比结果。

## 8. 解释边界

若 E3 低于基线，先核对数据、标签、输入、预训练和优化差异，不仅凭单个差值断言方法失败；若 E3 高于基线，也不能忽略弱 capture 标签和潜在预训练重叠。Stage 20 Test 已在 Stage 22 中评价，因此 Stage 23 是已暴露同任务上的开发性诊断，不是新的独立外部验证。任何后续泛化主张须另行冻结未被方法选择触及的数据或协议。
