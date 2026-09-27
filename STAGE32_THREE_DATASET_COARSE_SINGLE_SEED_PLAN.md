# Stage 32 — 三数据集粗粒度三路等权闭集对照（单 seed）

状态：**仅计划，未启动训练或测试**。本计划取代尚未完成的 Stage 31 四数据集细粒度自动队列；Stage 31 已完成及中断的产物保留，不冒充正式结果。USTC-TFC2016 不进入本轮。

## 研究问题与边界

检验本项目的 **TrafficFormer + FIG/TAGCN + YaTC 三路特征级等权融合**，在更粗的语义标签下，是否比同一批流上的本项目历史 E3 闭集结果更好。只跑 ISCX-VPN、ISCXTor2016、VNAT 各一个预先固定的 protocol/训练 seed，不跑单分支消融，不训练比较方法，不运行 Unknown 检测。粗粒度指标升高首先表示**任务变简单**；只有在同一粗标签、同一 flow ID、同一划分下的配对差值，才可称为模型改进证据。

## 预先固定的三个实验单元

| 数据集 | 冻结样本/划分 | 本轮粗标签 | 训练 seed | 历史对照口径 |
|---|---|---|---:|---|
| ISCX-VPN | Stage 20 全部 10,955 个 Service 流；Train/Val/Test = 8,764/1,098/1,093 | Stage 25 固定映射：Chat、Email、VoIP 合为 Communication；其余 File-Transfer、P2P、Streaming 不变，共 4 类 | 2022 | Stage 25 中同一 seed、相同 Test flow 的 E3 coarse hard-remap 和 E3 coarse-head；YaTC coarse hard-remap 只作背景参照 |
| ISCXTor2016 | Stage 20 全部 11,181 个 Service 流；Train/Val/Test = 8,946/1,118/1,117 | 同一映射，再保留 Browsing，共 5 类 | 2022 | 同上，使用 Tor 的同一 seed 和同一 flow |
| VNAT | Stage 14B `medium_seed2025` 原有 Known Train/Val/Test = 15,704/1,960/1,960；Unknown applications = zoiper、vimeo、sftp，全部排除在本次闭集拟合和评价之外 | 按 VNAT 原论文 Table II 的 application→category 映射。此 protocol 的 Known applications 实际只覆盖 4 个 category：Streaming={netflix,youtube}；Chat={skype}；Command & Control={rdp,ssh}；File Transfer={rsync,scp} | 2022 | 优先用现有本项目 F2/cleaned-pipeline `medium_seed2025` 的逐样本 Known-Val 预测做相同粗标签 hard-remap；若可无训练地重放其冻结 checkpoint 并验证 ID/argmax parity，再补同一 Known Test 配对；否则 Test 只报新方法绝对值，不声称配对提升 |

VNAT 的 `2025` 是**冻结划分编号**，不是额外训练 seed；每个单元仅训练一次，模型训练随机 seed 固定为 `2022`。不得因 VNAT Medium-2025 较难而换成 Medium-2026，也不得根据验证/测试结果改映射或删样本。三个数据集标签空间不同，分别呈报，不把它们的 Macro-F1 直接平均成“总体准确率”。

VNAT 五类原始定义（包含本 protocol 中 absent 的 VoIP={zoiper}）以原论文为依据：<https://arxiv.org/pdf/2205.05628>，Table II。当前 PCAP→flow 的 application 标签仍具有 capture-derived 弱标签限制；官方 HDF 的五类结果不能当作与本项目 23,449 个 clean PCAP flows 逐条对齐的标签。

## 模型与训练：只测本项目三路方法

1. 复用已完成的 **同一数据/flow ID** 三路 encoder checkpoint，不重新训练单分支模型。ISCX 两个单元使用 Stage 20/Stage 30 的三路 frozen representation；VNAT 使用 Stage 31 `Medium-2025` 已完成且通过哈希审计的三路 branch checkpoint。训练前必须逐项核对 checkpoint SHA256、class order、flow ID、Train/Val 覆盖；不能拿 Stage 31 的 ISCX `medium-2022` *application* 分支充当 Stage 20 *Service* 分支。
2. 三路分别为 TrafficFormer 768-D、FIG/TAGCN 128-D、YaTC 192-D。只在 Known Train 上拟合各路标准化；沿用 Stage 30 `T0_equal`：每路 64-D Gaussian adapter（推理用 mean），三路固定 1/3 权重后拼接，宽 128 的两层分类头。只将监督标签替换为本计划预注册的粗标签，不改输入/网络/损失/优化设置：30 epochs、batch 256、Adam 1e-3、Known-Val Macro-F1 选 checkpoint、seed 2022。原 fine branch 和 fine head 全部保留不覆盖；新 coarse head 是**新增能力**，不是把旧 fine 性能悄悄替换掉。
3. 训练和选择只使用各单元 Known Train/Val。Unknown flow 不得用于 encoder、adapter、scaler、checkpoint 或阈值；Known Test 只在三个 coarse head 与对照提取规则全部冻结后读取一次。禁止 Test 调参、按结果更换 protocol/seed、额外引入动态权重。

## 顺序执行和停止门槛

1. **数据/来源 gate**：锁定三个冻结 manifest、标签映射与历史 checkpoint 哈希；审计每个 split 的唯一 flow ID、类别支持数及三视图的 100% 同 ID 覆盖。检查 VNAT `Medium-2025` 三 branch 是否确实完整；检查 ISCX Stage 20 Test 三视图能否按原模型无损提取。任一单元缺失则记录 `BLOCKED` 数量与原因，不缩小样本集合。
2. **历史对照 gate**：在新训练前固定每个对照的文件、seed、样本 ID、原标签与粗标签转换方式。把 hard-remap、重新训练 coarse head、三路 coarse head 三种机制分列，避免混作一项。VNAT 若不能形成同 ID 的历史 Test 对照，预先标记为“Validation 配对、Test 非配对”。
3. **单 seed 训练**：按 VPN → Tor → VNAT 顺序，每数据集恰好一套三路 coarse adapter/head；最多使用用户授权的三张 GPU，启动每项 GPU 工作前重新检查可用显存与进程归属。保存完整曲线、最佳 epoch、checkpoint 与 SHA256、Known-Val 逐样本预测。
4. **一次性 Test**：确定全部 checkpoint 后才读取 Known Test；输出每条流 ID、真值粗标签、预测、概率、所属 split，并独立重放计算 Accuracy、Macro-F1、Weighted-F1、各类 Precision/Recall/F1/support、混淆矩阵。保留原细标签作为元数据，用于检查哪些困难应用被粗标签合并掩盖。
5. **配对报告**：逐数据集列 `新 T0 − 历史本项目 E3 hard-remap` 及 `新 T0 − 历史本项目 E3 coarse-head` 的 Accuracy/Macro-F1/Weighted-F1 差值（仅 exact-flow matched 时）；列 YaTC 历史结果作竞争力参照，不重训 YaTC。VNAT 只与已验证同流的历史本项目结果配对。显示每类变化与 bootstrap 区间作为不确定性描述；单 seed 不报告 seed 稳定性或显著跨 seed 结论。

## 解释与交付

- 主结果必须是**完整全流**三路模型的闭集 Known Test 指标。不能过滤 1 包流或只保留 ≥2 包来美化主表；若某 branch 无法覆盖短流，停止对应单元并记录，而不是改变样本总体。
- 同时列历史 fine→coarse 的任务简化收益，以及新 coarse-head 相对**同一粗任务**历史 E3 的真正配对差值。VNAT 的应用级 Unknown 集与 coarse category 有重叠，本阶段不评价开集，也不延续“Unknown category”主张。
- 已曝光的 Stage 20/VNAT Test 只属于 development，不能写成 untouched external validation。单 seed 仅支持探索性判断；若提升为负也如实保留。
- 新建独立 `stage32_three_dataset_coarse_single_seed/` 保存预检、配置、运行日志、三个 checkpoint、逐样本结果、配对表、`RESULTS.md`、完整报告、`manifest.json` 和独立复核；不覆盖 Stage 20–31 的数据/模型/结果。
- 本计划**不自动启动训练**。下一步先完成数据/对照 gate；若 gate 通过，再运行三路 coarse head。最终直接回答每个数据集是否提升、提升来自粗标签还是模型、是否超过历史 YaTC 参考值，并明确未跑 USTC/多 seed/Unknown 检测。
