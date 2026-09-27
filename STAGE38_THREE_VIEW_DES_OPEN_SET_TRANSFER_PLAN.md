# Stage 38 — 三路融合表示的 DES 开集迁移实验

英文名：*Staged Three-View Representation for Unknown-Free Detection*

状态：**PLAN_ONLY，尚未启动训练、Unknown 特征提取或开集评估**。本文件记录实施前的任务边界；后续若修改协议、分数或 Gate，必须先留下带日期的修订记录，不覆盖原定义。

## 1. 研究问题与方法边界

用 Stage32 效果较好的“TrafficFormer + FIG/TAGCN + YaTC 分支分别训练、特征层固定等权融合”替代 Stage37 的端到端联合训练，检查：

1. 更好的 Known 闭集表示是否同时提高 Unknown 检测；
2. 以前的经验支持解耦 DES-v0 和全局＋局部 DES-v1 在新融合表示上是否仍有价值；
3. 改善或失败发生于哪些 Unknown application，以及是否伴随 Known 拒识代价。

本阶段保留 Stage15B 的机制经验，但不将 Stage32 分类头的 MSP 冒称为 Open-Detect Native，也不将跨编码器组合冒称为历史同编码器 H1。Stage37 联合训练权重不进入主实验。Unknown 检测和闭集分类必须使用同一个已冻结的三路融合模型；不同模型之间的比较另标为系统级比较。

## 2. 冻结数据与语义

主要任务是既有 **application-level Unknown**，不是新 Service-level LOSO。Known/Unknown 按 application 整类划分；同一 Unknown application 的 VPN/non-VPN 样本都不得进入训练或校准。所有 Unknown 类、flow ID、Train/Validation/Test membership 在模型训练前固定。

| 数据集 | 协议 | Known Train/Val/Test | Unknown Test | Unknown application | 模型处理 |
|---|---|---:|---:|---|---|
| VNAT | Stage14B `medium_seed2025` | 15,704 / 1,960 / 1,960 | 3,825 | sftp、vimeo、zoiper | 直接冻结 Stage33 六类三路融合模型 |
| ISCX-VPN | Stage12 `medium-2022` | 12,910 / 1,611 / 1,621 | 6,000 | Hangouts、Skype、YouTube | 按 Stage32 配方在 Known-only 协议上重新训练 |
| ISCXTor2016 | Stage12 `medium-2022` | 8,876 / 1,107 / 1,113 | 4,000 | P2P、Skype | 同上 |

VNAT 的 Stage33 是 Stage32 分开训练＋固定等权特征融合方法的六类版本，不是 Stage37 联合训练。VNAT `vimeo` 与 Known Streaming、`sftp` 与 Known 文件传输应用存在语义邻近，必须作为困难 application-level Unknown 如实报告，不能称作未知 Service。Stage33 六类 taxonomy 是此前观察四类 Test 后确定的，因此 VNAT 结论仅属开发性证据。

Stage32 的 ISCX-VPN/Tor 原权重训练时已见过其全体 Service，**不可**直接用于严格 Unknown-Free 评估。ISCXTor 的 Stage12 medium 将 P2P 留出，Known 粗类预计只有四类而非 Stage32 原五类；新协议的闭集 Macro-F1 不得与原五类分数作为同任务直接比较。实现前需按冻结 metadata 审计粗标签映射与各角色的非空支持；任何映射歧义必须在读取 Unknown 结果前解决。Stage12 的 application-level Unknown 任务与新 Service-level Unknown 任务不可混称。

## 3. 实施顺序与输入审计

### 38A：VNAT 冻结模型先行

1. 只读校验 Stage14B freeze hash `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`、Stage31 三个分支和 Stage33 adapter/head checkpoint SHA256、样本 ID、标签与角色。
2. 复用 Stage36 已完成的 Known-Val packet/cache/logit parity 检查，但先重新验证其结果及相关哈希；Stage36 的历史产物保留，不改写已登记的四分数计划。旧 Stage36 等待 controller 已停止，不以其陈旧 `WAITING` JSON 作为运行证据。
3. 冻结 Stage33 六类模型，提取融合分类头之前的 128 维特征 `h(x)` 和分类 logits。先完成 Known Train/Validation，再在分数公式及阈值实现测试通过后读取 Known Test/Unknown Test。

### 38B：ISCX-VPN/Tor Known-only 模型迁移

1. 验证 Stage12 medium split 和原始输入来源哈希；固定一个训练 seed `2022`，不重新选择 Unknown 类、不重切 flow。
2. 按 Stage32 的三路输入、分支结构、分支单独训练、Known-Train-only 标准化、64 维每路 adapter、固定 `1/3` 特征权重、融合分类头和 Known-Val checkpoint selection 重新训练；只用该协议的 Known Train/Val。具体 epoch、batch、LR、损失、预训练初始化和代码版本须在首个 GPU run 前从 Stage32/其源训练记录逐项锁定并写入 run config，不因 Unknown 结果调整。
3. Stage31 已完成的分支或缓存仅可在同协议、同标签、同输入公式、同 seed、完整日志和 SHA256 全部匹配时只读复用；否则在新的 Stage38 目录重训，绝不修改或“续跑”已中止的 Stage31 历史目录。
4. Known-Val 选好三个分支、adapter 与融合头 checkpoint 后统一冻结。报告新协议的 Known-Val/Test Accuracy、Macro-F1、Weighted-F1 与逐类指标，不能用 Known Test 回调参。

运行顺序固定为 VNAT → ISCX-VPN → ISCXTor。若 VNAT 的输入/哈希 Gate 失败，先记录并停止 VNAT 对应单元，不借用其他数据集的结果填空；ISCX 可按各自独立 Gate 继续。训练和推理均用项目指定 Conda 环境、命名 tmux 与实时 GPU 容量选择，不干扰现有进程。

## 4. 冻结的开集分数

所有新表示分数在**同一个冻结的融合特征 `h(x)`、同一批样本**上计算，数值越大表示越异常：

| ID | 分数 | 拟合信息 |
|---|---|---|
| C0 | `1 - max softmax(logits)`（MSP） | 不拟合支持；分类头辅助基线 |
| C1 | `-logsumexp(logits)`（Energy，温度 1） | 不拟合支持；分类头辅助基线 |
| D0 | `DES-v0 = min_y ||h(x)-c_y||²` | `c_y` 仅由 Known Train 的对应类均值得到 |
| D1 | DES-v1：最近经验中心类 `y*`，全局 `d_g=D0`，局部 `d_l` 为该类 Known-Train `kNN-10` 欧氏距离均值；`0.5 Z_g + 0.5 Z_l` | `k=10`、权重 `0.5/0.5` 固定；`Z` 的 median/MAD 仅由 Known Validation 求得，分母加 `1e-8` |

每个分数独立使用 Known Validation P95（`numpy method="higher"`）作为 operating threshold；分数**严格大于**阈值判 Unknown。Unknown 为 positive class。不得使用 Unknown/Test 拟合中心、kNN、scaler、median/MAD、经验 CDF、阈值、权重或选择最佳 score。C0/C1 是辅助诊断，不是以前的 OD Native。

## 5. 与旧方法的公平比较及 H1 边界

- **机制主比较**：在同一个新融合表示上比较 D1−D0、D0/D1−C0/C1，以判断经验支持解耦与局部支持是否增益；样本和阈值规则完全相同。
- **系统级对照**：若历史 Open-Detect Native、DES-v0/v1 和 H1 的冻结分数能与本轮 exact flow ID、Known/Unknown 真值及协议逐项对齐，可列出同样本结果；其 encoder 不同，不能据此单独归因于 support 机制。
- **历史 H1** 定义为 `max(A_OD,A_DES0)`，其中每个 `A` 是 Known-Val ECDF percentile。Stage32 没有 OD learned-prototype/posterior 分数，因此本轮不把 C0/C1 代入并称为原 H1。若预先验证旧 OD 分数与新 D0 的 exact-ID 对齐，可另设次要的 `H1-X = max(A_OD_old,A_D0_new)`，明确标为**跨编码器适配版**；公式必须在 Unknown 值打开前登记，不能用 Test 选择是否报告。历史 H1 仅为 `HYBRID_PARTIAL` 开发候选，不能预设其会稳定修复 rsync/scp 型 support overlap。
- 不训练 gating model，不搜 `k`、融合权重、温度或阈值百分位，也不因 VNAT 结果为 ISCX 临时改变方法。

## 6. 评价、完整性与解释

每个 dataset × score 保存逐流 ID、真实 application、Known/Unknown、原始分数、Known-Val 统计量、P95 阈值和最终判决。报告：Unknown AUROC、AUPRC、UFAR、Known FRR、Known Acceptance；Known 闭集 Accuracy、Macro-F1、Weighted-F1；各 Unknown application 的召回/UFAR及分数分布。AUPRC 同时报告 Unknown prevalence。若跨数据集汇总，保留每个数据集的原值和差值，不只给总均值；一协议一 seed 不宣称“稳定多 seed”。

重点分析 VNAT `sftp/vimeo/zoiper`、VPN `Hangouts/Skype/YouTube`、Tor `P2P/Skype`，以及 nearest Known class、empirical-support overlap、OD 与 DES 决策互补。样本级配对计数和 bootstrap 区间属于探索性分析；同 capture flow 可能相关，不能把逐流 CI 当作独立采集验证。

执行前后必须通过：split 与 checkpoint SHA256、Train/Val/Test/Unknown ID 不重叠、Unknown 参与训练/支持/校准计数均为 0、Test 参与参数选择计数为 0、逐样本指标独立重放、历史 Stage12/14B/31/32/33/36 文件哈希不变。任一失败仅停止对应 run 并保存故障证据，不静默跳过或改变协议。

## 7. Gate 与论文表述

本轮是**单 seed、已暴露数据上的开发性诊断**，不设置“外部确认成功”的 Gate。结果只能归入：

- `PROMISING_DEVELOPMENT`：D0/D1 相对新表示的 C0/C1 有一致实用增益，并且与可对齐历史 OD 的系统级比较无明显 AUROC/AUPRC、UFAR 或 Known FRR 代价；
- `MIXED_OR_INCONCLUSIVE`：数据集/Unknown application 方向冲突，或闭集表示、输入对齐、样本支持不足以区分原因；
- `NO_USEFUL_GAIN`：新支持分数未产生有用的开集改善，或代价明显。

无论哪一类，都不能凭一 seed 将“跨数据集稳定优于 Open-Detect”写成确认结论。若结果正向，下一阶段应先冻结一个方法，再用多 setting/seed 和未用于开发的 capture/group-disjoint 或外部数据验证；不得在本轮 Test 上反复修改公式。

## 8. 交付与停止条件

实施时新建隔离的 `stage38_three_view_des_open_set_transfer/`，至少交付协议与 source-hash 审计、Known-only 训练配置/历史与 checkpoint、逐样本 open-set score、逐方法/逐数据集/逐 Unknown 类 CSV、配对比较、完整报告、`RESULTS.md`、`manifest.json`、`completion_verification.json` 和可读队列进度。Stage36 原目录只读保留；Stage20、Stage12、Stage14B、Stage31–33、Stage15B 历史产物不覆盖。

先完成 VNAT 的冻结试验并报告其所有预注册分数；再完成 VPN/Tor Known-only 模型与检测。各阶段的失败、部分结果和资源中断都保留，不能以闭集提升代替开集结论。**当前仅落地计划，不自动启动 Stage38。**
