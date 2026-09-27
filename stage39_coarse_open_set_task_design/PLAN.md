# Stage 39 — 闭集性能保持的粗服务级开集检测

英文名：Known-Preserving Coarse-Service Open-Set Detection。

状态：任务定义优化与 metadata 可行性审计完成后即可用于协议冻结；本文件不是训练完成报告。当前只执行 `audit_semantics.py`，未训练、未生成正式 split、未重算或搜索 Unknown 检测分数。既有 Stage38 是开发依据，不能再称为 untouched external validation。

## 1. 目标及语义

用户目标：优先保持 Known 分类质量，发现粗粒度的新流量类型即可，不要求给 Unknown 再细分类。

Stage38 已经将所有 Unknown 输出为一个标签。因此本轮关键改变是 **Unknown 的定义由未见 application 改为未见 service/family**。已知服务中的新应用不再要求一定被拒识，例如训练含 Netflix 的 Streaming 服务后，新出现的 Vimeo 可被接受为 Streaming。未见服务统一输出 Unknown。

这是降低任务粒度、对齐训练目标的研究任务修订，不是对历史 application-level Unknown 结果的修正。闭集高分不能保证 Unknown 可分；新任务仍须实际验证拒识能力。

## 2. 已有证据

- VPN 当前 6,000 条 Unknown 中，Hangouts 2,000 条属于 Communication；Skype 918 条属于 Communication、1,082 条属于 File-Transfer；YouTube 2,000 条属于 Streaming。三种服务都出现于原 Known Train。
- Tor 的 4,000 条 Unknown 中，P2P 2,000 条是真正未见服务；Skype 的 1,768 条 Communication、232 条 File-Transfer 属于原已知服务。
- 按下面拟定的 VNAT 四个服务族定义，原 sftp/vimeo/zoiper 分别属于已有的文件传输/流媒体/通信服务。这是任务 taxonomy，不应宣称是所有流的权威活动标注。
- 即使只看 Tor 未见的 P2P，Stage38 D1 AUROC=0.913526、UFAR=0.7060；高 AUROC 尚未产生足够低的误接收率。
- 历史 Stage16S 已在另一批 3,065 条弱标签流上做过 Service LOSO，OD-Native 平均 AUROC=0.623635。Service 留出不保证任务一定容易，该结果也不能与本轮直接配对。

来源：Stage12 frozen split metadata、Stage14B manifest、Stage38 的 `stage38_unknown_application_results.csv`、Stage16S `RESULTS.md`。

## 3. 服务族与 Known 分类头

| 数据集 | 用于定义新颖性的服务族 | Known 分类标签 |
|---|---|---|
| ISCX-VPN | Communication / File-Transfer / P2P / Streaming | 保持 Stage38 的粗服务标签，去掉被留出的服务 |
| ISCXTor2016 | Browsing / Communication / File-Transfer / P2P / Streaming | 保持 Stage38 粗服务标签；P2P 原本没有 Known Train |
| VNAT | Streaming={netflix,youtube,vimeo}; File-Transfer={rsync,scp,sftp}; Communication={skype,zoiper}; Remote-Access={rdp,ssh} | 保留原六类中剩余的标签：streaming、rdp、rsync、scp、skype、ssh，不强制把全部 Known 合成四类 |

ISCX 按每个 capture 的 `official_category` 映射；Skype 等多活动 application 不使用单一 application→service 映射。Chat/Email/VoIP 合为 Communication，Audio/Video 合为 Streaming。标签保留 capture-derived weak-label 标记。

VNAT 的服务族是开发性约定；若后续 capture 元数据不支持某个约定，应在训练前记录修订，不能由验证/测试分数反推标签。

## 4. 保持原始样本角色的协议构造

先使用当前 Stage38 同一批 flow 与 metadata。只写入新的 Stage39 目录，旧协议及结果保持不变。

每次从**原 Known Train 出现的服务族**中完整留出一个族 `u`：

1. 新 Known 集合 = 原 Known 服务集合去掉 `u`。
2. 所有不在新 Known 服务集合的样本都属于 Unknown；同服务的所有应用、所有 VPN/non-VPN 或 Tor/non-Tor 域整体移出训练、验证、支持拟合和阈值校准。
3. 剩余原 Known flow 完全保留原 Train/Val/Test membership。原 Known Test 不转入 Train 或 Val。
4. 原 Unknown 中属于新 Known 服务的 flow 单独进入 `Known-service / unseen-application` 附加评估；**不加入 Train/Val**，也不丢弃。这里只评价粗服务与接受率，不要求把 sftp 分为已知的 rsync/scp。
5. 原 Unknown 中属于被留出服务的 flow 与所有被留出服务的原 Known flow 一起作为 Unknown 评估；其角色必须在新模型训练前确定。
6. 校验五部分数量守恒、flow ID 不重叠、Train/Val 无 Unknown。正式协议冻结时记录 role manifest/hash、checkpoint/source config 初始哈希。

这样所有 flow 都有角色，既不会为了提高分数删除困难样本，也不会把之前的 Test 重新用作训练。

### 第一批：VPN/Tor，单 seed 2022

- VPN：4 个固定服务留出 setting 全部报告，按服务名排序执行，不根据 Test 挑选最容易的 Unknown。
- Tor：原本完全未见的 P2P 始终为 Unknown。先保留 `Known={Browsing,Communication,File-Transfer,Streaming}; Unknown={P2P}` 作为可复用原 Stage38 checkpoint 的开发性对照；Skype 另作已知服务新应用评估。
- Tor 的 4 个新训练 setting 再分别留出 Browsing/Communication/File-Transfer/Streaming，因此每个 setting 包含 P2P 加被留出的服务，共 2 个 Unknown 服务族。不能把这 4 个 setting 称作五折单类 LOSO。
- 第二批 VNAT：4 个服务族逐个留出，训练 seed 2022；底层样本源仍是 Stage14B medium_seed2025，区分 protocol seed 和 training seed。

共 12 个拟定新训练 setting、1 个无需新训练的 Tor 对照；不是多 seed 实验。`service_holdout_feasibility.csv` 列出每个 setting 的样本数和最小类支持数，**不等于已生成或冻结了正式 split**。

## 5. 保住闭集能力的方法约束

沿用当前成功的 Stage32/38 配方：TrafficFormer、FIG/TAGCN、YaTC 三路编码；每路 64 维 adapter；特征层固定等权融合；融合分类头。训练配置从 Stage38 `stage38b_training_config_lock.json` 及 VNAT 源配置逐项读取并在启动前锁定。

新的服务留出 setting 中，三个分支均需从相同、允许的初始化重新训练，排除该服务的监督样本。训练过被留出服务的 Stage32/33/38 checkpoint 不能只删掉输出类别后就宣称 Unknown-Free。原始官方通用预训练的来源与可能数据接触范围另行如实记录；不得称其从未接触过任意相似服务。

用 Known Val 选择 checkpoint，然后冻结 encoder、adapter、classifier。检测器只读取冻结特征，不回传梯度到分类头。这样检测阶段不会改变原闭集 logits，但**拒识仍会拒掉一部分 Known**，必须额外报告这一代价。

相较旧 Stage38，Known 类集合不同，旧 Macro-F1 不能直接作为同任务提升量。对同一新模型分别报告拒识前分类质量、拒识后全部 Known 上的质量，拒掉的 Known 计作错误；不要只在被接受样本上报告高准确率。

## 6. 开集模块与阈值

先只优化任务语义，主方法保留 DES-v1：Known Train 类中心与类内 kNN-10；全局/局部 0.5/0.5；median/MAD 来自 Known Validation。保留 MSP 为固定参照，所有 setting 使用同样两种分数，不能逐数据集按 Test 选择更好的公式。

主 operating point 保持 Known Validation P95、`method="higher"`、`score > threshold` 拒识。同步预登记 P90/P99 作为运行条件曲线，三者全部报告，不根据 Unknown 结果选阈值。P90 通常会牺牲更多 Known 接受率；不能把这种变化写成检测器本身改善。

在某些 setting Known Val 类支持很小（例如 VNAT rdp 原本只有 4 条）时，报告阈值尾部估计不足。首轮不叠加类别阈值、learned gating 或新融合分数；若后续需要改边界，应独立登记并只用 Known Train/Val 开发。

## 7. 指标与建议验收目标

以下是本轮开发目标，不是保证达到的性能，也不是已经取得的新结果：

- Known：Accuracy / Macro-F1 / Weighted-F1 / per-class Recall，建议目标 Macro-F1 ≥0.93；附加拒识后全体 Known Macro-F1 和覆盖率。
- Unknown：AUROC / AUPRC（同时报告 prevalence）、UFAR、Unknown Recall=1−UFAR、每个 Unknown 服务的 UFAR。
- 主要运行目标：P95 下 UFAR ≤0.30，Known Test FRR ≤0.07。P95 只保证验证上的目标接受率，不自动保证 Test。
- AUROC ≥0.90 是排序目标；AUROC 高但 UFAR 不合格，仍不得宣称已满足实用拒识。
- 每个留出 setting 单独报告；再报告按 setting 宏平均、最差 setting 和通过目标的数量。全部 setting 完成后才给数据集级结论；不剔除难 setting。
- 不使用跨任务闭集 F1 差、同一数据集挑选最优 score 或最易 Unknown 证明方法提高。

粗服务未见测试与同服务新应用接受测试构成两个不同问题：前者需要拒识，后者允许接受。二者均需报告，才能区分“发现新类型”和“把所有新样本一律拒绝”。另外把原 Known Test 与上述已知服务新应用样本合并，报告部署口径的 Known FRR、粗服务 Macro-F1、AUROC/AUPRC；不能只把困难旧 Unknown 改名后从主要部署指标中移除。原六类等较细 Known 分类指标仍在具有合法对应标签的原 Known Test 上独立报告。

## 8. 数据边界、证据与执行顺序

ISCX 标签来源于 capture，无法宣称权威逐流服务真值。VPN P2P 仅一个 capture，1,006 条全是 VPN，须标注 `PROTOCOL_LIMITED` 并报告每个 domain 的 Unknown UFAR，防止把域差异当成服务新颖性。Tor 原池也以 NonTor 为主；必须保留 domain 支持数，不能仅因数据集名为 Tor 就称其为纯 Tor 实验。原 flow split 可能共享 capture；保留 group/capture 审计信息，不能据此宣称跨采集泛化。

当前阶段已产出 metadata overlap、每服务支持数和候选 setting 数量表。后续顺序：冻结 role derivation 与训练配置 → Known-only 训练 → Known-Val 闭集检查及冻结 → Known-only 支持拟合和阈值校准 → 单次 Test 评估 → 逐样本独立重放与结果保全。

资源上沿用最多三张可用 GPU、named tmux 和自动队列；每个 setting 的三个分支是同一方法的组成部分。只在训练前重新查询空闲物理卡，不占用其他任务进程。训练/evaluation 的正式启动是后续执行阶段，当前 metadata 审计没有使用 GPU。

输出需保留 `protocol.json`、逐 flow 角色表及哈希、三路 checkpoint、训练历史、逐流 logits/scores、Known/Unknown 按服务结果、已知新应用结果、`RESULTS.md`、`manifest.json` 和重放验证。历史 Stage38 结果不覆盖，本阶段所有数据都按已暴露的 development data 描述。
