# Stage 24 — 双分支流内结构增强与跨流上下文可行性实验

状态：预注册；本文件必须在任何 Stage 24 模型结果产生前固定。

## 科学问题与边界

24A 检验现有 E3 图分支不足是节点信息、包/burst 结构，还是单流短包限制。24B 只审计跨流上下文是否具备因果、无泄漏、分组不交叉的条件。两者不得混入同一公平表。

第一轮仅使用 Stage 20 冻结的 ISCX-VPN Service-6（10,955 flows）与 ISCXTor Service-7（11,181 flows）。现有 Stage 22 E1/E3 的 2022/2023 checkpoint、Known Train/Validation 表示与结果仅复用，不改写。Stage 20 Known Test 已暴露，Stage 24 的方法开发与选型只看 Known Validation；任何以后使用的 Stage 20 Test 只算开发性复核，不称独立验证。

## 24A 固定输入与方法

- 每个样本只取同一流前 8 个真实包；缺失包用 mask，不能用零填充作真实节点。任何新增行为特征只从这 8 包可获得的长度、相对时间、方向及方向连续 burst 推导。源 IP、capture ID、文件名、标签不能成为模型特征。
- Known-Train-only 拟合节点/统计量标准化。Known Validation 只用于 checkpoint 选择；Known Test/Unknown Test 不用于特征、权重、阈值或模型选择。
- 2022/2023 的 768-D E1 表示固定且在所有分支共用。每个候选行为分支输出 128-D；按 Stage 22 的 Known-Train 分支 z-score 拼成 896-D，再用相同的线性融合头（30 epoch、batch 256、Adam LR 1e-3）训练。
- 行为训练沿用 Stage 22：50 epoch、batch 64、Adam LR 1e-3、Known Validation Macro-F1 选最佳。S1/G1/G2 分别独立训练，不共享标签信息或 Test 信息。

| ID | 候选 | 主要可识别的比较 |
| --- | --- | --- |
| B0 | 冻结 TrafficFormer E1 | 字节基线 |
| B1 | 已完成 E3：7-D、8 包 FIG/TAGCN | 当前融合基线 |
| S1 | 早期 8 包统计和 burst 汇聚 → 128-D MLP | 行为信息，无图拓扑 |
| G1 | 原 FIG 节点再加入相邻包 log-IAT 和 signed log-长度变化，TAGCN 拓扑不变 | 显式包间特征 |
| G2 | 相同 G1 节点，包→burst→flow 层级编码，128-D | 层级关系的额外效用 |
| G2-shuffle | 节点及训练规则与 G2 相同，但以固定 seed 打乱 burst 顺序 | 时序结构负对照；不参与选最优模型 |

S1/G1/G2 参数数目、峰值显存和运行时间必须报告；参数量不匹配时不允许将差异单独归因于结构。G2-shuffle 不能借真实标签或 Validation 结果决定打乱方式；其随机种子预设为 run seed。

第一轮为两个数据集 × 2022/2023 两种 seed。先审计 B0/B1 的逐流错误、包数分层及类别混淆，再运行新候选。只在 Known Validation 上按下述固定 Gate 选择至多一个新增候选。如果有候选通过，两数据集再补 2024–2026：E1 按 Stage 22 同一固定配置训练一次/seed 并被候选共用，不据 Test 重新调参。

## 评价与 Gate

主指标是 Known Validation Macro-F1；同时保存 Accuracy、Weighted-F1、逐类 P/R/F1、混淆矩阵、E1-only/candidate-only 决策计数、按 1、2、3–8 包分层结果、参数/时长/显存。每个比较必须逐 seed 配对，报告均值、极值及一致性；只有 2 seed 的 pilot 不能声称稳定性。

正式闭集晋级条件：相对于 B0 **和** B1，在 VPN/Tor 各自的五 seed 平均 Macro-F1 至少 +0.01；每个数据集 ≥4/5 正增益；不得用少数类别的大幅提升掩盖重复出现的难类退化。仅少数类或较长流受益则结论为 `CLASS_CONDITIONAL_BENEFIT`，不得称全局瓶颈。

只有在闭集 Gate 通过且预训练来源/Unknown-Free 边界通过核查后，才可另行冻结并执行开集评价。开集分数和 P95 阈值只能由 Known Train/Validation 确定，Unknown 只在最终评估使用；AUROC、AUPRC、UFAR、Known FRR 均需报告。官方 TrafficFormer 预训练语料的 Unknown 暴露尚未证实，未解决前不得声称严格 Unknown-Free。Stage 24 不自动发明新 detector。

## 24B：跨流上下文，只读可行性 Gate

从冻结 provenance 审计每条流的 capture、起止时间、可用端点/会话、历史邻居数、邻居标签/capture 绑定；不使用模型结果选择上下文窗口。合格的未来跨流协议须满足：仅使用预测时已经出现的历史流；训练不能读取 Validation/Test/Unknown 特征构成全局图；Train/Val/Test 的 capture/group 不交叉且每个类别均有支持；无原始端点或 capture ID 作为模型特征；必须设置边/邻居打乱负对照。当前 VPN P2P 只有一个 capture，故完整六类 Stage 20 不能声称 capture-disjoint 跨流泛化。未过 Gate 则只输出 `INTERFLOW_NOT_FEASIBLE`，不训练跨流 GNN。

## 完整性与产物

运行前后验证 Stage 20 manifest、Stage 21 cache、Stage 22 输入及 E1 checkpoint SHA256。Stage 20–23 文件只读。所有新脚本、配置、原始输出、失败日志、`RESULTS.md`、`manifest.json`、完成校验和项目实验索引保留在本目录或项目根；禁止覆盖旧实验。训练任务经固定 Conda 环境和命名 tmux 运行，最多同时占用三张本用户可用物理 GPU。
