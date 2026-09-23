# Unknown Traffic / Unknown Attack Detection 项目过程、实验计划修订与当前进度

> 更新时间：2026-09-21（UTC）  
> 项目目录：`Projects/unknown_traffic_project`  
> 文档性质：持续交接与当前执行依据  
> 当前状态：Stage 18 已完成；最终 Gate=E4_FAIL_NO_FUSION；未启动 E3+E4 融合或新的 detector 设计

## 1. 文档目的与证据边界

本文档把项目从 USTC-TFC2016 数据准备、TrafficFormer 适配、Known-space 分布诊断，到 Open-Detect、DES、VNAT 和 Known-only Hybrid 的全过程汇总到一个可持续维护的交接文件中。它重点回答：

1. 我们实际完成了什么；
2. 原始实验计划为什么以及如何被证据修订；
3. 当前哪些结论成立，哪些结论不能再写；
4. 当前正式方法、数据角色和冻结约束是什么；
5. 项目下一步处于什么状态。

事实和数值优先取自项目内现有 `RESULTS.md`、CSV/JSON、冻结协议和验证文件。聊天记录只用于定位任务，不作为实验结论来源。

原实验计划仍保留在：

- `未知流量_未知攻击检测_三核心问题_学生详细实验计划_公式优化版.md`
- `未知流量_未知攻击检测_三核心问题_学生详细实验计划_公式优化版_before_stage3_revision.md`

当前计划相对 Stage 3 修订前版本增加 405 行、删除 66 行，加入了 2026-09-12 evidence-based Master Plan Revision。但该计划文件的主状态仍停留在早期阶段；Stage 3–15B 的实际进展和最终边界以本文档及各阶段正式结果为准。

## 2. 当前一句话结论

项目已经证明：Known traffic representation 中存在 class-dependent local-support complexity，且 Open-Detect learned prototype、Known-Train empirical support 和 posterior uncertainty 具有真实互补性；但固定 Multi-GMM、DGSB、DES-v0、DES-v1 和现有 percentile-max Hybrid 都没有证明在所有数据集、protocol 和 hard class composition 上稳定优于 Open-Detect。

因此当前正式定位是：

- **Open-Detect Native 仍是正式基线；**
- **H1 = `max(A_OD, A_DES0)` 是唯一推荐继续研究的简单候选，但结论仅为 `HYBRID_PARTIAL`；**
- **DES-v1 不能升级为论文最终主方法；**
- **Adaptive K、semantic mode、Unknown Discovery 均未获证据支持，继续后移。**

## 3. 研究主线如何变化

### 3.1 原始计划

最初路线是：

```text
Known multimodality
→ Adaptive Multi-Prototype
→ Prototype-Specific Boundary
→ Fine-Grained Unknown Discovery
```

其核心假设是每个 Known class 内部包含多个真实且可解释的子模式，可先通过 BIC/GMM 找到 K，再将 component 作为 prototype。

### 3.2 第一次修订：从“真实多模态”转为“local-support sufficiency”

Stage 2.5 发现，在 896 维融合表示 `z_f` 上，PCA + full covariance 的 K=1 已显著强于 diagonal covariance 的 K=4/5；8/8 个代表性比较均显示 diagonal covariance misspecification 解释了 apparent multimodality 的相当部分。

同时，full covariance 下 K>1 的 held-out NLL 仍在 8/8 个代表性比较中改善。因此结论被修订为：

> 一个 global class-conditional support model 不一定足以描述 Known representation，但 residual component 不能直接解释为真实或语义子群。

Stage 2.6 和跨表示审计进一步表明 component 常与 packet count、total bytes、directional statistics 等简单流量统计关联。于是研究术语由 `semantic multimodality` 改为：

- `local-support heterogeneity`；
- `class-conditional support complexity`；
- `traffic-statistics-associated heterogeneity`。

### 3.3 第二次修订：从 density fit 转为 Unknown Detection utility

Stage 3 不再继续扩 K，也不实现 Adaptive K，而是在相同 Open-Detect representation 下严格比较 Single-Full 与固定 Multi-Full-K2。

结果为 `Gate D — MIXED`：

- A-1 Multi−Single UFAR：`+0.101176`，明显恶化；
- A-2：`+0.002689`，基本无改善；
- A-3：`−0.022899`，有所改善。

后续归因表明 A-1 恶化和 A-3 改善由少数 Known/Unknown class pair 主导，Known-validation NLL 改善不能预测 Unknown utility。固定 Multi-GMM 因此不再是主方法。

### 3.4 第三次修订：从 Multi-GMM 转为 representation–support decoupling

CipherSpectrum Stage 9 的 DGSB 在 operating point 上出现 tradeoff，没有稳定降低 UFAR。Stage 10B 的 score decomposition 显示：

- 最大恢复来自 learned prototype 替换为 empirical Known-Train centroid；
- covariance mismatch 是次要机制；
- posterior uncertainty 通常保留有用信号。

于是研究重点由“找更多 Gaussian component”改为：

> 将 representation training prototype 与 detection-time empirical support 解耦，检验数据锚定 support 是否更可靠。

Stage 11A 的 data-anchored prototype 训练失败，Stage 11B 的 decoupled readout 在 USTC 上仍然 scenario-dependent；但 Stage 12 在 ISCX-VPN 和 ISCXTor2016 上，DES-v0 相比 Native Open-Detect 的 mean ΔAUROC 分别为 `+0.023452` 和 `+0.069734`，为 decoupling 提供了外部支持。两套数据使用 `FLOW_DISJOINT_ONLY` fallback，因此不能声称 unseen-capture generalization。

### 3.5 第四次修订：Global + Local support

Stage 13A-2 在 USTC development 上使用固定：

```text
S_GL = 0.5 * Z_centroid + 0.5 * Z_kNN10
```

取得 `GO`：A1/A2/A3 AUROC 分别为 `0.994475/0.953415/0.991972`。但这仍是 USTC development 结果，不是独立外部确认。

VNAT Stage 14D 在同一 Native checkpoint 下比较：

- M0：Open-Detect Native；
- M1：DES-v0 empirical centroid；
- M2：DES-v1 fixed global + local。

总体 AUROC 为 `0.837535/0.851706/0.853085`。M1−M0 只有 `6/15` protocols 提升，bootstrap CI 跨零；M2−M1 仅 `+0.001378`，虽将 UFAR 平均降低 `0.030033`，但 AUROC 增益不稳定。最终为：

```text
DECOUPLING_ONLY_PARTIALLY_CONFIRMED
```

### 3.6 第五次修订：从固定 DES 转向互补性与 Known-only Hybrid

Stage 15A 对 VNAT、USTC、ISCX-VPN、ISCXTor 的冻结结果做机制诊断，确认 Open-Detect 与 DES 存在双向 rescue：

- VNAT P95：OD-only rescue `15,129`，DES-only rescue `11,585`；
- VNAT exact ranking：OD-only 正确 `6,203,271` 对，DES-only 正确 `7,741,572` 对；
- support overlap 与 DES gain：Spearman `rho=-0.343555`，探索性 `p=0.020854`；
- posterior uncertainty 在 21 个 DES-loss 单元中的 19 个保留补充信号。

Stage 15A Gate 为 `COMPLEMENTARITY_CONFIRMED`，但这只允许设计 Hybrid，不等于 Hybrid 已成功。

Stage 15B 仅测试三个预注册 Known-Val percentile-max 公式：

```text
H1 = max(A_OD, A_DES0)
H2 = max(A_OD, A_DES1)
H3 = max(A_proto, A_DES0, A_Vpost)
```

H1/H2/H3 在 VNAT 与 USTC pooled primary 上的 ΔAUROC 为 `+0.021876/+0.027272/+0.012776`，但 negative protocols 分别为 `12/11/14`，都没有充分减少 DES-v0 的 12 个 negative protocols；rsync/scp 失败也没有被修复。因此最终 Gate 为：

```text
HYBRID_PARTIAL
```

H1 因公式最简单、hard-class 行为优于 H2/H3，被保留为唯一后续候选；它不能被描述为已确认的最终方法。

## 4. 数据与表示基础

### 4.1 USTC-TFC2016 Stage 0/1

- Stage 0：构建 `489,101` 条双向流，20 类完整保留；FIG 共 `4,088,199` 节点；420/420 项抽样核验通过。
- TrafficFormer 短流适配：主策略允许至少 1 个真实包，保持官方 bigram、`[SEP]`、64 bytes/packet、最多 5 包和下游 PAD 语义；严格 min3 仅作为敏感性对照。
- Model A TrafficFormer：Test Accuracy `0.9892`，Macro-F1 `0.9920`。
- Model B FIG→TAGCN：Accuracy `0.6998±0.0236`，Macro-F1 `0.7329±0.0132`。
- Model C 融合：Accuracy `0.9893±0.0001`，Macro-F1 `0.9921±0.0001`。

### 4.2 表示术语

必须继续区分：

```text
z_t : TrafficFormer representation
z_g : FIG/TAGCN graph representation
z_f : train-standardized [z_t ; z_g]，896 维
mu_x: Open-Detect deterministic latent mean
```

Stage 2.5/2.6 使用的是 `z_f`，不是纯 `z_t`。Open-Detect 系列使用 `mu_x`。

## 5. 各阶段完成情况

| 阶段 | 状态 | 核心结论 |
|---|---|---|
| Stage 0–1 | 完成 | 20 类数据与 TrafficFormer/FIG/fusion closed-set backbone 验收通过 |
| Stage 2 | 完成 | diagonal GMM 显示 global simple model 描述不足，但不能证明真实 multimodality |
| Stage 2.5 | 完成，结论 B | covariance misspecification 解释相当部分 apparent multimodality |
| Stage 2.6 | 完成 | component 常由简单流统计解释，不应解释为 semantic modes |
| Stage 3 | 完成，Gate D | Fixed K2 utility 在 A1/A2/A3 不一致，Multi superiority 未建立 |
| Stage 4 | 完成，Diagnosis C | Known coverage 需要 local boundary，但 Unknown utility 未确认 |
| Stage 5/5.5 | 冻结后阻断 | CSTNET 协议有 Global Gate 退化和 sparse calibration 风险，`NOT_READY` |
| Stage 6–9 | 完成 | CipherSpectrum DGSB one-shot Test 为 operating-point tradeoff |
| Stage 10B | 完成 | learned-prototype mismatch 为主机制，covariance mismatch 次之 |
| Stage 11A | 完成，NO_GO | data-anchored prototype training 明显退化 |
| Stage 11B | 完成，NO_GO | decoupled support 在 USTC scenario-dependent |
| Stage 11C | 完成，WEAK_SIGNAL | Known-only support complexity selector 证据不足 |
| Stage 12 | 完成，EXTERNAL_CONFIRMED | DES-v0 在 ISCX-VPN/Tor 平均提升，但仅 flow-disjoint claim |
| Stage 13A-1/A-2 | 完成，GO | USTC 上 kNN local 与 centroid fusion 有开发价值 |
| Stage 14A/A.5 | 完成 | VNAT 数据质量和 group feasibility 已审计 |
| Stage 14B/B.5 | 完成并冻结 | 23,449 clean flows，15 protocols，Strict Unknown-Free PASS |
| Stage 14C 系列 | 完成 | Native baseline、特征诊断、训练失败定位与 cleaned 15-run 完成 |
| Stage 14D | 完成，PARTIAL | VNAT decoupling/global-local 平均正向但 protocol 不稳定 |
| Stage 15A | 完成，CONFIRMED | OD/DES 双向互补和 support-overlap failure regime 已确认 |
| Stage 15B | 完成，PARTIAL | H1 平均改善但未消除 negative protocols 或 rsync/scp 失败 |

## 6. VNAT 专项过程与结果

### 6.1 数据审计与协议冻结

- 原始 inventory：172 files，165 PCAP；163/165 完整可读。
- Stage 14A 构建：23,454 flows。
- 永久排除：1 条 duplicate-PCAP flow、4 条 rsync/sftp cross-app duplicate rows；两个 truncated PCAP 整体不进入 freeze。
- 最终 clean pool：`23,454 - 1 - 4 = 23,449`。
- 所有 10 个 application 均保留；VPN/non-VPN 是 metadata，不拆成语义类别。
- Low/Medium/High：Unknown class 数 `2/3/4`，seeds `2022–2026`，共 15 protocols。
- Known flow split：按 application 分层随机 `8:1:1`；不是 capture/group-disjoint。
- Freeze hash：`c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`。

主要有效性风险：

- application label 来自 filename/capture metadata，不是独立 per-flow ground truth；
- class size 从 44 到 13,563，极端不均衡；
- RDP Known Validation/Test 常各只有 4 条；
- VPN/non-VPN 分布严重不均衡；
- flow-random split 允许 capture/group 跨 split，因此不能声称 unseen-capture generalization。

### 6.2 闭集表示训练的失败与修复

初始 F0/F2 闭集结果偏低：F0 Macro-F1 `0.614834`，F2 `0.652969`；released Native Open-Detect 为 `0.830904`。

Stage 14C-3 证明数据和 F0 byte representation 对齐无误，主因是：

- batch 512 导致更新不足；
- patience 5 过早停止；
- seed-sensitive optimization；
- prototype reset 旧实现会断开 optimizer link。

Cleaned pipeline 固定为：

- F2 representation；
- batch 128；
- 100 epochs，无 early stopping；
- Adam LR 0.001；
- MultiStepLR `[50,80]`；
- epoch 51/81 prototype in-place reset，并清理 prototype optimizer state；
- checkpoint 只依据 Known Validation Accuracy/Macro-F1 harmonic mean。

15-run cleaned 结果：

| Scope | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Low | 0.876386 | 0.791160 | 0.867275 |
| Medium | 0.840436 | 0.802217 | 0.821985 |
| High | 0.888714 | 0.869709 | 0.885014 |
| Overall | 0.868512 | 0.821029 | 0.858092 |

相对 Native Open-Detect，mean Δ Accuracy/Macro-F1/Weighted-F1 为 `+0.000451/−0.009875/−0.001386`。因此 cleaned pipeline 达到可用且稳定，但没有证明闭集性能优于 Native。

Medium-2025 的低分主要来自 Known composition：rsync 与 scp 同时为 Known 时，185/191 rsync validation samples 被预测为 scp；当 scp 在 Medium-2026 被 held out 后，rsync 为 191/191。该差距主要不是随机 seed。

## 7. 当前开放集核心结果

### 7.1 Stage 14D VNAT

| Method | AUROC | AUPRC | UFAR | Known FRR |
|---|---:|---:|---:|---:|
| M0 Open-Detect | 0.837535 | 0.874068 | 0.589837 | 0.046577 |
| M1 DES-v0 | 0.851706 | 0.884966 | 0.599280 | 0.046029 |
| M2 DES-v1 | 0.853085 | 0.888365 | 0.569247 | 0.048026 |

解释：

- DES-v0 提高平均 AUROC，但只赢 6/15，并使平均 UFAR 略增；
- DES-v1 相比 DES-v0 主要改善 operating point/UFAR，AUROC 增益很小；
- hard unknown composition 尤其 rsync/scp/sftp 会改变结论；
- 不足以把 DES-v0 或 DES-v1 定义为最终主方法。

### 7.2 Stage 15A failure regimes

- DES-v0 的主要优势类：sftp `+0.202935`、netflix `+0.087576`、rdp `+0.047975`、zoiper `+0.039113`。
- 主要失败类：rsync `−0.161182`、scp `−0.045337`、youtube `−0.038862`。
- rsync/scp empirical support overlap 为 `0.981433/0.958865`。
- sftp 不是平均失败类；它具有 unusually large normalized prototype gap `2.460952`。
- prototype-centroid mismatch 的全局相关性未建立：`rho=0.200659, p=0.186278`。
- 最稳定的可解释信号是 support overlap，而非单独的 prototype mismatch。

### 7.3 Stage 15B Hybrid

| Dataset | H0 OD | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| VNAT | 0.837535 | 0.858118 | 0.859116 | 0.849366 |
| USTC | 0.939528 | 0.962698 | 0.972491 | 0.953250 |
| ISCX-VPN | 0.553024 | 0.561670 | unavailable | 0.521592 |
| ISCXTor | 0.524614 | 0.572999 | unavailable | 0.555676 |

H1 是当前唯一推荐候选，但必须附带以下限定：

- pooled primary ΔAUROC `+0.021876`，CI `[+0.004168,+0.041910]`；
- only `18/30` primary protocols 提升；
- negative protocols 仍为 12 个；
- rsync 平均/最差 ΔAUROC `−0.105989/−0.145269`；
- scp 仍为 `−0.029232/−0.047784`；
- 不能声称解决了 support-overlap failure。

## 8. 当前数据集角色

| 数据集 | 当前角色 | 边界 |
|---|---|---|
| USTC-TFC2016 | Development / mechanism benchmark | 已多次用于设计、诊断和 Gate，不是 untouched validation |
| VNAT | Development / external-dataset benchmark | 已用于 Stage 14–15 开发，不再是 untouched external dataset |
| ISCX-VPN | 已完成外部验证与 retrospective check | flow-disjoint-only，不支持 unseen-capture claim；之后也不再 untouched |
| ISCXTor2016 | 已完成外部验证与 retrospective check | flow-disjoint-only，不支持 unseen-capture claim；之后也不再 untouched |
| CipherSpectrum | 已完成 one-shot external Test | DGSB 为 tradeoff；labels 与 domain/endpoint shortcut 有风险 |
| CSTNET-TLS1.3 | Sparse-calibration stress candidate | Stage 5.5 `NOT_READY`，须另行冻结 protocol v2 后才可继续 |
| CIC-IDS-2017 | Supplementary diagnostic | endpoint/time/capture shortcut 风险，不作为主要 encrypted-traffic benchmark |

若要给 H1 或后续方法做确认性结论，必须使用新的、在公式和 Gate 冻结后才打开的 external protocol；不能继续把 USTC/VNAT 结果称为独立确认。

## 9. 当前冻结方法与不可更改项

### 9.1 正式 baseline

`M0 Open-Detect Native`：released learned-prototype Gaussian KL score，同一 protocol 下 encoder checkpoint 完全共享。

### 9.2 诊断方法

- `M1 DES-v0`：Known-Train empirical centroid minimum squared Euclidean distance。
- `M2 DES-v1`：固定 k=10、Known-Val median/MAD、`0.5 Z_global + 0.5 Z_local`。

### 9.3 当前唯一继续候选

`H1 = max(A_OD, A_DES0)`，其中两个 percentile 都只能由 Known Validation ECDF 得到。

### 9.4 继续实验必须保持

- Unknown 不参与 encoder training、support fitting、normalization、threshold 或权重选择；
- Test 不参与参数或公式选择；
- 阈值只用 Known Validation P95；
- 相同 protocol 的方法共享同一 encoder 和完全相同样本；
- 不搜索连续融合权重；
- 不临时调 k、threshold percentile 或 per-class threshold；
- 不基于开发集结果重新选择 Unknown classes、split 或 seed；
- 保存全部单 run、per-class 和 per-sample 证据。

## 10. 已证实、部分证实与未证实

### 已证实

- diagonal covariance misspecification 会制造 apparent multimodality；
- 一些类别存在 persistent local-support structure；
- component 经常与简单 traffic statistics 关联；
- OD 与 DES 存在真实双向 complementarity；
- empirical support overlap 是 DES failure 的主要可解释 regime；
- VNAT rsync/scp composition 会同时影响闭集和开放集结果；
- 原 F0/F2 闭集低性能主要来自训练机制，而不是数据错位。

### 部分证实

- representation–support decoupling 在平均意义上有收益，但 protocol 稳定性不足；
- global + local support 可改善部分 operating point，但 AUROC 增益较小；
- H1 可提高平均 AUROC/AUPRC/UFAR，但尚未控制 worst-case 与 hard-class failures。

### 未证实或已否定为通用结论

- 每类至少存在多个真实/语义 Gaussian 子群；
- BIC 选择的 K 是真实 mode 数；
- Fixed K2 必然优于 Single support；
- DGSB、DES-v0 或 DES-v1 稳定优于 Open-Detect；
- posterior uncertainty max-fusion 能自动修复 support overlap；
- Adaptive K 有实际 Unknown Detection 收益；
- Prototype-Specific Boundary 已获外部确认；
- Unknown Discovery 已具备启动条件。

## 11. 最终实验计划修订版

### 11.1 当前论文主线

建议论文叙事调整为：

```text
1. 诊断：learned prototype、empirical support 与 uncertainty 的几何失配
2. 机制：representation–support decoupling 与双向 error complementarity
3. 候选：Known-Val calibrated H1 percentile-max fusion
4. 边界：support-overlap hard regime 仍未解决
5. 确认：只有在新的 untouched protocol 上通过稳定性 Gate 后，H1 才能成为主方法
```

不再使用：

```text
真实多模态 → Adaptive K → 每个 component 是语义 prototype → Unknown clustering
```

作为当前已完成结论。

### 11.2 下一阶段若获授权

下一阶段不应继续搜索更多 max/weighted-sum 公式。应先预注册一个边界清晰的 Known-only continuation，目标是减少 H1 的 negative protocols 和 rsync/scp support-overlap failure；公式、输入信号、Gate 和新 external protocol 必须在打开任何 Unknown Test 前冻结。

最低 Gate 应继续包括：

- 新 external dataset mean AUROC/AUPRC 不低于 Open-Detect；
- paired ΔAUROC CI 支持正向结果；
- negative protocols 明显少于 H1/DES-v0；
- hard classes 不出现系统性大幅下降；
- UFAR 改善不能以显著 Known FRR 为代价；
- 全程不使用 Unknown calibration 或 learned test-time gating。

在该 Gate 通过前：

- Open-Detect 保持正式 baseline；
- H1 只称 development candidate；
- 不启动 RQ3 Unknown Discovery；
- 不把 VNAT 再称为 untouched external validation。

## 12. 当前文件与运行状态

### 12.1 Git

- 独立 Git 根目录：当前项目目录。
- Branch：`main`。
- HEAD：`ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc`。
- 当前无 staged files。
- 用户已有 tracked modifications：`EXPERIMENT_RESULTS.md`、`README.md`、`configs/dataset/ustc_tfc2016.yaml` 和当前详细实验计划；本文档不覆盖这些修改。
- GitHub remote：`sendsbrainsonly/unknown_traffic_project`。
- GitHub 推送尚未执行。
- 推荐发布包：代码、文档及小型核心结果约 `9.81 MiB`，预计 Git 压缩约 `3.13 MiB`；模型权重、cache、latent arrays、raw data 和大型逐样本表不得进入普通 Git。

### 12.2 存储清理

2026-09-19 存储清理完成，范围严格限于用户授权的 Python cache 和 75 个已完成 run 的 `latest_checkpoint.pt`：

- 删除前重新核验：Stage 14C.5 `45/45`、Stage 14C Native `15/15`、Stage 14C.6 `15/15` 个 latest/best 对应关系全部 PASS。
- 已删除 75 个 `latest_checkpoint.pt`，共 `24,270,861,345` bytes（`22.604 GiB`）。这些文件是完成 run 的末 epoch/恢复点；删除后不能从项目目录恢复，必要时只能重新训练生成。
- 已删除 93 个 Python cache 目录（89 个 `__pycache__`、4 个 `.pytest_cache`），共 `2,704,414` bytes；未发现额外 `.pyc/.pyo` 文件。
- 总删除量 `24,273,565,759` bytes，约 `22.606 GiB`；项目 apparent size 从约 `91 GiB` 降至约 `68 GiB`，文件系统可用空间由约 `388 GiB` 增至约 `410 GiB`。
- 独立复核：目标目录中 latest checkpoint 剩余 `0`；正式 best checkpoint 保留 Stage 14C.5/Native/Stage 14C.6=`45/15/15`，共 75 个、`24,271,014,094` bytes。
- 未删除任何 best checkpoint、结果表、manifest、frozen protocol、split、score、latent、smoke/interrupted/preliminary 证据或其他实验目录。
- 执行证据：`.tmux-task/cleanup_delete_preflight_20260919_002/`、`.tmux-task/cleanup_delete_execute_20260919_001/`、`.tmux-task/cleanup_delete_verify_20260919_001/`。

同日完成第二轮空间审计与用户授权的清理：

- 已删除另外 33 个历史已完成实验的 `latest/last` checkpoint，共 `10,699,129,825` bytes（`9.964 GiB`）。范围为 OpenDetect USTC audit 1 个、Stage 11A 16 个、Stage 14C-4 4 个、Stage 3 6 个、Stage 7 6 个。
- 删除前逐项验证文件路径、文件大小及对应 `best`；删除后独立复核 `targets_present=0`、`best_missing=0`。上述五个实验目录仍保留 865 个非 latest/last 文件，包括 best 权重、配置、指标、日志、manifest、逐样本/latent 数据及报告。
- 第二轮删除后项目 apparent size 为约 `57.443 GiB`，文件系统可用空间为约 `415.101 GiB`。
- 当前剩余 smoke 产物约 `4.137 GiB`、interrupted 产物约 `0.931 GiB`、Stage 14D preliminary 产物约 `0.291 GiB`、两份 superseded CICIDS shortcut backup 约 `0.056 GiB`。这些文件属于失败、smoke 或修订历史证据，删除前需再次明确授权。
- Stage 14C.5 F1/F3 的 30 个 best checkpoint 共 `9.042 GiB`；若不再做 F1/F3 inference，可作为高收益但会损失消融模型的候选。全部 F1/F2/F3 45 个 best 共 `13.563 GiB`。
- Stage 14C-3/14C-4 诊断 checkpoint 总计 `3.925 GiB`；可由报告和 metrics 支撑结论，但删除会失去复现诊断模型输出的能力。
- 当前正式 Stage 14C Native 15 个 best、Stage 14C.6 15 个 best、Stage 14B frozen protocol 和 Stage 15A/15B 核心证据仍列为必须保留。

### 12.3 进程

当前没有新的训练计划在执行。检测到三个历史 tmux payload shell 仍显示 running，但均为 0% CPU 的旧审计进程：

- `cipher_label_mismatch_audit_v1`
- `s14a-hdfschema-20260917`
- `s15a_manifest_keys_20260918_1604`

本文档任务未停止它们。它们应单独核对日志/exit status 后再决定是否关闭，不能当作正在运行的正式训练。

### 12.4 第三轮存储清理（A+B）

- 目标：按用户授权删除 smoke 模型权重、两份 superseded CICIDS shortcut backup、interrupted 中的大型权重/数组以及 Stage 14D preliminary artifacts；保留正式模型、正式结果、smoke/中断日志、配置和指标。
- 起始状态：项目 apparent size `57.443 GiB`、实际分配空间 `49.024 GiB`；预审计预计 A+B 去重后释放约 `5.288 GiB` 实际磁盘空间。
- 已删除 181 个文件：14 个 smoke 权重、10 个 interrupted 权重/数组、79 个 Stage 14D preliminary 文件、78 个 superseded backup 文件。
- 删除量：`5,752,696,342` logical bytes（约 `5.358 GiB`），实际分配空间 `5,678,365,184` bytes（约 `5.288 GiB`）。删除后项目 apparent size `52.086 GiB`、实际分配空间 `43.736 GiB`。
- 保留证据：338 个非权重 smoke 文件和 33 个非二进制 interrupted 文件仍在；正式 Stage 14C Native/Stage 14C.6/Stage 14C.5/旧 Stage 14C checkpoint 数分别为 `15/15/45/15`，Stage 3 和 Stage 7 正式 best 分别为 `3/3`。
- Git 检查：tracked deletion `0`。Stage 14D formal results/report 和当前 CICIDS component shortcut audit 均保留。
- 第一次独立核验使用了错误的 Stage 14C.6 文件名模式，误报 cleaned count 为 0；修正为实际 `*_best.pt` 命名后，第二次核验完整通过。
- 执行证据：`.tmux-task/cleanup_ab_preflight_20260919_001/`、`.tmux-task/cleanup_ab_delete_20260919_001/`、`.tmux-task/cleanup_ab_verify_20260919_002/`。
- 状态：`complete`。被删除的 smoke/interrupted 权重无法从项目目录恢复，只能重新运行对应实验生成；正式模型和正式结论未受影响。

## 13. 当前 handoff

- 2026-09-23 修订：本节原 Stage 15R handoff 已被后续阶段取代；历史记录保留在下方各阶段条目中。
- 最近完成的科学阶段：`Stage 21 — OURS-E3-T8 Coarse Service Closed-Set Benchmark`，状态 `CLOSED_SET_DIAGNOSTIC_COMPLETE`。
- 当前活动阶段：`Stage 22 — Pretrained TrafficFormer + E3 Closed-Set Comparison`，状态 `running`；最终指标尚未生成。
- 当前正式粗粒度协议：Stage 20 VPN-6 / TOR-7 closed split 与 13 个 LOSO protocols。
- 当前方法边界：Stage 21 E3 闭集表现中等，尚未优于既有高分 baseline；Stage 22 用于检验随机初始化/训练预算是否是主要原因。
- Storage cleanup：已完成三轮；前两轮删除 108 个 latest/last checkpoint 和全部 Python/pytest cache，第三轮删除 181 个 A+B 候选文件并实际释放约 `5.288 GiB`。当前项目 apparent size `52.086 GiB`、实际分配空间 `43.736 GiB`；正式 checkpoint 与正式结果均保留。
- GitHub publication：正在按源码、文档和轻量核心证据的白名单重新发布；大模型、数据和动态运行产物不进入普通 Git。
- 安全恢复点：不得干扰 Stage 22 三卡训练；不得把 running 状态写成完成；不得继续批量删除历史 checkpoint。

## 14. 关键证据入口

- 根级结果索引：`EXPERIMENT_RESULTS.md`
- TrafficFormer 短流修复：`docs/TRAFFICFORMER_SHORT_FLOW_ADAPTATION.md`
- Stage 2.5：`outputs/stage2_5_covariance_diagnosis/diagnosis_summary.md`
- Stage 3：`stage3_unknown_utility/README.md`
- CipherSpectrum one-shot：`stage9_cipherspectrum_final_test/RESULTS.md`
- Score decomposition：`stage10b_opendetect_score_decomposition/RESULTS.md`
- Dual external DES-v0：`stage12_dual_external_validation/RESULTS.md`
- USTC Global+Local：`stage13a2_global_local_fusion/RESULTS.md`
- VNAT protocol：`stage14b_vnat_protocol_freeze/RESULTS.md`
- VNAT lineage：`stage14b5_vnat_flow_retention_audit/RESULTS.md`
- Native Open-Detect VNAT：`stage14c_native_opendetect_vnat/RESULTS.md`
- F0/F2 failure diagnosis：`stage14c3_vnat_closed_set_failure_diagnosis/RESULTS.md`
- Cleaned 15-run：`stage14c6_cleaned_pipeline_full15_closed_set/RESULTS.md`
- VNAT open-set：`stage14d_vnat_frozen_open_set_evaluation/RESULTS.md`
- Complementarity：`stage15a_failure_regime_complementarity_diagnosis/RESULTS.md`
- Hybrid：`stage15b_known_only_hybrid_detector/RESULTS.md`

## 15. 四数据集当前结果边界（2026-09-19）

- 当前科学进度停在 `Stage 15B — Known-Only Hybrid Detector Development`；Gate 为 `HYBRID_PARTIAL`，尚未得到跨四数据集稳定胜出的最终 detector。Open-Detect Native 仍是正式 baseline，H1 仅是后续受控验证候选。
- Known-Test 闭集均值（15 个 frozen protocols）：USTC Accuracy/Macro-F1=`0.994324/0.945177`；VNAT Native=`0.858043/0.829606`；ISCX-VPN Native=`0.743350/0.731127`；ISCXTor2016 Native=`0.722965/0.689683`。四者协议和类别组成不同，绝对值不能视为严格同质 benchmark 排名。
- Open-set AUROC（Open-Detect/DES-v0/DES-v1）：USTC=`0.939528/0.970982/0.979954`；VNAT=`0.837535/0.851706/0.853085`；ISCX-VPN=`0.553024/0.576476/NA`；ISCXTor2016=`0.524614/0.594347/NA`。
- DES-v0 相对 Open-Detect 的 mean delta AUROC：USTC `+0.031454`（12/15 positive）、VNAT `+0.014171`（6/15）、ISCX-VPN `+0.023452`（14/15）、ISCXTor2016 `+0.069734`（14/15）。VNAT 提升不稳定；ISCX 两组相对提升稳定，但绝对 AUROC 仍低且 UFAR 分别约 `0.957/0.953`，不能表述为实用检测成功。
- Stage 15A 证明 OD/DES 信号具有双向互补，Gate=`COMPLEMENTARITY_CONFIRMED`；Stage 15B 的 H1/H2/H3 均未通过 worst-case Gate，最终为 `HYBRID_PARTIAL`。H1 被保留为最简单的后续候选，但不是已确认主方法。
- 证据范围：USTC/VNAT 已用于 development；ISCX-VPN/ISCXTor2016 在 Stage 12 提供过冻结外部验证，但进入 Stage 15 retrospective analysis 后不得再称为 untouched external datasets。DES-v1 尚无 ISCX 冻结结果。
- 状态：`complete`（只读汇总，无训练、无重拟合、无新 Test 打开）。

## 16. Stage 15R — Representation Bottleneck Audit（2026-09-19）

- 目标：在不使用 Unknown Test 进行训练、验证、特征选择或调参的前提下，审计 USTC/VNAT/ISCX-VPN/ISCXTor2016 的数据、特征和闭集表示瓶颈，并按 Gate 顺序执行 E0–E3 受控对照。
- 状态：`complete`。Stage 15R-0、20/20 E0–E3 pilots、Gate、报告、冻结资产哈希和 9/9 完成核验均已结束。
- 独立输出目录：`stage15r_representation_bottleneck_audit/`；已用实验留存工具创建 `RESULTS.md` 和 `manifest.json`。
- 冻结边界：Stage 12、14B/14C/14D、15A、15B 原有 checkpoint、score、manifest、protocol 和报告保持不变；Known Test 仅限方案冻结后闭集评价；Unknown Test 用量必须为 0。
- 已核对运行环境：`CONDA_PREFIX` 和 `sys.prefix` 均为固定 DGL Python 3.10 环境；所有 shell 载荷使用项目 tmux helper。
- 起始工作树已有大量历史用户实验文件与未跟踪产物，本阶段必须仅增量写入 Stage 15R 目录和本执行记录，不整理或覆盖其他变更。
- 已验证缓存：USTC `489101/489101`、ISCX-VPN `22142/22142`、ISCXTor `15096/15096` flow 严格闭合；三者均记录 `known_test_or_unknown_test_values_used_for_selection=false`。VNAT development cache 为 `23447/23449`，缺少的两条 ssh flow 在冻结协议中只作为 Known Test，特征值按边界未打开，并非预处理丢失。
- E0 Known-Validation Macro-F1：ISCX-VPN `0.769169`、ISCXTor `0.676900`、VNAT hard `0.739755`、VNAT normal `0.983318`、USTC A-2 `0.977545`；均复用冻结 checkpoint/预测，不重新训练且 Test 用量为 0。
- Pilot Gate：E1/E2/E3 mean ΔMacro-F1=`-0.072320/-0.166239/-0.128971`，均仅 `1/5` protocol 正向，worst Δ=`-0.148221/-0.384359/-0.302445`；clear/diagnostic Gate 全部失败，`full15_status=NOT_RUN_GATE_FAILED`。
- 工程失败留存：E1/E2 首次启动在 epoch 1 前因项目 TMP 路径触发 `AF_UNIX path too long`；原日志保存在 `failed_attempts/afunix_workers4/`。VPN/VNAT 临时用 workers=0 完成；随后实现项目本地 `/proc/self/fd/<fd>` 短路径，USTC/Tor 在不改科学配置的情况下恢复 workers=4。USTC 首次 workers=0 的慢速中断和 ISCXTor tshark partial 也分别完整留存。
- LightGBM `4.6.0` 安装在 Stage 15R 自己的 `vendor/` 中；共享 Conda 环境未修改。
- 最终结论：`DATA_OR_PROTOCOL_BOTTLENECK_IDENTIFIED`。同输入 CE-only 未稳定超过 Native，length+IAT 和简单统计也未形成稳定收益；主要证据指向细粒度 class overlap、Known composition、capture/group 协议与数据规模差异。
- 完整性：Known Test 用于选择=`0`、Unknown Test 使用=`0`、1,797 个受保护文件前后 SHA256 完全一致、20 个新/复用 checkpoint hash 全部验证、9/9 completion checks PASS。
- 下一步：Stage 15R 到此停止。若后续获授权，先做 capture/group-disjoint sensitivity 与 class-pair overlap audit；之后才考虑预训练 byte+temporal hierarchical encoder，不能从本阶段直接修改 DES、H1 或设计 Adaptive Hybrid。

## 17. Stage 15F-0 — Literature-Guided Feature Registry and Feasibility Audit（2026-09-19）

- 状态：`complete / STAGE15F0_PASS`；仅完成文献、特征、数据 lineage、窗口覆盖和加密可见性审计，Stage 15F-1 至 15F-5 均为 `NOT_RUN`。
- 文献：19/19 项一手来源登记完成，每项均记录 16 个输入/表示字段及 `PAPER_CONFIRMED / CODE_CONFIRMED / INFERRED / UNKNOWN` 证据标签；Deep Packet、MT-FlowFormer、FlowLens 未验证到官方代码，MIETT 官方仓库当前仍为 code-coming-soon。
- 特征注册表：冻结 B0--B4、T1--T2、S1--S2、M1--M3、P1 共 13 个配置；P1 在 provenance/license/preprocessing parity/pretraining overlap 审计前保持 blocked。
- Known Train/Val 窗口缓存：USTC `432537`、VNAT `23447`、ISCX-VPN `18121`、ISCXTor `11783`；Known Test/Unknown Test 特征值用量=`0/0`，所有协议 membership unmatched=`0`。
- N=8 协议均值截断率：USTC `31.47%`、VNAT `17.42%`、ISCX-VPN `11.82%`、ISCXTor `28.52%`；N=32 降至 `14.49%/7.59%/5.45%/16.90%`。类别异质性显著，但覆盖率不能替代性能实验或证明因果。
- 完整性：四数据集 cache parity PASS；2,928 条覆盖聚合；19 文献、13 特征、4 数据集、8/16/32/64 四个窗口全部通过 fail-closed verifier；3/3 单元测试通过；192 个初始保护文件和最终扩展范围 193 个文件前后均未变化。
- 失败证据：最初 ISCX-VPN Scapy 大 PCAP 慢路径被停止并完整保存在 `failed_attempts/iscx_vpn_scapy_slow/`；最终使用语义一致的 Stage15R-verified tshark 流并通过逐 flow parity。
- 下一阶段矩阵已写入 `stage15f1_preregistered_matrix.csv`，但没有启动训练；`single_feature_results.csv` 等未来正式结果文件按要求保持不存在。
- 关键入口：`stage15f_literature_guided_feature_benchmark/stage15f_report.md`、`feature_lineage_audit.md`、`packet_window_coverage.csv`、`completion_verification.json`。

## 18. Stage 15F-1A — Packet Window Sufficiency and Class-Conditional Benchmark（2026-09-19）

- 状态：`completed / E4_FAIL_NO_FUSION`。
- 目标：在五个 Stage 15R 预注册 Known-only pilot 上，以同一 T1 特征、同一两层 1D-CNN、同一训练预算和 seed 比较 T8/T16/T32；T8 先与 Stage 15R E2 做 parity，并审计 mask 对卷积与 pooling 的影响。
- 冻结边界：只读取 Known Train/Validation；Known Test/Unknown Test 特征值用量必须为 0；不修改 Stage 12--15R、Stage 15F-0、Open-Detect、DES 或 H1。
- 已核对 Stage 15R E2 配置：signed `log1p(frame_length)`、`log1p(IAT_us)`、valid mask；Known-Train median/IQR、clip `[-4,4]`；3→64→128 两层 Conv1d、masked mean pooling、CE；Adam 0.001、batch 128、100 epochs、MultiStepLR `[50,80]`、seed 2022、按 Known Validation Accuracy 选 checkpoint。
- 当前动作：审计现有缓存最大窗口、预测留存和可复用路径；随后创建独立实验包与 fail-closed 测试。

### Stage 15F-1A terminal record

- Status: `complete`
- Final Gate: `CLASS_CONDITIONAL_WINDOW_BENEFIT`
- Formal runs: `15/15` (T8/T16/T32 x five frozen Known-only pilots).
- T16 mean Delta Macro-F1 vs T8: `+0.015363`; positive protocols `5/5`; worst `+0.005718`.
- Known Test / Unknown Test usage: `0 / 0`; T64 and multiview training: `NOT_RUN`.
- Evidence: `stage15f1a_packet_window_benchmark/RESULTS.md` and `completion_verification.json`.

## 19. Stage 15F-1B — Statistical & Burst Feature Sufficiency Benchmark（2026-09-20）

- 状态：`complete / E4_FAIL_NO_FUSION`。
- 目标：在五个冻结 Known Train/Validation pilot 上，复核 Stage 15R E3 的 S12 parity，并以固定 LightGBM 配置比较 FULL_FLOW 的 S12/S-A/S-AB/S-ABC/S-ABCD/S-Burst 与 EARLY_16 的 S12/S-ABCD/S-Burst。
- 冻结边界：Known Test/Unknown Test 特征值用量必须为 0；不修改 protocol、Open-Detect、DES-v0/v1、H1 或既有 checkpoint/score；Pilot Gate 未通过时不运行 full15。
- 已验证起点：Stage 15F-1A Gate=`CLASS_CONDITIONAL_WINDOW_BENEFIT`；T16 mean ΔMacro-F1=`+0.015363`，T32 无稳定收益；Stage 15R E3 配置为 LightGBM 4.6.0、500 estimators、learning rate 0.05、num_leaves 31、Known-Val logloss early stopping 50、seed 2022。
- 输出目录：`stage15f_literature_guided_feature_benchmark/stage15f1b_statistical_burst_benchmark/`；实验包已初始化为 `stage15f1b-statistical-burst-benchmark-20260920-v1`。
- 当前动作：审计并冻结 feature definition、缺失值边界、FULL_FLOW/EARLY_16 lineage 与 protected hashes；随后先执行 S12 parity。
- 里程碑：S12 parity `5/5 PASS`，精确复用 Stage 15R E3 checkpoint，并复算得到一致的 Known-Validation 指标和 confusion matrix。
- 行为缓存：USTC `389982`、VNAT `21135`、ISCX-VPN `14521`、ISCXTor `9983` 个五协议 Known Train/Validation union flow；四者均与 Stage 15F-0 packet count/bytes/duration/direction 基础统计 parity `PASS`，Known Test/Unknown Test feature values=`0/0`。
- 已保存失败证据：USTC 首次方向编码误读；两次无科学语义变化的性能中断（重复 schema I/O、未合并分位数/重复 Early-16 计算）。修复后缓存完整通过，没有失败缓存进入正式训练。
- 当前动作：运行预注册的 40 个新增 LightGBM 配置；FULL_FLOW S12 的 5 个复用 run 与新增配置合计 45 个正式 Pilot。

### Stage 15F-1B terminal record

- Status: `complete`.
- Conclusions: `STATISTICAL_FEATURE_BENEFIT, BURST_COMPLEMENTARITY_CONFIRMED, CLASS_CONDITIONAL_BEHAVIOR_BENEFIT`.
- Formal runs: `45/45`; S12 parity: `5/5 PASS`; protected hash and completion verification: `PASS`.
- S-ABCD-S12 mean ΔMacro-F1: `+0.019473`; S-Burst-S-ABCD: `+0.006094`.
- Known Test / Unknown Test: `0 / 0`; full15, Stage15F-1C, open-set evaluation, DES/H1 changes, and Byte-Behavior model: `NOT_RUN`.
- Evidence: `stage15f_literature_guided_feature_benchmark/stage15f1b_statistical_burst_benchmark/RESULTS.md`.

## 20. Stage 15F-DQ — Cross-Paper Performance Gap Attribution（2026-09-20）

- 状态：`in_progress`。
- 当前授权范围：严格依次执行 DQ-0～DQ-3，并提交共享 31 个 ISCX-VPN PCAP 的中期报告；DQ-4～DQ-7、ISCXTor 扩展及跨模型公平训练保持 `NOT_RUN`。
- 数据边界：仅使用开发允许的 Known Train/Validation 与既有冻结 Known Validation 预测；Known Test/Unknown Test 不用于样本定义、标签映射、模型选择或诊断协议设计。
- 起始状态：独立输出目录尚不存在；Stage 12～15F-1B 历史资产保持原位；工作树已有大量历史未跟踪实验文件和 tracked 用户修改，本任务不整理或覆盖它们。
- 执行顺序：先完成跨项目 PCAP/flow 定义与哈希审计，再做 CATE/TrafficFormer 样本资格对账和冻结预测错误归因；只有 flow 对齐、标签映射和支持数 Gate 通过后，才允许执行 DQ-3 匹配的 Fine/Coarse Known-only 训练。
- 关键风险：TrafficFormer 的 service 标签可能由 capture/activity 而不是单一 application 决定；若固定 application→service 映射存在重大歧义，将按任务要求停止相关 F2/F3 分支而不静默改标签。
- 下一步：初始化实验留存 bundle，核对 31 个共享 PCAP、TFE-GNN CATE、TrafficFormer 过滤实现、Stage 12 split/provenance 和 Native Known Validation 预测的可连接键。

### Stage 15F-DQ DQ-0--DQ-3 interim terminal record

- 状态：`complete_with_blocked_branch`；DQ-0=`PASS`，DQ-1=`PASS_DESCRIPTIVE`，DQ-2=`PASS_DIAGNOSTIC`，DQ-3=`BLOCKED_MAPPING_AMBIGUITY`。
- DQ-0：Native/TFE-GNN/TrafficFormer 仅共同覆盖 31 个 VPN PCAP；Native 全量为 137 个 capture、22,142 个选定 session。共享 31 个 PCAP 上完整 flow manifest 23,645 行，其中 Native A 集 4,224 行；TrafficFormer parent-flow multiset 与既有 processing audit 31/31 精确一致。
- DQ-1：A/B/C/D flow 数=`4,224/2,720/1,984/1,804`；A 中 CATE MATCHED/UNMATCHED=`2,720/1,504`。UNMATCHED 只表示未与目标清单唯一匹配，不被解释为错误标签或背景流。
- DQ-2：共享31 Known Validation N=`335`，Accuracy/Macro-F1/Weighted-F1=`0.802985/0.770803/0.817823`；1--2 包流错误率 `15.74%`，>=3 包错误率 `29.00%`，当前子集不支持“短流单位样本更难”；Fine 错误中 22/66 映射后 Coarse 正确。
- DQ-3 Gate：Facebook 在当前 Known 集同时对应 Chat/VoIP，完整 Fine prediction→Service 映射不可识别；映射覆盖仅 `321/335=95.82%`。未引入多数映射或真实 capture 反推，未训练新的 F1/F3；Coarse 正式主任务仍为 `INSUFFICIENT_EVIDENCE`。
- 边界：DQ-4--DQ-7、ISCXTor、新 encoder/分类器训练均为 `NOT_RUN`；Known Test/Unknown Test 用量=`0/0`。
- 完整性：208 个冻结输入 SHA256 前后完全一致；独立完成性核验 PASS，21 个必需文件存在，31 shared PCAP、23,645 manifest rows、4,224 Native A rows 和 NOT_RUN 状态均复核通过。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq_performance_gap_attribution/RESULTS.md`、`performance_gap_attribution.md`、`completion_verification.json`。

## 21. Stage 15F-DQ-3R — Service Label Reconstruction（2026-09-20）

- 状态：`in_progress`。
- 目标：仅在共享 31 个 ISCX-VPN VPN PCAP 的冻结 Known Train/Validation 样本上，重建可审计的 capture/activity-level Service 标签来源，检查 Service 与 capture/group 支持和 Unknown-Free 语义，并预注册 DQ-3F；不训练模型。
- 冻结边界：Known Test/Unknown Test 特征值用量必须为 `0/0`；不修改 Stage 12--15F-DQ、DES、H1、历史 Fine/Open-Set 结果；不启动 Byte--Behavior。
- 独立输出：`stage15f_literature_guided_feature_benchmark/stage15f_dq3r_service_label_reconstruction/`；实验 ID=`stage15f-dq3r-service-label-reconstruction-20260920-v1`。
- 起始证据：DQ-0 已重建 31/31 TrafficFormer parent-flow multiset；共享31 Native A 集 4,224 flows。DQ-3 的完整 F2 因 Facebook application→Chat/VoIP 一对多而阻断，但这不自动否定使用 capture/activity Service 标签直接训练 F3。
- 标签原则：允许同一 application 在不同 capture 对应不同 Service；capture 文件名/项目映射只能作为 capture-level 弱标签，除非存在独立逐流定义，不得宣称为逐流权威真值。
- 工程记录：首次 bundle 初始化命令误用了 `--run-dir/--experiment-id`，exit 2，日志保留于 `.tmux-task/s15fdq3r_init_20260920/`；正确接口重试成功，无科学语义影响。

### Stage 15F-DQ-3R terminal record

- 状态：`complete / SERVICE_TASK_FEASIBLE_WITH_WEAK_LABELS`；本阶段只完成标签来源、支持度、group 可行性、Open-Set 语义与 DQ-3F 预注册，训练次数=`0`。
- 标签重建：31/31 个共享 VPN capture 的 Stage 12 与 TrafficFormer capture/activity 映射一致；冻结 Known Train/Validation 共 `3,065` flows（`2,730/335`），全部获得确定的 capture-derived Service 标签。
- 证据强度：`VERIFIED_DEFINITION/WEAK_CAPTURE_LABEL/AMBIGUOUS/UNAVAILABLE = 0/3065/0/0`。这些标签只说明 capture 的目标实验活动，不能宣称为逐 flow 权威真值，也未利用 Validation 结果反推标签。
- Service 支持：Chat=`132/17`、Email=`145/23`、File-Transfer=`332/43`、P2P=`804/100`、Streaming=`854/104`、VoIP=`463/48`（Train/Validation）。所有 Service 均有流级支持。
- group 边界：P2P 只有 1 个独立 capture，因此共享31子集不支持完整六类 capture-disjoint Train/Validation 泛化结论；当前 flow-level DQ-3F 只能作为弱 capture-label 诊断。
- Open-Set 语义：既有 Fine Unknown application 在 Service 空间与 Known Service 重叠，原 Low/Medium/High split 不可直接改名复用；未来若采用 Service Open-Set，必须独立重建并冻结 Known/Unknown Service。
- DQ-3F：F1/F3 使用完全相同的 3,065 flow IDs 和相同 Train/Validation membership；F2 只作可确定映射样本上的 `PARTIAL` 评价；配置已预注册，状态=`PREREGISTERED_NOT_RUN`。
- 完整性：17 个冻结输入 SHA256 前后一致；独立 verifier PASS；实验保存校验 `status=success artifacts=18 bundle_files=18`；Known Test/Unknown Test 特征值=`0/0`，Byte--Behavior 未启动，DES/H1 未修改，旧实验未覆盖。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq3r_service_label_reconstruction/RESULTS.md`、`dq3r_report.md`、`completion_verification.json`。

## 22. Stage 15F-DQ-3F — Fine vs Service Matched-Sample Training Benchmark（2026-09-20）

- 状态：`in_progress`。
- 目标：在 DQ-3R 冻结的 3,065 条共享31 ISCX-VPN Known Train/Validation flow 上，用相同输入、membership、模型主体和训练预算配对比较 Fine Application（F1）与直接 Service 监督（F3），并生成不训练新模型的 F2-partial。
- 冻结边界：Known Test/Unknown Test 特征值必须保持 `0/0`；不修改 Stage 12--15F-1B、DQ/DQ-3R、Open-Detect、DES-v0/v1、H1；DQ-4--DQ-7、TFE-GNN/TrafficFormer 新训练和 Byte--Behavior 保持 `NOT_RUN`。
- 已核对预注册配置：ResNet18、1 channel、latent 128、100 epochs、batch 128、Adam 0.001、Open-Detect lambda 0.005、MultiStepLR `[50,80]`、prototype reset zero-based `[50,80]`、Known-Validation Accuracy checkpoint selection、early stop patience 10/min epoch 82。
- 重复规则：DQ-3R 固定单次 seed=2022，但未登记 repeat count；按 DQ-3F 任务的缺省规则，在结果打开前冻结配对 seeds=`2022,2023,2024`，F1/F3 获得完全相同训练预算。
- 当前动作：先完成 3,065 条 flow 与 Stage 12 Known Train/Validation image array 的行序、标签、hash、Unknown-free 和输入元数据排除 Gate；不一致则停止训练。
- Preflight Gate：`PASS`。3,065 flow IDs 唯一，Train/Validation=`2730/335`，11 Fine classes、6 Services；Train/Validation flow-ID 与 exact-image overlap 均为 `0`；Stage12 NPZ target/排序/array digest 均通过；Known Test/Unknown Test feature values=`0/0`。
- GPU 运行：六个 paired jobs 使用项目 GPU selector 在单卡串行执行。前两次启动均在 epoch 1 前因当前 PyTorch/CUDA 不支持预初始化 `reset_peak_memory_stats` 而失败；失败目录与 tmux 日志完整保留，移除非科学必需的 reset 后正式队列已正常进入 epoch 循环，未修改模型、数据或训练配置。

### Stage 15F-DQ-3F terminal record

- 状态：`complete / COARSE_TRAINING_BENEFIT`；附加限制=`CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`。
- 正式运行：F1 Fine 与 F3 Service 各 3 seeds（2022/2023/2024），共 `6/6 SUCCESS`；相同 2,730 Train、335 Validation flow、32x32 byte image、Native architecture/loss/optimizer/budget/checkpoint rule，仅监督标签和输出类数不同。
- F1 Fine 三 seed mean±std：Accuracy=`0.784080±0.033625`，Macro-F1=`0.635049±0.046443`，Weighted-F1=`0.775730±0.040095`。
- F2-partial common subset：coverage=`322--324/335`，Accuracy=`0.913223±0.041867`，Macro-F1=`0.871102±0.063815`；它是预测依赖的部分评价，不能替代 F3。
- F3 Service full 335：Accuracy=`0.890547±0.041458`，Macro-F1=`0.856050±0.064200`，Weighted-F1=`0.888036±0.044738`；F3 Macro-F1 在 3/3 seeds 高于 F1，但该差值反映任务粒度而非同任务模型提升。
- 相同 F2-evaluable subset 上，F2 比 F3 mean Accuracy/Macro-F1 高 `+0.017528/+0.008203`；因此“直接 Service 训练优于预测映射”不成立，F3 的优势是覆盖全部335样本和独立监督目标。
- Fine error decomposition：三个 seed 合计 217 个 Fine 错误中，126 个（`58.06%`）在 deterministic Service 语义下兼容；标签粒度是显著因素，但不是全部原因。
- F3 mean Recall：Chat=`0.745098`、Email=`0.855072`、File-Transfer=`0.705426`、P2P=`0.950000`、Streaming=`0.900641`、VoIP=`0.979167`。P2P/Streaming 大类影响 Accuracy，但 Chat/Email 未崩溃；File-Transfer 与 Chat 最困难，seed2023 较弱。
- 边界：P2P 只有一个 capture，不能宣称 capture-disjoint 泛化；所有 Service targets 仍为 weak capture labels。DQ-4 可作为下一步过滤/目标流审计，但 DQ-7 在统一 Service 数据、flow、标签、group split 和指标冻结前尚不公平。
- 完整性：21 个受保护输入前后 SHA256 一致，6 checkpoint SHA256 验证，独立 verifier PASS，实验包 `status=success artifacts=92 bundle_files=92`；Known Test/Unknown Test feature values=`0/0`；DQ-4--DQ-7、TFE-GNN/TrafficFormer 新训练、Byte--Behavior=`NOT_RUN`，DES/H1 未修改。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq3f_fine_service_matched_benchmark/RESULTS.md`、`dq3f_report.md`、`completion_verification.json`。

## 23. Stage 15F-DQ-4 — Flow Selection and Filtering Attribution（2026-09-20）

- 状态：`in_progress`。
- 目标：只在 DQ-3R/DQ-3F 冻结的 3,065 条 Known Train/Validation flow 上，重建 A（全部）、B（CATE matched）、C_PARENT/C_FINAL（TrafficFormer eligible/final Service inclusion）和 D（B∩C），分离 evaluation-population effect 与 training-selection effect。
- 冻结边界：Known Test/Unknown Test 特征值用量保持 `0/0`；不修改 DQ-3F、TrafficFormer、TFE-GNN、DES/H1 或历史结果；不启动 DQ-5～DQ-7、Byte--Behavior、Open-Set。
- 已确认母集：Train/Validation=`2730/335`；旧 DQ-0 A=`4224` 只作来源审计，不能作为本阶段样本。
- 当前动作：对 3,065 条 flow 逐条连接既有 CATE/TrafficFormer lineage，核验 C_PARENT 与 final Service manifest 的关系，先执行六类 Train/Validation 支持 Gate；通过的 B/C/D 才按 DQ-3F F3 冻结配置训练。

### Stage 15F-DQ-4 terminal record

- 状态：`complete`；Gates=`EVALUATION_POPULATION_EFFECT, FILTERING_EFFECT, CLASS_CONDITIONAL_SELECTION_EFFECT`；长期限制=`WEAK_CAPTURE_LABEL, CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`。
- 冻结母集 A=`3,065`（Train/Val=`2,730/335`）；重新求交得到 B=`2,101`、C_PARENT=C_FINAL(Service)=`1,551`、D=`1,440`。旧 DQ-0 的 A/B/C/D=`4,224/2,720/1,984/1,804` 未被错误复用。
- TrafficFormer 实际过滤：17,769 parent rows 中 `<2048 captured bytes` 删除 16,243，成功 1,526；Service final manifest 1,526 与 success multiset 精确一致。Native session 与 TrafficFormer parent 并非一一对应，因此 C 是 parent-derived selection subset。
- 六类 Gate：B/C/D Train/Val 均保留 6 Services，但 B/D VoIP Val 仅 4 条、C 最小 Val=6，所有 per-Service 结论均标记低支持。
- M-A 复用 DQ-3F 3 个 checkpoint 并通过 array/prediction/hash parity；M-B/M-C/M-D 各 3 seeds，共 9/9 新 run SUCCESS，不删除或重跑弱 seed。
- common D_val Macro-F1 mean：M-A/M-B/M-C/M-D=`0.865662/0.673803/0.898537/0.675056`；相对 M-A delta=`−0.191859/+0.032875/−0.190606`。M-C 正向 `2/3` seeds，seed2024 为 `−0.024805`，因此 filtering benefit 是部分且 seed-sensitive，不是稳定全胜。
- M-B/M-D 在 seed2023 出现 optimization collapse（own Macro-F1=`0.200651/0.221315`）；这是冻结配置的真实结果，未调参。全 A stress Macro-F1：M-A/M-B/M-C/M-D=`0.856050/0.431240/0.622356/0.400851`，显示 selection-induced distribution shift 明显。
- Track E 不重训：同一 M-A 在 B/C/D_val 相对 A_val Macro-F1 delta=`−0.019417/+0.003782/+0.009612`，说明仅改变评价人口的影响较小且方向混合，不能解释 M-B/M-D 的大幅退化。
- 完整性：9 个新 checkpoint hashes、3 个 M-A frozen hashes、受保护输入前后 hashes、12 个 verifier checks 全部 PASS；Known Test/Unknown Test feature use=`0/0`；DQ-5～DQ-7、TrafficFormer/TFE-GNN 新训练、Byte--Behavior、Open-Set=`NOT_RUN`，DES/H1 未修改。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq4_flow_selection_filtering_attribution/RESULTS.md`、`dq4_performance_gap_attribution.md`、`completion_verification.json`。

## 24. Stage 16 — Coarse Service Classification and Open-Detect Benchmark（2026-09-20）

- 状态：`in_progress / method identity gate`。
- 目标：先核验 DQ-3F Service-6 模型与 Open-Detect 的方法身份，并对 A（2730/335）和 C（1367/184）协议做 flow membership、Service 标签及哈希 parity；只有两个方法实体确实独立时才允许新训练。
- 冻结边界：Known Test/Unknown Test 特征值用量保持 `0/0`；不修改 DQ-3F、DQ-4、Open-Detect、DES/H1 或历史 checkpoint/prediction；全部标签仍为 `WEAK_CAPTURE_LABEL`。
- 首轮证据：DQ-3F `train_dq3f.py` 直接从 `Projects/Open-Detect` 导入并实例化 `CorrectedOpenDetectNet`，同时复用 Open-Detect `run_epoch`、`reset_prototypes_in_place` 和 `weight_init`；其冻结配置明确写为 `training_protocol=corrected-paper`。
- 风险处理：不得把 corrected Open-Detect reproduction 重命名为独立“我们的方法”，也不得静默换入未参与 DQ-3F 的 Stage14C-6 F2 模型。若最终身份 Gate 不通过，将保存阻断审计和真实历史结果，但不启动伪两方法训练。

### Stage 16 terminal record

- 状态：`partial / BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`；数据 parity=`PASS`，方法身份 Gate=`FAIL_EXPECTED`。
- 直接证据：DQ-3F trainer 将 Open-Detect `code/` 与 `reproduction/` 加入 import path，实例化 `CorrectedOpenDetectNet(OpenDetectNet)`，并复用 Open-Detect `run_epoch`、in-place prototype reset、`weight_init`、VAE/prototype loss 与 Known-Val Accuracy checkpoint selection；这只能解释为 corrected Open-Detect reproduction，不能作为独立“我们的方法”。
- A 协议复核：Train/Val=`2730/335`；C 协议复核：`1367/184`；6 类标签与 flow IDs 唯一性通过，canonical membership/label hashes 已保存。
- 历史 DQ-3F corrected Open-Detect 三 seed：Accuracy=`0.890547±0.041458`、Macro-F1=`0.856050±0.064200`、Weighted-F1=`0.888036±0.044738`；这些真实结果保留，但不再误记为独立 OURS。
- 新 Open-Detect 训练、两方法 paired delta、paired errors 和两方法 C 对照均标记 `NOT_RUN_IDENTITY_GATE / NOT_IDENTIFIABLE`；训练次数=`0`，GPU用量=`0`，Known Test/Unknown Test=`0/0`，Open-Set/DES/H1/Byte–Behavior 均未启动。
- 完整性：19 个 DQ-3F/DQ-4/Open-Detect 受保护文件前后 SHA256 一致；独立 verifier `PASS`；正式证据位于 `stage16_coarse_service_opendetect_benchmark/`。
- 后续只能二选一重新预注册：准确命名的 corrected-vs-released Open-Detect 变体研究，或选择真正独立的 Stage14C-6 F2 own-method 在 A/C Service 数据上重建公平比较。

## 25. Stage 16S — Service-Level Open-Set Benchmark（2026-09-20）

- 状态：`in_progress / protocol and method reuse audit`。
- 目标：在六类 Service 的 Leave-One-Service-Out flow-level 协议上，为每个留出 Service 与 seed 训练一个严格 Unknown-Free 的 corrected Open-Detect 五类模型，并共享其 encoder/classifier 比较 OD-Native、DES-v0、DES-v1 和 H1。
- 方法冻结：DES-v0 使用 posterior mean 到 Known-Train empirical centroid 的最小平方欧氏距离；DES-v1 固定 k=10、Known-Val median/MAD、global/local=0.5/0.5；H1 固定为 OD 与 DES-v0 Known-Val ECDF percentile 的 max；阈值均为 Known-Val P95 (`method=higher`)。
- 协议设计：保留历史 Known Train 作为新 Known Train；历史 Known Validation 按 Service 和 exact-image group 确定性 1:1 拆成新 Known Validation/Test；被留出 Service 的全部3065-pool样本作为 Unknown Test。这样不把旧 Known Test/Unknown Test 移入训练，也不让 exact image 跨角色。
- 已核验开发池：Train/旧Val=`2730/335`；32 个重复表示行仅形成7个同角色同Service group，旧Train/Val exact-image交集=0，跨Service exact-image group=0；flow-ID交集=0。capture 在角色间重叠，故所有结论限定为 flow-level weak-label，P2P 单capture轮次额外标记 `PROTOCOL_LIMITED`。
- 当前动作：生成并冻结六轮 manifest、方法/资产 hashes 和 preflight Gate；通过后先跑单个 pilot，再执行其余预注册轮次，不按 Unknown 结果删轮次。
- Preflight：`PASS`。六轮 manifest 共18,390行；flow-ID与exact-image跨角色交集均为0；Unknown fitting=0；所有Known类训练支持均满足kNN-10。Unknown样本数 Chat/Email/File-Transfer/P2P/Streaming/VoIP=`149/168/375/904/958/511`。
- Pilot：`loso_chat/seed2022` SUCCESS，GPU0，runtime=`133.83s`，Known-Test closed Macro-F1=`0.956907`。OD/DES-v0/DES-v1/H1 AUROC=`0.678985/0.757718/0.820554/0.735948`；对应UFAR=`0.932886/0.798658/0.798658/0.865772`，说明排序改善未自动解决P95误接收。
- Pilot完整性：一个checkpoint由四方法共享；Unknown train/val/support/normalization/threshold样本均为0；Known-Test selection=0；467条评分样本唯一、四类分数有限、checkpoint/score hashes已保存。
- 正式网格：通过live GPU selector选择物理GPU `0,5,6,7`（均约45.5GiB free、0% util）并启动剩余冻结任务；session=`s16s_full_grid_20260920`，不使用繁忙GPU1--4。

### Stage 16S terminal record

- 状态：`complete / PASS_WITH_PRESERVED_RECOVERY`；正式 checkpoint=`18/18`，四方法指标=`72`行，配对比较=`54`行。
- 原始训练执行：17 个 run 完整 SUCCESS；`Streaming-2023` 完成100 epoch并保存 epoch-79 best checkpoint 后，在后处理 parity 断言处失败。失败目录、日志和 `FAILURE.json` 原样保留，未重训、未删除。
- 失败定位：checkpoint-selection 的 `PIL→ToTensor` 与手写 `float/255` 输入最大差 `5.96e-8`，使塌缩模型中 1/114 个近零 margin Validation 样本翻转；统一原验证预处理后 checkpoint Accuracy 精确恢复为 `0.552632`。18 个冻结 checkpoint 随后在独立 `canonical_evaluation/` 完成纯推理。
- 总体 OD/DES-v0/DES-v1/H1 AUROC=`0.623635/0.652011/0.686584/0.642470`；AUPRC=`0.776025/0.789242/0.806845/0.787600`。
- 相对 OD：DES-v0 ΔAUROC=`+0.028376`（11/18 wins）；DES-v1=`+0.062949`（13/18）；H1=`+0.018835`（8/18）。DES-v1 排名最佳，但 File-Transfer/Streaming Service mean 退化；H1 在 Streaming/VoIP 的 3/3 seeds 均退化。
- P95 工作点仍差：OD/DES-v0/DES-v1/H1 UFAR=`0.927674/0.927872/0.914446/0.915263`；Known FRR=`0.035700/0.039963/0.038198/0.042734`。因此只能支持 Service-conditional ranking benefit，不能宣称未知拒识已解决。
- 训练稳定性：seed2023 在六个 LOSO 协议均出现 optimization collapse，Known-Test Macro-F1=`0.255009--0.377309`；全部保留在正式均值中，没有选择性重跑。
- 边界：六类均完成三 seed，但全部为 `WEAK_CAPTURE_LABEL` flow-level 协议；P2P 只有一个 capture，`CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`，不能作跨capture或外部泛化主张。
- 完整性：Strict Unknown-Free、共享 encoder、Unknown/Test calibration=`0/0`、18 checkpoint hashes、score hashes、72/54 行数、16 个受保护历史资产前后 SHA256 均 PASS；3/3 单元测试通过。
- 证据：`stage16s_service_open_set_benchmark/stage16s_report.md`、`RESULTS.md`、`completion_verification.json`、`service_open_set_six_metrics.csv`、`paired_vs_opendetect.csv`。
## 27. Stage 17-0/17-1 — Historical Encoder Recovery and Open-Set Pilot（2026-09-20）

- 状态：`complete / STRICT_PILOT_COMPLETE_PROMOTION_NOT_YET_SUPPORTED`。
- 目标：恢复历史 TrafficFormer、FIG/TAGCN 与 concat fusion 的真实代码、特征和 checkpoint 血缘，并在 Stage 16S 冻结的 Email/Streaming × seeds 2022--2024 上进行最小 Service-LOSO 编码器比较。
- 已确认：Model A 与 Model B 的实现和 USTC-20 checkpoint 存在；Model C 仅找到可执行的 train-zscore + concat + linear-probe 代码，尚未发现历史成品 checkpoint。历史 checkpoint 不能直接作为当前五类 Known 的 LOSO 模型。
- 暴露边界：TrafficFormer 官方预训练权重可加载，但公开 README 未披露预训练语料组成，记为 `PRETRAINING_EXPOSURE_UNVERIFIED`；严格主轨不把该权重混入 Unknown-Free 比较。
- 输入恢复：Stage 12 provenance 保留原 PCAP 与 packet refs；Stage 17 将复用已经验证的 Stage 12 双向 session 状态机，从 31 个共享 PCAP 为冻结 3,065 flow 恢复 TrafficFormer 前5包字节和 FIG 前30包图，不重划分数据。
- 输出：`stage17_encoder_recovery_and_open_set_pilot/`，实验 ID=`stage17-historical-encoder-recovery-open-set-pilot-20260920-v1`。
- 输入与恢复核验：使用 Stage 12 双向 session 状态机从 31 个共享 PCAP 精确恢复冻结 3,065 flows；缓存 SHA256=`aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff`，TrafficFormer/FIG/fusion forward-backward、确定性、历史 A/B checkpoint 严格加载均 PASS。最初基于 tshark 的恢复因本机不支持 `frame.raw` 失败，失败日志已保留，随后使用相同 flow 构造规则的 Scapy 流式读取成功。
- 正式运行：Email/Streaming × seeds 2022--2024 共 6 个冻结协议全部完成；E0 复用 6 个 Stage 16S checkpoint，E1/E2/E3 新训练 checkpoint 共 18 个；Unknown 用于训练、验证、support、normalization、threshold 的数量均为 0，20,028 条逐样本预测已保存。
- Email mean：E0/E1/E2/E3 Known-Val Macro-F1=`0.698090/0.099522/0.485736/0.907480`，DES-v1 AUROC=`0.718684/0.605146/0.617569/0.801853`，UFAR=`0.948413/1.000000/0.912698/0.805556`。
- Streaming mean：E0/E1/E2/E3 Known-Val Macro-F1=`0.694651/0.119760/0.483253/0.795084`，DES-v1 AUROC=`0.674952/0.504437/0.565708/0.720970`，UFAR=`0.883090/0.990257/0.713640/0.783925`。
- 配对结论：E3 相对 E0 的 AUROC 在 6/6 run 提升；Email/Streaming mean ΔAUROC=`+0.083169/+0.046018`，mean ΔUFAR=`-0.142857/-0.099165`。但绝对 UFAR 仍高，E1 严格随机初始化发生明显塌缩/欠拟合，且 E3 同时改变表示与线性 probe，不能把收益单独归因于图分支。
- 决策：仅支持两个 development Services 上的历史表示恢复与 pilot 证据，四数据集正式升级=`NOT_YET_SUPPORTED`；未启动下一阶段。
- 完整性：6/6 formal runs、18/18 新 checkpoint、6/6 复用 checkpoint、Strict Unknown-Free、17 个受保护资产前后 hash、153 个 bundle 文件及 manifest hash 校验均 PASS。
- 证据：`stage17_encoder_recovery_and_open_set_pilot/stage17_report.md`、`RESULTS.md`、`completion_verification.json`、`encoder_pilot_results.csv`、`manifest.json`。

## 28. Stage 18 — PP-OpenNet-inspired Independent Packet Encoder（2026-09-20）

- 状态：`complete / E4_FAIL_NO_FUSION`。
- 目标：在 Stage 17 冻结的 Email/Streaming × seeds 2022--2024 Service-LOSO 协议上，实现不依赖 Open-Detect 训练循环、模型或原型机制的独立 E4 包级时间—长度编码器，并完成闭集、MSP/Energy/Feature-Distance 开集检测及预注册消融。
- 文献边界：PP-OpenNet 官方公开代码未找到；本阶段只复用论文明确披露的 metadata-only、多尺度、循环融合思想。网络宽度、归一化、优化器细节和拒识实现均标记为项目自主设计，结论为 `approximate / inspired independent reconstruction`，不是作者等价复现。
- 数据边界：复用 Stage 17 SHA256=`aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff` 的 3,065-flow 缓存；真实字段为方向、caplen 和相对首包时间，可计算 IAT。缓存最多30包，不能冒充论文的1000包/2秒 retroactive slicing。
- 冻结协议：`loso_email`、`loso_streaming`，seeds `2022/2023/2024`；Known Train/Validation 仅用于训练、归一化、checkpoint 选择、支持和阈值；Known Test/Unknown Test 仅最终评价。
- 独立目录：`stage18_pp_opennet_inspired_e4/`；实验 ID=`stage18-pp-opennet-inspired-e4-20260920-v1`。不覆盖 Stage 17 或 Stage 16S 资产。
- 执行结果：6/6 正式 run、42/42 变体和 checkpoint 完成。E4_FULL Known Test Macro-F1 为 Email `0.749697`、Streaming `0.798280`；主 feature-distance AUROC 为 `0.627692/0.560439`，UFAR@Val-P95 为 `0.882937/0.838900`。
- 配对结论：相对 E0，总体 ΔMacro-F1=`+0.077618`，但 ΔAUROC=`-0.056970`，仅 `3/6` AUROC 胜出；相对 E3，总体 ΔAUROC=`-0.167346`，`0/6` 胜出。闭集改善没有转化为稳定开集收益。
- 消融结论：严格的 Time+Length−Length 为 `+0.055788 Macro-F1/-0.035383 AUROC`，Time+Length−Time 为 `+0.176222 Macro-F1/-0.031306 AUROC`；Direction 为 `+0.035479 AUROC`；Multi-scale 为 `+0.011391 Macro-F1/+0.009839 AUROC`；RNN 为 `+0.016056 Macro-F1/-0.007519 AUROC`。当前 30 包单分支不足以降低绝对 UFAR。
- 完整性：42/42 checkpoint 独立重放最大绝对差 `0.0`，19 个受保护 Stage 16S/17 资产前后 SHA256 一致，Unknown 训练/验证/归一化/support/threshold 使用均为 `0`。
- 决策：预注册 Gate 四项失败，融合未启动。详细落地文档为 `CURRENT_PROGRESS_2026-09-21.md`；证据位于 `stage18_pp_opennet_inspired_e4/RESULTS.md`、`stage18_report.md`、`completion_verification.json`。

## 29. Stage 19 — RoNeTC Four-Dataset Fair Comparison（2026-09-21）

- 状态：`in_progress / input-lineage and comparability audit`。
- 目标：将 `Projects/RoNeTC` 中恢复的 RoNeTC 源码接入主项目 Stage 15R 冻结的四数据集五个代表协议，在相同 flow IDs、Known/Unknown 类、Train/Validation/Test membership、seed、训练预算和评价口径下，与既有 E0/Open-Detect 及项目方法结果做配对比较；允许的调参仅使用 Known Train/Validation。
- 冻结协议：ISCX-VPN `medium-2022`、ISCXTor2016 `medium-2022`、VNAT `medium-2025/2026`、USTC-TFC2016 `A-2`。历史协议、checkpoint、预测与结果不得覆盖。
- 已确认 RoNeTC 状态：作者历史源码已恢复并完成结构 smoke；现有真实数据运行仅为 CIC-IDS-2017 三类近似实验，尚无上述四数据集 checkpoint 或结果，不能直接宣称四数据集已完成。
- 输入边界：RoNeTC 需要每流前三视图（IP header、transport header、payload）与前 4/8/12/16 包；Stage 15R 缓存只保存时间、长度、方向，因此必须从已有 Stage-0 PKL/原始 PCAP 按冻结 flow ID 和原 sessionization 规则重建，不允许用 32x32 Native 图像伪装 RoNeTC 三视图。
- 当前动作：完成五协议逐样本 packet-view 可回溯性 Gate，冻结 Known-only 小规模调参方案与计算预算，随后实现独立 Stage 19 adapter、训练、测试、配对汇总及完整性核验。

### Stage 19 user-requested interruption record

- 状态：`paused_by_user / no owned GPU processes`。
- 输入 Gate：`PASS`。冻结协议 manifest 共 523,029 行，Unknown Train/Validation=`0/0`；ISCX-VPN、ISCXTor、VNAT、USTC 三视图缓存分别恢复 `22,142/15,096/23,449/438,893` 条 flow，缺失均为 0。
- VNAT 首次缓存因只处理 `tcp.stream/udp.stream` 而漏掉 12 条 frozen `other:*` IP flow；失败缓存完整保留，解析器改为与 Stage14A/14C 相同的 canonical endpoint+IP protocol identity 后，23,449/23,449 全部通过。
- RoNeTC dense adapter 单卡 parity 和 USTC 两卡 DataParallel smoke 均 PASS；后者保持全局 batch=128，单卡峰值显存约 17.46 GB，未启动 USTC 正式训练。
- 用户要求停止当前用户全部 GPU 进程。共向 7 个属主为 `birkenwald` 的 GPU PID 发送 SIGTERM：Stage19 四个训练，以及三个先前存在的 LLM replay/activation 服务；无需 SIGKILL。用户 `nlp` 的 GPU7 训练未触碰。
- Stage19 中止点：ISCXTor=`35/100`、ISCX-VPN=`24/100`、VNAT-2025=`2/100`、VNAT-2026=`2/100`；四个 session 均 exit 143。checkpoint、history、tmux log 与 `INTERRUPTED.json` 已保留，均不得作为正式结果。
- 完成边界：Known Test/Unknown Test evaluation 未运行，USTC formal 未启动，RoNeTC 与基线的正式四数据集比较尚未完成。
- 当前 GPU 核验：`CURRENT_USER_GPU_PROCESSES=0`；唯一剩余 GPU7 进程属主为 `nlp`。
- 恢复规则：仅在用户明确要求后，把四个 partial run 目录移入 `interrupted_attempts/` 永久保存，再从零启动正式 run；不得从 best-so-far checkpoint 继续并冒充同一预注册训练。

### Stage 19 maximum-three-GPU resume record

- 状态：`running / maximum 3 physical GPUs`。
- 用户授权最多三卡并行且无需持续监控；队列每15秒更新 `stage19_ronetc_four_dataset_comparison/progress.txt`。
- 四个中止 run 已完整移至 `interrupted_attempts/user_stop_20260921/`，正式 run 从零开始，不复用 best-so-far checkpoint。
- USTC 三卡 DataParallel smoke 在物理 GPU `0,4,5` PASS：全局 batch=128，dense adapter parity 最大误差=`0.0`，三卡峰值 allocated memory 约 `11.75/11.74/11.46 GB`。
- 正式调度：首轮最多三个单卡 run；第四个在一个 slot 释放后启动；前四个全部成功后，USTC 独占三卡启动。任何 run 失败均不自动重试，且会阻止 USTC 启动。
- 监控入口：`stage19_ronetc_four_dataset_comparison/progress.txt` 与 `queue_status.json`；正式 Test 指标仅在对应 run 完成100 epoch后生成。

### Stage 19 independent single-GPU scheduling correction

- 状态：`running / independent one-GPU-per-process / maximum 3 physical GPUs`。
- 用户澄清三卡是并发上限，不要求把一个任务的 batch 拆到多卡；调度目标改为每个独立训练进程占一张卡。当前仅剩 VNAT-2026 和 USTC A-2，因此同时使用两张卡，第三张保持空闲。
- 原四卡 takeover 已于 `2026-09-21T10:24:30Z` 通过监督进程 SIGTERM 干净停止；VNAT=`7/100`、USTC=`0/100`，部分产物分别保存在 `stage19_ronetc_four_dataset_comparison/interrupted_attempts/three_gpu_reconfigure_20260921/vnat/medium_seed2026/` 与 `ustc/A-2/`，均有 `INTERRUPTED.json` 且不得作为正式结果。
- USTC 单卡 full-batch smoke：batch=`128`，`data_parallel=false`，峰值 allocated memory=`34,887,070,720` bytes，finite forward/backward/optimizer=`PASS`，dense-adapter parity 最大误差=`0.0`。证据：`stage19_ronetc_four_dataset_comparison/smoke_runs/ustc/A-2/smoke_result.json`。
- 当前正式进程：VNAT-2026 使用物理 GPU0；USTC A-2 使用物理 GPU1，不传 `--data-parallel`。启动后两卡 live memory 均约 `36,953 MiB`；GPU2/3=`0 MiB`，GPU7 的其他用户任务未触碰。
- tmux session：`rontec_stage19_independent_single_gpu`；当前状态入口仍为 `stage19_ronetc_four_dataset_comparison/queue_status.json` 和 `progress.txt`；唯一 takeover 证据为 `takeover_independent_single_gpu_20260921.json`。
- 当前任务仍为 `in_progress`。安全下一步：只读监控两个 PID、epoch history、tmux exit status 与 GPU ownership；任一失败不得自动重试，必须先保存失败目录与日志。

### Stage 19 USTC three-GPU correction

- 状态：`running / USTC DataParallel on exactly 3 physical GPUs`。
- 用户随后明确要求剩余 USTC A-2 使用三卡并行。单卡 USTC 在 `7/100` 轮干净停止，完整保存在 `stage19_ronetc_four_dataset_comparison/interrupted_attempts/single_gpu_to_three_gpu_20260921/ustc/A-2/`，并标记为非正式结果；checkpoint 不含 optimizer/scheduler state，因此三卡 run 从 epoch 1 重启。
- VNAT-2026 已有 `PASS` 的 100/100 正式结果，监督脚本新增 `--reuse-completed-vnat`，只复用完成标记和原日志，不重新启动 VNAT。
- GPU 拓扑核验：物理 GPU0/1/2 位于同一 NUMA 侧，启动前均约 `45,469 MiB` free、0% util；GPU selector 以 `--allowed 0,1,2 --count 3 --min-free-gb 38 --max-utilization 10` 精确选择三卡。
- 首次三卡启动训练路径正常，但状态文件把完成的 VNAT 历史 GPU 计入活动卡数，显示4而实际为3；在 epoch 1 前停止并保存在 `interrupted_attempts/three_gpu_status_metadata_fix_20260921/`，修正后重新启动。
- 当前正式 session=`rontec_stage19_ustc_three_gpu_final`；训练 PID=`1828122`；物理 GPU0/1/2 live memory 约 `12,919/12,899/12,585 MiB`，均有活动负载；`queue_status.json` 断言 `active_physical_gpus=3`、`ustc_data_parallel=true`、`physical_gpu=0,1,2` 全部 PASS，启动日志无 OOM/Traceback/RuntimeError。
- 当前任务仍为 `in_progress`。安全下一步：只读等待首轮 `history.csv` 生成，用实际三卡 epoch 时间更新 ETA；不得把两个中断尝试纳入正式结果。

### Stage 19 user-requested stop and completed-protocol comparison

- 状态：`complete / ABORTED_BY_USER_WITH_FOUR_COMPLETED_PROTOCOLS`。
- 用户要求释放当前训练进程，然后仅比较已完成且协议对应的数据集结果。
- 终止前快照：Stage 19 总进度 `404/500`；ISCXTor/ISCX-VPN/VNAT-2025/VNAT-2026 均为 `100/100 SUCCESS`；USTC A-2 为 `4/100 RUNNING`。
- 目标会话=`rontec_stage19_ustc_three_gpu_final`，目标训练 PID=`1828122`，物理 GPU=`0,1,2`。GPU7 上其他用户训练 PID=`1363550` 不在本次范围内，不得触碰。
- USTC 已完成四轮 history 和 best checkpoint 将作为中断证据保留，不得当作正式 Test 结果。
- 执行结果：项目本地 `STOP_QUEUE` 已完成温和终止；USTC 最终停在 `5/100`，训练子进程返回码 `-15`，未运行 Known Test/Unknown Test，已写入 `runs/ustc/A-2/INTERRUPTED.json` 并排除于正式比较。
- 释放验证：Stage 19 目标进程组和训练脚本匹配数均为 0；物理 GPU0/1/2 均为 `0 MiB, 0%`。GPU7 的非目标训练 PID `1363550` 仍在运行，未受影响。
- 最终 live 快照中 GPU3 于 Stage 19 停止后被另一个 AcMAS 服务 PID `2836757` 使用，其 cwd 在另一项目；它与 GPU7 训练都不属于本任务，均未触碰。
- 中断证据：USTC epoch5 Validation Accuracy/Macro-F1=`0.977083/0.979953`，checkpoint/history SHA256=`d2332dfc...d6e82/f9bca161...4c1ee`；只作中断证据，不作正式 Test 结果。
- 对应协议比较：四个 `100/100 SUCCESS` 协议上，RoNeTC/H1 非加权平均 AUROC=`0.674164/0.689248`，AUPRC=`0.816834/0.821828`，UFAR=`0.886431/0.884747`，Known FRR=`0.055273/0.049348`；AUROC 单协议胜场为 `2/2`。
- 证据入口：`stage19_ronetc_four_dataset_comparison/completed_protocol_comparison.md`、`completed_protocol_comparison.json`、`RESULTS.md`和 `manifest.json`。
- 安全下一步：当前无活动 Stage 19 训练。只有用户明确要求完成四数据集比较时，才从 epoch 1 重新启动 USTC；不得将当前部分 checkpoint 当作正式结果。

## 30. Stage 20 — Dual Coarse Service Task Switch（2026-09-21）

- 状态：`in_progress / protocol freeze`。
- 用户决策：后续 ISCX-VPN 与 ISCXTor2016 主任务由细粒度 Application 分类切换为粗粒度 Service 分类；历史 Stage 12 细粒度协议、模型和结果只降为诊断证据，不删除、不覆盖，也不与新任务作同任务数值比较。
- 已核实历史依据：ISCX-VPN Service-6 的 DQ-3F corrected Open-Detect 三 seed 闭集 Accuracy=`0.890547±0.041458`、Macro-F1=`0.856050±0.064200`，说明标签粒度是显著因素；Stage 16S 已完成 VPN 六类 Service LOSO，但标签为 `WEAK_CAPTURE_LABEL`，P2P 仅一个 capture。
- 新协议目标：VPN 使用 `Chat/Email/File-Transfer/P2P/Streaming/VoIP` 六类；TOR 使用 `Browsing/Chat/Email/File-Transfer/P2P/Streaming/VoIP` 七类，其中官方 `Audio/Video` 归并为 `Streaming`。每个数据集先冻结 Service-level closed split，再为每个 Service 构建 LOSO Unknown-Free 协议。
- 数据复用边界：只复用 Stage 12 已生成的 32×32 流图像与逐流 provenance，按 capture 的官方 category 重标注；不重新解析原始 PCAP，不把旧 Fine Unknown application 直接改名为 Service Unknown。
- 输出目录：`stage20_dual_coarse_service_protocol/`；实验 ID=`stage20-dual-coarse-service-protocol-20260921-v1`。
- 安全下一步：完成 deterministic manifest、flow/image 跨角色隔离审计、独立 verifier 和实验包校验；本阶段不启动 GPU 训练。

### Stage 20 terminal record

- 状态：`complete / PROTOCOL_FROZEN_NOT_TRAINED`。
- 冻结数据：VPN `10,955` flows、6 Services；TOR `11,181` flows、7 Services；总计 `22,136` flows。TOR 官方 `Audio/Video` 归并为 `Streaming`。
- 协议：VPN 6 + TOR 7，共 `13` 个 LOSO；manifest `143,997` 行。每个 held-out Service 仅进入 `unknown_test`，Unknown training/validation/support/normalization/threshold 使用量均为 `0`。
- 完整性：逐协议 flow-ID 跨角色交集=`0`、exact-image 跨角色交集=`0`，独立 verifier `PASS`。闭集 Service split 也按 exact-image group 冻结。
- 失败保留：首轮发现一个 VPN 32×32 image hash 同时属于 File-Transfer 与 VoIP，使两个 LOSO 出现 Known/Unknown 图像交叉；Gate 正确失败。首轮清单、审计和日志已保留，成功版在抽样前整体排除该冲突图像组（3 rows），未放宽检查。
- 科学边界：Service 标签仍为 `WEAK_CAPTURE_LABEL`；并非统一 capture-disjoint，VPN P2P 只有一个 capture；粗粒度分数不能冒充细粒度同任务涨点。
- GPU/训练：本阶段 GPU 使用=`0`，新模型训练=`0`。输出：`stage20_dual_coarse_service_protocol/`。
- 安全下一步：若继续实验，在不改变 membership 的前提下训练选定模型，并同时报告闭集 Accuracy/Macro-F1 与开集 AUROC/AUPRC/UFAR/Known-FRR。

## 31. Stage 21 — OURS-E3-T8 Coarse Service Closed-Set Benchmark（2026-09-22）

- 状态：`in_progress / implementation and cache parity`。
- 用户要求：在 Stage 20 冻结的粗粒度任务上用 `unknown_traffic_project` 的方法重新训练和测试，不得把 corrected Open-Detect 基线冒充为 OURS。
- 方法身份：采用项目 Stage 17 已恢复并获得 6/6 pilot 正向信号的 E3 路线，即 TrafficFormer 分支 + FIG/TAGCN 分支、Known-Train per-branch z-score、768+128 concat、线性分类头；本阶段闭集主指标来自 E3，不使用 Open-Detect 网络、原型或 loss。
- 本地适配：Stage 20 输入由首 8 包冻结，因此 TrafficFormer 使用前 5 包，FIG 使用前 8 包，方法名固定为 `OURS-E3-T8`；这是透明的任务适配，不冒充 Stage 17 的 30 包 E3。
- 正式设计：ISCX-VPN Service-6 与 ISCXTor2016 Service-7，各 seeds `2022/2023/2024`，共 6 个闭集 run；checkpoint 只按 Known Validation Macro-F1 选择，Test 仅最终评价。
- 输出目录：`stage21_coarse_service_ours_e3_benchmark/`；历史 Stage 12/16/17/20 与 TrafficFormer 结果只读。
- 安全下一步：构建并审计 Stage20 exact membership 的 E3-T8 输入缓存，完成单 run smoke 和 GPU 容量核验后，最多三张自动选择的空闲 GPU 并行正式训练。

### Stage 21 terminal record

- 状态：`complete / CLOSED_SET_DIAGNOSTIC_COMPLETE`。
- 用户最终将范围调整为双卡、seeds `2022/2023`；VPN/TOR 共 `4/4` 正式 run 完成。最初 GPU4 上的 seed2024 收到 SIGTERM 后释放，日志和 `INTERRUPTED.md` 保留，正式聚合明确排除。
- 实际训练为一进程一卡：seed2022/2023 分别使用物理 GPU1/2，不使用 DataParallel；每 run 约 `367.18--378.62 s`，峰值 allocated memory=`8.802 GiB`。
- E3 主结果：VPN Accuracy/Macro-F1/Weighted-F1=`0.520128/0.524935/0.512889`；TOR=`0.622202/0.591124/0.617643`。VPN/TOR Macro-F1 seed 标准差=`0.047917/0.014553`。
- 多方位核验：同时保存 Accuracy、Balanced Accuracy、Macro/Weighted P/R/F1、MCC、Kappa、最差类别、逐类 P/R/F1/support、混淆矩阵、seed 稳定性、验证—测试差距和 E3−E1/E2 配对增益。E3 相对 E2 Macro-F1 mean delta 为 VPN `+0.114874`、TOR `+0.212758`，四个 paired run 全为正。
- 主要短板：VPN File-Transfer mean recall=`0.2925`；TOR Chat mean recall/F1=`0.1250/0.174656`。E1 严格随机初始化明显欠拟合，不能把 E3 的融合收益解读为 TrafficFormer 单分支已恢复到高性能。
- 完整性：Stage20 membership、缓存 finite/nonempty、4 个 SUCCESS、run artifact hashes、validation-only best epoch 和逐样本 Test 指标重放共 `98/98` checks PASS；Test 选模样本=`0`。
- 证据：`stage21_coarse_service_ours_e3_benchmark/RESULTS.md`、`multiview_summary.json`、`multiview_confusion_matrices.csv`、`completion_verification.json`、`manifest.json`。
- 安全下一步：若要提升分数，另开实验仅用 Known Train/Validation 调整 TrafficFormer 训练预算/预训练策略，并比较 FIG `T8/T16/T30`；不得覆盖本 Stage21 baseline。

## 32. 精简仓库重新发布（2026-09-23）

- 状态：`blocked_on_explicit_remote_disclosure_approval`。
- 目标：将代码、配置、测试、项目说明、各阶段 `RESULTS.md`、`manifest.json` 和小型核心结果重新整理并推送到 GitHub；模型权重、数据、embedding、缓存、运行日志和大体积中间产物继续保留在本地，不进入普通 Git。
- 起始状态：工作树实际占用约 `54 GiB`、逻辑大小约 `65 GiB`；已跟踪文件约 `0.81 MiB`；现有忽略规则之外仍有约 `3.25 GiB` 未跟踪文件，其中包含大于 GitHub 普通 Git 单文件限制的 Parquet/NPZ 产物。
- 科学状态：Stage 21 已完成并通过 `98/98` checks；Stage 22 正在 `stage22-pretrained-e3-grid-20260923` 会话中运行，当前不得记录为完成，也不得提交其动态 checkpoint、run 目录或日志。
- 既有脏工作树：`README.md`、`EXPERIMENT_RESULTS.md`、`configs/dataset/ustc_tfc2016.yaml` 和研究计划文档已有未提交修改；本次保留这些修改，不回滚。
- 当前动作：补齐忽略规则和公开仓库说明，生成稳定的当前状态文档，按白名单暂存可复现源码与轻量证据，然后审计 staged 文件大小、敏感信息、大文件和 Git 历史后再提交、推送。
- 发布白名单：源码、配置、测试、Markdown/Word 文档、稳定 JSON/manifest/hash，以及小于 2 MiB 的命名汇总 CSV；Stage 22 仅含方法、配置、预检和 `running` 状态，不含动态 run/checkpoint/log。
- 提交前核验：暂存 `1,382` 个文件、`15.24 MiB`，最大文件小于 `1 MiB`；禁止的模型/数组/数据扩展和 `runs/artifacts/checkpoints/cache` 路径均未暂存；敏感模式扫描 PASS；`410` 个新增 Python 文件语法编译 PASS；`532` 个 JSON 解析 PASS；核心 Markdown 相对链接 PASS；USTC YAML 解析 PASS。
- 已知格式边界：全量 `git diff --check` 会报告第三方 UER 源码、历史 CSV 的 CRLF/尾随空白及 Markdown hard-break；这些是原始历史内容，不在本次批量改写，新增的核心发布文档单独检查通过。
- 本地提交：`9422136 Publish reproducible project snapshot through Stage 21`；连同既有未推提交 `ffcb0df`，本地 `main` 当前领先 `origin/main` 两个提交。
- 推送状态：尚未推送。HTTPS push 实际停在 GitHub 用户名提示；复核历史上下文后确认此前使用 SSH。脱敏 SSH 探针返回 `successfully authenticated`，GitHub 的 exit 1 仅表示不提供 shell，不是认证失败。远端 `main` 仍为 `25cd43c`。
- 当前阻塞：向 `git@github.com:sendsbrainsonly/unknown_traffic_project.git` 推送时，外部披露安全审查要求用户明确确认该具体目的地；SSH 身份本身已经可用。
- 安全下一步：收到对 `sendsbrainsonly/unknown_traffic_project` 的 `main` 分支明确授权后，执行一次性 SSH 非强制 push，随后核对远端分支 SHA；不得使用 force push。
