# Unknown Traffic / Unknown Attack Detection
# 当前实验进度与研究状态

> 本文件是 2026-09-21 的历史快照。最新状态请查看 [`CURRENT_PROGRESS.md`](CURRENT_PROGRESS.md)；本文件不再作为当前进度入口。

> 更新时间：2026-09-21（UTC）  
> 项目目录：/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project  
> 文档性质：当前状态落地与后续交接，不替代各阶段正式 RESULTS.md

## 0. 最新任务决策：切换为粗粒度 Service 主任务

用户已决定后续 ISCX-VPN / ISCXTor2016 不再以细粒度 Application 分类作为主任务。Stage 20 已冻结新的双数据集粗粒度协议：VPN 六类 `Chat/Email/File-Transfer/P2P/Streaming/VoIP`；TOR 七类，在上述六类基础上增加 `Browsing`，并将官方 `Audio/Video` 归并为 `Streaming`。

共选择 `22,136` 条流，冻结 `13` 个 Leave-One-Service-Out 协议；逐协议 flow-ID 与 exact-image 跨角色交集均为 `0`，Unknown fitting 使用量为 `0`。当前状态为 `PROTOCOL_FROZEN_NOT_TRAINED`，尚无新 TOR 粗粒度模型结果。历史 Fine Application 结果完整保留并降为诊断证据。详情见 `stage20_dual_coarse_service_protocol/RESULTS.md`。

## 1. 当前结论

项目已经完成从数据审计、Open-Detect 复现、VNAT 协议冻结、DES 支持解耦、Service-level benchmark、历史编码器恢复，到独立包级 E4 编码器实验的主要实验链条。

当前最后完成的阶段是 Stage 18 — PP-OpenNet-inspired Independent Packet Encoder，最终 Gate 为 E4_FAIL_NO_FUSION。

E4 在闭集分类上有一定价值，但没有在 Email/Streaming 两类 Unknown 设置上稳定改善开集 AUROC 和绝对 UFAR，因此没有启动 E3+E4 融合，也没有自动进入下一阶段。

当前没有正在运行的 Stage 18 训练、汇总或重放进程。

## 2. 研究主线演化

### 2.1 原始假设

原始计划试图通过高维 GMM 和多 prototype 建模 Known class 的真实多模态结构，并继续发展 Adaptive K 和 Unknown Discovery。

### 2.2 已完成的关键修订

1. Stage 2.5 证明 diagonal covariance misspecification 可以解释 apparent multimodality 的相当部分，因此不再把 GMM component 直接解释为真实语义子群。
2. Stage 3 证明 Known-validation NLL 改善不能稳定预测 Unknown Detection utility，固定 Multi-GMM 不再作为主方法。
3. Stage 10B–13A 将研究重点转为 representation–support decoupling：比较 learned prototype、Known-Train empirical centroid 和 local kNN support。
4. Stage 15A 确认 Open-Detect 与 DES 存在双向互补，但 Stage 15B 的 Hybrid 只有 HYBRID_PARTIAL，没有形成稳定最终 detector。
5. Stage 15R/15F/DQ 系列将主要瓶颈进一步定位到表示、窗口、流量统计、标签粒度和样本选择，而不是单一 GMM K 值。
6. Stage 18 对 PP-OpenNet 启发的包级时间—长度—方向表示进行了独立实验，但没有达到开集晋级标准。

## 3. 主要阶段状态

| 阶段 | 状态 | 当前可支持的结论 |
|---|---|---|
| Stage 2.5 | 完成 | diagonal covariance misspecification 是 apparent multimodality 的重要来源；不能把 component 直接称为真实子群 |
| Stage 3 | 完成 | Multi-GMM 对 Unknown utility 的收益不稳定，不作为主方法 |
| Stage 12 | 完成 | DES-v0 在 ISCX-VPN / ISCXTor2016 的 flow-disjoint 协议上有外部支持，但不能宣称 capture-disjoint 泛化 |
| Stage 13A-2 | 完成 | 固定 global-local support fusion 在 USTC development 上通过 GO |
| Stage 14A–14B | 完成 | VNAT 清理后 23,449 flows，15 个 Low/Medium/High frozen protocols，freeze hash 保持不变 |
| Stage 14C | 完成 | 15 个 VNAT Open-Detect encoder 全部完成；部分 Medium run 明显较弱 |
| Stage 14C.3–14C.6 | 完成 | 定位 batch/early stopping/prototype optimizer link 等训练问题，完成 cleaned pipeline 15-run 闭集验证 |
| Stage 14D | 完成 | VNAT 上 DES-v0/v1 有一定平均提升，但只得到 DECOUPLING_ONLY_PARTIALLY_CONFIRMED |
| Stage 15A | 完成 | Open-Detect 与 DES 存在真实双向互补，Gate=COMPLEMENTARITY_CONFIRMED |
| Stage 15B | 完成 | H1/H2/H3 仅 HYBRID_PARTIAL，未稳定解决 rsync/scp failure |
| Stage 15R | 完成 | 表示瓶颈和数据/协议瓶颈被确认，E1/E2/E3 全 15-run 未继续扩展 |
| Stage 15F-1A | 完成 | T16/T32 仅显示类别条件窗口收益，不是全局窗口瓶颈 |
| Stage 15F-1B | 完成 | 统计与 burst 特征有类别条件收益，不足以直接冻结新主表示 |
| DQ-3F | 完成 | Service 任务可训练，但标签为弱 capture-activity label；不能宣称严格 capture-disjoint 泛化 |
| DQ-4 | 完成 | TrafficFormer/过滤选择会产生明显 selection-induced distribution shift，收益对 seed 敏感 |
| Stage 16 | 部分完成 | 方法身份 Gate 阻断了把 DQ-3F corrected Open-Detect 误称为独立 OURS |
| Stage 16S | 完成 | Service-level DES-v1 有条件 ranking benefit，但 P95 UFAR 仍很高 |
| Stage 17 | 完成 | 历史 encoder recovery pilot 完成，但不足以支持四数据集正式推广 |
| Stage 18 | 完成 | E4 独立包级编码器未通过开集 Gate，融合未启动 |

## 4. 关键实验结果

### 4.1 VNAT Stage 14D

同一冻结 Native checkpoint 下：

| 方法 | AUROC |
|---|---:|
| Open-Detect Native | 0.837535 |
| DES-v0 | 0.851706 |
| DES-v1 | 0.853085 |

DES-v0 相对 Native 的提升只出现在 6/15 protocols；DES-v1 相对 DES-v0 的增益较小。因此结果只能支持部分 decoupling benefit，不能宣称全局稳定提升。

### 4.2 Service-level Stage 16S

| 方法 | AUROC |
|---|---:|
| Open-Detect Native | 0.623635 |
| DES-v0 | 0.652011 |
| DES-v1 | 0.686584 |
| H1 | 0.642470 |

DES-v1 相对 Native 的平均 ΔAUROC 为 +0.062949，但 P95 UFAR 仍为约 0.914446，因此排序改善没有转化为可靠拒识。

### 4.3 Stage 17 历史 encoder pilot

E3 相对 E0 的 AUROC 在 6/6 pilot run 提升：

- Email 平均 ΔAUROC：+0.083169
- Streaming 平均 ΔAUROC：+0.046018

但绝对 UFAR 仍高，且 E3 同时改变表示与 probe，不能把收益单独归因于图分支；因此未推广到四个数据集正式实验。

### 4.4 Stage 18 E4

正式协议为 Email/Streaming × seeds 2022–2024，共 6 个 run、7 个变体、42 个 checkpoint。

E4_FULL：

| Unknown 设置 | Accuracy | Macro-F1 | Weighted-F1 | AUROC | AUPRC | UFAR@Val-P95 |
|---|---:|---:|---:|---:|---:|---:|
| Email | 0.743100 | 0.749697 | 0.741966 | 0.627692 | 0.646520 | 0.882937 |
| Streaming | 0.809117 | 0.798280 | 0.809736 | 0.560439 | 0.924559 | 0.838900 |
| Overall | 0.776108 | 0.773988 | 0.775851 | 0.594065 | 0.785539 | 0.860918 |

相对 Open-Detect Native：

- ΔMacro-F1：+0.077618
- ΔAUROC：−0.056970
- AUROC 胜出：3/6
- ΔUFAR@95：−0.031723

因此闭集提升没有转化为稳定开集收益。

## 5. Stage 18 消融结论

严格单因素配对结果：

| 消融 | ΔMacro-F1 | ΔAUROC | ΔUFAR@95 |
|---|---:|---:|---:|
| Time+Length − Length | +0.055788 | −0.035383 | +0.049821 |
| Time+Length − Time | +0.176222 | −0.031306 | +0.075293 |
| Full − Time+Length（Direction） | +0.000022 | +0.035479 | −0.077190 |
| Full − No-MS（Multi-scale） | +0.011391 | +0.009839 | −0.052604 |
| Full − No-RNN（Recurrent） | +0.016056 | −0.007519 | +0.029344 |

当前解释：

- Length 是主要闭集判别信号；
- Time 继续改善闭集，但未改善 feature-distance AUROC；
- Direction 对闭集影响很小，却对开集分数有明显帮助；
- Multi-scale 有小幅正向收益；
- Recurrent 模块改善闭集，但开集收益不稳定；
- 当前最多 30 包的单分支不能解决 support overlap 和阈值迁移。

## 6. 数据、标签与协议边界

1. VNAT Stage 14B 冻结数据为 23,449 flows，freeze hash：  
   c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f
2. Stage 17/18 使用 3,065-flow cache，SHA256：  
   aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff
3. Service 标签主要是 WEAK_CAPTURE_LABEL，不能等价于逐流权威语义标签。
4. P2P 仅由单个 capture 提供，不能宣称 capture-disjoint 泛化。
5. Stage 14B 的 VNAT flow-random split 不是 capture-disjoint split。
6. Stage 16S/17/18 的结论主要是 flow-level development/pilot 证据，不应称为新的 untouched external validation。

## 7. 完整性状态

最近一次 Stage 18 验收：

- 6/6 formal runs 成功；
- 42/42 checkpoint 保存并记录 SHA256；
- 42/42 checkpoint 独立重放通过，最大张量差异 0.0；
- Unknown training/validation/normalization/support/threshold 使用量均为 0；
- Known Test checkpoint-selection 使用量为 0；
- 19 个受保护 Stage 16S/17 资产前后 SHA256 一致；
- 3/3 单元测试通过；
- 实验包 204 个文件完成逐文件哈希验证；
- 当前无残留 Stage 18 训练进程。

## 8. 已完成与未启动内容

### 已完成

- Open-Detect 基线与 Native detector 复现；
- DES-v0、DES-v1、H1 受控比较；
- VNAT 数据审计、清理、协议冻结和 15-run Open-Detect 训练；
- Service-level open-set benchmark；
- 历史 TrafficFormer/FIG/TAGCN encoder recovery pilot；
- 独立 PP-OpenNet-inspired E4 训练、消融、开集评估和重放验证；
- 每个主要阶段的结果、哈希、日志和实验包保存。

### 尚未启动或明确不应直接启动

- E3+E4 多分支融合；
- Adaptive K / K=20 或 K=30 扩展；
- 根据 Test 结果调整阈值、融合权重或 Unknown 类；
- DQ-5、DQ-6、DQ-7；
- Byte–Behavior 新方法；
- 四数据集上的 PP-OpenNet 等价复现；
- 未经重新预注册的长窗口 T64 或新 Hybrid detector。

## 9. 当前下一步状态

当前没有自动排定的下一阶段。若继续推进，必须新建独立预注册阶段，优先明确：

1. 是否获取并验证更长包窗口，而不是直接继续调 E4 网络；
2. 如何处理 Known support overlap 和 Validation-to-Test threshold shift；
3. 是否要选择真正独立于 Open-Detect 的方法实体；
4. 是否能够获得更可靠的 flow-level/service-level 标签。

在这些问题没有新协议和新证据前，不应把 E4 或 H1 宣称为最终主方法。

## 10. 主要证据入口

- [持续执行日志](EXECUTION_PROGRESS.md)
- [实验总索引](EXPERIMENT_RESULTS.md)
- [Stage 14D VNAT open-set](stage14d_vnat_frozen_open_set_evaluation/RESULTS.md)
- [Stage 15A 互补性诊断](stage15a_failure_regime_complementarity_diagnosis/RESULTS.md)
- [Stage 16S Service-level benchmark](stage16s_service_open_set_benchmark/RESULTS.md)
- [Stage 17 历史 encoder pilot](stage17_encoder_recovery_and_open_set_pilot/RESULTS.md)
- [Stage 18 完整报告](stage18_pp_opennet_inspired_e4/stage18_report.md)
- [Stage 18 结果摘要](stage18_pp_opennet_inspired_e4/RESULTS.md)
- [Stage 18 完成验证](stage18_pp_opennet_inspired_e4/completion_verification.json)
