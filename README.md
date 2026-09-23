# Unknown Traffic / Unknown Attack Detection

面向未知流量 / 未知攻击检测（Open-Set）的研究项目。当前 evidence-based 主线为 **Persistent Local-Support Modeling**：

> **任务粒度决策（2026-09-21）**：后续 ISCX-VPN / ISCXTor2016 主评测切换为粗粒度 **Service classification + Leave-One-Service-Out unknown detection**。VPN 使用 6 类，TOR 使用 7 类；冻结协议见 [`stage20_dual_coarse_service_protocol/RESULTS.md`](stage20_dual_coarse_service_protocol/RESULTS.md)。历史细粒度 Application 实验全部保留，但仅作为诊断证据，不能与新任务作同标签空间的直接涨点比较。

- **RQ1**：一个 global class-conditional support model 是否足以描述 Known traffic representation？
- **RQ2（后移）**：只有 Multi local support 先证明 Unknown Detection utility 后，才检验 local-support-specific boundary。
- **RQ3（后移）**：只有 detection 有效且 rejected Unknown buffer 质量合理后，才做 Fine-Grained Unknown Class Discovery。

严格 Unknown-Free：Unknown 类只允许进入 Final Test，不得进入 encoder/scaler/PCA/Gaussian/GMM/K/threshold 的训练、拟合、选择或校准。Stage 3 已按冻结顺序 `A-1 → A-2 → A-3` 完成；Stage 4 已完成 Known-only boundary diagnosis，Stage 5 已冻结 DGSB 和 CSTNET 外部协议。本任务未进入 Adaptive K、Unknown Discovery 或 CSTNET 正式 Unknown 实验。

## 当前状态（2026-09-23 UTC）

- **当前主任务**：ISCX-VPN 6 类与 ISCXTor2016 7 类粗粒度 Service classification；Stage 20 的 13 个 LOSO open-set 协议已经冻结。
- **最近完成阶段**：Stage 22 官方预训练 TrafficFormer/E3 闭集诊断，4/4 正式 run 与 99 项完成性检查均通过。VPN 的 E1/E3 Macro-F1 为 `0.8606±0.0040/0.8595±0.0070`，TOR 为 `0.8213±0.0064/0.8222±0.0076`。
- **当前运行阶段**：无。Stage 22 已结束；任何 open-set 使用都需要先单独冻结外部预训练暴露协议。
- **已停止阶段**：Stage 19 RoNeTC 四数据集完整训练应用户要求停止；只有四个已完成协议可用于受限比较，USTC 部分训练不计入正式结果。
- **科学结论**：现有证据不支持 fixed Multi-GMM 或当前 E3 配置稳定优于基础方法。E3 相对同 run TrafficFormer 在 VPN 略低、TOR 略高，两个数据集都只有 `1/2` seeds 的 Macro-F1 为正；粗粒度任务改善与细粒度结果不属于同一标签空间，不能作为同任务“涨点”。

详细状态见 [`CURRENT_PROGRESS.md`](CURRENT_PROGRESS.md)，完整实验索引见 [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)，持续执行记录见 [`EXECUTION_PROGRESS.md`](EXECUTION_PROGRESS.md)。

## 公开仓库边界

普通 Git 只保存源码、配置、测试、说明文档、各阶段 `RESULTS.md`、`manifest.json`、完成性核验和经过筛选的小型汇总指标。原始/派生数据、checkpoint、embedding、缓存、运行日志和大体积中间产物保留在本地；具体规则见 [`PUBLIC_REPOSITORY_CONTENTS.md`](PUBLIC_REPOSITORY_CONTENTS.md)。

当前没有选定可称为“最终模型”的权重，因此本次发布不上传 checkpoint。Stage 22 没有证明 E3 稳定优于同 run TrafficFormer；后续只有在另行选定正式发布模型后，才通过 Git LFS 或 GitHub Release 发布权重及 SHA256。

## 历史基础阶段（截至 Stage 8B，保留用于追溯）

Completed:

- USTC closed-set representation
- Stage 2 Gaussian audit
- Stage 2.5 covariance diagnosis
- Stage 2.6 component shortcut audit
- CICIDS2017 cross-dataset diagnosis
- Open-Detect USTC encoder audit
- Cross-representation correspondence audit
- Three-dataset suitability audit
- CipherSpectrum labeling audit
- Open-Detect protocol audit and Stage 3 protocol freeze
- Stage 3 Unknown Detection utility and post-hoc failure diagnosis
- Stage 4 Known-only local boundary calibration diagnosis
- Stage 5 DGSB method freeze and CSTNET strict Unknown-free protocol freeze
- Stage 5.5 CSTNET external-validation final preflight audit
- Stage 6 CipherSpectrum canonical-120k Strict Unknown-Free protocol freeze
- Stage 7 CipherSpectrum Low/Medium/High Known-only encoder training
- Stage 8A CipherSpectrum Known-only representation, density and boundary freeze
- Stage 8B CipherSpectrum DGSB-v2 Known-only rule freeze

## External Dataset Roles（frozen before Stage 6 training）

- **USTC-TFC2016 = Development / Mechanism benchmark**：用于方法发现与诊断，不再作为独立外部验证。
- **CipherSpectrum = Primary Independent External benchmark**：使用排除 MIX/getpocket 的 canonical 120k，并严格按 `split_group_id` 划分。
- **CSTNET-TLS1.3 = Sparse-Calibration Stress benchmark**：保留 many-class / sparse-validation 压力定位。
- **CIC-IDS-2017 = Supplementary NIDS benchmark**：受 day/PCAP 与 endpoint shortcut 混杂影响，不作为主要 encrypted-traffic benchmark。

| 阶段 | 状态 | 关键结果 |
|---|---|---|
| Stage 0 数据处理 | ✅ 完成 | 489,101 条双向流（20 类全量）；FIG 全量生成（4,088,199 节点）；交叉校验全绿；≥50 流人工校验 420/420 项通过 |
| Stage 1 Closed-Set Backbone | ✅ 完成 | Model A（TrafficFormer 微调）test Acc 0.9892 / Macro-F1 0.9920；Model B（FIG→TAGCN，seed 0/1/2）0.6998±0.0236 / 0.7329±0.0132；Model C 融合 0.9893±0.0001 / 0.9921±0.0001，§7.5 验收 PASS |
| Stage 2 Gaussian audit | ✅ 完成 | 原始 diagonal GMM 结果显示 simple global model 不充分，但不能单独证明真实/语义 multimodality |
| Stage 2.5 covariance diagnosis | ✅ 完成 | 896D fusion `z_f`；full K=1 显著变强，covariance misspecification 解释相当部分 apparent multimodality；full K>1 仍有 residual held-out NLL 改善 |
| Stage 2.6 component shortcut audit | ✅ 完成 | `z_f -> PCA64 -> full GMM`；四个代表类的 component 多与 packet/byte/directional statistics 高度相关 |
| CICIDS2017 cross-dataset diagnosis | ✅ 完成 | class-dependent：DDoS residual、PortScan endpoint/time associated、GoldenEye unstable、Hulk degenerate；不支持 all-classes multimodality |
| Open-Detect USTC encoder audit | ✅ 完成 | `mu_x -> train-only StandardScaler -> PCA64 -> full GMM`；四类 K=2/3 held-out NLL 改善，Gate A CROSS-REPRESENTATION PERSISTENCE |
| Cross-representation correspondence audit | ✅ 完成 | component correspondence 因类而异，Gate C SIMPLE-STATISTICS PERSISTENCE；不得解释为 semantic modes |
| Three-dataset suitability audit | ✅ 完成 | CSTNET Tier A primary external；CipherSpectrum secondary；CICDDoS2019 supplementary only |
| CipherSpectrum labeling audit | ✅ 完成 | primary 40 classes / 160,200 flows；必须按 `split_group_id` grouped splitting；41st local class 不进入主实验 |
| Open-Detect protocol audit / Stage 3 freeze | ✅ 完成 | A-1/A-2/A-3 class lists 已恢复并冻结；作者五个样本 folds 不可恢复；本地采用官方 class scenarios + 已冻结 80/10/10 flow split |
| Stage 3 Unknown Detection utility | ✅ 完成 | 三个 primary setting 的 strict Unknown-Free audit 全部 PASS；Multi−Single UFAR 为 A-1 `+0.101176`、A-2 `+0.002689`、A-3 `−0.022899`，方向不一致，最终 **Gate D — MIXED**；未证明稳定实际效用 |
| Stage 3 Failure Diagnosis | ✅ 完成 | A-1 净恶化全部来自 Htbot 多吸收 86 个 Tinba；A-3 净改善全部来自 Virut 少吸收 236 个 Neris；K2 Known-validation NLL 改善不能预测 Unknown utility，最终 **Diagnosis A — DENSITY-UTILITY MISMATCH** |
| Stage 4 Local Boundary Diagnosis | ✅ 完成 | Global 对 Known component coverage 严重不均衡；Component-P05 将平均 component gap 从 Validation/Test 的 `0.1692/0.1719` 降至 `0.0720/0.0704`，最终 **Diagnosis C — LOCAL BOUNDARY NECESSARY**；该结论仅限 Known coverage，USTC Unknown 结果为 post-hoc，候选规则尚未独立验证 |
| Stage 5 DGSB / CSTNET Protocol Freeze | ✅ READY | DGSB 严格 AND Gate 已冻结，USTC Known Validation acceptance 为 A-1/A-2/A-3 `0.9500/0.9500/0.9495`，Known Test FRR `0.0490/0.0505/0.0501`；CSTNET 原始/eligible 类 `120/119`，Low/Medium/High folds 与 group-aware split 哈希全部 PASS；未训练 CSTNET、未访问 Unknown score |
| Stage 5.5 CSTNET Preflight | ⛔ NOT_READY | 26/26 frozen/known-asset hash PASS；Global-exclusive rejection A-1/A-2 仅 `0.0395%–0.0739%`，A-3 Validation/Test 均 `0%`；Global/Local class mismatch `0%`；600/600 实际 32×32 输入中 domain/SNI/token `0%`，IP 全掩码但 port 全可见；class/component calibration 样本压力建议单独冻结 protocol v2；未训练或访问 CSTNET Unknown score |
| Stage 6 CipherSpectrum Protocol Freeze | ✅ READY | canonical 120k=`40×3,000`，MIX/getpocket=0；Low/Medium/High seeds `42/43/44` 的 class-held-out 与 70/15/15 group-aware splits 已冻结；Unknown-Free 与16个协议哈希独立验证 PASS；未训练或访问 CipherSpectrum Unknown score |
| Stage 7 CipherSpectrum Known-only Training | ✅ 完成 | Low/Medium/High 三个 Open-Detect encoder 独立从零训练；Known Validation Acc `0.7692/0.7153/0.7467`、Macro-F1 `0.7691/0.7045/0.7297`；全部 smoke/formal bundle 与 checkpoint 哈希 PASS；Known Test/Unknown 打开或使用量均为 0，未执行 Unknown inference |
| Stage 8A CipherSpectrum Known-only Freeze | ✅ READY_FOR_FINAL_TEST | 三档 deterministic 128D `mu_x`、train-only StandardScaler/PCA64/full K1/K2、Known-Validation-only Native/Global/Class-P05/Component-P05 边界与正式 evaluation config 已冻结；102/102 个 K2 收敛；Low/Medium/High component fallback `2/76,2/68,4/60`；全部 bundle hash PASS；Known Test/Unknown 打开量均为 0 |
| Stage 8B CipherSpectrum DGSB-v2 Freeze | ✅ READY_FOR_ONE_SHOT_FINAL_TEST | 固定 Global P02.5 AND predicted-class Local P02.5；Low/Medium/High 均为 `DUAL_GATE_ACTIVE`，Known Validation AND acceptance `0.9682/0.9689/0.9689`；原五种 Stage 8A 方法逐字段未修改；paired bootstrap=1000 与 Final Test 指标已预注册；Known Test/Unknown 打开量均为 0 |

当前正式结论：

> Single global Gaussian is not always descriptively sufficient, but residual components are frequently associated with simple traffic statistics and cannot be interpreted as semantic modes. Under the frozen strict Unknown-Free test, fixed K2 multi-local support did not demonstrate consistent practical Unknown Detection utility across A-1/A-2/A-3.

Stage 3 final decision:

> **Gate D — MIXED.** A-1 significantly worsened, A-2 was statistically inconclusive for UFAR, and A-3 significantly improved; multi-local-support utility is therefore not established.

Stage 3 结果与可复核证据见 `stage3_unknown_utility/outputs/summary/` 和各 setting 的独立实验包；post-hoc failure diagnosis 见 `stage3_failure_diagnosis/outputs/`。当前未执行 Adaptive K、Unknown Discovery 或外部数据集正式 Unknown 实验。

Stage 4 固定 K2，只比较 Global、Class-P05、Component-P05 boundary。A-3/Virut 的 Global Known-Validation component acceptance 为 `0.0972/0.9863`，说明统一阈值存在严重局部 coverage 偏差；Component-P05 在 Known Test 上保持更均衡。但 USTC Unknown 已被观察，且 post-hoc UFAR 在 A-1/A-3 变差，因此只能冻结 CSTNET-TLS1.3 外部验证候选，不能声称方法提升。

Stage 5 将 absolute density 和 winning-component local support 组成严格 AND Gate：`G(x) >= tau_global_dual AND L(x) >= tau_local_y*k*`。规则只用 USTC Known Validation 校准，并只用 Known Test 做 sanity；DGSB 未读取 USTC Unknown。CSTNET 使用 `sample_count>=100`（119 类）和首包 UTC 1 分钟 capture-group 代理冻结三组 class-held-out folds。由于 domain 即标签且 endpoint 尚未解析，协议明确保留 `PROTOCOL_RISK`，DGSB 尚无独立 Unknown 性能结论。

Stage 5.5 只做正式实验前预检。A-3 的 Global Gate 在 Known Validation/Test 都没有任何独立拒绝，触发唯一阻断条件；A-1/A-2 的独立拒绝也仅为 `0.0395%–0.0739%`。固定 600-PCAP 审计表明 raw PCAP 可见 domain/SNI 为 `8.67%`，但实际 Open-Detect 32×32 输入为 `0%`；IP 已 mask，port 仍可见。Low/Medium/High Validation P05 样本数仅 `11.6/11.0/12.0`，K2 小 component 的 n<30 fallback 风险很高。因此 Final Gate 为 **NOT_READY**，本阶段没有训练 CSTNET、提取正式 latent、拟合模型、校准边界或访问 Unknown score。

## 仓库结构

```
├── configs/encoder/          trafficformer_input / fig_graph 生成配置
├── docs/                     技术说明（ADAPTATION、FIG、Model B、服务器部署）
├── scripts/
│   ├── 00_prepare_data.py            Stage 0：pcap → 双向流 → pkl + flow_index
│   ├── 01_verify_stage0.py           Stage 0：全量验证（ALL CHECKS PASSED）
│   ├── task03_generate_trafficformer_input.py   TrafficFormer TSV（min1/min3 双策略）
│   ├── task04_generate_fig_graph.py  FIG 全量生成（jsonl + index + label_map）
│   ├── task05_generate_checklist.py / task05_verify_checklist.py   ≥50 流人工校验（程序化核对）
│   ├── task06_generate_splits.py     官方 80/10/10 分层切分（train/val/test TSV）
│   ├── task07_train_model_b.py       Model B：FIG→TAGCN 从零训练 + z_g 导出
│   ├── task08_export_model_a_zt.py   Model A：z_t 导出（GPU 流式，78.8 分钟全量）
│   ├── task09_model_c_fusion.py      Model C：z_f=[z_t;z_g] 融合 + §7.5 验收
│   ├── task10_stage2_audit.py        Stage 2：GMM K=1..5 + BIC + held-out NLL + bootstrap + 小簇
│   ├── task10b_bic_extension.py      BIC 延伸 K=6..10（饱和诊断）
│   ├── task10c_figures.py            UMAP/t-SNE 可视化
│   ├── task10d_separation_metrics.py 可视化分离度数值化（silhouette + 混边率）
│   ├── task10e_valnll_curve.py       val NLL 随 K 曲线
│   ├── task10f_covariance_diagnosis.py  Stage 2.5：z_f covariance diagnosis
│   └── task10g_component_shortcut_audit.py Stage 2.6：component shortcut audit
├── src/preprocessing/        flow_split / trafficformer_input / fig_graph
├── src/stage1/               fig_dataset / tagcn（Model B）
├── tf_runtime/code/          官方 TrafficFormer/UER 固定 commit 源码快照（MIT）
├── stage3_protocol/          冻结的 Open-Detect A-1/A-2/A-3 Strict Unknown-Free 协议
├── stage3_failure_diagnosis/ 冻结 Stage 3 的 post-hoc class/component failure attribution
├── stage4_local_boundary_diagnosis/ Known-only Global/Class/Component-P05 boundary diagnosis
├── stage5_dual_gate_boundary/ DGSB method freeze + CSTNET strict Unknown-free protocol freeze
├── stage5_5_cstnet_preflight/ Stage 5.5 gate/input/calibration final preflight (no training)
├── stage6_cipherspectrum_protocol/ CipherSpectrum canonical-120k Strict Unknown-Free protocol freeze
├── stage7_cipherspectrum_known_training/ CipherSpectrum 三档 Known-only encoder 训练与冻结 checkpoint
├── stage8a_cipherspectrum_known_density/ CipherSpectrum Known-only representation/density/boundary freeze
├── stage8b_cipherspectrum_dgsbv2/ CipherSpectrum DGSB-v2 Known-only rule and one-shot test configuration freeze
├── dataset_suitability_audit/ 三外部数据集 suitability 证据
├── cipherspectrum_labeling/  CipherSpectrum 40/41 类标签与 grouped-split manifest
└── tests/                    单元测试（test_trafficformer_input / test_fig_graph / test_model_b）
```

数据与产物**不入普通 Git**，保留在项目本地工作目录：

`data/`、`outputs/`、各阶段 `runs/`、`artifacts/`、`checkpoints/` 与缓存目录不会被删除；Git 中保留相应结果说明、清单、哈希和复现代码。

## 复现路径

环境：Python 3.8+（本地 miniconda）、torch 2.4.1+cpu、scikit-learn 1.3.2、scapy、joblib、umap-learn、python-docx。Model A 微调需 GPU 服务器（AutoDL 2× V100-32GB，见 docs/STAGE1_AUTODL_SETUP.md）。TrafficFormer/UER 源码版本与本地调用方式见 `docs/TRAFFICFORMER_RUNTIME_SOURCE.md`；SMB 20 类映射、实际训练参数和权重校验见 `docs/SMB_20CLASS_FINETUNE_REPAIR.md`。模型权重不入普通 Git。

1. **Stage 0**：`python scripts/00_prepare_data.py` → `python scripts/01_verify_stage0.py` → `task03`（TSV）→ `task04`（FIG）→ `task05`（校验清单）
2. **Stage 1**：`task06`（切分）→ 服务器 `task08`（z_t 导出）→ 本地 `task07`（Model B + z_g）→ `task09`（融合验收）
3. **Stage 2**：`python scripts/task10_stage2_audit.py`（建议 loky 20 workers + `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`），补充诊断 `task10b/10d/10e`

## 已知偏差与事实（报告需注明）

- 官方管线会匿名化 IP/端口/TCP 时间戳，本仓库的 clean-room 复刻保留真实字节（对 FIG 对齐有利）。
- 官方 <3 包过滤主要消灭 benign 短流类（Facetime/BitTorrent/Skype 全灭），closed-set 会偏向恶意类。
- 数据集 1 包/方向之谜：短流类双向流仅 2 包（1 数据包 + 1 ACK），四点证据确认是作者构造方式而非镜像压体积。

## Stage 9 — CipherSpectrum One-Shot Final Test

Completed strict frozen Low/Medium/High external evaluation. Final result: `B — OPERATING-POINT TRADEOFF`. Evidence: [`stage9_cipherspectrum_final_test/README.md`](stage9_cipherspectrum_final_test/README.md).
