# Unknown Traffic / Unknown Attack Detection（USTC-TFC2016）

面向未知流量 / 未知攻击检测（Open-Set）的研究项目。当前 evidence-based 主线为 **Persistent Local-Support Modeling**：

- **RQ1**：一个 global class-conditional support model 是否足以描述 Known traffic representation？
- **RQ2（后移）**：只有 Multi local support 先证明 Unknown Detection utility 后，才检验 local-support-specific boundary。
- **RQ3（后移）**：只有 detection 有效且 rejected Unknown buffer 质量合理后，才做 Fine-Grained Unknown Class Discovery。

严格 Unknown-Free：Unknown 类只允许进入 Final Test，不得进入 encoder/scaler/PCA/Gaussian/GMM/K/threshold 的训练、拟合、选择或校准。Stage 3 已按冻结顺序 `A-1 → A-2 → A-3` 完成：三者均为 primary benchmark，不是 secondary sensitivity，分别独立训练并全部报告。本任务止于最终 Gate，未进入 Adaptive K、Boundary 或 Discovery。

## Current Status（截至 2026-09-12）

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

当前正式结论：

> Single global Gaussian is not always descriptively sufficient, but residual components are frequently associated with simple traffic statistics and cannot be interpreted as semantic modes. Under the frozen strict Unknown-Free test, fixed K2 multi-local support did not demonstrate consistent practical Unknown Detection utility across A-1/A-2/A-3.

Stage 3 final decision:

> **Gate D — MIXED.** A-1 significantly worsened, A-2 was statistically inconclusive for UFAR, and A-3 significantly improved; multi-local-support utility is therefore not established.

Stage 3 结果与可复核证据见 `stage3_unknown_utility/outputs/summary/` 和各 setting 的独立实验包。本任务不继续下一阶段。

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
├── dataset_suitability_audit/ 三外部数据集 suitability 证据
├── cipherspectrum_labeling/  CipherSpectrum 40/41 类标签与 grouped-split manifest
└── tests/                    单元测试（test_trafficformer_input / test_fig_graph / test_model_b）
```

数据与产物**不入仓库**（本地 `D:\unknown_traffic_project`）：

```
D:\unknown_traffic_project\
├── data\flows\                24 个 pkl（4GB，flow_id → 包级元组）
├── data\trafficformer_input\  compatible_min1（489,101 流）/ strict_min3（311,345 流）TSV
├── data\fig_graph\all_flows\  fig_all.jsonl + fig_index.csv + label_map.json
├── data\splits\               train/val/test 三套 TSV
└── outputs\                   stage0_data\ / stage1\（modelA/B/C）/ stage2\（六件套 + 图 + 补充诊断）
```

## 复现路径

环境：Python 3.8+（本地 miniconda）、torch 2.4.1+cpu、scikit-learn 1.3.2、scapy、joblib、umap-learn、python-docx。Model A 微调需 GPU 服务器（AutoDL 2× V100-32GB，见 docs/STAGE1_AUTODL_SETUP.md）。TrafficFormer/UER 源码版本与本地调用方式见 `docs/TRAFFICFORMER_RUNTIME_SOURCE.md`；SMB 20 类映射、实际训练参数和权重校验见 `docs/SMB_20CLASS_FINETUNE_REPAIR.md`。模型权重不入普通 Git。

1. **Stage 0**：`python scripts/00_prepare_data.py` → `python scripts/01_verify_stage0.py` → `task03`（TSV）→ `task04`（FIG）→ `task05`（校验清单）
2. **Stage 1**：`task06`（切分）→ 服务器 `task08`（z_t 导出）→ 本地 `task07`（Model B + z_g）→ `task09`（融合验收）
3. **Stage 2**：`python scripts/task10_stage2_audit.py`（建议 loky 20 workers + `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`），补充诊断 `task10b/10d/10e`

## 已知偏差与事实（报告需注明）

- 官方管线会匿名化 IP/端口/TCP 时间戳，本仓库的 clean-room 复刻保留真实字节（对 FIG 对齐有利）。
- 官方 <3 包过滤主要消灭 benign 短流类（Facetime/BitTorrent/Skype 全灭），closed-set 会偏向恶意类。
- 数据集 1 包/方向之谜：短流类双向流仅 2 包（1 数据包 + 1 ACK），四点证据确认是作者构造方式而非镜像压体积。
