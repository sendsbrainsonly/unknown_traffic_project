# Unknown Traffic / Unknown Attack Detection（USTC-TFC2016）

面向未知流量 / 未知攻击检测（Open-Set）的研究项目。围绕三个核心问题：

- **RQ1**：Known 类内表示是否普遍多模态 → Adaptive Multi-Prototype（GMM）
- **RQ2**：局部模式边界是否异质 → Prototype-Specific Boundary（Q95 半径）
- **RQ3**：Unknown 本身是否非单一类 → Fine-Grained Unknown Class Discovery（ARI/AMI）

执行边界：只做 Stage 0-4（数据处理 → Closed-Set Backbone → Known-Space Distribution Audit → Single Prototype → Multi-Prototype），Stage 4 验收后再决定是否进入 Boundary / Discovery 阶段。严格 Unknown-Free：Unknown 类只在最终 test 集出现（详见计划文档 v2 §29）。

## 当前进展（截至 2026-09-10）

| 阶段 | 状态 | 关键结果 |
|---|---|---|
| Stage 0 数据处理 | ✅ 完成 | 489,101 条双向流（20 类全量）；FIG 全量生成（4,088,199 节点）；交叉校验全绿；≥50 流人工校验 420/420 项通过 |
| Stage 1 Closed-Set Backbone | ✅ 完成 | Model A（TrafficFormer 微调）test Acc 0.9892 / Macro-F1 0.9920；Model B（FIG→TAGCN，seed 0/1/2）0.6998±0.0236 / 0.7329±0.0132；Model C 融合 0.9893±0.0001 / 0.9921±0.0001，§7.5 验收 PASS |
| Stage 2 Distribution Audit | ✅ 完成 | 20/20 类 consensus K>1、held-out val NLL 120/120 类-seed 改善（zf 中位 +46.6% / zt +38.2%）、bootstrap P(K=5)=1.0；§8.11 判据 4/4 通过 → RQ1 多模态证据成立，进入 Stage 3 |
| Stage 3 Single Prototype | ⏳ 待启动 | 计划 §9：单高斯原型 + 阈值开集检测 baseline |

Stage 2 的两个重要结论（详见各 summary md）：

1. **BIC 饱和**：训练集 BIC 在 K≤10 内无最小值（延伸诊断 18/18 单元 K*=9/10）。BIC 罚项 ≈0.5·ln(N)·1793≈1 万 loglik，远小于新分量增益（千万级）。结论只能表述为「每类 ≥5 个子群（下界）」；Stage 4 的 K 决定改用 val-NLL 曲线 + 小簇合并（N_min）+ 预算上限，不用 BIC 点估计。
2. **reg_covar 偏离**：GMM 的 reg_covar 由 sklearn 默认 1e-6 调为 1e-3（类内近常数维度会劫持似然尺度），偏离已在脚本头与 summary md 中记录。

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
├── src/preprocessing/        flow_split / trafficformer_input / fig_graph
├── src/stage1/               fig_dataset / tagcn（Model B）
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

环境：Python 3.8+（本地 miniconda）、torch 2.4.1+cpu、scikit-learn 1.3.2、scapy、joblib、umap-learn、python-docx。Model A 微调需 GPU 服务器（AutoDL 2× V100-32GB，见 docs/STAGE1_AUTODL_SETUP.md）。

1. **Stage 0**：`python scripts/00_prepare_data.py` → `python scripts/01_verify_stage0.py` → `task03`（TSV）→ `task04`（FIG）→ `task05`（校验清单）
2. **Stage 1**：`task06`（切分）→ 服务器 `task08`（z_t 导出）→ 本地 `task07`（Model B + z_g）→ `task09`（融合验收）
3. **Stage 2**：`python scripts/task10_stage2_audit.py`（建议 loky 20 workers + `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`），补充诊断 `task10b/10d/10e`

## 已知偏差与事实（报告需注明）

- 官方管线会匿名化 IP/端口/TCP 时间戳，本仓库的 clean-room 复刻保留真实字节（对 FIG 对齐有利）。
- 官方 <3 包过滤主要消灭 benign 短流类（Facetime/BitTorrent/Skype 全灭），closed-set 会偏向恶意类。
- 数据集 1 包/方向之谜：短流类双向流仅 2 包（1 数据包 + 1 ACK），四点证据确认是作者构造方式而非镜像压体积。
