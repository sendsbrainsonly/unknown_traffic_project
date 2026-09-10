# Stage 1 Model B：FIG → TAGCN（closed-set 单分支）

v2 计划 §7.1 的 Model B：

```text
FIG
↓
TAGCN
↓
z_g
↓
Classifier
```

## 1. 数据

- 图数据：Task 0.4 产物 `data/fig_graph/all_flows/`（fig_all.jsonl + fig_index.csv，
  489,101 图，节点 = 每流前 30 包，7 维特征，burst 内链式边 + burst 间首首/尾尾边）。
- 划分：**与 Model A 完全相同的 splits**（`data/splits/compatible_min1/`，
  官方 41/42 分层 80/10/10 → 391,280/48,910/48,911），按 flow_id 经
  fig_index 的 byte_offset 随机访问抽取。装载时校验 split 内所有 flow_id
  都在 fig_index 中（未对齐直接报错）。
- label：fig_index.label_id 经 split 内唯一类排序重映射为连续 0..C-1
  （全量 20 类本为 0-19；冒烟子集自动重映射）。

## 2. 模型（src/stage1/tagcn.py）

TAGCN 按 FEC-OSL 原文（Yang et al., IEEE TIFS 2026）公式 5-6 的
K 跳多项式滤波器：

```
H = Σ_{k=0..K} (Â^k X) W_k + b,   Â = D^-1/2 (A+I) D^-1/2
```

- K=2（默认，可配 --k-hops）
- readout：真实节点 masked **mean** pooling → z_g（hidden=128，默认）
- 分类头：Linear(hidden → labels_num)
- forward 返回 (logits, z_g)；z_g 供 §7.4 导出与 Model C 融合

## 3. 训练（scripts/task07_train_model_b.py）

- 配方沿用 FEC-OSL 原文报告值：**SGD lr 1e-4 / 50 epoch / batch 64**，
  仅分类交叉熵（Stage 1 不上聚类分支）
- 每 epoch 末 dev 评估，按 dev Macro-F1 保存 best（modelB_best.pt）
- 结束后 test 评估：§7.3 全套（Acc / Macro P/R/F1 / per-class F1 / 混淆矩阵）
- §7.4 导出：embedding_{train,val,test}.npy（z_g）+ labels_*.npy + flow_ids_*.npy
  + run_summary.json（超参、dev 轨迹、test 指标、标准化常数）

实测速度：本地 CPU（无 GPU）**5 ms/批**，全量 50 epoch ≈ 30 分钟；
图极小（≤30 节点）是主因，Model B 无需服务器。

## 4. 实现选择与已知偏差（写报告时注明）

- FEC-OSL **未公开官方代码**；TAGCN 多项式滤波器按论文公式 5-6 实现，
  原文 "readout→hg" 未指定池化形式，取 masked mean。
- **特征标准化**：用训练集逐特征全局 mean/std（7 维统一），验证/测试共用。
  未逐图标准化：单节点图 std=0 会退化。论文未说明标准化方案，此为自定。
- 图批处理：变长图 padding 到批内最大节点数 + mask；padding 节点无自环
  （零行零列），k 跳传播后仍为零，不泄漏进 readout（有单元测试保证）。
  不违反 FEC-OSL "图尺寸可变不补齐" 的定义（那是建图规则，批处理是训练实现）。
- 节点特征 7 维按 FEC-OSL 顺序：direction/length/timestamp_delta/
  burst_packet_count/burst_byte_count/prev_burst_packet_ratio/prev_burst_byte_ratio
  （首 burst 比率特征填 0，见 docs/FIG_GRAPH.md）。
- 原始流量未匿名化（与 Task 0.3 一致，FIG 对齐有利）。

## 5. 运行

```bash
# 单元测试（合成图 + 真实冒烟）
python -m unittest tests.test_model_b -v

# 冒烟（2 类 320 流；lr 1e-4 下 3 epoch 不会收敛，仅验证链路）
python scripts/task07_train_model_b.py --split smoke_gmail_zeus --epochs 3

# 全量（本地 CPU ≈ 30 分钟）
python scripts/task07_train_model_b.py
```

产物在 `outputs/stage1/modelB/`。
