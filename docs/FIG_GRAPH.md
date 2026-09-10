# Task 0.4：FEC-OSL FIG 输入构建（FIG_INPUT）

本文说明 FIG（Flow Interaction Graph）输入的定义、构建流程、输出格式与对齐约束。

## 1. 任务定义

对 Stage 0 的每一条双向流，用**与 Task 0.3 完全相同的 `flow_id` 集合**构造一个
Flow Interaction Graph。必须保存 `flow_id` 与图本身；**任何流都不得被丢弃**，
包括 Task 0.3 `strict_min3` 会过滤掉的短流（学生验收清单第 12 项）。

FIG 的下游是 TAGCN + readout（Stage 1 / Task 2），与 TrafficFormer 的 z_t 拼接为
z_f=[z_t;z_g]。Task 0.4 只负责建图落盘，不训练任何模型。

## 2. FIG 定义（依据论文）

依据 FEC-OSL 原文（Yang et al., IEEE TIFS 2026）与其遵循的奠基工作
（Shen et al., IEEE TIFS 2021, DOI 10.1109/TIFS.2021.3050608）。

### 2.1 节点

每条流的前 **30** 个包（Nv=30，FEC-OSL 实现细节）。节点数 = min(30, 包数)，
**不补齐、不合成**；单包流就是单节点图。

### 2.2 burst（切分规则）

burst = **同方向连续包段**。Shen 2021 Algorithm 1 按方向切分，**没有时间阈值**。
FEC-OSL 的 "predefined burst threshold" 措辞继承自经典定义，但文中说明其过程
遵循 [35]（即 Shen 2021），故按方向切分。

### 2.3 节点特征（7 维，按 FEC-OSL 顺序）

| # | 特征 | 定义 |
|---|---|---|
| 1 | direction | 发起者 +1，响应方 -1 |
| 2 | length | 包长度（caplen；与 wirelen 在全语料仅 6 包不同） |
| 3 | timestamp_delta | 相对流首包的秒差 |
| 4 | burst_packet_count | 所在 burst 的包数 |
| 5 | burst_byte_count | 所在 burst 的字节数 |
| 6 | prev_burst_packet_ratio | 本 burst 包数 / 前一 burst 包数 |
| 7 | prev_burst_byte_ratio | 本 burst 字节数 / 前一 burst 字节数 |

首 burst 没有前驱，特征 6-7 填 `first_burst_ratio_fill`（默认 0.0）。
前一 burst 字节数为 0 时（理论不可能，防御性处理）特征 7 同样填 fill。

### 2.4 边（无向，Shen 2021 Algorithm 1）

- **burst 内边**：同一 burst 中按时间顺序相邻节点两两相连（链式）。
- **burst 间边**：相邻 burst 之间，首节点连首节点、尾节点连尾节点。
- 两个 burst 都是单包时首=尾，边对去重后只留 1 条（Shen 原文约束）。
- 边以 `(min, max)` 索引对存储，排序去重。

## 3. 构建流程

```
24 个 Stage 0 pkl（(ts, caplen, wirelen, direction, raw_frame)）
  ↓ 逐流：按 ts 稳定排序（Stage 0 已按 ts 存，此为 no-op）
  ↓ 取前 min(30, 包数) 个包
  ↓ direction(1=发起者,0=响应方) → 符号 ±1 → 切 burst
  ↓ 逐节点算 7 特征（含跨 burst 比率）
  ↓ 连边：burst 内链式 + burst 间首首/尾尾，去重
  ↓ 序列化 JSON 记录 + 写 fig_index.csv
  ↓
data/fig_graph/all_flows/
  ├── fig_all.jsonl          # 每行一个图（与 fig_index 行序一致）
  ├── fig_index.csv          # flow_id ↔ byte_offset ↔ 统计字段
  ├── label_map.json         # 与 Task 0.3 相同的 class_to_id
  ├── stats_by_class.csv     # 逐类节点/边/burst 统计
  └── generation_summary.json
```

## 4. 输出格式

### 4.1 fig_all.jsonl（每行一个 JSON 对象）

```json
{"flow_id":"BitTorrent__000001","node_count":2,"burst_count":2,
 "features":[[1,100,0.0,1,100,0.0,0.0],[-1,40,0.001,1,40,1.0,0.4]],
 "edges":[[0,1]]}
```

- `features`：node_count × 7，整数不写小数点，浮点保留 6 位小数。
- `edges`：0 基节点索引对，已排序去重。
- 行序 = `fig_index.csv` 的 `jsonl_line_number` 顺序。

### 4.2 fig_index.csv

`sample_index, jsonl_line_number, flow_id, byte_offset, label_id, split,
class_name, source_file, source_pkl, packet_count, used_packet_count,
packet_count_bucket, node_count, burst_count, edge_count, policy, graph_sha256`

- `byte_offset`：该行在 jsonl 中的字节偏移，供 Stage 1 加载器 seek 随机访问。
- `label_id`：由排序类名生成，**与 Task 0.3 的 label_map.json 完全一致**。
- `graph_sha256`：该行 JSON 字符串的 sha256，供 Task 0.5 一致性核对。

## 5. 对齐约束（验收要点）

1. flow_id 集合 = Task 0.3 `compatible_min1` 的全部 489,101 条（20 类全保留）。
2. label_id 与 Task 0.3 一致（同一排序规则）。
3. 包数、包序、时间戳、方向均来自同一份 Stage 0 pkl，FIG 按 ts 排序与
   TrafficFormer 的存储序一致（Stage 0 已验证 ts 非降）。
4. 短流不丢弃：1 包流 = 单节点 0 边；2 包反向流 = 2 单包 burst + 1 条边。

## 6. 已知偏差（写报告时注明）

- FEC-OSL 未公开官方代码；本实现按论文文字 + Shen 2021 算法逐条复刻。
  burst 切分按方向（无时间阈值），若后续需要时间阈值变体可加 config 开关。
- 原始流量未匿名化（与 Task 0.3 一致，对 FIG 对齐有利）；FEC-OSL 在
  USTC-TFC2016 上训练时的预处理细节（是否匿名化 IP 等）论文未说明。
- 前 30 包窗口是 FEC-OSL 原文设定（其 Limitations 节也承认依赖前 30 包），
  TrafficFormer 用前 5 包——窗口不同是设计使然。

## 7. 运行

```bash
python scripts/task04_generate_fig_graph.py \
    --flows-dir D:/unknown_traffic_project/data/flows \
    --output-dir data/fig_graph/all_flows
```

默认输出目录为 `configs/encoder/fig_graph.yaml` 的 `outputs.root_dir/policy`。
重复运行默认拒绝覆盖，需 `--overwrite`。测试：`python -m unittest tests.test_fig_graph`。
