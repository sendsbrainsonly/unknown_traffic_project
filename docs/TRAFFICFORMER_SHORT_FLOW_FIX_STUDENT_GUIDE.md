# TrafficFormer 短流修复：学生执行指南

- 日期：2026-08-31
- 适用任务：Stage 0 / Task 0.3
- 当前状态：修复代码和合成测试完成；真实 USTC-TFC2016 数据尚未运行。

## 一、为什么需要修复

TrafficFormer 官方微调数据生成器会拒绝少于 3 个包的 TCP/UDP 流。USTC-TFC2016 中部分类别包含大量 1–2 包流，严格复制该过滤条件会导致部分类别完全或近乎完全消失，无法继续执行项目要求的 20 类 closed-set 和后续 open-set 实验。

这次修复的目标不是修改 TrafficFormer，而是让短流能够使用同一种官方输入格式进入模型。

## 二、修复前后对比

| 项目 | 官方短流过滤 | 本项目主策略 |
|---|---|---|
| 最少包数 | 3 | 1 |
| 最多使用包数 | 5 | 5 |
| 每包字节 | 以太网头后 64 字节 | 相同 |
| 文本格式 | 每个真实包前 `[SEP]`，官方 bigram | 相同 |
| 不足 5 包 | 提前过滤 `<3` 包流 | 使用已有真实包，随后 token PAD |
| 是否构造空包 | 否 | 否 |
| `[CLS]` | 模型读取器添加 | 相同 |
| PAD | 模型读取器补到固定 token 长度 | 相同 |

修复的唯一核心变化是：

```text
官方：packet_count < 3 -> 丢弃
本项目：packet_count < 1 -> 丢弃
```

空流仍然无效；1 包和 2 包流被保留。

## 三、数据如何流动

```text
Stage 0 flow PKL
  flow_id -> [(timestamp, caplen, wirelen, direction, raw_frame), ...]
        │
        ├─ 保持原始捕获顺序
        ├─ 只取前 min(packet_count, 5) 个真实包
        ├─ 每包取 raw_frame[14:78]
        ├─ 官方等价 bigram
        └─ 每个真实包前添加 [SEP]
        │
        ├─ trafficformer_all.tsv
        └─ flow_map.csv -> 保留 flow_id 对齐关系
        │
        └─ 下游官方 tokenizer 添加 [CLS] 并将 token 序列 PAD 到 320
```

注意：生成器不会补出第 2、3、4、5 个“假包”，也不会复制最后一个真实包。

## 四、修复文件清单

### 4.1 本次新增文件

| 文件 | 学生需要知道的作用 |
|---|---|
| `src/preprocessing/trafficformer_input.py` | 核心编码器；实现官方等价 bigram、短流策略和 PKL→TSV/映射生成 |
| `scripts/task03_generate_trafficformer_input.py` | Task 0.3 命令入口 |
| `configs/encoder/trafficformer_input.yaml` | 统一记录字节范围、包数、序列长度和两种策略 |
| `tests/test_trafficformer_input.py` | 单元测试和 CLI 合成端到端测试 |
| `docs/TRAFFICFORMER_SHORT_FLOW_ADAPTATION.md` | 教师/开发者使用的完整技术修复说明 |
| `docs/TRAFFICFORMER_SHORT_FLOW_FIX_STUDENT_GUIDE.md` | 本学生执行指南 |

### 4.2 明确保留、没有修改的原文件

| 文件 | 保留原因 |
|---|---|
| `src/preprocessing/flow_split.py` | Stage 0 的 canonical flow 构建不能被某个 encoder 的过滤策略污染 |
| `scripts/00_prepare_data.py` | 原始 PCAP→flow PKL 入口保持不变 |
| `scripts/01_verify_stage0.py` | 原有 Stage 0 校验逻辑保持不变 |
| `configs/dataset/ustc_tfc2016.yaml` | 原数据配置保持不变 |
| `未知流量_未知攻击检测_三核心问题_学生详细实验计划_公式优化版.md` | 原实验计划保持不变 |

## 五、两种策略

### 5.1 主实验：compatible_min1

```yaml
compatible_min1:
  min_packets: 1
```

用途：保留全部非空流，维持 20 类覆盖。

### 5.2 对照实验：strict_min3

```yaml
strict_min3:
  min_packets: 3
```

用途：量化 `<3` 包过滤带来的逐类样本损失。该策略只复现官方的最小包数条件，不包含官方的 `<2KB` 文件过滤、小类别过滤或 IPv6 处理，因此不能称为“完整官方管线”。

## 六、输出文件清单

主策略默认输出到：

```text
data/trafficformer_input/compatible_min1/
```

| 输出 | 内容 |
|---|---|
| `trafficformer_all.tsv` | 官方分类器需要的 `label`、`text_a` 两列 |
| `flow_map.csv` | TSV 行号到 `flow_id` 的一对一映射 |
| `label_map.json` | 类别名和数字标签的确定性映射 |
| `generation_summary.json` | 策略、格式、总流数、保留数和丢弃数 |
| `retention_by_class.csv` | 每类保留率和 `1 / 2 / 3–4 / >=5` 包分桶统计 |

`flow_map.csv` 还包含：

```text
packet_count
used_packet_count
packet_count_bucket
token_count
is_short_flow
policy
text_sha256
```

这些字段用于后续 TrafficFormer/FIG 对齐和短流分层评估。

## 七、学生执行步骤

所有命令必须在项目根目录、固定 Conda 环境和命名 tmux 会话中运行。

### Step 1：确认 Stage 0 PKL 已存在

```bash
../../skills/tmux-task-execution/scripts/tmux_task.sh \
  run student-task03-check-YYYYMMDD-HHMMSS "$PWD" -- \
  bash -c 'rg --files data/flows -g "*.pkl" | sort'
```

若没有 PKL，停止；不要生成空 TSV。

### Step 2：运行测试

```bash
../../skills/tmux-task-execution/scripts/tmux_task.sh \
  run student-task03-tests-YYYYMMDD-HHMMSS "$PWD" -- \
  env PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -p test_trafficformer_input.py -v
```

预期：`Ran 10 tests` 和 `OK`。

### Step 3：生成主策略输入

```bash
../../skills/tmux-task-execution/scripts/tmux_task.sh \
  run student-task03-min1-YYYYMMDD-HHMMSS "$PWD" -- \
  python scripts/task03_generate_trafficformer_input.py \
    --policy compatible_min1
```

### Step 4：生成 `<3` 包过滤对照

```bash
../../skills/tmux-task-execution/scripts/tmux_task.sh \
  run student-task03-min3-YYYYMMDD-HHMMSS "$PWD" -- \
  python scripts/task03_generate_trafficformer_input.py \
    --policy strict_min3
```

默认不覆盖已有输出。如果确实需要重跑，应先核对旧结果是否已经归档，再显式使用 `--overwrite`。

## 八、学生验收清单

提交前逐项确认：

- [ ] `compatible_min1` 中所有非空 PKL flow 都有一条 TSV 记录。
- [ ] `flow_map.csv` 中 `flow_id` 无重复。
- [ ] `tsv_line_number` 能准确定位对应 TSV 行。
- [ ] 1 包流有 1 个 `[SEP]`，2 包流有 2 个 `[SEP]`。
- [ ] 超过 5 包的流只使用前 5 个真实包。
- [ ] 没有人工构造的空包、复制包或补零字节包。
- [ ] `strict_min3` 只比主策略少 `<3` 包流。
- [ ] 两种策略共同 cohort 的 `text_sha256` 完全一致。
- [ ] `retention_by_class.csv` 中主策略仍覆盖 20 类。
- [ ] 使用官方词表后检查 token 长度 320 和 PAD mask。
- [ ] 随机抽查至少 50 条流的包顺序、字节范围、标签和 `flow_id`。
- [ ] 后续 FIG 使用同一组 `flow_id`，不能单独删除短流。

## 九、明确禁止

学生不得：

1. 修改 `flow_split.py` 以迎合 TrafficFormer；
2. 复制最后一个包凑够 5 包；
3. 创建 `[SEP] + [PAD]` 形式的假包；
4. 在 TSV 中手工添加 `[CLS]`；
5. 重新训练或修改官方词表；
6. 用 Unknown test 结果决定是否保留短流；
7. 只提交截图，不提交 TSV、CSV、配置和日志；
8. 把 `strict_min3` 称为完整官方复现；
9. 在没有真实数据验证时声称 20 类已经全部保留。

## 十、当前已验证和未验证边界

已验证：

- 新增代码语法和 YAML 配置可解析；
- 10 项测试全部通过；
- 1/2 包保留、3 包共同 cohort 等价、最多 5 包、短帧和防覆盖行为通过；
- 合成 PKL 经真实 CLI 成功生成 TSV 和映射；
- 原有跟踪文件没有改动。

尚未验证：

- 真实 USTC-TFC2016 的 20 类保留数量；
- 官方词表 token IDs 和 PAD mask；
- TrafficFormer 模型 embedding；
- 与 FIG 的真实 `flow_id` 对齐；
- closed-set 和 open-set 指标。

因此当前可报告为：

> Task 0.3 短流兼容生成器已实现并通过合成验证；真实数据和模型级验证待 Stage 0 PKL、官方词表与模型资产就绪后执行。
