# TrafficFormer 短流兼容修复说明

- 日期：2026-08-31
- 状态：新增实现与合成测试完成；真实 USTC-TFC2016 数据生成尚未执行。

## 1. 问题

TrafficFormer 官方微调数据生成代码会拒绝少于 3 个包的 TCP/UDP 流，并拒绝 IPv6。对当前 Stage 0 统计所描述的 USTC-TFC2016 短流分布，直接复制该过滤行为会造成部分类别完全或近乎完全丢失，无法维持项目计划中的 20 类 closed-set 基线。

官方参考：

- 仓库：<https://github.com/IDP-code/TrafficFormer>
- 固定提交：`6d0ba64d82e74fb130c6c7301ef20885dbfbdf29`
- 短流过滤：`data_generation/finetuning_data_gen.py:get_feature_flow`
- bigram：`data_generation/utils.py:bigram_generation`

## 2. 修复决策

本修复不修改 Stage 0 原始流构建代码，也不修改 TrafficFormer 模型。新增 Task 0.3 生成器，并提供两个共用同一编码实现的策略：

| 策略 | 最少真实包数 | 用途 |
|---|---:|---|
| `compatible_min1` | 1 | 主实验；保留所有非空流 |
| `strict_min3` | 3 | 官方短流过滤的覆盖率/敏感性对照 |

修复只改变“是否接受 1–2 包流”，不改变以下输入语义：

- 使用捕获顺序中的前 `min(packet_count, 5)` 个真实包；
- 每包跳过 14 字节以太网头，最多读取后续 64 字节；
- 使用官方等价的 overlapping two-byte bigram 文本；
- 每个真实包前写入 `[SEP]`；
- TSV 不写入 `[CLS]`；
- 不构造空包、不复制包、不补零字节；
- PAD 仍由下游 TrafficFormer reader 在 token 序列尾部完成，目标长度为 320。

RIFA 不属于本次短流修复。当前生成器输出确定性基础输入，`rifa_applied=false`；若后续加入 RIFA，应作为固定种子的 train-only augmentation，不能污染 validation/test。

## 3. 新增文件

- `src/preprocessing/trafficformer_input.py`：纯函数编码、策略控制、PKL→TSV/映射生成。
- `scripts/task03_generate_trafficformer_input.py`：命令入口。
- `configs/encoder/trafficformer_input.yaml`：输入格式与双策略配置。
- `tests/test_trafficformer_input.py`：单元测试和真实 CLI 合成烟测。
- `docs/TRAFFICFORMER_SHORT_FLOW_ADAPTATION.md`：本说明。

原有 `flow_split.py`、Stage 0 脚本、数据配置和实验方案均未修改。

## 4. 输出

默认主策略输出目录：

```text
data/trafficformer_input/compatible_min1/
├── trafficformer_all.tsv
├── flow_map.csv
├── label_map.json
├── generation_summary.json
└── retention_by_class.csv
```

`trafficformer_all.tsv` 保持官方分类器所需的 `label`、`text_a` 两列。`flow_map.csv` 以 `sample_index` 和 `tsv_line_number` 将 TSV 行映射回 `flow_id`，同时记录包数、实际使用包数、短流标志、token 数量、策略和文本 SHA-256。

生成器默认拒绝覆盖已有输出；只有显式传入 `--overwrite` 才替换本生成器管理的五个文件。

## 5. 运行方式

在项目根目录通过工作区要求的 tmux 助手运行：

```bash
../../skills/tmux-task-execution/scripts/tmux_task.sh \
  run unknown-task03-min1-YYYYMMDD-HHMMSS "$PWD" -- \
  python scripts/task03_generate_trafficformer_input.py \
    --policy compatible_min1
```

生成 `<3` 包过滤对照：

```bash
../../skills/tmux-task-execution/scripts/tmux_task.sh \
  run unknown-task03-min3-YYYYMMDD-HHMMSS "$PWD" -- \
  python scripts/task03_generate_trafficformer_input.py \
    --policy strict_min3
```

## 6. 已完成验证

固定 Conda 环境：

```text
/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310
```

执行：

```bash
python -m unittest discover -s tests -p test_trafficformer_input.py -v
```

结果：`10 tests passed`。覆盖内容：

- 官方 bigram 黄金向量和参考实现等价；
- 1 包、2 包流在主策略中保留；
- 2 包流在 `strict_min3` 中过滤；
- 最多使用 5 个真实包且不构造 PAD 包；
- `packet_count >= 3` 的共同 cohort 在两个策略下文本完全一致；
- 短捕获帧不会被人为扩展；
- TSV 与 `flow_id` 映射一致；
- 默认不覆盖旧输出；
- CLI 合成端到端生成成功。

## 7. 尚未验证与下一步

当前仓库没有 `data/flows/*.pkl`，因此还不能确认真实数据上的 20 类保留数，也没有执行官方词表 tokenization 或模型推理。拿到 Stage 0 PKL 后应依次：

1. 分别运行 `compatible_min1` 和 `strict_min3`；
2. 对比 `retention_by_class.csv`，确认主策略仍有 20 类；
3. 验证两个策略共同 cohort 的 `text_sha256` 完全一致；
4. 使用固定官方词表检查 token IDs、长度 320 和 PAD mask；
5. 按 `1 / 2 / 3–4 / >=5` 包分桶报告 closed-set 和 Unknown Detection 指标；
6. Task 0.4 的 FIG 输入必须使用同一组 `flow_id`，不得单独丢弃短流。

在完成真实数据和官方词表验证前，本修复只能称为“格式兼容短流适配”，不能称为 TrafficFormer 官方预处理的完整复现。
