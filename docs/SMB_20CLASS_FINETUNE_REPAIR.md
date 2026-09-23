# SMB 20 类映射与 Model A 微调修复说明

## 修复目标

USTC-TFC2016 原始目录中 `SMB-1.pcap`、`SMB-2.pcap` 是同一个业务类别的两个
采集分片。如果把文件名直接当作类别，会得到 21 类；如果只保留其中一个文件，
又会静默丢数据。本修复将两个分片统一映射为单一 `SMB` 类，并重新生成完整的
20 类 TrafficFormer 训练数据和 `finetuned_model.bin`。

原始 PCAP 始终只读；已有 Stage 0--2 结果没有被覆盖。

## 新增或修改的代码文件

- `src/preprocessing/ustc20_view.py`：构造并校验 20 类符号链接视图；
- `scripts/task00_build_ustc20_view.py`：命令行入口；
- `tests/test_ustc20_view.py`：SMB 合并、20 类完整性及原始文件只读回归测试；
- `configs/dataset/ustc_tfc2016.yaml`：将 Stage 0 输入指向项目内兼容视图。

原有 `src/preprocessing/flow_split.py`、TrafficFormer 输入编码器和官方
TrafficFormer/UER 模型源码均未因 SMB 映射而修改。

## 映射语义

兼容视图中的关键关系为：

```text
原始 SMB-1.pcap  -> data/raw_views/ustc_tfc2016_20class/Benign/SMB/SMB-1.pcap
原始 SMB-2.pcap  -> data/raw_views/ustc_tfc2016_20class/Benign/SMB/SMB-2.pcap
标签              -> SMB
```

视图使用符号链接，不复制、不移动、不修改源 PCAP。Weibo 的 4 个分片同样保持在
单一 `Weibo` 类中。最终标签集合严格为 10 个 benign + 10 个 malware。

## 实际执行命令

以下命令均在项目规定的固定 Conda 环境和具名 tmux 会话中执行：

```bash
python scripts/task00_build_ustc20_view.py
python scripts/00_prepare_data.py
python scripts/01_verify_stage0.py
python scripts/task03_generate_trafficformer_input.py
python scripts/task06_generate_splits.py --project-root "$PWD" --policy compatible_min1

PYTHONPATH=tf_runtime/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tf_runtime/code/fine-tuning/run_classifier.py \
  --vocab_path tf_runtime/code/models/encryptd_vocab.txt \
  --config_path tf_runtime/code/models/bert/base_config.json \
  --pretrained_model_path tf_runtime/code/models/pretrained_model.bin \
  --train_path data/splits/compatible_min1/train_dataset.tsv \
  --dev_path data/splits/compatible_min1/valid_dataset.tsv \
  --test_path data/splits/compatible_min1/test_dataset.tsv \
  --output_model_path outputs/stage1/modelA/finetuned_model.bin \
  --epochs_num 3 --earlystop 3 --batch_size 32 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible \
  --seq_length 320 --learning_rate 6e-5 --pooling first --seed 7
```

计划参数 `batch_size=64` 在两张 L20 各已有约 15.5 GiB 外部显存占用时，于首次
完整反向传播发生 OOM。因此正式运行只把全局 batch 调整为 32（每卡 16）；数据、
模型结构、学习率、epoch、seed 和划分均保持不变。这是资源适配，不应表述为严格的
原官方 batch-size 复现。

## 数据核验结果

- Stage 0：20 类，489,101 条双向流；
- SMB：38,937 条流，其中 SMB-1 为 32,661，SMB-2 为 6,276；
- `scripts/01_verify_stage0.py`：`RESULT: ALL CHECKS PASSED`；
- TrafficFormer `compatible_min1`：保留 489,101/489,101 条流，未伪造数据包；
- 标签表：只有 `SMB`，没有 `SMB-1` 或 `SMB-2`；
- 固定分层划分：train 391,280，validation 48,910，test 48,911；
- 三个 split 两两无交集，并集等于全部 489,101 条输入。

`compatible_min1` 保留至少 1 包的流，最多编码前 5 包、每包 64 字节并在包前加入
`[SEP]`。不足 5 包时由 TrafficFormer reader 在 token 序列尾部 PAD；没有构造
虚假数据包。详见 `docs/TRAFFICFORMER_SHORT_FLOW_ADAPTATION.md`。

## 训练与模型结果

训练使用官方预训练权重和 TrafficFormer/UER 固定源码 commit
`6d0ba64d82e74fb130c6c7301ef20885dbfbdf29`。验证结果逐轮为：

| epoch | validation accuracy | validation Macro-F1 |
|---:|---:|---:|
| 1 | 0.9841 | 0.9869 |
| 2 | 0.9883 | 0.9911 |
| 3 | 0.9893 | 0.9919 |

最佳权重在独立测试集上的结果：

- accuracy / Micro-F1：0.9895（48,399 / 48,911）；
- Macro-F1：0.9921；
- Weighted-F1：0.9895。

模型文件：

```text
outputs/stage1/modelA/finetuned_model.bin
size:   528647077 bytes
sha256: df866633fa15541d1a1325d9a5d683a36d359499c462769d5be0c2a202fdfe52
```

完成检查确认该文件可由 `torch.load(..., weights_only=True)` 加载，共 201 个 tensor
key，无 NaN/Inf；最终分类层 `output_layer_2.weight` 的形状为 `[20, 768]`。

## 复现边界

- 这是对当前项目 20 类任务的可复核微调产物，不是 TrafficFormer 作者发布的
  20 类 checkpoint；
- batch size 因共享 GPU 容量由 64 调整为 32；
- 当前复刻官方脚本采用流级分层随机划分，同一源 PCAP 的不同流可能分布在不同
  split；高准确率不能直接解释为对独立采集环境的泛化能力；
- 权重被 `.gitignore` 排除，不会随普通 Git push/clone 分发；需要单独保存并按
  上述 SHA-256 校验；
- 本次任务只完成 Model A 微调，没有启动 Stage 2.5、Stage 3 或 Adaptive K。
