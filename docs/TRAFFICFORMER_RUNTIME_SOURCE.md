# TrafficFormer / UER 仓库内源码说明

## 为什么必须补充

Model A 的训练脚本依赖 TrafficFormer，embedding 导出脚本依赖其中的
`uer` Python 包。只有本项目的调用脚本而没有 TrafficFormer/UER 源码时，
`task08_export_model_a_zt.py` 会在启动阶段报 `No module named 'uer'`。

## 已纳入的源码

仓库新增 `tf_runtime/code/`，内容来自官方仓库：

- 上游：<https://github.com/IDP-code/TrafficFormer>
- 固定 commit：`6d0ba64d82e74fb130c6c7301ef20885dbfbdf29`
- 许可证：MIT，原始许可证保存在 `tf_runtime/code/LICENSE`
- 关键目录：`uer/`、`fine-tuning/`、`pre-training/`、`data_generation/`
- 关键输入：`models/encryptd_vocab.txt`、`models/bert/base_config.json`

本次导入保留上游源码内容，但排除上游误提交的
`__pycache__/*.pyc` 编译缓存，以及本项目不使用的 Windows
`data_generation/SplitCap.exe` 二进制文件。本项目继续使用已审计的
Stage 0 流切分脚本。版本、边界和逐文件 SHA-256 记录在
`tf_runtime/UPSTREAM.md` 与 `tf_runtime/SOURCE_MANIFEST.sha256`。

## 没有纳入的文件

上游 commit 本身没有提交模型权重，本项目也继续通过 `.gitignore` 排除
`*.bin`、`*.pt` 和 `*.pth`。模型权重始终是本地或外部产物，不会随普通 Git clone 下载：

1. TrafficFormer 官方 `pretrained_model.bin`：已于 2026-09-11 下载到当前工作副本，
   来源、大小、SHA-256 和结构验证见 `tf_runtime/PRETRAINED_MODEL.md`；
2. 本项目 20 类训练产生的 `finetuned_model.bin`：已于 2026-09-11 完成并验证，
   当前工作副本路径为 `outputs/stage1/modelA/finetuned_model.bin`；训练参数、指标、
   大小和 SHA-256 见 `docs/SMB_20CLASS_FINETUNE_REPAIR.md`。

官方 README 给出的预训练模型链接位于
`tf_runtime/code/README.md`。官方下载对象实际是 ZIP，不能直接改名为 `.bin`；
必须提取指定成员并核对 `tf_runtime/PRETRAINED_MODEL.md` 中的完整 SHA-256。

## 使用方式

训练 Model A 时，将仓库内源码加入 `PYTHONPATH`，再调用官方训练入口：

```bash
PYTHONPATH="$PWD/tf_runtime/code" python \
  tf_runtime/code/fine-tuning/run_classifier.py \
  --vocab_path tf_runtime/code/models/encryptd_vocab.txt \
  --config_path tf_runtime/code/models/bert/base_config.json \
  --pretrained_model_path <pretrained_model.bin> \
  --train_path data/splits/compatible_min1/train_dataset.tsv \
  --dev_path data/splits/compatible_min1/valid_dataset.tsv \
  --test_path data/splits/compatible_min1/test_dataset.tsv \
  --output_model_path outputs/stage1/modelA/finetuned_model.bin \
  --epochs_num 3 --earlystop 3 --batch_size 32 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible \
  --seq_length 320 --learning_rate 6e-5 --pooling first --seed 7
```

导出 `z_t` 时使用新增的仓库内启动器，而不是直接调用原 Task 08：

```bash
python scripts/task08a_export_model_a_zt_repo.py \
  --vocab_path tf_runtime/code/models/encryptd_vocab.txt \
  --config_path tf_runtime/code/models/bert/base_config.json \
  --pretrained_model_path outputs/stage1/modelA/finetuned_model.bin \
  --train_path data/splits/compatible_min1/train_dataset.tsv \
  --dev_path data/splits/compatible_min1/valid_dataset.tsv \
  --test_path data/splits/compatible_min1/test_dataset.tsv \
  --flow-map-path data/trafficformer_input/compatible_min1/flow_map.csv \
  --split-dir data/splits/compatible_min1 \
  --out-dir outputs/stage1/modelA/embeddings \
  --seq_length 320 --batch_size 64 --device cuda
```

当前本地训练产物已经完成结构、有限数值、日志退出状态和独立测试集复核；这不等于
作者提供了同一 20 类 checkpoint。由于共享 GPU 显存约束，实际全局 batch size 从
计划的 64 调整为 32，完整复现边界见 `docs/SMB_20CLASS_FINETUNE_REPAIR.md`。
