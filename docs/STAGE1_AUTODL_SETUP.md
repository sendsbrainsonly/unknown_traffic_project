# Stage 1 AutoDL 服务器操作指南（V100）

目标：在 AutoDL V100 上跑 Model A（TrafficFormer 官方微调，20 类全量封闭集），
拿到 closed-set 基线；随后 Model B（TAGCN）与 Model C（融合）。

## 0. 上传文件（本地执行一次）

打包产物（本地 D 盘根目录）：

| 文件 | 内容 | 大小 |
|---|---|---|
| `autodl_stage1_data.tar.gz` | compatible_min1 的 TSV+flow_map+label_map、fig_graph/all_flows、data/splits（主划分+smoke）、configs | ~250MB（压缩后） |
| `autodl_tf_code.tar.gz` | 老师仓库 code/（官方代码+修改+requirements+**官方 pretrained_model.bin**，SHA-256 be9dcc1e…已验证） | ~684MB |

上传（本地 Git Bash，AutoDL 实例页面的"登录指令"里有 host/port）：

```bash
scp -P <端口> /d/autodl_stage1_data.tar.gz root@<host>:/root/autodl-tmp/
scp -P <端口> /d/autodl_tf_code.tar.gz   root@<host>:/root/autodl-tmp/
```

（配好 SSH 免密后可由 Claude 代传；VSCode 远程窗口拖拽亦可，scp 更稳。）

## 1. 环境验证（服务器）

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：V100、torch 2.0.1、True（镜像：PyTorch 2.0.1 / Python 3.8 / CUDA 11.8）
```

镜像 torch 版本不对（≠2.0.1）时：pip install torch==2.0.1 torchvision==0.15.1
（用 AutoDL 国内镜像源，注意是 CUDA 11.8 版）。

## 2. 解压 + 装依赖

```bash
cd /root/autodl-tmp
tar xzf autodl_stage1_data.tar.gz
tar xzf autodl_tf_code.tar.gz
cd code && pip install -r requirements.txt   # torch 2.0.1 已装会自动跳过
```

解压后结构：`/root/autodl-tmp/data/…`、`/root/autodl-tmp/code/…`

## 3. 冒烟测试（2 类 400 流，1 epoch，几分钟）

```bash
cd /root/autodl-tmp
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python -u code/fine-tuning/run_classifier.py \
  --vocab_path code/models/encryptd_vocab.txt \
  --train_path data/splits/smoke_gmail_zeus/train_dataset.tsv \
  --dev_path   data/splits/smoke_gmail_zeus/valid_dataset.tsv \
  --test_path  data/splits/smoke_gmail_zeus/test_dataset.tsv \
  --pretrained_model_path code/models/pretrained_model.bin \
  --output_model_path smoke_finetuned.bin \
  --config_path code/models/bert/base_config.json \
  --epochs_num 1 --earlystop 1 --batch_size 32 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible \
  --seq_length 320 --learning_rate 6e-5 --pooling first --seed 7 \
  2>&1 | tee smoke.log
```

通过判据：无 Traceback/OOM；日志末尾出现 `Test set evaluation.` 和有限数值指标；
准确率显著高于 50%（2 类随机水平）且损失在下降。

## 4. 正式训练 Model A（20 类全量，先 3 epoch）

```bash
cd /root/autodl-tmp
mkdir -p runs/modelA_ustc20_min1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python -u code/fine-tuning/run_classifier.py \
  --vocab_path code/models/encryptd_vocab.txt \
  --train_path data/splits/compatible_min1/train_dataset.tsv \
  --dev_path   data/splits/compatible_min1/valid_dataset.tsv \
  --test_path  data/splits/compatible_min1/test_dataset.tsv \
  --pretrained_model_path code/models/pretrained_model.bin \
  --output_model_path runs/modelA_ustc20_min1/finetuned_model.bin \
  --config_path code/models/bert/base_config.json \
  --epochs_num 3 --earlystop 3 --batch_size 64 \
  --embedding word_pos_seg --encoder transformer --mask fully_visible \
  --seq_length 320 --learning_rate 6e-5 --pooling first --seed 7 \
  2>&1 | tee runs/modelA_ustc20_min1/train.log
```

**2026-09-05 实测：单卡 V100-32GB 上 batch 64 会在第一个反向传播 CUDA OOM**
（30.23/31.73GiB，崩溃日志存 runs/modelA_ustc20_min1/train_oom_b64.log）。
两种解法：① 单卡 → `--batch_size 32 --gradient_accumulation_steps 2`（有效 batch 仍 64）；
② 双卡（2026-09-05 用户升级 2× V100-32GB 后采用）→ `CUDA_VISIBLE_DEVICES=0,1` + batch 64，
官方代码 device_count>1 自动 DataParallel，每卡分 32，正是冒烟验证过的显存足迹。
老师在他们服务器上 batch 64 能跑是因为显存更大。

超参说明：seed 7 / First pooling / seq 320 / batch 64 / lr 6e-5 沿用老师复现配方；
epoch 先 3（489k 流 batch64 ≈ 6,100 step/epoch，双卡 V100 粗估 0.7-1.3 小时/epoch），
看 val 曲线再决定是否加。labels_num 由官方代码自动从数据计数，无需传。

后台运行（断 SSH 不杀进程；容器无 systemd。本地 ssh 直接起后台任务会挂起，
用子壳 + setsid 包裹、末尾 exit 0 可干净返回）：

```bash
cd /root/autodl-tmp && (setsid nohup bash -c 'export PYTHONPATH=/root/autodl-tmp/code; CUDA_VISIBLE_DEVICES=0,1 /root/miniconda3/bin/python -u code/fine-tuning/run_classifier.py [同上全部参数]' < /dev/null > runs/modelA_ustc20_min1/train.log 2>&1 &)
```

## 5. 完成判定与收结果（沿用老师规范）

1. 退出码 0；2. `train.log` 含 `Test set evaluation.` 且指标有限无 NaN；
3. `finetuned_model.bin` 非空可加载；4. 无 Traceback/OOM/Killed。

结果回传本地存档（D:\unknown_traffic_project\outputs\stage1\modelA\）：
train.log + finetuned_model.bin + 测试段指标。

训练完成后**立即关机**（计费按小时），数据盘保留，下次直接开机续用。

## 6. 基线参照（无需重跑）

老师已复现的官方管线 USTC 基线（14 类/6,122 流/官方过滤）：
Acc 0.9690、Macro-F1 0.9731（TF）/ 0.9772（w/ EA）；论文 0.9784 / 0.9830。
我们的主设定是 20 类全量 489,101 流，数字不可直接比较，报告中并列说明。
