# Stage 44B — 同协议 Open-Detect Native 对照结果

- 状态：`COMPLETE / INDEPENDENT_REPLAY_PASS`；2026-09-29。

## Data and split

- 范围：VNAT 四个冻结 Service-LOSO fold、seed 2022，23,449 个唯一 flow UID。四折各留出一个完整 Service 为 Unknown；Known Train/Validation/Test 的 capture/group 不交叉。
- 两方法使用完全相同的冻结角色、Known Test 和 Unknown Test flow UID。各自只用 Known Validation 的 P95 校准阈值；Unknown 为正类；判定规则均为 `score > threshold`。下表为自然样本比例、P95。

## Core results

| Unknown Service | Known Test / Unknown Test | Known Macro-F1 OD / 三路 | AUROC OD / 三路 | AUPRC OD / 三路 | Unknown F1 OD / 三路 | UFAR OD / 三路 | Known FRR OD / 三路 |
|---|---:|---:|---:|---:|---:|---:|---:|
| communication | 1,380 / 2,208 | 0.9968 / 1.0000 | 0.8176 / 0.9954 | 0.8591 / 0.9955 | 0.3424 / 0.9848 | 0.7889 / 0.0005 | 0.0348 / 0.0486 |
| file_transfer | 954 / 5,870 | 0.9969 / 0.9969 | 0.5997 / 0.9948 | 0.8367 / 0.9977 | 0.0030 / 0.9934 | 0.9985 / 0.0056 | 0.0996 / 0.0461 |
| remote_access | 1,008 / 13,607 | 0.9941 / 1.0000 | 0.5888 / 0.9876 | 0.9216 / 0.9991 | 0.0574 / 0.9798 | 0.9704 / 0.0376 | 0.0238 / 0.0288 |
| streaming | 1,461 / 1,764 | 0.9953 / 0.9979 | 0.9607 / 0.9911 | 0.9291 / 0.9876 | 0.9064 / 0.9778 | 0.1298 / 0.0000 | 0.0602 / 0.0548 |
| **四折等权均值** | — | **0.9958 / 0.9987** | **0.7417 / 0.9922** | **0.8866 / 0.9950** | **0.3273 / 0.9840** | **0.7219 / 0.0109** | **0.0546 / 0.0445** |

“三路”指 Stage 44 已冻结的三路等权表示及 DES-v1；OD 指本次重新训练的 Open-Detect Native。三路在四折的 AUROC 都更高，等权平均差 `+0.2505`；Unknown F1 高 `+0.6567`，UFAR 低 `0.7110`。闭集 Known Macro-F1 两者均很高，差约 `0.0029`，但开集表现差别显著。尤其 file_transfer、remote_access 的 OD Native 在 P95 下几乎不拒识 Unknown。Streaming 是 OD 最接近三路的一折，但三路仍占优。

自然比例的 AUPRC 会受到 Unknown 占比影响，特别是 remote_access。使用与 Stage 44 完全一致的确定性 1:1 子集，四折等权均值 AUROC 为 OD `0.7417` / 三路 `0.9925`，AUPRC 为 `0.6824` / `0.9893`，Unknown F1 为 `0.3202` / `0.9732`；结论不依赖自然类别比例。逐折原始数据见 `matched_comparison.csv`；类别级数据见 `per_unknown_application_comparison.csv`。

## Configuration and execution

- 四个 OD 模型使用原生 ResNet-18 VAE、128 维 latent、F0 32×32 输入和未改动的 Stage 14C Native 训练循环：Adam 0.001、batch 128/64、100 epochs、lambda 0.005、MultiStepLR 50/80、原型重置和 seed 2022。最佳 epoch 依次为 `13 / 40 / 49 / 46`，只根据 Known Validation Accuracy 选 checkpoint。
- 四折仅 Known Train/Validation 进入训练与 checkpoint 选择。Unknown/Test 不用于训练、归一化、阈值或模型选择。阈值仅来自 Known Validation。四个 checkpoint 及逐样本分数均保留。
- 第一轮评估在 remote_access 因 1 个临界验证样本的预处理差异触发校验失败：直接 uint8→Tensor 与原生 PIL/ToTensor 得到不同预测。原生加载器与训练保存的混淆矩阵完全相同。失败日志和首轮评估保存在 `evaluation_attempt1/`；仅修改新建评估适配层，统一按 Native PIL/ToTensor 重算四折，没有重训前三折模型，也没有修改 Stage 44/14B 冻结资产或 Open-Detect 原生源码。
- 独立逐样本重放 `PASS`：24 条 Native 指标、24 条同流配对行和 36,442 条逐样本记录，指标最大绝对差 `0`；4 个 checkpoint SHA256 通过。Stage 44 freeze SHA256 前后均为 `0fd8e1bea6a326fc31fb0ee8405c47d6551df131e2694d34a11869a935667792`。

## Limitations

这是**同一数据协议**的重新训练对照，但不是同一 encoder 或同一输入特征：OD Native 使用其 F0 图像与学习原型 KL 分数，三路方法使用 Stage 44 的三路表示和 DES-v1。因此结果证明三路完整方法在这四个服务留出任务上更强，**不能单凭此实验把全部差距归因于检测头/支持解耦**。VNAT 已参与方法开发，本结果不是 untouched external validation；只有一个 seed，remote_access/streaming 还存在已记录的 capture/group 不均衡。未根据 Test 重选服务类别、阈值、特征或模型。

## Preserved evidence

- `od_run_results.csv`：四折 × 自然/1:1 × P90/P95/P99 的 Native 指标。
- `matched_comparison.csv`、`closed_set_comparison.csv`、`method_summary.csv`：同流配对、闭集和均值汇总。
- `per_unknown_application_comparison.csv`：rsync/scp/sftp/ssh 等 Unknown application 明细。
- `evaluation/*/sample_scores.csv`：逐样本 Native 原始分数及 P95 决策。
- `completion_verification.json`、`independent_replay_verification.json`：哈希、角色与重放验证。
- `queue_failure.json`、`resume_progress.json`、`evaluation_attempt1/`：保留失败及修复证据。
- `COMPARISON_RECORD.md`：便于教学和后续论文核对的同协议逐折数据快照。

## Conclusion and next step

本次四折同协议对照完成：三路完整方法的开集指标稳定高于 Open-Detect Native，但由于 encoder/输入特征不同，不能把优势单独归因于检测机制。本阶段到此停止，不据此自动修改方法或启动下一阶段。
