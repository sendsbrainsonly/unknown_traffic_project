# VNAT 同协议 Open-Detect Native 与三路方法对照记录

记录日期：2026-09-29。实验状态：`COMPLETE / INDEPENDENT_REPLAY_PASS`。本文件是已有 Stage 44B 结果的固定摘要，**没有重新训练或重新评估**。

## 数据协议与指标口径

- 数据：VNAT clean pool 的 23,449 个唯一 flow UID；四个 Service-LOSO fold，seed 2022。每折留出一个完整 Service 作为 Unknown，Known Train/Validation/Test 按 capture/group 分离。
- 两方法的 Known Test、Unknown Test 样本及 flow UID 完全一致。Unknown 为正类；各方法分别使用 Known Validation P95 阈值，判定为 `score > threshold`。不使用 Unknown/Test 选模型或阈值。
- 下表使用 Test 的自然类别比例。UFAR 为 Unknown 被错误接纳为 Known 的比例；Known FRR 为 Known 被错误拒绝的比例；Unknown F1 是二分类 F1，不是 Known Macro-F1。

| Unknown Service | Known Test | Unknown Test | Known Macro-F1 OD / 三路 | AUROC OD / 三路 | AUPRC OD / 三路 | Unknown F1 OD / 三路 | UFAR OD / 三路 | Known FRR OD / 三路 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| communication | 1,380 | 2,208 | 0.9968 / 1.0000 | 0.8176 / 0.9954 | 0.8591 / 0.9955 | 0.3424 / 0.9848 | 0.7889 / 0.0005 | 0.0348 / 0.0486 |
| file_transfer | 954 | 5,870 | 0.9969 / 0.9969 | 0.5997 / 0.9948 | 0.8367 / 0.9977 | 0.0030 / 0.9934 | 0.9985 / 0.0056 | 0.0996 / 0.0461 |
| remote_access | 1,008 | 13,607 | 0.9941 / 1.0000 | 0.5888 / 0.9876 | 0.9216 / 0.9991 | 0.0574 / 0.9798 | 0.9704 / 0.0376 | 0.0238 / 0.0288 |
| streaming | 1,461 | 1,764 | 0.9953 / 0.9979 | 0.9607 / 0.9911 | 0.9291 / 0.9876 | 0.9064 / 0.9778 | 0.1298 / 0.0000 | 0.0602 / 0.0548 |
| **四折等权均值** | — | — | **0.9958 / 0.9987** | **0.7417 / 0.9922** | **0.8866 / 0.9950** | **0.3273 / 0.9840** | **0.7219 / 0.0109** | **0.0546 / 0.0445** |

OD 指新训练的 Open-Detect Native；“三路”指 Stage 44 冻结的三路等权表示 + DES-v1。四折等权均值的三路减 OD：AUROC `+0.2505`、AUPRC `+0.1084`、Unknown F1 `+0.6567`、UFAR `−0.7110`、Known FRR `−0.0101`。这些是四折的描述性差值，不是跨独立数据集的置信区间。

## 比例敏感性与关键现象

remote_access 的 Unknown Test 有 13,607 条、Known Test 仅 1,008 条，自然比例 AUPRC 受正类占比影响较大。在与 Stage 44 完全相同的确定性 1:1 评估子集上，四折等权 AUROC 为 OD `0.7417` / 三路 `0.9925`，AUPRC 为 `0.6824` / `0.9893`，Unknown F1 为 `0.3202` / `0.9732`；排序和 P95 工作点的主要差异仍存在。

file_transfer 和 remote_access 是 OD Native 的明显失败折：在 P95 下，Unknown 的误接纳率分别为 `0.9985` 和 `0.9704`，但对应 Known Macro-F1 仍为 `0.9969` 和 `0.9941`。这说明本协议中的闭集分类质量不能代替未知类拒识证据。Streaming 是 OD 最接近三路的一折，三路 AUROC 仍高 `0.0304`。

## 可复核性与解释边界

- 四个 OD checkpoint 都完成 100 epoch，仅 Known Validation Accuracy 决定最佳 epoch；逐样本原始分数、预测与阈值保留。独立重放 24 条 Native 指标、24 条配对行、36,442 条逐样本记录，最大绝对差为 `0`；四个 checkpoint SHA256 和冻结协议哈希均通过。
- 首轮评估适配层与原生 PIL/ToTensor 路径有 1 条验证样本预测差异，触发校验失败。失败输出保留于 `evaluation_attempt1/`，修复后的四折评估统一使用原生路径；前三折 checkpoint 未重训。
- 这是**同协议、同测试流**对照，不是同特征或同 encoder 的消融。只能说“三路完整方法”在本次四折上更好，不能将全部提升归因于 DES-v1 检测头。VNAT 已用于方法开发，本次单 seed 结果不能称为 untouched external validation；group/capture 数量不均衡也是限制。

原始证据：[逐折配对 CSV](matched_comparison.csv)、[方法均值 CSV](method_summary.csv)、[闭集 CSV](closed_set_comparison.csv)、[Unknown application 明细](per_unknown_application_comparison.csv)、[独立重放核验](independent_replay_verification.json)、[完整实验报告](RESULTS.md)。冻结 Stage 44 协议 SHA256：`0fd8e1bea6a326fc31fb0ee8405c47d6551df131e2694d34a11869a935667792`。
