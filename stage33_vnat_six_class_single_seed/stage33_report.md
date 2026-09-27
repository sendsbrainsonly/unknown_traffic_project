# VNAT 六类细化闭集结果

本轮是 Stage 32 四类 Known Test 满分之后，由用户提出的**事后任务难度诊断**。六类映射在本轮新模型训练前固定：`streaming={netflix,youtube}`，`rdp`、`rsync`、`scp`、`skype`、`ssh` 各单独成类。没有更换 Stage 14B `medium_seed2025` 的七个 Known application 或 15,704/1,960/1,960 个 Train/Val/Test flow；`zoiper/vimeo/sftp` 仍为 Unknown，未用于训练、归一化、选择或评价。原三路 encoder 不重新训练，只按 Stage 32 配置训练六类 T0 adapter/head，一次 seed 2022，30+30 epoch，Known Validation 选最佳模型。

| 任务 | Known Test 流数 | 类别数 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Stage 32 原四类 | 1,960 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Stage 33 本六类 | 1,960 | 6 | 0.963776 | 0.943109 | 0.963750 |

两个任务使用相同 flow membership，但标签空间和新训练的分类头不同，**不能把差值解释为模型性能下降**。六类任务的 Known Validation Accuracy/Macro-F1/Weighted-F1=`0.972449/0.956710/0.972422`。最佳 adapter/head epoch=`16/6`；完成全部 30+30 轮，无 early stopping。训练、Test 两个独立 tmux 任务均退出 0，物理 GPU 都为 0。

## Known Test 逐类结果

| 六类 | Train/Val/Test 流数 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| streaming | 438/54/54 | 1.0000 | 1.0000 | 1.0000 |
| rdp | 36/4/4 | 1.0000 | 1.0000 | 1.0000 |
| rsync | 1530/191/191 | 0.8191 | 0.8063 | 0.8127 |
| scp | 1832/229/229 | 0.8405 | 0.8515 | 0.8460 |
| skype | 1017/126/126 | 1.0000 | 1.0000 | 1.0000 |
| ssh | 10851/1356/1356 | 1.0000 | 1.0000 | 1.0000 |

71 条 Test 错误全部集中于 `rsync→scp` 37 条与 `scp→rsync` 34 条。可复核混淆矩阵见 `runs/vnat/known_test_evaluation/confusion_matrix.csv`，逐 flow 预测见同目录 `sample_predictions.csv`。这与此前 VNAT 困难类别分析一致：四类时 `rsync/scp` 同属 File Transfer，二者的相互错误被合并标签掩盖。

## 完整性与解释边界

独立脚本 `verify_six.py` 用冻结 manifest 真值和保存的 logits 重算 1,960 行 Accuracy/F1、argmax 与混淆矩阵，全部一致；所选两份 checkpoint SHA256 与评估前一致。Stage 32 frozen source 和 Test cache SHA256 未变；Stage 14B 四个受保护文件的 SHA256 与原始 freeze 记录一致，freeze hash 保持 `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`。

这是单 seed、已曝光 Test 上的 development 证据。VNAT 仍按 flow 随机划分，同一 capture 可能跨 Train/Val/Test；原 application 标签有 capture-derived 弱标签限制。RDP Test 仅 4 条，1.0 F1 的不确定性极高。不能据此声称跨 capture 泛化或 Unknown 检测改善，也不应为了让指标下降继续根据 Test 反复调标签。到此停止，不修改旧实验或启动下一阶段。
