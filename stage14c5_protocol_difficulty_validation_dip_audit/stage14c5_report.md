# Stage 14C-5 — VNAT Protocol Difficulty & Validation Dip Audit

## 审计结论

**`PIPELINE_READY_WITH_KNOWN_TRANSIENT`**

Medium-2025 与 Medium-2026 的闭集差距主要由 **Known 类组成难度**导致，而不是训练随机种子。Medium-2025 同时把 `rsync` 与 `scp` 作为 Known；其最佳验证 checkpoint 中，`rsync` 的 191 个样本有 185 个被预测成 `scp`。Medium-2026 将 `scp` 整类留作 Unknown，`rsync` 达到 191/191 正确。

观测回放与原 Stage 14C-4 逐 epoch 核心指标严格一致，最大绝对差异为 `1.110e-16`。大幅 validation drop 没有与 epoch 51/81 的 scheduler/reset 同步；reset 会造成 prototype norm 的预期离散跳变，但 reset 当轮 Macro-F1 变化小且方向不一致，随后恢复。因此属于已知、可恢复的验证波动，不构成新的实现错误。

## 1. 边界与完整性

- Stage 14B freeze hash：`c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`
- Known Test 使用量：`0`
- Unknown Test 使用量：`0`
- DES 执行：`False`
- 原训练代码/配置修改：`False`
- 两个 telemetry replay parity：`True`
- 本回放只增加只读观测；没有参与 checkpoint 选择，也没有改变训练随机状态。

## 2. Frozen protocol 组成

| protocol | Known | Unknown | Train | Val |
|---|---|---|---:|---:|
| Medium-2025 | netflix|rdp|rsync|scp|skype|ssh|youtube | zoiper|vimeo|sftp | 15704 | 1960 |
| Medium-2026 | netflix|rdp|rsync|skype|ssh|vimeo|zoiper | sftp|youtube|scp | 15328 | 1911 |

### 每个 Known 类 Train / Validation 数量

| protocol_id     | application   |   train_samples |   validation_samples |
|:----------------|:--------------|----------------:|---------------------:|
| medium_seed2025 | netflix       |             165 |                   20 |
| medium_seed2025 | rdp           |              36 |                    4 |
| medium_seed2025 | rsync         |            1530 |                  191 |
| medium_seed2025 | scp           |            1832 |                  229 |
| medium_seed2025 | skype         |            1017 |                  126 |
| medium_seed2025 | ssh           |           10851 |                 1356 |
| medium_seed2025 | youtube       |             273 |                   34 |
| medium_seed2026 | netflix       |             165 |                   20 |
| medium_seed2026 | rdp           |              36 |                    4 |
| medium_seed2026 | rsync         |            1530 |                  191 |
| medium_seed2026 | skype         |            1017 |                  126 |
| medium_seed2026 | ssh           |           10851 |                 1356 |
| medium_seed2026 | vimeo         |             976 |                  121 |
| medium_seed2026 | zoiper        |             753 |                   93 |

`rdp` 在两组中均仅有 36/4 个 Train/Val 样本，但两个 selected checkpoint 的 rdp F1 分别为 0.8571 和 1.0000；它会增加方差，却不是两组约 0.25 Macro-F1 差距的主因。`sftp` 在两组均为 Unknown，不参与训练/验证，也不是该差距来源。

## 3. 闭集指标与 seed 对照

| protocol_id     |   protocol_training_seed |   primary_val_accuracy |   primary_val_macro_f1 |   primary_val_weighted_f1 |   fixed2022_val_macro_f1 |   within_protocol_seed_delta_macro_f1 |   best_epoch |   epochs_completed |
|:----------------|-------------------------:|-----------------------:|-----------------------:|--------------------------:|-------------------------:|--------------------------------------:|-------------:|-------------------:|
| medium_seed2025 |                     2025 |               0.892857 |               0.727247 |                  0.864414 |                 0.766133 |                              0.038887 |           93 |                100 |
| medium_seed2026 |                     2026 |               0.994767 |               0.980668 |                  0.994803 |                 0.980992 |                              0.000324 |           53 |                100 |

- 主 runs 的 Medium-2026 − Medium-2025 Macro-F1：`0.253421`。
- 固定相同训练 seed=2022 后差距仍为：`0.214858`。
- 相同 seed 下仍保留主差距的 `84.78%`，所以协议类别组成是主因，随机 seed 是次要因素。

## 4. Selected checkpoint 类别级结果

| protocol_id     | class   |   train_support |   val_support |   precision |   recall |       f1 |
|:----------------|:--------|----------------:|--------------:|------------:|---------:|---------:|
| medium_seed2025 | netflix |             165 |            20 |    0.722222 | 0.65     | 0.684211 |
| medium_seed2025 | rdp     |              36 |             4 |    1        | 0.75     | 0.857143 |
| medium_seed2025 | rsync   |            1530 |           191 |    0.375    | 0.031414 | 0.057971 |
| medium_seed2025 | scp     |            1832 |           229 |    0.542079 | 0.956332 | 0.691943 |
| medium_seed2025 | skype   |            1017 |           126 |    0.976744 | 1        | 0.988235 |
| medium_seed2025 | ssh     |           10851 |          1356 |    1        | 0.999263 | 0.999631 |
| medium_seed2025 | youtube |             273 |            34 |    0.8      | 0.823529 | 0.811594 |
| medium_seed2026 | netflix |             165 |            20 |    0.863636 | 0.95     | 0.904762 |
| medium_seed2026 | rdp     |              36 |             4 |    1        | 1        | 1        |
| medium_seed2026 | rsync   |            1530 |           191 |    0.989637 | 1        | 0.994792 |
| medium_seed2026 | skype   |            1017 |           126 |    0.991935 | 0.97619  | 0.984    |
| medium_seed2026 | ssh     |           10851 |          1356 |    0.997788 | 0.997788 | 0.997788 |
| medium_seed2026 | vimeo   |             976 |           121 |    0.991597 | 0.975207 | 0.983333 |
| medium_seed2026 | zoiper  |             753 |            93 |    1        | 1        | 1        |

### 非零混淆项

| protocol_id     | true_class   | predicted_class   |   count |
|:----------------|:-------------|:------------------|--------:|
| medium_seed2025 | rsync        | scp               |     185 |
| medium_seed2025 | scp          | rsync             |      10 |
| medium_seed2025 | netflix      | youtube           |       7 |
| medium_seed2025 | youtube      | netflix           |       5 |
| medium_seed2025 | rdp          | skype             |       1 |
| medium_seed2025 | ssh          | skype             |       1 |
| medium_seed2025 | youtube      | skype             |       1 |
| medium_seed2026 | skype        | ssh               |       3 |
| medium_seed2026 | vimeo        | netflix           |       3 |
| medium_seed2026 | ssh          | rsync             |       2 |
| medium_seed2026 | netflix      | vimeo             |       1 |
| medium_seed2026 | ssh          | skype             |       1 |

Medium-2025 的性能损失高度集中在 `rsync ↔ scp`，尤其是 `rsync → scp = 185`；不是所有类别都普遍退化。Medium-2026 没有 `scp` 参与 Known 决策，rsync/scp 的语义边界冲突不再出现。

## 5. Validation dip 与 reset/scheduler 对应关系

实现顺序经原代码确认：每轮先训练；epoch 51/81 在验证前原位重置 prototype 并清除该参数的 optimizer state；验证完成后才调用 `scheduler.step()`。因此 epoch 51/81 使用前一轮 milestone 后已经降低的 LR，同时也发生 reset。

### 显著下降汇总

“显著下降”预注册为：相邻轮 Macro-F1 或 Accuracy 下降至少 0.15，或 validation loss 至少增至 3 倍且绝对增加至少 0.10。

| protocol_id     |   obvious_drop_count |   obvious_drops_at_reset_epochs |   obvious_drops_before_first_lr_reduction |   obvious_drops_after_first_lr_reduction |   recovered_within_3_epochs_count |   reset_epoch_51_delta_macro_f1 |   reset_epoch_81_delta_macro_f1 |   reset_epoch_51_prototype_norm_jump |   reset_epoch_81_prototype_norm_jump |   max_val_macro_f1_after_epoch_82 |
|:----------------|---------------------:|--------------------------------:|------------------------------------------:|-----------------------------------------:|----------------------------------:|--------------------------------:|--------------------------------:|-------------------------------------:|-------------------------------------:|----------------------------------:|
| medium_seed2025 |                    4 |                               0 |                                         4 |                                        0 |                                 2 |                        0.112349 |                       -0.036812 |                              5.40801 |                             1.6078   |                          0.727247 |
| medium_seed2026 |                    7 |                               0 |                                         7 |                                        0 |                                 5 |                       -0.031978 |                        0.033186 |                              6.73937 |                             0.894947 |                          0.96435  |

两个 exact-config primary replay 中，显著下降全部发生在首次 LR 降低前，epoch 51/81 均没有命中显著下降规则。reset 当轮 Macro-F1 有正有负，且负变化仅约 0.03–0.04；这与“reset 导致大幅崩塌”的假设不符。

对 Stage 14C-4 的四条完整历史轨迹（两条 primary + 两条 fixed-2022 control）使用同一规则复核，共发现 24 个显著下降事件：Medium-2025 为 4/6 个（primary/control），Medium-2026 为 7/7 个；其中 reset epoch 为 0 个，epoch 51 以后为 0 个。该结果不是单一训练 seed 的偶然现象。

### epoch 49–52 与 79–82 原始观测

| protocol_id     |   epoch |   learning_rate_used | prototype_reset   |   val_loss |   val_accuracy |   val_macro_f1 |   delta_val_macro_f1 |   prototype_norm_after_train |   prototype_norm_after_reset |   prototype_norm_reset_delta |   prototype_gradient_norm_mean |   encoder_gradient_norm_mean |
|:----------------|--------:|---------------------:|:------------------|-----------:|---------------:|---------------:|---------------------:|-----------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|-----------------------------:|
| medium_seed2025 |      49 |               0.001  | False             |   0.211492 |       0.886735 |       0.601356 |             0.135328 |                      3.70353 |                      3.70353 |                     0        |                       0.701575 |                     0.484951 |
| medium_seed2025 |      50 |               0.001  | False             |   0.254191 |       0.877041 |       0.492722 |            -0.108634 |                      3.73441 |                      3.73441 |                     0        |                       0.715783 |                     0.706185 |
| medium_seed2025 |      51 |               0.0001 | True              |   0.207341 |       0.870408 |       0.605071 |             0.112349 |                      3.73786 |                      9.14587 |                     5.40801  |                       0.700963 |                     0.420992 |
| medium_seed2025 |      52 |               0.0001 | False             |   0.19727  |       0.888776 |       0.659034 |             0.053963 |                      9.14405 |                      9.14405 |                     0        |                       0.682983 |                     1.10848  |
| medium_seed2025 |      79 |               0.0001 | False             |   0.196534 |       0.888776 |       0.661929 |            -0.010305 |                      9.12123 |                      9.12123 |                     0        |                       0.686545 |                     1.17381  |
| medium_seed2025 |      80 |               0.0001 | False             |   0.198469 |       0.893367 |       0.702785 |             0.040856 |                      9.12044 |                      9.12044 |                     0        |                       0.683755 |                     1.14716  |
| medium_seed2025 |      81 |               1e-05  | True              |   0.223444 |       0.87551  |       0.665973 |            -0.036812 |                      9.12054 |                     10.7283  |                     1.6078   |                       0.677066 |                     0.744184 |
| medium_seed2025 |      82 |               1e-05  | False             |   0.196865 |       0.889286 |       0.694931 |             0.028958 |                     10.7278  |                     10.7278  |                     0        |                       0.683867 |                     1.16339  |
| medium_seed2026 |      49 |               0.001  | False             |   0.103274 |       0.977499 |       0.908034 |            -0.038435 |                      4.48301 |                      4.48301 |                     0        |                       0.144801 |                     0.321729 |
| medium_seed2026 |      50 |               0.001  | False             |   0.046463 |       0.993197 |       0.96354  |             0.055505 |                      4.52931 |                      4.52931 |                     0        |                       0.136301 |                     0.230953 |
| medium_seed2026 |      51 |               0.0001 | True              |   0.06381  |       0.993197 |       0.931561 |            -0.031978 |                      4.53507 |                     11.2744  |                     6.73937  |                       0.116203 |                     0.177063 |
| medium_seed2026 |      52 |               0.0001 | False             |   0.049705 |       0.993721 |       0.959074 |             0.027512 |                     11.2686  |                     11.2686  |                     0        |                       0.093008 |                     0.488141 |
| medium_seed2026 |      79 |               0.0001 | False             |   0.069202 |       0.994767 |       0.931045 |            -0.032456 |                     11.2768  |                     11.2768  |                     0        |                       0.058113 |                     0.277764 |
| medium_seed2026 |      80 |               0.0001 | False             |   0.071673 |       0.994244 |       0.917175 |            -0.01387  |                     11.2785  |                     11.2785  |                     0        |                       0.043047 |                     0.214304 |
| medium_seed2026 |      81 |               1e-05  | True              |   0.08169  |       0.994767 |       0.950361 |             0.033186 |                     11.2787  |                     12.1736  |                     0.894947 |                       0.04707  |                     0.241483 |
| medium_seed2026 |      82 |               1e-05  | False             |   0.074674 |       0.994767 |       0.950361 |             0        |                     12.1733  |                     12.1733  |                     0        |                       0.060851 |                     0.318508 |

prototype norm 在 reset 后明显跳变，这是重算 prototype 的直接结果；但梯度范数没有在 reset 当轮爆炸，且下一轮/随后数轮验证指标恢复。scheduler 降低 LR 后，大幅下降事件反而消失。

### 显著下降时类别 Recall 的最大变化

| protocol_id     |   drop_epoch | class   |   support |   previous_recall |   drop_epoch_recall |   delta_recall |
|:----------------|-------------:|:--------|----------:|------------------:|--------------------:|---------------:|
| medium_seed2025 |           21 | scp     |       229 |          0.9869   |            0        |      -0.9869   |
| medium_seed2026 |           10 | netflix |        20 |          0.95     |            0        |      -0.95     |
| medium_seed2025 |           44 | ssh     |      1356 |          0.999263 |            0.151917 |      -0.847345 |
| medium_seed2026 |            7 | netflix |        20 |          0.8      |            0        |      -0.8      |
| medium_seed2026 |           15 | netflix |        20 |          0.85     |            0.05     |      -0.8      |
| medium_seed2026 |            4 | netflix |        20 |          0.75     |            0        |      -0.75     |
| medium_seed2026 |           10 | rdp     |         4 |          0.75     |            0        |      -0.75     |
| medium_seed2026 |            4 | ssh     |      1356 |          0.878319 |            0.171829 |      -0.70649  |
| medium_seed2025 |            6 | youtube |        34 |          0.647059 |            0        |      -0.647059 |
| medium_seed2026 |           10 | ssh     |      1356 |          0.997788 |            0.365044 |      -0.632743 |
| medium_seed2026 |            7 | vimeo   |       121 |          0.842975 |            0.247934 |      -0.595041 |
| medium_seed2025 |           44 | rdp     |         4 |          0.75     |            0.25     |      -0.5      |
| medium_seed2025 |           44 | youtube |        34 |          0.970588 |            0.588235 |      -0.382353 |
| medium_seed2025 |            6 | scp     |       229 |          0.956332 |            0.598253 |      -0.358079 |
| medium_seed2025 |           21 | skype   |       126 |          1        |            0.722222 |      -0.277778 |
| medium_seed2025 |           33 | ssh     |      1356 |          0.999263 |            0.7441   |      -0.255162 |

下降主要表现为少数易混类别的决策边界短暂切换，而不是所有类别同步失效。训练 loss、梯度范数与数值检查均未显示 NaN 或系统性爆炸；这更符合高 LR 阶段、类别不平衡和难类边界共同造成的验证瞬态。

## 6. 必须回答的问题

1. **Medium-2025 与 Medium-2026 差距的主要原因**：Known class composition。Medium-2025 同时包含高度混淆的 rsync/scp；Medium-2026 把 scp 留作 Unknown。相同 seed 对照仍保留 84.8% 的主差距，随机 seed 不是主因。
2. **validation dip 是否由 scheduler/prototype reset 引起**：没有支持“大幅 dip 由它们触发”的证据。显著 dip 没有发生在 reset epoch；LR 降低后显著 dip 消失。reset 与 LR 下降在同一轮共现，不能用本观察审计完全分离二者的微小即时效应，但可以排除其造成已观察大幅崩塌。
3. **dip 是正常动态还是新实现问题**：属于可恢复的已知训练瞬态。exact replay parity 通过、没有 NaN/崩溃、没有 reset/梯度异常对应，且后续达到更高验证分数。
4. **清理后 pipeline 能否冻结**：可以，需把高 LR 阶段 validation 波动作为已知 transient 记录，不应因单轮 dip 提前停止或修改协议。
5. **能否进入完整 15-run 重跑**：从本审计的训练完整性和稳定性证据看可以；本阶段本身没有启动该重跑。

## 最终结论

```text
PIPELINE_READY_WITH_KNOWN_TRANSIENT
```

该结论不是声称训练轨迹无波动，而是确认波动可恢复、没有与 prototype reset 构成稳定因果对应，并且主要闭集差距可由冻结协议中的类别组合严格解释。
