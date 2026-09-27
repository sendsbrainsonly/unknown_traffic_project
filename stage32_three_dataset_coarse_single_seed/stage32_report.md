# Stage 32 — 三数据集粗粒度三路等权闭集实验

Status: completed diagnostic; all scores are on the frozen **Known-only closed-set** task.

## 单 seed 结果

| 数据集 | 类别数 | Known Test 流数 | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| iscx_vpn | 4 | 1093 | 0.917767 | 0.915828 | 0.919350 | 0.914487 |
| iscx_tor | 5 | 1117 | 0.909662 | 0.896150 | 0.899375 | 0.896717 |
| vnat | 4 | 1960 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## 同流、同粗标签配对

| 数据集 | 评价集 | 历史对照 | 历史 Macro-F1 | 新 T0 Macro-F1 | ΔMacro-F1 | 95% bootstrap CI |
|---|---|---|---:|---:|---:|---:|
| iscx_vpn | known_test | E3_hard_remap | 0.899913 | 0.919350 | +0.019438 | [+0.007504, +0.031536] |
| iscx_vpn | known_test | E3_coarse_head | 0.900100 | 0.919350 | +0.019250 | [+0.006860, +0.032669] |
| iscx_vpn | known_test | YaTC_hard_remap_context | 0.929493 | 0.919350 | -0.010142 | [-0.022482, +0.001351] |
| iscx_tor | known_test | E3_hard_remap | 0.853614 | 0.899375 | +0.045761 | [+0.026430, +0.065705] |
| iscx_tor | known_test | E3_coarse_head | 0.855214 | 0.899375 | +0.044161 | [+0.023698, +0.063876] |
| iscx_tor | known_test | YaTC_hard_remap_context | 0.871463 | 0.899375 | +0.027912 | [+0.016400, +0.039672] |
| vnat | known_validation | historical_F2_hard_remap | 0.994538 | 1.000000 | +0.005462 | [+0.000000, +0.013608] |

## 解释边界

- ISCX 两项 Test 对照是 seed 2022 的 exact-flow 历史结果；VNAT 历史 F2 仅与 Known Validation 配对，且其训练 seed 为 2025。VNAT Test 没有经同流冻结重放验证的旧 F2 对照，不能声称 Test 相对 F2 的增益。
- 粗标签抹平了部分原细类错误；高分不是原细粒度任务性能提升，也不是开集性能。三路原 fine encoder 与 fine head 保留。
- VNAT 按 flow 随机划分，同 capture 可能同时在 Train/Val/Test，且标签主要来自 capture；特别高的分数不证明跨 capture 或跨环境泛化。
- 三个 Test 曾在历史项目中暴露，因此是 development evidence；仅一个训练 seed，不能宣称稳定多 seed 优势。
- 所有对照和主模型均未使用 Unknown 流拟合；Test 未参与 checkpoint、阈值或参数选择。

完整逐流预测、逐类指标、训练曲线和 checkpoint 位于 `runs/`；独立重放见 `completion_verification.json`。
