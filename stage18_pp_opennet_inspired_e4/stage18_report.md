# Stage 18：PP-OpenNet 启发的独立 E4 包级编码器实验报告

## 1. 最终状态与复现边界

- 正式协议：`Email / Streaming × seeds 2022, 2023, 2024`，共 6 个 run。
- 受控变体：7 个，共训练并保存 42 个 checkpoint。
- 最终 Gate：`E4_FAIL_NO_FUSION`；E3+E4 融合未启动。
- Unknown 用于训练、验证、归一化、支持集、阈值或 checkpoint 选择的样本数：均为 `0`。

本实验是 PP-OpenNet 启发的独立近似实现，不是作者代码复现。论文能够确认包级时间/长度元数据、多尺度卷积和循环特征融合等总体思路，但未找到可核验的作者官方公开代码；无法从论文确认的实现细节均标记为项目自主设计。

## 2. 冻结数据与协议

复用 Stage 17 已冻结的 3,065-flow 缓存：

```text
stage17_encoder_recovery_and_open_set_pilot/feature_cache/
service3065_historical_inputs.npz
SHA256 = aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff
```

缓存可还原每条 flow 前 30 个真实包的相对时间、捕获长度、方向与 mask。没有重新下载/采集数据，也没有改变 Stage 16S/17 的 Known/Unknown 划分。

| 协议 | Known Train | Known Validation | Known Test | Unknown Test |
|---|---:|---:|---:|---:|
| Email Unknown | 2,585 | 155 | 157 | 168 |
| Streaming Unknown | 1,876 | 114 | 117 | 958 |

Known Test 和 Unknown Test 只用于最终评价；模型选择、归一化、类中心和阈值均只使用 Known Train/Validation。

## 3. 输入、模型与训练

每个有效包使用 `log1p(caplen)`、`log1p(non-negative IAT in microseconds)` 和方向符号 `+1/-1`。长度与 IAT 只用 Known Train 拟合 median/IQR，标准化后裁剪到 `[-8,8]`；方向不标准化。padding 标准化后置零，并在卷积、统计池化和循环模块中显式使用 mask。单元测试确认改变 padding 区域数值不会改变输出。

E4_FULL 从随机初始化：并行 Conv1d `k=1/3/5` 提取多尺度局部特征，masked mean/std/max 构成统计分支，packet GRU 与 fusion GRU 聚合序列，最终输出 128 维 flow embedding 和 5 类 Known 分类头。E4 未调用 Open-Detect encoder、训练循环、learned prototype 或 loss。

训练固定为 100 epochs、batch 128、AdamW、LR `1e-3`、weight decay `1e-4`、MultiStepLR `[50,80]`、inverse-sqrt class-weighted cross entropy；checkpoint 仅按 Known Validation Macro-F1、再按 Accuracy 选择。

## 4. 闭集结果

| Unknown 协议 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Email | 0.743100 ± 0.007944 | 0.749697 ± 0.012937 | 0.741966 ± 0.008488 |
| Streaming | 0.809117 ± 0.022433 | 0.798280 ± 0.019562 | 0.809736 ± 0.023025 |
| Overall 6 runs | 0.776108 | 0.773988 | 0.775851 |

E4_FULL 类别级均值：

| 协议 | Known service | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Email | Chat | 0.7704 | 0.7037 | 0.7271 |
| Email | File-Transfer | 0.5583 | 0.7273 | 0.6316 |
| Email | P2P | 0.9667 | 0.5400 | 0.6925 |
| Email | Streaming | 0.6520 | 0.8526 | 0.7389 |
| Email | VoIP | 0.9583 | 0.9583 | 0.9583 |
| Streaming | Chat | 0.7783 | 0.7407 | 0.7549 |
| Streaming | Email | 0.8727 | 0.7500 | 0.8063 |
| Streaming | File-Transfer | 0.6711 | 0.7424 | 0.7049 |
| Streaming | P2P | 0.8450 | 0.8000 | 0.8213 |
| Streaming | VoIP | 0.8706 | 0.9444 | 0.9040 |

相对 E0 Open-Detect Native，E4_FULL Macro-F1 的服务级均值变化为 Email `+0.051607`、Streaming `+0.103628`，总体 `+0.077618`。但均值包含 E0 的弱 seed，闭集提升不能直接解释为开集成功。

## 5. 开集结果

主分数为 Known-Train 标准化 embedding 到最近 Known-Train 类中心的平方距离，阈值由 Known Validation 经验分位数给出。

### 5.1 固定 Known Acceptance 目标

| Unknown | Val target | AUROC | AUPRC | UFAR | 实际 Known Test acceptance |
|---|---:|---:|---:|---:|---:|
| Email | 90% | 0.627692 | 0.646520 | 0.815476 | 0.917197 |
| Email | 95% | 0.627692 | 0.646520 | 0.882937 | 0.972399 |
| Email | 99% | 0.627692 | 0.646520 | 0.978175 | 0.989384 |
| Streaming | 90% | 0.560439 | 0.924559 | 0.739736 | 0.846154 |
| Streaming | 95% | 0.560439 | 0.924559 | 0.838900 | 0.920228 |
| Streaming | 99% | 0.560439 | 0.924559 | 0.897008 | 0.977208 |

Val-P95 工作点下，6-run 总体 AUROC=`0.594065`、AUPRC=`0.785539`、UFAR=`0.860918`、Known Test acceptance=`0.946314`。Streaming 的 AUPRC 受 958 Unknown 对 117 Known 的类别比例影响，不能替代 AUROC 与绝对 UFAR。

### 5.2 同一 embedding 的三种拒识分数

| Score | Overall AUROC | Overall AUPRC | UFAR@Val-P95 | Known Test acceptance |
|---|---:|---:|---:|---:|
| Feature distance（主分数） | 0.594065 | 0.785539 | 0.860918 | 0.946314 |
| MSP | 0.644934 | 0.809060 | 0.892859 | 0.931642 |
| Energy | 0.600162 | 0.764396 | 0.928735 | 0.816802 |

MSP 的排序性能高于 feature distance，但 Val-P95 UFAR 更差；Energy 也未改善工作点。这说明分数定义确实影响排序，但当前 embedding 的 support overlap 与 validation-to-test 阈值迁移也同时存在。

## 6. 与 E0、E3 的配对比较

| Unknown | 对手 | ΔMacro-F1 | ΔAUROC | ΔAUPRC | ΔUFAR@95 |
|---|---|---:|---:|---:|---:|
| Email | E0 Open-Detect Native | +0.051607 | +0.025276 | +0.040750 | -0.037698 |
| Streaming | E0 Open-Detect Native | +0.103628 | -0.139215 | -0.018141 | -0.025748 |
| Email | E3 DES-v1 Fusion | -0.157783 | -0.174161 | -0.116730 | +0.077381 |
| Streaming | E3 DES-v1 Fusion | +0.003195 | -0.160531 | -0.023871 | +0.054976 |

总体 E4−E0：ΔMacro-F1=`+0.077618`、ΔAUROC=`-0.056970`、ΔAUPRC=`+0.011305`、ΔUFAR=`-0.031723`，AUROC 仅 `3/6` 胜出。总体 E4−E3：ΔMacro-F1=`-0.077294`、ΔAUROC=`-0.167346`、ΔAUPRC=`-0.070301`、ΔUFAR=`+0.066178`，AUROC `0/6` 胜出。

## 7. 严格单因素特征与模块消融

Time/Length 使用不含 direction 的 E4_TIME_LENGTH 作为左侧，避免把 direction 混进单特征贡献。所有数值是 6 个相同 protocol/seed 的配对均值。

| 严格比较 | ΔMacro-F1 | ΔAUROC | ΔAUPRC | ΔUFAR@95 |
|---|---:|---:|---:|---:|
| Time+Length − Length（Time） | +0.055788 | -0.035383 | -0.046578 | +0.049821 |
| Time+Length − Time（Length） | +0.176222 | -0.031306 | -0.070588 | +0.075293 |
| Full − Time+Length（Direction） | +0.000022 | +0.035479 | +0.048322 | -0.077190 |
| Full − No-MS（Multi-scale） | +0.011391 | +0.009839 | +0.019840 | -0.052604 |
| Full − No-RNN（Recurrent） | +0.016056 | -0.007519 | -0.014214 | +0.029344 |
| Full − Base（Joint modules） | +0.001372 | +0.001428 | +0.013561 | -0.052755 |

结论：Length 是主要闭集信号；Time 与 Length 组合各自改善闭集，但在 feature-distance detector 下反而降低 AUROC。Direction 对闭集均值几乎无影响，却恢复 `+0.035479` AUROC 并降低 UFAR。Multi-scale 带来小幅一致收益；RNN 改善闭集但未改善开集。Full 与 Base 的 AUROC 仅差 `+0.001428`，不能声称两个模块的组合解决了开集表示瓶颈。

## 8. 五个核心问题

1. **E4 能否同时获得较好的闭集与开集性能？** 否。闭集 Macro-F1 达 Email `0.7497`、Streaming `0.7983`，但主分数总体 AUROC 仅 `0.5941`、UFAR@95 为 `0.8609`。
2. **包级时间—长度序列是否有额外价值？** 对闭集有明确价值，direction 对开集也有增量；但 30 包单分支不足以形成稳定、低 UFAR 的 detector。
3. **多尺度和循环贡献多少？** 多尺度为 `+0.0114 Macro-F1/+0.0098 AUROC`；循环为 `+0.0161 Macro-F1/-0.0075 AUROC`；联合模块相对 Base 的 AUROC 仅 `+0.0014`。
4. **瓶颈是表示还是分数/阈值？** 两者都有。MSP 将总体 AUROC 提到 `0.6449`，说明分数有影响；但 UFAR 反而升到 `0.8929`，三种分数仍无法解决 Streaming 重叠与工作点迁移。
5. **是否具备与 E3 融合的依据？** 不具备。Streaming ΔAUROC=`-0.1392`、总体 ΔAUROC 为负、仅 3/6 胜出，且 Email UFAR 超过预注册上限。

## 9. 完整性、资源与重放

- 6/6 正式 run 成功；42/42 checkpoint 保存并记录 SHA256。
- 42/42 checkpoint 独立重放通过，最大 tensor 绝对差 `0.0`。
- 19 个 Stage 16S/17 受保护资产前后 SHA256 完全一致。
- 逐样本预测 35,049 行；闭集 42 行；开集 378 行；per-class 210 行；epoch 动态 4,200 行。
- 最终单元测试 3/3 通过；语法检查、GPU smoke 和请求级 verifier 均 PASS。
- 第一次 GPU smoke 因缺少 scaler 参数失败，失败日志保留；修复后重跑通过。
- E4_FULL 每个 100-epoch run 用时约 `16.34–23.19 s`，PyTorch 峰值显存约 `58–59 MB`。

## 10. 局限与最终结论

- 当前缓存最多前 30 包，不能等价于论文的 1,000-packet/two-second slice。
- 未实现论文 background class、ARPL、PVRP，以保持当前 Strict Unknown-Free 比较边界。
- 服务标签为弱 capture-activity 标签，不能证明 capture-disjoint 泛化。
- 只执行 Stage 17 当前可运行的 Email、Streaming Unknown 设置；不是新的 untouched external validation。
- 未定位到可核验的 PP-OpenNet 官方代码，只允许声称 “PP-OpenNet-inspired independent reconstruction”。

最终结论为 `E4_FAIL_NO_FUSION`。包级时间、长度和方向对闭集与部分拒识信号有价值，但当前 E4 没有同时达到稳定 AUROC 和可接受绝对 UFAR。按预注册规则封存结果，不启动 E3+E4 融合。
