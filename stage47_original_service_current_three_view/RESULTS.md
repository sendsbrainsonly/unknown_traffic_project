# 表4-2补充：当前三视图的原服务标签同流闭集对照

Status: complete / INDEPENDENT_REPLAY_PASS

Experiment ID: stage47-table42-current-threeview-original-services-2022-2023; type: matched-closed-set-benchmark; claim scope: diagnostic.

## Data and split

沿用Stage20原服务标签：VPN6类、Train/Val/Test8764/1098/1093；Tor7类、8946/1118/1117。encoder/training seed成对为2022/2022、2023/2023。所有flow保留，不重新划分、不合并标签。

## Configuration and execution

复用fine-trained TF/FIG/YaTC分支与冻结表示，直接导入当前Stage31/32模型、encode/infer、normalize。重新训练Gaussian适配器30完整epoch(meanCE+0.05KL)，等权1/3拼接头30完整epoch(CE)，batch256/Adam1e-3；两阶段均Known-Val选模。预先指定CPU4线程串行，未占用新GPU。四run/8checkpoint全部冻结后才Test。

主数值流程完成、逐样本/模型重放PASS；初次外层归档校验退出1，原因是manifest使用不支持的status=complete和本文件缺六个规范标题，不是训练/指标失败。原始失败日志保留；仅修正文档/元数据后重新完整验证，不重训、不更改数字。

## Core results

2022、2023两个encoder/training seed，均值±总体标准差(ddof=0)，单位%。保留七条旧配置并新增当前三视图，共八配置。

### ISCX-VPN（六类）

| 方法 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| OURS-E3-T8-pretrained | 85.00 ± 0.82 | 85.95 ± 0.70 | 84.64 ± 0.74 |
| Open-Detect corrected-paper | 72.87 ± 1.42 | 74.43 ± 1.42 | 72.54 ± 1.35 |
| RoNeTC-runnable-reconstruction | 70.68 ± 0.69 | 71.57 ± 0.81 | 70.13 ± 0.51 |
| TFE-GNN-8-UDP-short | 64.87 ± 1.92 | 65.85 ± 1.46 | 64.45 ± 1.63 |
| TrafficFormer-pretrained | 85.32 ± 0.41 | 86.06 ± 0.40 | 84.74 ± 0.43 |
| Trident-early8-86D | 61.34 ± 0.23 | 63.22 ± 0.06 | 60.65 ± 0.05 |
| YaTC-official-pretrained-Stage20 | 87.83 ± 1.19 | 88.71 ± 1.16 | 87.65 ± 1.26 |
| OURS-current-threeview-equal | **88.47 ± 0.64** | **89.37 ± 0.55** | **88.36 ± 0.60** |

当前方法减基线的双seed平均Macro-F1差值（百分点）：
- OURS-E3-T8-pretrained: +3.420
- Open-Detect corrected-paper: +14.943
- RoNeTC-runnable-reconstruction: +17.796
- TFE-GNN-8-UDP-short: +23.514
- TrafficFormer-pretrained: +3.309
- Trident-early8-86D: +26.149
- YaTC-official-pretrained-Stage20: +0.661

### ISCXTor2016（七类）

| 方法 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| OURS-E3-T8-pretrained | 83.21 ± 0.67 | 82.22 ± 0.76 | 83.39 ± 0.60 |
| Open-Detect corrected-paper | 62.85 ± 8.33 | 57.66 ± 9.97 | 61.48 ± 9.13 |
| RoNeTC-runnable-reconstruction | 60.43 ± 1.07 | 53.40 ± 1.98 | 59.34 ± 1.74 |
| TFE-GNN-8-UDP-short | 60.07 ± 0.09 | 55.36 ± 1.38 | 58.81 ± 0.87 |
| TrafficFormer-pretrained | 83.12 ± 0.58 | 82.13 ± 0.64 | 83.24 ± 0.51 |
| Trident-early8-86D | 63.29 ± 0.27 | 60.53 ± 0.19 | 63.24 ± 0.33 |
| YaTC-official-pretrained-Stage20 | 84.38 ± 0.13 | 82.74 ± 0.08 | 84.36 ± 0.15 |
| OURS-current-threeview-equal | **87.82 ± 0.36** | **86.63 ± 0.15** | **87.88 ± 0.34** |

当前方法减基线的双seed平均Macro-F1差值（百分点）：
- OURS-E3-T8-pretrained: +4.414
- Open-Detect corrected-paper: +28.975
- RoNeTC-runnable-reconstruction: +33.231
- TFE-GNN-8-UDP-short: +31.271
- TrafficFormer-pretrained: +4.506
- Trident-early8-86D: +26.106
- YaTC-official-pretrained-Stage20: +3.892

## Preserved evidence

PLAN.md、run.py、tests.py、progress.py、preflight.json、source_hashes_before/after.json、四组输入与scaler/身份表、四组训练曲线、8个checkpoint、Val/Test逐样本预测/logits/混淆矩阵/per-class、run_results.csv、all_methods_run_results.csv、method_summary.csv、paired_comparison.csv、per_class_results.csv、table4_2_supplement.md、completion_verification.json、manifest.json；原tmux日志/exit及归档纠正与重放脚本另行保存。

## Limitations

复用原fine-trained TF/FIG/YaTC encoder及冻结表示，重新训练当前Gaussian适配器与等权拼接头，非Stage27其他融合头、非Stage30旧分数、非Stage46联合结果。每run30+30epoch，Known-Val选模，全部四run冻结后Test。

CPU4线程串行轻量模块训练，与历史GPU训练可能有浮点/随机数实现差异；模型、loss、epoch、batch、数据角色及标签规则未改。两个seed不足以充分证明稳定性；旧Test已暴露，只属于development对照。该闭集结果不能证明开集拒识改进。

原表及历史图未覆盖。table4_2_supplement.md是待整合正文的新表，保留本次独立实验来源。无Unknown访问/encoder更新/Test选参，不自动开始下一阶段。

## Conclusion and next step

当前三视图双seed平均Macro-F1为VPN89.37%、Tor86.63%，在本表八配置均值中最高；相对YaTC为+0.661/+3.892个百分点，相对早期双视图为+3.420/+4.414个百分点。两个seed及已暴露Test不足以宣称普遍统计显著性或开集优势。完成归档复验后报告本次补充表，不自动调整方法/启动实验。
