# Experiment results: stage40-opendetect-metric-supplement-20260927

- Status: `success`（仅已完成的 USTC A-2 指标补算；CIC 未完成）
- Experiment type: `metric-audit`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-27T02:41:41Z`
- Objective: Recompute Open-Detect paper and released-code metrics from frozen Stage40 USTC sample scores under validation P95 and separate retrospective Test-oracle thresholds

## Data and split

USTC-TFC2016 A-2 使用原有 17 个 Known 类和整类 Unknown（Geodo、Htbot、Tinba）；Known Test 4,333、Unknown Test 558。所有 4,891 个逐样本 ID 唯一，测试角色与 Known/Unknown 标签一致。没有重新训练、重新划分或改变分数。

## Configuration and execution

固定原 `stage40_ustc_cic_open_set/ustc_a2/detection/sample_scores.csv` 与 `calibration.json`，执行 `python stage40_opendetect_metric_supplement/recompute_metrics.py`。正式口径维持 Known Validation P95；另列发布代码的带标签 Test `argmax(TPR-FPR)` 阈值作为 retrospective oracle。论文表格主要报告 Accuracy/F1；发布代码额外报告 AUROC/Precision/Recall。详细定义与边界见 `METRIC_DEFINITIONS.md`。

## Core results

闭集 Known Test：Accuracy `0.984306`，Weighted-F1 `0.984315`，Macro-F1 `0.987649`（Weighted Precision `0.984360`，Weighted Recall `0.984306`）。

| 分数 | 阈值口径 | 二分类 Accuracy | Unknown Precision | Unknown Recall | Unknown F1 | UFAR | Known FRR |
|---|---|---:|---:|---:|---:|---:|---:|
| MSP | Known-Val P95 | 0.894705 | 0.542574 | 0.491039 | 0.515522 | 0.508961 | 0.053312 |
| MSP | Test-oracle | 0.820282 | 0.387997 | 0.996416 | 0.558513 | 0.003584 | 0.202400 |
| Energy | Known-Val P95 | 0.895318 | 0.547718 | 0.473118 | 0.507692 | 0.526882 | 0.050312 |
| Energy | Test-oracle | 0.734819 | 0.300162 | 0.994624 | 0.461155 | 0.005376 | 0.298638 |
| Centroid | Known-Val P95 | 0.896136 | 0.553648 | 0.462366 | 0.503906 | 0.537634 | 0.048004 |
| Centroid | Test-oracle | 0.895113 | 0.546584 | 0.473118 | 0.507205 | 0.526882 | 0.050542 |
| DES-v1 | Known-Val P95 | 0.898794 | 0.558442 | 0.539427 | 0.548769 | 0.460573 | 0.054927 |
| DES-v1 | Test-oracle | 0.831118 | 0.399851 | 0.958781 | 0.564346 | 0.041219 | 0.185322 |

阈值无关 AUROC/AUPRC：MSP `0.916286/0.629202`、Energy `0.871426/0.584590`、Centroid `0.677617/0.439162`、DES-v1 `0.945727/0.711960`；两个阈值口径共用同一分数，值不变。完整阈值、混淆矩阵计数及原始精度见 CSV。

DES-v1 在正式 P95 工作点为 TP/FP/TN/FN=`301/238/4095/257`；Test-oracle 为 `535/803/3530/23`。Test-oracle 大幅提高 Unknown 召回，但误拒更多 Known；不能作为独立测试性能宣称。

## Preserved evidence

- `open_detect_metric_supplement.csv`：四种分数 × 两种阈值的完整指标和 TP/FP/TN/FN。
- `closed_set_metrics.json`：Known Test 闭集指标。
- `completion_verification.json`：样本/决策重放和冻结输入前后 SHA256。
- `recompute_metrics.py`、`METRIC_DEFINITIONS.md`、`manifest.json`。

## Limitations

- 只有 USTC A-2 单次运行，不能给出论文表格的五折均值±标准差。
- CIC Stage40 队列失败，两个留一攻击设置尚无开集逐样本分数，不能补算。
- 论文未显式定义 F1 平均方式；这里按发布代码的二分类 Unknown-positive F1 报告，闭集另按 weighted F1 报告。
- Test-oracle 读取了 Test 真值来选择阈值，数值只用于理解发布代码口径和阈值取舍，不能用于正式模型比较或部署。

## Conclusion and next step

已补齐当前可计算的 Open-Detect 发布代码指标，并保留正式 Known-Val P95 与回顾性 Test-oracle 的边界。正式 DES-v1 USTC 结果为二分类 Accuracy `0.898794`、Unknown F1 `0.548769`、AUROC `0.945727`；其 UFAR 仍达 `0.460573`。CIC 待原 Stage40 训练/评分完成后才可用同一脚本扩展，当前不启动后续实验。
