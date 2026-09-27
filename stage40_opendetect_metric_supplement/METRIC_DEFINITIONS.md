# Open-Detect 指标口径审计

Open-Detect 论文的开世界结果表（例如 Table V）列出 Accuracy 和 F1-Score，并给出五折平均±标准差；论文正文未明确定义 F1 的平均方式或正类。发布代码 `../Open-Detect/code/test.py` 的 `auroc_score` 则明确将 Known=0、Unknown=1，打印 AUROC、Accuracy、Precision、Recall、F1，且 `f1_score` 使用默认二分类口径。闭集 `inner_acc` 使用 weighted Precision/Recall/F1。故本补充以“发布代码口径”命名，不声称论文表格的 F1 定义已被严格证明。

同一批 Stage40 USTC A-2 测试样本（Known 4,333，Unknown 558）计算两种阈值结果：

1. `known_validation_p95`：原 Stage40 已冻结阈值；`score > threshold` 判 Unknown。它是正式 Known-only 阈值评估；没有使用 Test 标签选阈值。
2. `released_code_test_oracle_youden`：严格按发布代码在带标签的 Known+Unknown Test 上计算 ROC，取首个 `argmax(TPR-FPR)` 对应阈值，`score >= threshold` 判 Unknown。这是回顾性代码复现，不是独立测试或可部署阈值。

对于两种口径，`accuracy_binary=(TP+TN)/N`，`precision_unknown=TP/(TP+FP)`，`recall_unknown=TP/(TP+FN)`，`f1_unknown_binary=2PR/(P+R)`，`UFAR=FN/(TP+FN)`，`Known FRR=FP/(TN+FP)`。AUROC/AUPRC 是同一冻结分数的阈值无关评估，故在两行中相同。

闭集指标仅用 Known Test 的原分类预测计算；F1-weighted 是发布代码的闭集口径，F1-macro 额外保留用于观察少数类。不要将闭集 F1 与二分类开集 F1、论文不同数据/类别数/训练量的结果直接比较。由于 Unknown 仅占测试样本 11.41%，二分类 Accuracy 对类别不平衡敏感，须同时看 Unknown Recall/UFAR 与 Known FRR。

来源：本地论文 `../Open-Detect/paper/Meng 等 - 2025 - Detection of Unknown Attacks Through Encrypted Traffic A Gaussian Prototype-Aided Variational Autoe.pdf`；发布代码 `../Open-Detect/code/test.py`；历史审计 `stage10a_opendetect_protocol_audit/outputs/paper/metric_definition_audit.md`。这些路径均相对本项目根目录。
