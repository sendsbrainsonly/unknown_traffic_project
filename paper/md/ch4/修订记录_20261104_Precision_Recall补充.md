# 第4章 Precision／Recall 补充与统计口径统一

- 日期：2026-11-04（Asia/Shanghai）；执行：[agent:codex]。
- 正文入口：[4(1).md](<4(1).md>)，对应表4-5；远端已删除的旧 `1.md` 不再作为正文入口。
- 数据来源：Stage22／23／23B／47已保存的Known Test逐样本预测；本次仅使用补算结果，不新增训练或模型推理。

## 数据与指标

ISCX-VPN六类的Train／Validation／Test为8,764／1,098／1,093；ISCXTor2016七类为8,946／1,118／1,117。八种配置均使用原冻结成员与标签，seeds为2022、2023。保留RoNeTC重建版、TFE-GNN短流／UDP适配版和Trident前8包适配版的方法身份。

新增Macro-Precision、Macro-Recall、Weighted-Precision、Weighted-Recall及Micro指标，`zero_division=0`。先计算每次运行指标，再对两个种子取均值和总体标准差（ddof=0）。单标签多分类的Weighted-Recall与Micro Precision／Recall／F1均等于Accuracy。

完整结果见[七指标表](../../../stage47_precision_recall_supplement_20261103/table_precision_recall.md)、[32条逐run指标](../../../stage47_precision_recall_supplement_20261103/run_metrics.csv)、[16条汇总](../../../stage47_precision_recall_supplement_20261103/method_summary.csv)和[208条类别级指标](../../../stage47_precision_recall_supplement_20261103/per_class_metrics.csv)。

## 正文修订

1. 在4.1.4补充Macro Precision／Recall及加权指标定义，说明Macro-F1不能由平均P、R直接求调和均值。
2. 表4-5补入P、R及早期E3双视图，形成八配置对照。正文原有部分数值与双种子汇总不一致，本次统一由原始CSV生成的均值±总体标准差替换，不再混用统计口径。
3. 本方法Macro-F1为VPN89.37±0.55%、Tor86.63±0.15%；相对YaTC的差值修正为+0.66／+3.89个百分点，相对早期E3为+3.42／+4.41个百分点。
4. 将该表的任务范围明确为VPN6／Tor7，不与后文VPN4／Tor5或USTC／CIC补充任务混用。
5. README增加当前正文入口和补充CSV链接；历史Stage47静态结果保持原样。本次未重编译远端已有PDF。

## 核验与证据

32条逐run和16条汇总的原Accuracy／F1均在1e-12容差内重现；31个原始输入文件SHA256不变。正文表4-5的80个显示单元格逐一与补充CSV匹配，其他正文段落和原有换行格式由独立脚本核对。旧正文、README和原manifest保存在[本次发布证据目录](../../../stage47_precision_recall_supplement_20261103/publication_20261104/)。

本次沿用论文写作skill的表格与说明规范，在Markdown正文中完成数据回填；两个种子及已暴露Test仅支持当前开发协议内对照，不新增显著性、未知拒识或单分支因果结论。
