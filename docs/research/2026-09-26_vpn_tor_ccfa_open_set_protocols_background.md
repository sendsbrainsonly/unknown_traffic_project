# VPN / Tor CCF A 开集协议背景补充

日期：2026-09-26。性质：文献与官方数据说明核查；未运行实验、未修改协议或原始数据。

## 核心结论

“Tor 开集识别效果很好”不能直接作为 ISCXTor 应用/服务级开集应当同样好的证据。CCS 的 Deep Fingerprinting 是逐次网站访问识别，使用长 Tor-cell 方向序列，且训练包含非监控网站背景类。它与当前项目的短 flow、整服务 Unknown 留出、Known-only 训练和阈值校准在样本单元、语义、监督边界均不同。

本补充检索未额外确认到一篇同时满足“直接使用 ISCXTor2016 + 严格 Unknown-Free + 完整公开类别/样本划分/阈值规则”的 CCF A 论文。不能把该检索范围内未找到说成不存在。主研究文件另行核实 FEC-OSL 与 INFOCOM 2020 Unknown-Application Filtering，本文件不重复其全文审计。

## 1. Deep Fingerprinting，ACM CCS 2018

CCS 位于 [CCF 官方网络与信息安全 A 类目录](https://www.ccf.org.cn/Academic_Evaluation/NIS/zgjsjxhtjgjxshy/al/2017-03-13/586204.shtml)。论文出版信息由 [作者论文](https://mjuarezm.github.io/assets/pdf/ccs18.pdf) 和 [作者代码库](https://github.com/deep-fingerprinting/df) 交叉确认。

| 项目 | 查证结果 |
|---|---|
| 数据集 | 作者采集的网站访问，不是 ISCXTor2016 |
| 语义类别/样本 | 网站主页 / 一次访问的 Tor trace |
| 监控类 | 95 个网站，每类 1,000 次访问 |
| 开集训练 | 85,500 监控 traces，加 900–20,000 非监控网站，每站一次 |
| 开集测试 | 9,500 监控 traces，加 20,000 非监控网站，每站一次；非监控训练/测试网站不同 |
| 输入 | 最长 5,000 cells 的方向序列，短样本补零 |
| 清理 | 去掉不足 50 packets 或缺少某方向的 traces 等异常访问 |
| 阈值 | 扫描置信阈值并报告 ROC/PR；正文未给出 Known-Val P95 制度 |
| 对当前项目的可比性 | 非监控背景进入训练，因此不是我们的 Strict Unknown-Free 协议 |

来源：正文 §2、§4、§5.1、§5.7。不要把正文开集 900/100 每类划分与官方代码 README 展示的闭集 800/100/100 混为一个实验；后者明示 76,000 Train / 9,500 Val / 9,500 Test。官方 README 也明确 open-world 模型训练包含 monitored 和 unmonitored。[正文](https://mjuarezm.github.io/assets/pdf/ccs18.pdf)、[官方 README](https://github.com/deep-fingerprinting/df/blob/master/README.md)

因此可借鉴其协议透明性、背景类别是否参与训练的披露方式和 PR/base-rate 报告；不能直接搬用其数字证明当前 flow-level detector 或数据集有错。

## 2. ISCXTor2016 官方标签边界

[UNB 官方 Tor 页面](https://www.unb.ca/cic/datasets/tor.html) 说明：同时采集工作站普通流量与网关 Tor 流量；先确认工作站流量多数来自目标 application X，再给对应 Tor PCAP 的所有 flow 标成 X。这是 capture 活动来源的弱逐流标签证据，不是每条 Tor flow 的应用因果鉴定。

同页列出 Browsing、Email、Chat、Audio-Streaming、Video-Streaming、FTP、VoIP、P2P 等语义；文字同时写“7 traffic categories”，列项的分组方式并不完全一致。因而论文中的“8 类”“16 类”和项目合并服务类必须对照具体 label map，不能只比较类别数字。

推论：某次 Skype/VoIP capture 中辅助连接被赋予活动标签，是合理的数据质量风险；不能据此直接宣称本地噪声率或把某组拒识失败全部归因于标签。

## 3. ISCXVPN2016 官方活动与 flow 的区别

[UNB 官方 VPN 页面](https://www.unb.ca/cic/datasets/vpn.html) 明确区分 7 个活动语义及 VPN/non-VPN 共 14 个 traffic categories，并说明 Hangouts 通话采集中也存在 Browsing flows。官方 VPN 采用 OpenVPN UDP mode。标签定义、应用名、活动名、隧道内/外观察点必须分别记录。

推论：把某 application 整类留下，不自动保证它对应的 service 在 Known 中消失；同一 application 可承担 Chat、VoIP、文件传输等活动。只有依据 capture 活动来源定义 service，才能审计 whole-service holdout。该风险与捕获级弱标签风险叠加，但仍需本地流映射审计才能量化。

## 4. 非 A 主证据的补充线索：PreDyn-IDS

[出版社页](https://www.sciencedirect.com/science/article/abs/pii/S1566253526004884) 的可检索正文描述 ISCXTor fine-tune 6 类、其余 10 类 Unknown，通用 Train/Val/Test 比例 8:1:1。还描述 203,817 条来自 ISCXBot2014/VNAT/ISCXVPN/ISCXTor 的无标签预训练数据。另一个 clustering 实验列 Tor CHAT/VOIP、VPN-App Netflix/Skype、VPN-Service Email/P2P 为 Unknown。

限制：本轮未核实该 venue 为 CCF A，不纳入 A 类核心证据；出版社打开返回 403，只有出版社索引全文片段可读。不能把不同节的 6/10 和 CHAT/VOIP 设置视为同一个协议，也无法据此核实预训练是否排除了以后 Unknown 的样本、具体 class IDs、每类数量和固定 seed。因此仅保留后续核查线索，不能当作严格匹配的直接基线。

## 5. 本项目可采用的判读原则

1. 保留当前 whole-service Unknown-Free 结果作为独立任务，不用网站指纹高分替换或否定它。
2. 文献横向表至少列：数据版本、样本单元、观察点、class map、Known/Unknown 语义、Unknown 是否进入预训练/训练/校准、长度过滤、各 split 数量及阈值来源。
3. 论文若允许辅助 Unknown/背景类训练，需要明确标注该监督条件；不应悄悄加入当前冻结实验。
4. 高 closed-set F1 只证明 Known 类间可分；对完全留出的服务能否拒识，仍需 score 分布与固定工作点下 UFAR/Known FRR。
5. 现有证据支持把“任务定义和监督边界差异”列为优先解释项；不足以判定“VPN/Tor 数据本身不能做开集”。

## 核查范围

主要一手来源：作者 CCS2018 PDF、官方 DF GitHub、UNB VPN/Tor 数据说明、CCF 官方 CCS 条目；补充出版社索引线索单独标注。没有将网页检索摘要中的高准确率当作同协议开集结论。研究仅新增本文件和项目内 tmux 读取日志。
