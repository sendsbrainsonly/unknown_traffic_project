# VPN/Tor 开集困难归因与 USTC/CIC 评测方向：原始文献和本地结果核查

日期：2026-09-26（UTC）。性质：只读研究、已有结果分析，不是新实验或新协议冻结。

## 1. 结论先行

**数据集、标签语义和评测协议很可能影响当前开集困难，但“闭集超过 Open-Detect”不能推出“开集失败不是模型的问题”。** 当前证据支持分别检查：Known 分类能力、Unknown 排序能力、固定阈值下的拒识能力、标签及采集偏差。

- USTC 值得作为下一项已知类分类 + 未知恶意家族拒识的候选实验；可对接本地冻结 A-2 和 Open-Detect 的原论文任务。
- CIC-IDS-2017 可作为补充未知攻击实验，但本地只验证了三类闭集，不能承诺其开集必然更容易，也不能宣称已覆盖完整 CIC 攻击种类。
- VPN/Tor 应保留为困难场景。不能只因新数据集分数高而删除现有失败结果，或把开发后反复查看的 Test 再称为 untouched external validation。
- 本轮直接核实的 VPN/Tor CCF A 工作采用了不同程度的辅助 Unknown、在线学习或不同样本单位，不能将其指标当作本项目 Strict Unknown-Free 协议下的直接成绩标尺。

## 2. 当前项目的直接证据

### 2.1 Stage38：闭集很好，开集表现不一致

来源：[Stage38 RESULTS](../../stage38_three_view_des_open_set_transfer/RESULTS.md)。固定一组 seed，每个数据集内部共享三路 representation；Known 类使用 capture-derived coarse labels，ISCX Known/Unknown 应用 membership 沿用 Stage12。

| 数据集 | Known Test Macro-F1 | 分数 | AUROC | UFAR @ Known-Val P95 | Known FRR |
|---|---:|---|---:|---:|---:|
| ISCX-VPN | 0.975277 | MSP | 0.583954 | 0.846500 | 0.053054 |
| ISCX-VPN | 0.975277 | DES-v0 | 0.530237 | 0.863167 | 0.048735 |
| ISCX-VPN | 0.975277 | DES-v1 | 0.559509 | 0.855500 | 0.054287 |
| ISCXTor2016 | 0.941121 | MSP | 0.901799 | 0.633000 | 0.050314 |
| ISCXTor2016 | 0.941121 | DES-v0 | 0.898750 | 0.488250 | 0.050314 |
| ISCXTor2016 | 0.941121 | DES-v1 | 0.924869 | 0.634750 | 0.043127 |

Unknown 为 positive；UFAR 是 Unknown 被接受为 Known 的比例，越低越好。这里的 MSP/Energy 不是 Open-Detect Native。

解释：

1. VPN 不只是某个阈值选得不好；这几个现有 score 的 AUROC 也弱，说明排序分离不足。不能通过在 Test 上换阈值来解决或证明解决。
2. Tor 不能概括成“模型完全不会开集”：DES-v1 AUROC 为 0.924869，但 P95 仍接受 63.475% 的 Unknown。
3. 同一 Tor representation 上，DES-v1 相比 DES-v0 的 AUROC 增加 0.026119，UFAR 却增加 0.146500。这是 score/工作点行为不同的直接证据，而不是纯粹数据集解释。
4. 闭集优化要求区分几个 Known 类，未必要求 Unknown 远离这些类；更高闭集 Macro-F1 不自动意味着更好的 support 几何、拒识分数或尾部校准。

### 2.2 Stage39：完整 Service 留出的首个结果仍困难

来源：[Stage39 RESULTS](../../stage39_coarse_open_set_execution/RESULTS.md)，研究时读取的中间快照；不是全体 setting 汇总结论。

VPN Communication 整类留出：Known Train/Val/Test = 6,823/851/858，Unknown = 10,528。

| 分数 | Known Test Macro-F1 | AUROC | AUPRC | UFAR P95 | Known FRR P95 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.995504 | 0.351050 | 0.899017 | 0.918788 | 0.051282 |
| DES-v1 | 0.995504 | 0.405230 | 0.923279 | 0.849354 | 0.058275 |

Unknown 占比为 0.924644，AUPRC 必须结合这一随机排序基准解释，不能把约 0.92 视为强检测能力。该单 setting 已说明：仅把标签变粗、把闭集分数提高，并不足以保证 Unknown 检测成功。不能由它推断全部 VPN/Tor setting 必然失败。

### 2.3 USTC/CIC 已有的是哪些证据

| 数据集及来源 | 当前闭集任务 | Train/Val/Test | Test Accuracy / Macro-F1 |
|---|---|---|---|
| [USTC Stage34](../../stage34_ustc_cic_closed_set/RESULTS.md) | A-2 的 17 个 Known 类；Geodo/Htbot/Tinba 未进入训练；每类每 split 确定性抽取 10% | 34,665 / 4,333 / 4,333 | 0.984306 / 0.987649 |
| [CIC Stage34B](../../stage34b_cic_balanced_retest/RESULTS.md) | BENIGN、DoS Slowhttptest、PortScan 三类；良性与恶意总量平衡 | 35,516 / 3,912 / 2,536 | 0.998028 / 0.997482 |

CIC Test 每类支持数为 1,268/132/1,136；使用五分钟 `(source_pcap, floor(timestamp/300))` 分组，不等于跨采集日泛化。两个攻击类别仍有采集文件/日期绑定风险。以上数字不能代替当前三路模型在 USTC/CIC 上的开集评测，也不构成同协议战胜 Native Open-Detect 的新证据。

## 3. 直接相关 CCF A 文献究竟如何划分

CCF 类型依据官方目录的 TIFS、INFOCOM、WWW 条目；这里将期刊 A 与会议 A 分开注明。文献原始协议与本项目严格复现结果不是一回事。

### 3.1 ISCX-Tor：FEC-OSL，TIFS 2026（CCF A 期刊）

论文：Yang 等，*End-to-End Open-Set Semi-Supervised Learning for Fine-Grained Encrypted Traffic Classification*，21:1347–1362。[出版记录](https://doi.org/10.1109/TIFS.2026.3653575)。核对本地正文第 VII 节、Fig.6 及其前后文字。

- 论文数据说明提到 ISCX-Tor 的 16 种类型，但开集给出的 class ratios 合计为 8；未核实到可逐样本复现的八类名单或选取 manifest，不能自行补写。
- Known:Unknown 为 6:2、4:4、2:6。**Unknown 中的一半无标签参加训练**，用于 energy boundary 和 clustering，另一半作为 novel Unknown 测试。
- 因此实际 Known / auxiliary Unknown / novel Unknown 类数分别是 **6/1/1、4/2/2、2/3/3**，不是我们全部 Unknown 完全不参与学习的设置。
- 采用 class-balanced sampling；原文未给出足以恢复各类精确 Train/Val/Test 样本 ID 的完整采样记录，不能据此声称 flow/capture-disjoint。
- Known 能量分布拟合 Weibull；Tor quantile 为 0.1，USTC/CIC-IDS-2018 为 0.05。不是本项目经验 Known-Val P95 的同一定义；所读实验段落未明确给出独立校准 split 的样本清单。
- 论文另一个 CIC 数据集是 **CIC-IDS-2018，不是 CIC-IDS-2017**。

意义：可以研究它如何利用额外 Unknown 信息塑造边界，但不能据其 Tor 高指标推断 Known-only DES 应达到同样水平。本地 `Projects/FEC-OSL` 标注为独立重建，不作为作者官方 split 的替代来源。

### 3.2 ISCX-VPN：Autonomous Unknown-Application Filtering，INFOCOM 2020（CCF A 会议）

论文：Zhang 等，*Autonomous Unknown-Application Filtering and Labeling for DL-based Traffic Classifier Update*。[作者论文](https://arxiv.org/pdf/2002.06359)，[正式 DOI](https://doi.org/10.1109/INFOCOM41043.2020.9155292)。定位 Table I/III、实验 IV-D。

- 使用 ISCX VPN-nonVPN 与自行采集应用的组合，单位是 **packet**，不是本项目的整条 flow。
- 三个场景都是 9 个 Known application，Unknown 分别 2/2/3 类。
- A 的 Unknown：Discord、GoogleMap；B：TorTwitter、SCP(download)；C：Netflix、Speedtest(download)、Tencent QQ(voice)。部分 Unknown 来自自行采集，而非全部来自本地 ISCX。
- 每个 Known application 的 P1 训练数据为 2,500 packets；P3 称为 validation，每类 1,200 packets。
- 活跃数据 P2 为每个 Known 类 500 packets，P4 为每个 Unknown 类 4,500 packets，连同部分 P1 进入 discriminator 自学习及 classifier 更新；另有每个 Unknown 类 1,200 packets 的 P5。
- 原文随后又用 P3/P4/P5 描述更新分类器的测试；不能替作者改写为我们当前严格独立的 Train/Val/Test 协议。
- 随机抽取包，原文未据此证明同一流/同一 capture 不跨集合；使用模型更新与初始置信度 CDF 规则，不是冻结 encoder + Known-Val P95 的同一实验。

意义：这是 Unknown filtering + discovery/update 研究，不能将更新后分类精度直接当成冻结模型的 Unknown detection AUROC，亦不能直接当作 ISCX 全流级公平对照。

### 3.3 USTC：Open-Detect，TIFS 2025（CCF A 期刊）

论文：Meng 等，*Detection of Unknown Attacks Through Encrypted Traffic: A Gaussian Prototype-Aided Variational Autoencoder Framework*。[正式 DOI](https://doi.org/10.1109/TIFS.2025.3612141)。已视觉核对 Table II/III。

- 原论文使用 USTC-TFC2016 与 Malicious_TLS，并非 ISCX-VPN/ISCXTor。
- USTC A-1：19 Known + Tinba 1 Unknown。
- USTC A-2：17 Known + Geodo/Htbot/Tinba 3 Unknown。
- USTC A-3：15 Known + Geodo/Htbot/Tinba/Miuref/Neris 5 Unknown。
- Train/Validation/Test = 8:1:1，类别整体留出，Known validation 以约 95% 接受率设阈值。
- Table II 的 USTC 采样为 19 类各 2,000 flows，Neris 79 flows；不能把该采样规模当成本地几十万条流的同一个实验。

意义：USTC A-2 最贴近本项目历史 Unknown-attack 问题，也已有 Known-only 三路权重可核查复用。评估前仍需校验全部分支、fusion、scaler 的训练 ancestry 和 sample IDs；不能直接调用在 20 类上监督训练过的模型。

### 3.4 USTC/CIC：UnDiff，WWW 2025（CCF A 会议）

论文：Lian 等，*Facing Anomalies Head-On: Network Traffic Anomaly Detection via Uncertainty-Inspired Inter-Sample Differences*。[作者全文](https://www.xoveexu.com/file/paper/25-04-WWW-UnDiff.pdf)。定位 §4.1、Table 1。

- 只用 benign 训练，是 normal-only anomaly detection，不是留出少量 application/service 的同一任务。
- 各数据集随机抽取 10,000 normal flows 训练；测试为 5,000 normal + 5,000 anomalous；五次不同随机种子。
- 论文 AUC：USTC **99.90±0.0%**，CIC-IDS2017 **88.88±0.4%**。同一论文也没有显示 CIC 必然容易。
- ACC 对应 optimal F1 工作点；所读正文未明确它的阈值搜索数据清单，不能把此 ACC/F1 与本项目 Known-Val P95 UFAR 等同，也不能未经代码证据就指控其在 Test 调参。

意义：支持把 Unknown attack 与 Unknown benign application 区分讨论；不支持把“benign-only vs all attacks”结果作为“多 Known 类 + 新家族留出”的直接对照。

### 3.5 Tor 网站指纹：Deep Fingerprinting，CCS 2018（CCF A 会议）

[作者全文 §5.7](https://mjuarezm.github.io/assets/pdf/ccs18.pdf)使用网站访问 trace，而不是 ISCXTor application/service flow：95 个 monitored 网站；开集训练为 85,500 monitored traces 加 900–20,000 unmonitored traces，测试为 9,500 monitored 加 20,000 unmonitored。背景训练/测试网站不同，但背景确实参与训练。输入最长 5,000-cell 方向序列，不能与本项目短流窗口等同。

正文扫描阈值绘制 PR/ROC，没有采用本项目的 Known-Val P95 校准制度。[官方代码说明](https://github.com/deep-fingerprinting/df/blob/master/README.md)的闭集 800/100/100 每网站划分也不能与正文开集 900/100 划分混用。其意义是展示另一种规范的 Tor open-world 任务，不是本地 ISCXTor 的可直接引用基准。

本轮没有额外核实到同时具备“直接 ISCXTor + 严格 Unknown-Free + 完整公开类别/样本/阈值划分”的 CCF A 实验；这是检索边界，不是声称此类论文不存在。

## 4. 官方数据定义能证明什么

### VPN

[UNB 官方说明](https://www.unb.ca/cic/datasets/vpn.html)按七种行为及 VPN/non-VPN 形成 14 类，区分 Browsing、Email、Chat、Streaming、File Transfer、VoIP、P2P。Skype 等 application 可产生多个 service。页面明确举例：Hangouts 通话 capture 中也有 browsing flows。

### Tor

[UNB 官方说明](https://www.unb.ca/cic/datasets/tor.html)说明：同步采集 workstation 与 Tor gateway PCAP，先确认 workstation 的多数 flows 来自应用 X，再将对应 Tor PCAP 的所有 flows 标为 X。这是 capture-derived label，不是逐流活动的权威核验。

页面文字“7 categories”与实际列出的八个活动项不一致；不同论文对 Audio/Video、Tor/nonTor 是否合并也需要逐项核查，不能只凭数据集名称对齐类别数。

**证据边界：**这些来源证明潜在标签污染与任务语义混合的风险，不提供本地 flow 的错标百分比。尚不能断言本地标签错误是主因，更不能按模型预测或 Test 分数修改标签。

## 5. 对“不是模型问题”的判断

| 假设 | 当前证据 | 能否下定论 |
|---|---|---|
| Known 分类器整体没学会 | Stage38/39 Known Macro-F1 很高 | 不支持这是全部原因 |
| 闭集越高开集必然越好 | Stage39 首个 setting 闭集 0.9955，DES AUROC 0.4052 | 直接反例，不成立 |
| VPN/Tor capture 标签有污染风险 | 官方采集/标签定义及本地弱标签规则 | 风险明确，实际数量未量化 |
| Unknown application 与 Known service 语义重叠 | Stage38 混合粒度；Stage39 整服务留出修正 | 部分任务确有风险；不能解释全部结果 |
| 检测 score/阈值的工作点不理想 | Tor D1 排序变好但 UFAR 比 D0 高 14.65 个百分点 | 证据明确；不是让我们在 Test 调阈值 |
| 换 USTC/CIC 一定成功 | 仅有当前闭集及他人不同协议结果 | 尚未证实 |

在完全同样本、同标签粒度的闭集对照中超过 Open-Detect，只能说明 Known 判别指标更好。监督表示仍可能把新业务投到已有 support 内，也可能学习 capture/endpoint shortcut；这些都属于表示与数据相互作用，不宜二选一归因。

## 6. 建议顺序：不据 Test 改协议

1. 保留并完成已经冻结的 Stage39 所有 settings，不用首个失败 setting 代表全数据集，不因本轮研究改变其队列/参数。
2. 优先考虑 **USTC A-2 当前三路模型的冻结开集评测**。先验证现有 17-class 分支/fusion 的 Known-only 来源，再冻结 Unknown 三家族与所有样本数量、采样规则；不根据测试效果选择“容易的”家族。
3. 如需判断是否比 Open-Detect 更好，在同一 A-2 样本、同一 Known-Val 校准规则下做配对比较。历史论文数字及不同样本规模结果单独列，不混为公平胜负。官方预训练权重的语料来源与潜在测试数据重叠需另行披露，不能仅凭下游 Known-only 就保证预训练阶段从未接触相关数据。
4. CIC 先考虑两个明确的 leave-one-attack-out 诊断：Known={BENIGN, PortScan}, Unknown={Slowhttptest}；以及反方向。必须重建 Known-only 训练，不能复用已经见过全部三类的监督权重来声称严格 Unknown-Free。
5. CIC 保留分组约束、报告少数攻击样本量/日期绑定风险。仅这两种攻击的结果不能推广为完整 CIC 未知攻击检测。
6. 所有任务固定 Known-Val 校准（主 P95，预先登记的 P90/P99 作代价曲线）；报告 AUROC/AUPRC、UFAR、Known FRR、每类结果和 Unknown prevalence。不能用 Test 调 threshold、反转 score 或挑选类。

这是研究建议，不是本轮启动的新实验；Stage39 和历史 frozen assets 没有因本报告被改写。

## 7. 检索和执行记录

- 只读取本地结果、原始论文及官方网页；提取文本保存在项目 `.artifacts/pdf/` 的 `open_detect_protocol_review.txt`、`fec_osl_protocol_review.txt`、`undiff_protocol_review.txt`。
- 视觉核对文件：`open_detect_table3_review.png`（Table II）、`open_detect_table3_page10.png`（Table III/IV）；文件名前者沿用首次命名，实际页码以图像内容为准。
- Shell 读取/提取通过项目 `.tmux-task/` 下 research/protocol 相关命名会话执行；固定既有 Conda 环境，无安装、GPU训练、detector拟合、阈值更新。
- 补充检索：[VPN/Tor CCF A 文献核查](2026-09-26_vpn_tor_ccfa_open_set_protocols_background.md)。Tor 网站指纹的 monitored/unmonitored website 不等于 ISCXTor application/service 分类。
- CCF 官方条目：[TIFS](https://www.ccf.org.cn/Academic_Evaluation/NIS/zgjsjxhtjgjxskw/al/2017-03-13/586136.shtml)、[INFOCOM](https://www.ccf.org.cn/Academic_Evaluation/CN/zgjsjxhtjgjxshy/al/2017-03-13/586270.shtml)、[WWW](https://www.ccf.org.cn/Academic_Evaluation/Cross_Compre_Emerging/zgjsjxhtjgjxshy/al/2017-04-25/592216.shtml)。部分复查请求返回 HTTP 405；原文出版信息与目录条目分别记录，不把网站短时不可读解释为评级变化。

最终判断：**优先补测 USTC 有研究依据；CIC 是有边界的补充方向。当前 VPN/Tor 困难应归为协议/标签、representation/support 和检测工作点的共同待解释问题，而不是已经证明“模型无问题”。**
