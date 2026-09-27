# Stage 23 方法准入与排除审计（2026-09-23）

本文件只判断能否在 **Stage 20 原样 22,136 条 flow、同一 Service-6/7 标签及 Train/Val/Test 成员关系** 下进入正式闭集配对表；不是对方法本身质量的判断。Stage 20 manifest SHA256：`6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb`。统计依据为 `native_method_coverage_audit_v2.json`、`yatc_mfr_input_audit.json`、各方法源码及 Stage 23 逐流输入审计。原始数据和其他项目均只读。

| 方法 | 正式表处理 | 冻结 flow 覆盖/证据 | 身份边界 |
|---|---|---|---|
| TrafficFormer E1 | 准入，复用 Stage 22 | 22,136/22,136，逐样本预测已重放 | 官方预训练权重 + 本任务微调 |
| OURS-E3-T8 | 准入，复用 Stage 22 | 22,136/22,136，逐样本预测已重放 | TrafficFormer 预训练 + FIG/TAGCN |
| Open-Detect corrected-paper | 准入，Stage 23 新训练 | 22,136/22,136；4/4 run 完成并独立重放 | 本地 corrected-paper 实现，非原仓库逐行复现 |
| RoNeTC | 准入，Stage 23 新训练 | 原 Stage 19 缓存缺 Tor 137 条；按冻结 packet refs 补全后 22,136/22,136 | 可运行重构，保留原三种 8 包 view 与训练配置 |
| YaTC | 准入，Stage 23 新训练 | 从冻结 packet refs 重建 5 包 40×40 MFR；Train/Val 已全覆盖，Test 仅在选模后物化 | 官方预训练权重与训练循环；输入为同 flow 的作者 MFR 规则 |
| ET-BERT strict flow | **不准入本轮正式表** | 冻结 flow 有 7,758 条单包、7,437 条两包，共 **15,195/22,136** 条先被 strict `>=3 packets` 排除；5 KiB 规则还会额外排除 | 原生 flow 规则见 `../ET-BERT/ET-BERT-main/FLOW_MATERIALIZATION.md`；改为 1 包 PAD 是有意义但**不同定义的适配实验**，不得混称原生公平结果 |
| TFE-GNN native TCP graph | **不准入本轮正式表** | 仅 Known Train/Val 第一包已见 VPN 6,242 条 UDP、Tor 2,074 条 UDP，合计 **8,316 条**；另 VPN Train 19 条非 TCP/UDP。原生构图只处理 TCP，并要求至少一个 payload 包、最多前 50 个 payload 包；100% coverage 不成立 | `../TFE-GNN/reproduction/PROTOCOLS.md` 与 `build_paper_dataset.py`。把 UDP/8 包直接塞入图属于新适配，不是当前已复现方法 |
| Trident native | **不准入本轮正式表** | 原 USTC 缓存 `80,000×86` 属另一批 flow、另一任务。官方未发布完整 USTC 预处理；本地 86 维重构使用完整双向流统计。Stage 20 只冻结每条流的前 1–8 个 packet refs，不能从该记录断言完整会话统计与作者方法同义 | `../Trident/code/reproduction/REPRODUCTION_HANDOFF.md` 与 `extract_ustc_flows.py`。仅取前 8 包计算 86 维会是另一个标注清楚的新变体 |
| UnDiff | **不准入原生闭集表** | 原生 one-class anomaly 任务无 Service-6/7 多类分类头 | 应在单独异常检测面板比较，不能把其 anomaly 指标写成 Macro-F1 |

TFE 数字只使用 Known Train/Val 的已物化 MFR 字节，未为准入判断提前读取 Known Test 特征。VPN Train/Val TCP 第一包 3,205/396，UDP 5,540/702；Tor TCP 7,105/885，UDP 1,841/233。即便 Test 全是 TCP，原生 TFE 也已无法覆盖冻结训练与验证全集。YaTC 的 1–4 包流按其作者规则零填充，因此它与 ET-BERT strict flow 的排除逻辑不同。

正式表目标为 5 种方法 × 2 个数据集 × 2 个 seeds = 20 行。未准入方法不能借用历史高分、只取共同交集，或在 Test 上补造缺失预测。适配版 ET-BERT/TFE/Trident 如需比较，必须另立清楚命名的协议并预先冻结训练配置；不得事后插入本轮原生主表。
