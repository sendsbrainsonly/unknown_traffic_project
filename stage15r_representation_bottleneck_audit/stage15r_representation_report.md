# Stage 15R — Representation Bottleneck Audit

## 1. 结论先行

最终 Gate：

```text
DATA_OR_PROTOCOL_BOTTLENECK_IDENTIFIED
```

在 5 个预注册 Known-Train/Validation pilot 上，E1/E2/E3 均未通过扩展 Gate：平均 `ΔMacro-F1` 分别为 `-0.072320/-0.166239/-0.128971`，且都只在 `1/5` 个 protocol 上超过 E0。由此不能把当前低闭集性能归因于单一的 Open-Detect 联合损失、缺少长度/IAT，或 ResNet18 架构不足。

证据更支持数据/任务组成是主要限制：ISCX-VPN 的混淆集中在 Spotify/Netflix/Vimeo，ISCXTor 集中在 Facebook/Hangouts/Vimeo/YouTube；VNAT 的 hard protocol 中 rsync→scp 达 `188/191 = 98.43%`，但当 scp 不在 Known 集合时同一数据集的 normal protocol Macro-F1 可达 `0.983318`。这说明 class composition 和经验支持重叠会显著改变难度。

本结论不等于“标签错误”或“数据质量差”。本阶段没有发现输入—标签错位、重复归一化或 flow-ID 缺失导致的训练错误；发现的是任务粒度、类别重叠、采集域/分组条件和样本规模共同限制了当前闭集判别。

## 2. 边界与完整性

- 新实验目录独立于 Stage 12、14B/C/D、15A/B。
- 训练与选择仅使用 Known Train/Validation。
- Known Test 特征值使用量：`0`。
- Unknown Test 特征值使用量：`0`。
- E0 复用冻结 checkpoint 和 Known-Validation 结果，不重新训练。
- E1/E2/E3 固定 seed `2022`；未根据 Test 或 Unknown 调参。
- 所有 20 个 `5 protocols × 4 experiments` pilot 均完成。
- full-15 未运行：预注册 Gate 失败；因此没有生成虚假的 `full15_closed_set_results.csv`。

## 3. 输入与数据链审计

### 3.1 E0/E1 字节图实际包含什么

当前输入不是任意“前 1024 个流字节”，而是最多前 8 个 packet 的固定槽位：

```text
8 packets × 128 bytes = 1024 bytes → row-major 32×32 uint8 image
```

每个 packet 槽位由 80 字节 IPv4/header serialization 与最多 48 字节 Scapy `Raw` payload 组成。Ethernet 不进入槽位；IPv4 源/目的地址被清零；TCP/UDP 端口及其他未显式清除的头字段仍可见。packet boundary 由固定 128-byte 槽位隐式保留，但方向和 IAT 不在 E0/E1 输入中。短流零填充，超过第 8 个 packet 的内容截断。

训练图像使用 `ToTensor()` 映射到 `[0,1]`；E1 训练期有 random crop 与 horizontal flip，验证无增强。没有额外 dataset mean/std scaler。

潜在 shortcut 审计结论：sample ID、filename、split 标记和绝对时间不进入模型；IP 地址已清零；端口仍是潜在 shortcut。ISCX/VNAT 的 application label 源于 capture/file metadata，而非独立逐流人工标签，因此不能声称完全消除了 capture-domain shortcut。

### 3.2 flow construction 并非四个数据集完全同义

- ISCX-VPN：按 Stage 12 的双向 IPv4 TCP/UDP、60 秒 timeout、TCP SYN/FIN/RST boundary 回放；本阶段用 tshark 恢复全流统计与前 8 packet 序列。
- ISCXTor2016：同一 Stage 12 sessionization；本阶段用 Scapy `PcapReader` 恢复，15,096/15,096 flow 严格闭合。
- VNAT：复用 Stage 14B/14C flow UID 与现有缓存；flow-random 8:1:1，不是 capture-disjoint。
- USTC：复用 Stage 0 PKL 和已核验的 flow-ID/image alignment。

因此四个数据集的结果可用于受控诊断，但不能把数值差异全部解释为 encoder 差异。

### 3.3 短流分布

| Dataset | Flow scope | 1 packet | 2–4 | 5–10 | 11–20 | >20 |
|---|---:|---:|---:|---:|---:|---:|
| ISCX-VPN | 22,142 | 25.39% | 61.44% | 3.81% | 2.68% | 6.68% |
| ISCXTor2016 | 15,096 | 44.01% | 25.24% | 10.17% | 3.68% | 16.91% |
| VNAT | 23,447 development-visible | 0.13% | 85.28% | 0.72% | 3.63% | 10.24% |
| USTC-TFC2016 | 489,101 | 2.47% | 55.12% | 20.08% | 6.89% | 15.43% |

VNAT 的 23,447 不是 Stage 14B clean pool 丢了 2 条。Stage 14C.5 cache 的定义是“至少在一个冻结 protocol 中进入 Known Train/Validation 的 flow union”；Stage 14B 的另外 2 条 ssh flow 在所有可见位置都只属于 Known Test，因此本阶段刻意没有打开其特征值。Stage 14B clean pool 仍为 23,449，协议未修改。

短流比例高是输入信息受限的事实，但不是充分原因：USTC 也有 `57.59%` 的 1–4 packet flow，却保持高验证性能；而显式长度/IAT 的 E2 并未稳定改善 ISCX。

## 4. 受控 pilot 结果

全部结果均为 Known Validation；不是 Known Test。

| Dataset / protocol | E0 Acc | E0 Macro-F1 | E1 Macro-F1 | E2 Macro-F1 | E3 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| ISCX-VPN / medium-2022 | 0.786468 | 0.769169 | 0.643527 | 0.384810 | 0.466724 |
| ISCXTor / medium-2022 | 0.686540 | 0.676900 | 0.528679 | 0.425010 | 0.487714 |
| VNAT / medium-2025 hard | 0.896939 | 0.739755 | 0.669661 | 0.761427 | 0.774562 |
| VNAT / medium-2026 normal | 0.995290 | 0.983318 | 0.960076 | 0.856865 | 0.846497 |
| USTC / A-2 | 0.976830 | 0.977545 | 0.983143 | 0.887379 | 0.926336 |

### 4.1 Paired Macro-F1 变化

| Dataset / protocol | E1−E0 | E2−E0 | E3−E0 |
|---|---:|---:|---:|
| ISCX-VPN / medium-2022 | -0.125641 | -0.384359 | -0.302445 |
| ISCXTor / medium-2022 | -0.148221 | -0.251890 | -0.189186 |
| VNAT / medium-2025 hard | -0.070093 | +0.021673 | +0.034807 |
| VNAT / medium-2026 normal | -0.023242 | -0.126453 | -0.136821 |
| USTC / A-2 | +0.005598 | -0.090166 | -0.051210 |
| **5-pilot mean** | **-0.072320** | **-0.166239** | **-0.128971** |

E1 是“相同 32×32 输入上的 ResNet18+CE 对照”，不是严格只替换一个 loss 的同构消融：它移除了 VAE/reconstruction、KL、prototype 路径，并采用普通分类头。因此 E1 下降只能否定“CE-only 已明显更好”，不能精确量化某一个 Open-Detect loss 分量的因果贡献。

### 4.2 Gate

| Candidate | Positive | Negative | Mean ΔMacro-F1 | Worst Δ | Clear Gate |
|---|---:|---:|---:|---:|---|
| E1 | 1/5 | 4/5 | -0.072320 | -0.148221 | FAIL |
| E2 | 1/5 | 4/5 | -0.166239 | -0.384359 | FAIL |
| E3 | 1/5 | 4/5 | -0.128971 | -0.302445 | FAIL |

预注册 clear gate 要求 mean `ΔMacro-F1 ≥ +0.03`、至少 `4/5` positive，且 worst decrease `> -0.02`。三个候选也都未达到 diagnostic signal 门槛。因此：

```text
full15_status = NOT_RUN_GATE_FAILED
```

## 5. 类别级失败模式

### ISCX-VPN

E0 的主要错误集中在相近流媒体/应用类别：Spotify→Vimeo `17.7%`，Netflix→Spotify `13.6%`，Netflix→Vimeo `13.6%`，Vimeo→Netflix `12.2%`。E1 后 Spotify→Vimeo 增至 `31.6%`。E2/E3 则出现多类系统性塌缩到 Facebook，说明简单时序/统计输入不足以独立恢复细粒度 application semantics。

### ISCXTor2016

E0 的主要错误包括 Facebook→Vimeo `38.8%`、Hangouts→Vimeo `29.0%`、YouTube→Vimeo `26.5%` 和 Vimeo→YouTube `19.7%`。E2/E3 大量类别被预测为 FTP，例如 Spotify→FTP 约 `49%`。Tor/Non-Tor 混合的细粒度应用任务并未被简单长度、IAT 或汇总统计解决。

### VNAT

hard Medium-2025 中 rsync→scp 为 `188/191`，E1 进一步为 `189/191`；E2/E3 将其降到 `136/191` 和 `132/191`，因此 hard protocol Macro-F1 小幅提高。但 Medium-2026 不包含 scp 作为 Known，E0 已达 `0.983318`，E2/E3 反而分别下降 `0.126453/0.136821`。这不是稳定的 feature gain，而是 class-composition-specific rescue。

### USTC

E0 已达 `0.977545` Macro-F1；E1 提升到 `0.983143`，绝对增益 `+0.005598`，不足以达到预注册候选门槛。主要残余混淆 Virut→Neris 从 E0 的 `19.2%` 降为 E1 的 `16.3%`。E2/E3 明显较弱，说明 USTC 的高性能仍依赖字节/header/payload 表示，而非仅靠前 8 packet 长度/IAT或汇总统计。

## 6. 对十个问题的明确回答

1. **ISCX-VPN 当前约 0.743 Known-Test Accuracy 的主要瓶颈是什么？** 本阶段不重新读取 Known Test；在相同 frozen protocol 的 Known Validation 上，主要证据指向细粒度 application class overlap、流量很短和 flow/capture protocol 限制的组合，而不是单一训练目标或简单特征缺失。E1/E2/E3 均显著低于 E0。
2. **ISCXTor2016 当前约 0.723 Known-Test Accuracy 的主要瓶颈是什么？** 同样以 validation 诊断；Tor/Non-Tor 下应用类别高度重叠，69.25% flow 只有 1–4 packet，且 flow-disjoint 协议不是 unseen-capture 协议。E1/E2/E3 全部下降，未发现简单替代表征可解决问题。
3. **VNAT 是否存在相同问题？** 部分相同，但更强地表现为 Known class composition 敏感性。rsync/scp 同时 Known 时极难；scp held out 后 E0 接近满分。VNAT 不能概括为统一的 feature bottleneck。
4. **USTC 为什么明显更高？** USTC 训练规模远大、类别由 benign applications 与 malware families 构成、类别可分性更高；其字节/header/payload 输入已有强信号。这个差异不能仅归因于数据是否加密。
5. **同样输入下 CE-only 是否超过 Open-Detect？** 不稳定。仅 USTC `+0.005598` Macro-F1；其余 4/5 protocol 下降，平均 `-0.072320`。答案是否。
6. **时序特征是否明显优于当前字节输入？** 否。E2 平均 `-0.166239`，仅 VNAT hard protocol `+0.021673`。
7. **简单统计特征是否已取得强分类结果？** 否。E3 在 USTC 尚可达 `0.926336` Macro-F1，但仍低于 E0；VPN/Tor 仅 `0.466724/0.487714`。它只对 VNAT hard composition 有局部增益。
8. **当前应优先修改什么？** 优先审计并严格化任务/协议：capture/group-disjoint sensitivity、class-pair overlap、标签来源和每类有效支持；不应先换复杂网络或直接融合更多特征。
9. **是否有标签、任务定义或 flow construction 问题证据？** 没有发现 label/image 错位、重复 normalization 或未解释的 flow 丢失。存在明确有效性限制：filename/capture-derived label、flow-disjoint fallback、VNAT flow-random split、端口可见和不同 dataset 的 sessionization/cache 语义。它们是协议风险，不等于已证明标签错误。
10. **下一阶段应开发哪种 representation？** 当前没有候选达到冻结标准，不建议立即开发并冻结新 representation。完成 group-aware/label-overlap audit 后，若仍需模型方向，最合理的单一候选是保留 packet boundary、direction、IAT 与 bytes 的 pretrained packet/flow hierarchical encoder，并在相同 frozen Known split 上与 E0 比较；不能用本阶段结果直接宣称 ET-BERT/YaTC 会提升。

## 7. 限制

- 每个 ISCX 数据集只运行一个 representative protocol；这是 pilot 诊断，不是全 protocol 性能估计。
- E2 只观察前 8 packet，以匹配 E0 information horizon；它不能否定更长时序模型。
- E3 的 VNAT full-flow forward/reverse counts 和 payload bytes 不可用，按预注册规则保留为 missing；VNAT direction changes 只来自前 8 packet。
- E0 与 E1 并非严格同构 loss-only 消融。
- 本阶段没有访问 Test 性能，也没有运行 Unknown detection，因此不对 AUROC 或 DES/H1 作新结论。

## 8. 最终决定

```text
Gate: DATA_OR_PROTOCOL_BOTTLENECK_IDENTIFIED
Full-15: NOT_RUN_GATE_FAILED
New representation frozen: NO
Known Test used for selection: 0
Unknown Test used: 0
DES/H1/Adaptive Hybrid modified: NO
```

Stage 15R 到此停止。
