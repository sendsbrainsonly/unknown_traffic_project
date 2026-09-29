# Stage 44 — VNAT Coarse Service-Level LOSO Open-Set Benchmark

## Objective

在不修改 Stage 14/39 资产的前提下，将 VNAT 的 10 个 application 合并为 4 个语义 Service，并用更严格的 group/capture-disjoint 协议评估当前三路模型的闭集与开集能力。

## Frozen service taxonomy

| Service | Applications |
|---|---|
| Streaming | netflix, vimeo, youtube |
| File-Transfer | rsync, scp, sftp |
| Communication | skype, zoiper |
| Remote-Access | ssh, rdp |

VPN/non-VPN 仅作为 domain metadata，不拆成新类别。一个 Service 被留作 Unknown 时，其所有 application 和两个 domain 整体留出。

## Protocol

- Source clean pool: Stage 14B 的 23,449 个唯一 flow。
- Seed: 2022，单 seed 诊断实验。
- Four LOSO folds: held-out Streaming / File-Transfer / Communication / Remote-Access。
- Known Service 的 Train/Validation/Test 目标比例为 8:1:1，但以 `group_id` 为不可拆分原子；若 group 极端不平衡，报告实际比例，不拆 group 追求表面比例。
- 一个 `group_id` 和其 capture 不得跨 Known Train/Validation/Test。
- Unknown flow 使用量：encoder training=0、validation=0、normalization=0、support fitting=0、threshold calibration=0。
- Known Test/Unknown Test 只用于最终评价。

## Model and execution

- 当前方法：TrafficFormer byte branch + FIG graph branch + YaTC MFR branch。
- 每路特征经 64-D variational adapter，三路等权后拼接为 192-D，再由 fusion head 分类。
- 复用 Stage 31/39 已验证的模型、损失和训练预算；只改变本阶段冻结的样本 membership 与标签粒度。
- 单卡串行；每次 GPU workload 前重新按实时容量选择一张物理 GPU。
- 不根据 Test 或 Unknown 结果调参。

## Scores and metrics

- Closed set: Accuracy, Macro-F1, Weighted-F1, per-service F1。
- Open set: MSP, Energy, empirical centroid, DES-v1。
- Primary threshold: Known Validation P95；P90/P99 仅敏感性分析。
- AUROC, AUPRC, Unknown-positive Binary F1, UFAR, Known FRR。
- 同时报告自然比例与固定 1:1 Known:Unknown 视图；1:1 只改变评价 membership，不重拟合模型或阈值。

## Interpretation boundary

本实验是已开发 VNAT 数据上的 coarse-task diagnostic，不是 untouched external validation。由于 Remote-Access 和 Streaming 存在极端 group imbalance，结果必须与 split audit 一起解释。Stage 39 的旧结果与失败证据保持不变。
