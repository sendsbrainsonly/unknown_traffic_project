# Stage46 — 原始初始化的三视图联合训练长预算诊断

Owner: [agent:codex]；登记日期2026-10-23；状态：执行前预注册。

## 问题与边界

验证Stage37的20epoch联合训练是否受训练预算限制。不是Stage45的分阶段权重warm-start续训：从与Stage37相同的官方TrafficFormer/YaTC预训练权重、随机FIG/适配器/分类头开始。
只跑ISCX-VPN四服务、ISCXTor五服务；沿用Stage20冻结flow/标签/Train-Val-Test，不删难样本，不运行Unknown/DES。最多两张实时满足容量要求的GPU，每卡一个worker。固定环境、不安装依赖。

## 固定配置

- 原Stage35 JointThreeView：TF768、FIG128、YaTC192，经原适配器各64维，等权1/3、拼接192维、原分类头；全部联合更新。
- fused deterministic-mean cross entropy，effective batch64，microbatch32，sample-weighted gradient accumulation；seed2022；原DataLoader生成器/随机采样规则。
- 完整100epoch，无early stopping；每epochKnown-Val评估。
- TF：官方UER AdamW，基础LR6e-5，原weight_decay_rate分组0.01/0，correct_bias=False；linear、10%warmup。
- FIG：Adam1e-3、原默认其余参数；fusion adapters/head：Adam1e-3、原默认其余参数。
- YaTC：AdamW5e-4、weight_decay0.05、layer_decay0.75；原10%warmup + cosine至1e-6（乘逐层lr_scale）。
- **total_steps = ceil(N_train/64) ×100**。warmup约10epoch；原20epoch约2epoch warmup。本轮20epoch前缀不是原20epoch调度的精确复现。此为长预算/调度视野敏感性实验，不能声称仅改epoch或等FLOPs因果对照；调度函数不改，仅按100epoch计算总步数。
- 归一化复用原函数：图节点scaler、初始化embedding scaler均只fit Known Train；保存与Stage37原scaler的数值差异，不因差异调参。
- 沿用Stage45公共GradScaler原子更新修复：缩放梯度溢出时恢复同批RNG并降低scale重放，所有优化器同步step、最多8次backoff，不丢样本/更新。不更改模型、LR或损失。此数值实现修正相对历史Stage37也明确披露。

## 预指定20/50/100预算

同一100epoch轨迹，在20/50/100边界分别固化“截至此预算Known-Val Macro-F1最高、并列取最早”的checkpoint；不使用Test挑预算/epoch。
保存每epoch验证logits/per-class、Train在线loss/Accuracy/Macro/Weighted-F1、学习率、梯度范数、显存和耗时；边界另做完整KnownTrain eval。
两数据集都完成100epoch且六个选点全部冻结后，才对六模型统一评估Known Test。Stage32分阶段与Stage37原J20只作历史同流参照，报告来源、相同flow-ID/标签和不等训练预算。

保存初始化权重、20/50/100各预算最佳权重，以及20/50/75/100完整恢复状态（优化器、scheduler、GradScaler、RNG、DataLoader generator）。不覆盖历史实验、不按Test增加训练轮数。若OOM或不可恢复非有限值，保留失败状态并停止对应run，不悄悄减batch/修改配置。

## 验收与判断

1. 2/2runs完整100epoch，成功step数闭合，每epoch覆盖每个KnownTrain flow一次。
2. 六个Known-Val选点在新Test访问前SHA256冻结；源协议/权重/源码/缓存哈希前后不变；Unknown访问0。
3. 报告20→50、20→100、50→100配对Accuracy/Macro-F1/Weighted-F1、最佳epoch、Train/Val loss和曲线；历史J20/分阶段A另表比较。
4. 从逐样本logits/身份/标签独立重算全部新指标和混淆矩阵；RESULTS.md、manifest、completion_verification齐全。
5. 单seed、已暴露Test，仅development诊断；不以Test最高分选后续配置。区分更长预算改善、过拟合/无一致收益和证据不足，不保证100epoch一定更好。

完成后停止，不自动启动开集、200epoch或调参。
