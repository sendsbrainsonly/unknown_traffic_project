# Stage47 — 表4-2当前三视图原服务标签补充对照

登记：2026-10-23，[agent:codex]。用户要求补跑当前主方法在表4-2的同协议结果。

## 固定协议

- ISCX-VPN六类、ISCXTor七类，原Stage20 Service标签，不合并Communication、不重划分、不删除样本。
- encoder seed与adapter/head seed配对为2022/2022、2023/2023；不是从Stage30五个head seed中挑最好者。
- Train/Val/Test：VPN8764/1098/1093，Tor8946/1118/1117；所有原始类别在三个角色都有支持。
- 复用各seed已经按相同标签与Known Train/Val完成训练的TF/FIG/YaTC分支及Stage22/27特征；不重复训练编码器、不使用Stage46长程联合结果。

## 当前方法保持不变

直接导入Stage31/32实际使用的Adapters、EqualFusion、encode/infer/metrics及Stage32 normalize，不重写模型。
三视图输入768/128/192，各64维Gaussian适配器；训练loss=三分支meanCE＋0.05meanKL，logvar clamp[-8,4]，Train随机重参数、Val确定性mu。
适配器30完整epoch，Adam1e-3、batch256，Known-Val平均单视图Macro-F1选模、并列取最早。冻结适配器后，固定1/3加权拼接192维、原128→128分类头/dropout0.1，CE、Adam1e-3、batch256、30完整epoch，Known-Val融合Macro-F1选模。
保持KnownTrain两级既有标准化：Stage22/27原始scaler及Stage32额外view scaler。不得因细标签结果改loss、网络、epoch或参数。旧Stage30同结构探索没有Stage32这一显式额外view标准化步骤，因此不直接把其历史分数作为当前配置结果。

## 运行设备与资源

本轮只训练缓存表示上的轻量适配器/分类头，**预先固定CPU、4线程、四run串行、float32**；CUDA_VISIBLE_DEVICES置空。不是GPU容量不足时静默fallback；不改任何已有GPU任务，Stage46仍按原队列独立推进。CPU与GPU的数值/随机数实现可能不同，在结果限制中披露，不宣称逐浮点位等同历史GPU训练。
开始前用Stage32已完成模型的Known-Val logits做CPU前向parity，验证数据标准化与模型结构复用正确。

## 冻结与输出

四run全部完成并冻结8个checkpoint SHA256后才能读取KnownTest feature keys；KnownTrain/Val仅用于训练/选模/scaler。Unknown使用0、Test选参0。历史Test已暴露，属于development，不称untouched external。
保存完整训练曲线、配置、scaler、8权重、Train/Val身份标签、Val/Test逐样本预测/logits/per-class/confusion、hash前后证据。
汇总四单run、两seed均值±总体标准差(ddof=0)，以Stage23B原28行作为六外部方法及旧E3参照，新增当前方法4行，生成八配置对照表。先报告真实结果，不保证高于YaTC。
独立重算逐样本指标并加载checkpoint再次前向重放。生成RESULTS.md、manifest.json、completion_verification.json、table4_2_supplement.md、run_results.csv、method_summary.csv、paired_comparison.csv。原表与历史实验不覆盖；本任务不自动开始开集或其他调参。
