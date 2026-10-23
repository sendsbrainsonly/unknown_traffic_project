# Stage44C — VNAT 同配置三视图分支消融

登记日期：2026-10-23；owner: codex。配置在本轮新训练和Test推理前写入。

## 研究问题

在现有三路适配器、分类头结构和训练规则不变的条件下，FIG、YaTC和TrafficFormer各自提供多少已知分类与未知拒识信息？

## 固定条件

沿用Stage44四个Service LOSO fold、seed2022、全部flow ID及Known/Unknown角色。三个编码器、三个高斯适配器和Known-Train scaler全部冻结，复用原checkpoint。此处隔离的是现有表示下的分支信息贡献，不重新比较历史双分支与三分支的整套训练流程。

四种配置为Full、去FIG、去YaTC、去TrafficFormer。所有配置保留三个64维输入槽，缺失分支置零，始终乘1/3后拼接为192维，使用相同192→128→128→C分类头。因此不会因改成两路而同时改变头宽度、输入缩放或初始化的随机数消耗。零槽对应的有效输入自由度减少是去信息操作本身的结果，不声称有效参数完全等价。

每折四个头均从相同随机初始化开始，以相同batch顺序和种子训练30轮，Adam lr1e-3、batch256、CE、dropout0.1。只按Known Validation Macro-F1选模；相同分数保留较早epoch。无early stopping，不调整超参数。共16个头，单GPU串行运行。

先完成所有折的训练及选模并保存HEADS_FROZEN.json，再读取本轮Test输入进行固定推理。历史Stage44 Test已暴露，本轮是development诊断，不称为独立外部验证。

## 评价

- 闭集：Known Test Accuracy、Macro-F1、Weighted-F1、per-service Precision/Recall/F1及confusion matrix。
- 开集：同一128维融合表示上的MSP、Energy、DES-v0、DES-v1；AUROC、AUPRC、Binary Accuracy、Unknown Precision/Recall/F1、UFAR、Known FRR。
- 支撑只用Known Train；median/MAD和P95只用Known Validation。k=10、global/local=0.5/0.5、higher分位数、score>threshold判Unknown。
- 同时报自然比例与Stage44相同规则的1:1子集；主比较是各去分支版本减去本轮重训Full。保留与历史Full的数值一致性检查，不能将新Full与旧Full混用。
- 四折逐折差值与描述性均值全部报告，不根据Test选择分支或删折；四折不是四个独立seed，不据此宣称统计显著。

## 原有协议限制

本轮保持现有已知测试应用覆盖缺口及VPN/non-VPN失衡，目的是避免同时改变划分和网络。结论仅适用于该协议覆盖的样本；不能补足Zoiper等应用作为Known的独立Test证据，也不能证明跨隧道条件的泛化。

## 交付

配置、源文件SHA256、16个best checkpoint、完整训练动态、逐样本预测/分数、逐类指标、单run与配对汇总、历史Full比较、独立重放及完整性核验、RESULTS.md和manifest.json。全部新增文件置于本目录，Stage44源文件只读。
