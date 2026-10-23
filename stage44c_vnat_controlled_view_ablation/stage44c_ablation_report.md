# Stage44C — VNAT 同配置三视图分支消融结果

日期：2026-10-23；执行：Codex；状态：**16/16训练和评估完成，独立重放PASS**。

## 1. 这轮实验回答了什么

本轮固定Stage44四个Service LOSO协议，分别去掉FIG、YaTC或TrafficFormer提供的信息，检验各分支对当前融合分类与DES-v1拒识的贡献。

三路编码器、适配器、Known-Train scaler全部共享原冻结checkpoint。四种配置均保留三个64维槽、1/3缩放、192维拼接、192→128→128→C融合头和0.1 dropout；去掉的槽填零。每个头从相同初始化开始，seed2022、Adam1e-3、batch256、CE、完整30轮，按Known Validation Macro-F1选择最佳epoch。不是直接在已训练的Full头上临时抹掉某个输入。

因此，本轮控制了适配器、分类头结构、训练预算、数据与选模差异，可以解释**现有冻结表示下的分支信息贡献**。它没有单独检验高斯适配器相对无适配器的增益，也没有单独检验MLP相对线性头的增益，不代表从头联合训练所有结构的消融。

四折各训练Full、without_fig、without_yatc、without_trafficformer，共16个头。全部选模并写出HEADS_FROZEN.json后才进行本轮Test输入推理。开集支撑只用Known Train，median/MAD和P95只用Known Validation；各配置使用相同阈值规则，阈值数值根据其自身验证分数确定。

## 2. 四折平均结果

以下是四个fold的等权平均，单seed2022；单位均为百分比。闭集指标来自未应用拒识的Known Test分类；Unknown F1是Unknown为正类的二分类F1。开集使用DES-v1、自然测试比例和Known-Val P95。

| 配置 | 闭集Accuracy | 闭集Macro-F1 | 闭集Weighted-F1 | AUROC | AUPRC | Unknown F1 | UFAR↓ | Known FRR↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整三路 | 99.887 | 99.871 | 99.887 | 99.222 | 99.500 | 98.398 | 1.091 | 4.455 |
| 去FIG，保留TF＋YaTC | 99.921 | 99.891 | 99.922 | 99.451 | 99.766 | 98.637 | 1.405 | 2.494 |
| 去YaTC，保留TF＋FIG | 99.885 | 99.873 | 99.885 | 96.919 | 98.745 | 89.827 | 14.803 | 4.913 |
| 去TrafficFormer，保留FIG＋YaTC | 99.748 | 99.742 | 99.748 | 98.348 | 98.806 | 94.144 | 8.530 | 4.811 |

四折间标准差、其余MSP/Energy/DES-v0分数、1:1测试视图均保留在CSV中。四折共享源数据、Unknown服务各不相同，标准差不能解释为独立多seed的不确定性。

## 3. 逐折AUROC

每行的服务为该折整体留出的Unknown服务。

| Unknown服务 | 完整三路 | 去FIG | 去YaTC | 去TrafficFormer |
|---|---:|---:|---:|---:|
| 通信 | 99.545 | 99.785 | 99.448 | 99.057 |
| 文件传输 | 99.476 | 98.917 | 98.695 | 97.882 |
| 远程访问 | 98.763 | 99.237 | 91.404 | 98.011 |
| 流媒体 | 99.106 | 99.865 | 98.128 | 98.441 |

去YaTC与去TrafficFormer均在4/4折降低AUROC。去FIG在3/4折提高AUROC，在文件传输折降低AUROC，不能把FIG描述为所有折均有收益的模块。

## 4. 逐折UFAR

UFAR表示未知流被错误接受为已知的比例，越低越好；同样单位为百分比。

| Unknown服务 | 完整三路 | 去FIG | 去YaTC | 去TrafficFormer |
|---|---:|---:|---:|---:|
| 通信 | 0.045 | 0.317 | 0.000 | 1.857 |
| 文件传输 | 0.562 | 1.601 | 15.383 | 24.974 |
| 远程访问 | 3.755 | 3.645 | 42.750 | 7.290 |
| 流媒体 | 0.000 | 0.057 | 1.077 | 0.000 |

### YaTC的贡献

最明显的是远程访问留出：去YaTC后的Known Test Macro-F1仍为100%，但AUROC从98.763%下降到91.404%，UFAR从3.755%升至42.750%，Unknown F1从97.980%降至72.698%。

这给出了本轮所需的直接对照：**已知类别仍然分得准，并不代表未知拒识仍然有效。**在当前冻结表示和训练配方下，YaTC包含了其余两个分支没有完全替代的拒识信息。不能只凭平均闭集Macro-F1几乎不变，认定可以安全去掉YaTC。

### TrafficFormer的贡献

去TrafficFormer后平均AUROC下降0.875个百分点，平均UFAR从1.091%升至8.530%；文件传输留出的UFAR从0.562%升至24.974%。这支持TrafficFormer对本协议的未知区分提供补充信息，但不意味着每一个工作点指标都必须在每折下降。

### FIG的贡献与取舍

去FIG后平均AUROC提高0.229个百分点，平均Known FRR降低1.961个百分点；但平均UFAR升高0.314个百分点。文件传输折中，保留FIG的Full在AUROC、UFAR上更好，同时Known FRR更高。

因此，本轮**没有证实FIG具有稳定的整体净收益，也没有证实完整三路在所有评价口径下优于TF＋YaTC**。FIG在不同Unknown服务及误拒/误接受指标之间存在取舍，不能只保留对三路有利的指标。此前细粒度任务的图分支证据与本轮粗粒度VNAT结果应分别报告。

## 5. 基准复现与完整性

- 四折新Full头与历史Stage44 Full头的权重逐参数一致，最大绝对差为0；最佳epoch均为1，但实际完整训练了30轮，没有early stopping。
- Full在32条方法×测试视图配对记录中的AUROC、AUPRC、UFAR、Known FRR和Unknown F1与历史Stage44差值全部为0。
- 16个best head均保存并冻结SHA256；四种配置在每折的初始状态哈希相同。
- 独立重放128行开集指标、145,768条逐样本分数记录，最大指标差为0；384个固定样本的CPU头前向及支撑距离重算通过。
- 97个登记文件前后哈希一致，其中包含原模型、适配器、scaler、角色清单及执行代码；另行核查7个输入缓存数组共124,046,106字节，前后SHA256一致。
- Unknown训练、Unknown验证、Test选模、Unknown支撑/校准用量均为0。没有生成新划分。
- 单卡物理GPU6串行执行，队列完成约319秒，tmux退出码0；独立重放及实验包哈希核查均PASS。

## 6. 仍然保留的限制

1. 当前已知测试没有覆盖所有application，VPN仅381/23,449且集中于通信服务；本轮不提供此前缺失的Known-Zoiper等应用测试证据，也不能证明均衡的跨隧道泛化。
2. Known Test服务类别只有三类且性能接近满分，闭集指标存在天花板效应。
3. 本轮复用已训练适配器，并固定它们的共同选模结果，结论是条件于这些冻结表示的分支消融。不同适配器或训练全过程的因果贡献仍需另行控制实验。
4. 去分支使用零槽保持头结构，但有效信息维度随消融变化。推理阶段为共享缓存仍计算了全部分支，运行耗时不能作为实际删分支后的端到端加速结果。
5. VNAT Test已用于历史开发，本轮是预登记的补充诊断；不得据此选择最有利的测试划分，或把四折当成四个独立seed。

## 7. 可用于论文的结论

> 在相同VNAT流样本、服务标签、冻结编码器与适配器、融合头结构及训练选择规则下，移除YaTC或TrafficFormer均降低了四个服务留出任务的未知检测AUROC。特别是远程访问留出任务，去除YaTC后已知分类Macro-F1仍为100%，而UFAR由3.755%升至42.750%，说明闭集精度不能替代未知拒识评价。FIG的效果依赖未知服务和工作点：去除FIG提高了平均AUROC并降低了已知误拒率，但平均未知误接受率略升，未观察到完整三路对TF＋YaTC的全面优势。

上述结论限定于本轮单seed、已开发VNAT协议及冻结表示。实验完成后停止，不自动变更主方法。

## 8. 文件与重放

- `run_results.csv`：128行方法×视图×分支配置×fold结果。
- `closed_results.csv`、`variant_summary.csv`：单run闭集结果与四折汇总。
- `paired_comparison.csv`：各去分支配置减Full的逐折差值。
- `historical_full_comparison.csv`：新Full与原Stage44的配对核对。
- `runs/<fold>/<variant>/`：best checkpoint、30轮训练动态、per-class指标、confusion matrix、校准参数、四角色表示、逐样本预测/分数。
- `independent_replay_verification.json`、`completion_verification.json`、`input_cache_hashes_before/after.json`：验收证据。
- 实际运行：在固定DGL Conda环境和命名tmux中，以GPU选择器`--min-free-gb 12 --allowed 6,7 --max-utilization 30`启动`python -B -u stage44c_vnat_controlled_view_ablation/scripts/run_ablation.py`；物理GPU实际选中6。
- 重放命令：`python -B stage44c_vnat_controlled_view_ablation/scripts/replay_results.py`，仍需遵循工作区tmux执行规则。
