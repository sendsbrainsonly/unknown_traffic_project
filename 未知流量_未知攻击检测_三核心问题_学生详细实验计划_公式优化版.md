# 未知流量 / 未知攻击检测：三核心问题详细实验计划

> **版本定位**：学生执行版  
> **核心主线**：Known-Space Multimodality → Prototype-Specific Boundary → Fine-Grained Unknown Class Discovery  
> **原则**：严格 Unknown-Free；逐阶段验收；每一步必须保留前一版本；所有正式结果至少 3 个随机种子并报告 Mean ± Std。

---

# 1. 研究目标

本课题面向 **Open-Set / Unknown Traffic & Unknown Attack Detection**。训练阶段仅能看到已知类别；测试阶段会出现训练期间从未出现过的流量或攻击类别。

与只做：

```text
Known / Unknown
```

的传统开放集检测不同，本研究进一步考虑三个问题：

## 核心问题 1：Known 类内部是否真的可以用一个单一 prototype 描述？

现实中的同一个已知流量类别，可能因为以下因素形成多个子模式：

- 不同攻击工具；
- 不同恶意软件变种；
- 不同网络环境；
- 不同客户端 / 服务端实现；
- 不同 TLS / 协议配置；
- 不同通信阶段；
- 不同 burst 行为；
- 不同时间段或采集条件。

因此真实的类条件表示分布更可能是：

$$
p(z\mid y=c)
=
\sum_{j=1}^{K_c}
\pi_{c,j}
\mathcal N(\mu_{c,j},\Sigma_{c,j})
$$

而不是简单的：

$$
p(z\mid y=c)
\approx
\mathcal N(\mu_c,\Sigma_c)
$$

或者：

$$
\text{Class}_c \rightarrow \mu_c
$$

本研究使用 **Adaptive Multi-Prototype** 建模每个已知类别内部的多模态结构。

---

## 核心问题 2：不同 local mode 的有效接受范围是否相同？

即使一个类别有多个 prototype，不同局部模式的离散程度也可能明显不同：

$$
\Sigma_{c,1}\neq \Sigma_{c,2}
$$

因此不能所有 prototype 共用统一全局阈值：

$$
D(z,P)<\tau
$$

本研究进一步为每个 prototype 建立独立接受边界：

$$
P_{c,j}\rightarrow r_{c,j}
$$

即 **Prototype-Specific Boundary**。

---

## 核心问题 3：Unknown 本身是否也不是单一类别？

目前大部分 open-set 方法的最终输出为：

```text
Unknown Attack A → Unknown
Unknown Attack B → Unknown
Unknown Attack C → Unknown
```

但真实安全运营中，更有意义的输出应为：

```text
Unknown Attack A → Unknown Class 1
Unknown Attack B → Unknown Class 2
Unknown Attack C → Unknown Class 3
```

因此在完成 Unknown Detection 后，本研究进一步进行：

$$
\boxed{\text{Fine-Grained Unknown Class Discovery}}
$$

目标是自动发现多个未知攻击 / 未知流量族，而不是把所有未知样本压缩成一个统一 `Unknown` 类。

---

# 2. 整体技术路线

```text
                               Raw PCAP
                                  │
                                  ↓
                                  
                                      ┌────────────────────────────┐
                  │  Known-Space Audit         │
                  │  Single vs Multi-Modal     │
                  └──────────────┬─────────────┘
                                 │
                                  
                                  
                        Bidirectional Flow
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
             TrafficFormer                  FIG/TAGCN
                    │                           │
                   z_t                         z_g
                    └─────────────┬─────────────┘
                                  │
                           Concat Fusion
                             z_f=[z_t;z_g]
                                  │
                                  ↓
              
                                 ↓
                     Adaptive Multi-Prototype
                                 │
                                 ↓
                  Prototype-Specific Boundary
                                 │
                                 ↓
                         Open-Set Decision
                         /              \
                     Known            Unknown
                       │                 │
                       ↓                 ↓
                 Known Class       Unknown Buffer
                                         │
                                         ↓
                              Unknown Class Discovery
                                         │
                            ┌────────────┼────────────┐
                            │            │            │
                       Unknown-1    Unknown-2    Unknown-3
```

---

# 3. 实验总原则

## 3.1 严格 Unknown-Free

对于任何 held-out Unknown 类：

$$
C_U
\cap
(D_{\mathrm{train}}\cup D_{\mathrm{val}})
=
\varnothing
$$

Unknown 类：

- 不进入 encoder 训练；
- 不进入 validation；
- 不用于 prototype 构建；
- 不用于 GMM / BIC；
- 不用于 boundary 估计；
- 不用于最终 threshold calibration；
- 不用于超参数选择；
- 只在最终 test 阶段出现。

**注意：真实 Unknown 标签只允许在实验结束后用于评估 AUROC、ARI、AMI 等指标。**

---

## 3.2 分阶段执行

不得一次实现完整系统。

严格按照：

```text
数据处理
↓
Closed-Set Backbone
↓
Known-Space Distribution Audit
↓
Single Prototype Baseline
↓
Adaptive Multi-Prototype
↓
Local Dispersion Audit
↓
Prototype-Specific Boundary
↓
Known-Only Calibration
↓
Oracle Unknown Discovery
↓
End-to-End Unknown Discovery
```

只有前一阶段达到验收标准，才进入下一阶段。

---

## 3.3 固定随机种子

开发阶段可先使用：

```text
seed = 0
```

正式实验至少：

```text
seed = 0
seed = 1
seed = 2
```

推荐最终使用 5 个 seed：

```text
0, 1, 2, 3, 4
```

论文报告：

$$
Mean\pm Std
$$

---

## 3.4 保留所有中间结果

不要只保存最终模型。

每一个阶段必须保存：

- 配置文件；
- random seed；
- train / val / test split；
- embedding；
- prototype 参数；
- GMM 参数；
- boundary；
- threshold；
- per-sample score；
- per-sample prediction；
- 最终指标；
- 运行日志。

---

# 4. 推荐项目目录结构

```text
project/
├── configs/
│   ├── dataset/
│   ├── encoder/
│   ├── openset/
│   └── clustering/
│
├── data/
│   ├── raw_pcap/
│   ├── flows/
│   ├── trafficformer_input/
│   ├── fig_graph/
│   └── splits/
│
├── src/
│   ├── preprocessing/
│   ├── encoders/
│   ├── fusion/
│   ├── prototypes/
│   ├── openset/
│   ├── clustering/
│   ├── metrics/
│   └── visualization/
│
├── scripts/
│   ├── 00_prepare_data.py
│   ├── 01_train_closedset.py
│   ├── 02_extract_embeddings.py
│   ├── 03_distribution_audit.py
│   ├── 04_single_proto_baseline.py
│   ├── 05_multi_proto.py
│   ├── 06_boundary_audit.py
│   ├── 07_proto_specific_boundary.py
│   ├── 08_oracle_unknown_discovery.py
│   └── 09_end2end_unknown_discovery.py
│
├── outputs/
│   ├── stage0_data/
│   ├── stage1_closedset/
│   ├── stage2_distribution_audit/
│   ├── stage3_single_proto/
│   ├── stage4_multi_proto/
│   ├── stage5_boundary/
│   ├── stage6_calibration/
│   ├── stage7_oracle_discovery/
│   └── stage8_end2end_discovery/
│
└── README.md
```

---

# 5. 第一批数据集安排

## 5.1 USTC-TFC2016

用途：

- 方法开发；
- Known-space multimodality audit；
- 主消融；
- open-set traffic / malware family discovery；
- 不同 openness 设置。

优先先把全部方法在该数据集上跑通。

---

## 5.2 CSE-CIC-IDS2018

用途：

- 真正的 Unknown Attack Type Detection；
- 验证不同攻击类型被 held-out 后能否被识别为 Unknown；
- 验证 Unknown Class Discovery 是否能恢复真实攻击类型。

---

## 5.3 CICIDS2017

用途：

- 第二个 IDS benchmark；
- 与 CICIDS2018 联合设计 cross-dataset unknown detection。

---

## 5.4 CTU-13

用途：

- unseen botnet / malware family；
- 验证未知恶意流量 family discovery。

---

# 6. 阶段 0：PCAP → Flow → 双编码器输入

## 6.1 目标

完成：

```text
PCAP
↓
Bidirectional Flow
↓
TrafficFormer Input
+
FIG/TAGCN Input
+
Label
```

本阶段不训练模型。

---

## 6.2 具体任务

### Task 0.1：读取原始 PCAP

要求：

- 保存 packet timestamp；
- 保存 packet length；
- 保存 source / destination IP；
- 保存 source / destination port；
- 保存 L4 protocol；
- 保存 payload / raw bytes；
- 保存 packet direction。

---

### Task 0.2：双向流构建

使用五元组：

$$
(srcIP,srcPort,dstIP,dstPort,protocol)
$$

反向五元组视作同一 bidirectional flow。

必须保证：

```text
A:port1 → B:port2
B:port2 → A:port1
```

属于同一 flow。

每条 flow 记录：

```text
flow_id
class_label
packet_count
start_time
end_time
forward_packets
backward_packets
```

---

### Task 0.3：TrafficFormer 输入

严格按照 TrafficFormer 原始方法要求构造。

必须保存：

```text
flow_id
trafficformer_input
```

不能只保存 tensor，必须保留 `flow_id` 用于与 FIG 对齐。

---

### Task 0.4：FIG 输入

同一个 `flow_id` 构造 Flow Interaction Graph。

必须保存：

```text
flow_id
graph
```

---

### Task 0.5：一致性检查

随机抽取至少 50 条 flow，人工检查：

- TrafficFormer 与 FIG 是否来自同一 flow；
- 标签是否一致；
- packet 数量是否一致；
- packet 顺序是否正确；
- timestamp 是否正确；
- direction 是否正确；
- 空 flow / 单包 flow 是否处理合理。

---

## 6.3 必须输出

```text
dataset_statistics.csv
flow_index.csv
data_validation_report.md
```

其中：

### dataset_statistics.csv

```text
class_name
flow_num
packet_num
avg_packets_per_flow
median_packets_per_flow
min_packets_per_flow
max_packets_per_flow
```

---

## 6.4 验收

随机一个 `flow_id` 必须能够得到：

```python
trafficformer_input
fig_graph
label
```

且三者完全对齐。

---

# 7. 阶段 1：Closed-Set Backbone 验证

## 7.1 目标

先确保 representation 本身可用。

训练：

### Model A

```text
TrafficFormer
↓
z_t
↓
Classifier
```

### Model B

```text
FIG
↓
TAGCN
↓
z_g
↓
Classifier
```

### Model C

```text
TrafficFormer → z_t ─┐
                     ├→ Concat → z_f → Classifier
FIG/TAGCN    → z_g ─┘
```

第一版固定：

$$
z_f=[z_t;z_g]
$$

暂时不加 Dynamic Fusion。

---

## 7.2 数据

USTC-TFC2016 全 20 类作为 Known：

```text
Train: 20 classes
Validation: 20 classes
Test: 20 classes
```

建议：

```text
60% / 20% / 20%
```

或者如果已有官方 split，则优先使用官方划分。

---

## 7.3 指标

- Accuracy；
- Macro-F1；
- Macro-Precision；
- Macro-Recall；
- per-class F1；
- confusion matrix。

---

## 7.4 保存 embedding

训练完成后分别保存：

```text
embedding_train.npy
embedding_val.npy
embedding_test.npy

labels_train.npy
labels_val.npy
labels_test.npy

flow_ids_train.npy
flow_ids_val.npy
flow_ids_test.npy
```

---

## 7.5 验收

两个单分支必须正常收敛。

双分支最好满足：

$$
F1_{\mathrm{TF+FIG}}
\ge
\max(F1_{\mathrm{TF}},F1_{\mathrm{FIG}})
$$

如果双分支下降明显：

优先检查：

1. 两个 flow 是否严格对齐；
2. embedding 数值尺度是否差异过大；
3. 是否需要 LayerNorm / Standardization；
4. classifier 是否过大；
5. concat 后学习率是否需要调整；
6. batch 内两个分支是否一一对应。

此问题未解决前，不进入 open-set。

---

# 8. 阶段 2：Known-Space Distribution Audit

> 这一阶段是整个课题的第一个关键证据实验。

## 8.1 研究问题

验证：

$$
\boxed{
\text{Known class representation 是否普遍具有 multimodal structure？}
}
$$

不要先假设答案是“有”。

---

## 8.2 数据

只使用：

```text
Known Training Embeddings
Known Validation Embeddings
```

不能出现任何 Unknown 类。

---

## 8.3 每类拟合 GMM

对于类别 $c$：

$$
Z_c=\{z_i:y_i=c\}
$$

依次拟合：

```text
GMM K = 1
GMM K = 2
GMM K = 3
GMM K = 4
GMM K = 5
```

第一版：

```python
covariance_type = "diag"
```

不使用 full covariance。

---

## 8.4 使用 BIC 自动选 K

$$
K_c^*
=
\arg\min_K
BIC(GMM_K(Z_c))
$$

记录每个类别：

```text
class
BIC_K1
BIC_K2
BIC_K3
BIC_K4
BIC_K5
selected_K
```

---

## 8.5 必须增加 held-out NLL 检验

仅用训练集选 GMM 后，在 Known validation 上计算：

### Single Gaussian

$$
NLL_{single}
$$

### Selected GMM

$$
NLL_{gmm}
$$

重点观察：

$$
NLL_{gmm}<NLL_{single}
$$

是否普遍成立。

原因：

只看训练集 BIC 仍可能被质疑为过拟合。

---

## 8.6 稳定性实验

至少三种 seed：

```text
0 / 1 / 2
```

对每个类别统计：

```text
K_seed0
K_seed1
K_seed2
```

再做 bootstrap：

```text
80% training samples
repeat 20 times
```

统计：

$$
P(K_c^*=k)
$$

如果某个类别：

```text
K=3 出现 18/20 次
```

说明 multimodality 较稳定。

---

## 8.7 小 cluster 处理

如果某个 GMM component：

$$
N_{c,j}<N_{\min}
$$

则不允许直接保留。

第一版建议尝试：

```text
N_min = max(20, 0.01 * N_c)
```

同时尝试：

```text
0.5%
1%
2%
```

作为敏感性实验。

或者设置最小 mixture weight：

$$
\pi_{c,j}\ge 0.01
$$

---

## 8.8 可视化

每种类型挑代表类别：

```text
K=1
K=2
K≥3
```

绘制：

- UMAP；
- t-SNE；
- GMM assignment；
- prototype center。

禁止根据 UMAP 图直接证明高维空间一定多峰。

可视化只作为辅助。

---

## 8.9 关键统计量

最终报告：

$$
R_{multi}
=
\frac{
|\{c:K_c^*> 1\}|
}{C}
$$

以及：

```text
mean selected K
median selected K
max selected K
```

---

## 8.10 阶段输出

```text
bic_per_class.csv
validation_nll.csv
selected_k_per_seed.csv
bootstrap_k_stability.csv
cluster_size_statistics.csv
distribution_audit_summary.md
figures/
```

---

## 8.11 验收标准

满足以下至少两条：

1. 较多类别稳定选择 $K> 1$；
2. GMM 在 Known validation 上 NLL 明显优于 single Gaussian；
3. bootstrap 下 selected K 具有稳定性；
4. 可视化出现与统计结果一致的多峰结构。

如果几乎所有类别都稳定为 $K=1$，则停止 Multi-Prototype 主线，先重新检查 embedding。

---

# 9. 阶段 3：Single Prototype Open-Set Baseline

## 9.1 目标

建立最基础：

$$
\text{1 class}\rightarrow\text{1 prototype}
$$

baseline。

---

## 9.2 第一组 Open-Set Split

USTC-TFC2016：

```text
Known = 16 classes
Unknown = 4 classes
```

Unknown 完全 held-out。

为了减少 split 偶然性：

至少构造 3 套不同 class split。

例如：

```text
split_A
split_B
split_C
```

正式实验中每套 split 再运行 3 个 seed。

---

## 9.3 Prototype

$$
\mu_c
=
\frac1{N_c}
\sum_{i:y_i=c} z_i
$$

---

## 9.4 距离

第一版同时跑两种：

### Euclidean

$$
D_E(z,\mu_c)=||z-\mu_c||_2
$$

### Diagonal Mahalanobis

$$
D_M(z,\mu_c)
=
(z-\mu_c)^T
\Sigma_c^{-1}
(z-\mu_c)
$$

目的：

后续避免性能提升只是因为“换了 Mahalanobis”。

---

## 9.5 Known-only threshold

在 Known validation 上计算：

$$
d_i=
\min_cD(z_i,\mu_c)
$$

第一版尝试：

```text
Q90
Q95
Q97.5
Q99
```

最终参数必须只根据 Known validation 的行为预先固定，不能看 Unknown test 最优结果后反选。

建议主结果固定：

$$
\tau=Q_{0.95}
$$

其余分位数作为 sensitivity。

---

## 9.6 指标

### Known Classification

- Accuracy；
- Macro-F1；
- Precision；
- Recall。

### Unknown Detection

- AUROC；
- AUPR-Unknown；
- FPR95；
- Unknown Recall；
- Known False Rejection Rate；
- Unknown False Acceptance Rate。

---

## 9.7 保存 per-sample 结果

```text
flow_id
gt_label
known_or_unknown_gt
predicted_known_class
min_distance
threshold
known_or_unknown_pred
```

后续 failure-case 分析必须依赖这些文件。

---

# 10. 阶段 4：Adaptive Multi-Prototype

## 10.1 目标

验证核心问题 1：

> Single Prototype 是否会错误描述 Known Space？

---

## 10.2 Prototype 构建

对于每个 Known 类：

$$
K_c^*
=
\arg\min_KBIC(GMM_K(Z_c))
$$

得到：

$$
P_{c,j}
=
\mathcal N(\mu_{c,j},\Sigma_{c,j})
$$

其中：

$$
j=1,\ldots,K_c^*
$$

第一版：

```text
diag covariance
```

并进行数值稳定：

$$
\Sigma_{c,j}
\leftarrow
\Sigma_{c,j}
+\epsilon I
$$

尝试：

```text
epsilon = 1e-6 / 1e-5 / 1e-4
```

若结果稳定则固定一个值。

---

## 10.3 测试距离

$$
D_{c,j}(x)
=
(z-\mu_{c,j})^T
\Sigma_{c,j}^{-1}
(z-\mu_{c,j})
$$

最终：

$$
D(x)=\min_{c,j}D_{c,j}(x)
$$

候选 known class：

$$
\hat c
=
\operatorname{class}
\left(
\arg\min_{c,j}D_{c,j}(x)
\right)
$$

---

## 10.4 必须比较

```text
A. Single Prototype
B. Fixed K=2
C. Fixed K=3
D. Fixed K=4
E. Adaptive K using BIC
```

重点验证：

$$
Adaptive\ K
>
Fixed\ K
$$

而不是只证明：

$$
More\ prototypes
>
Single\ prototype
$$

---

## 10.5 关键 failure-case 分析

### Case A：Known false rejection

找出：

```text
Single Prototype → Unknown
Adaptive Multi-Prototype → Known
GT = Known
```

统计这类样本数量和比例。

---

### Case B：Unknown false acceptance

找出：

```text
Single Prototype → Known
Adaptive Multi-Prototype → Unknown
GT = Unknown
```

重点可视化这些样本是否位于：

```text
mode A —— artificial single center —— mode B
```

附近。

---

## 10.6 Artificial Known Region 分析

建议增加一个专项实验：

对于每个 class：

1. 计算 single centroid；
2. 找到最近的两个 GMM modes；
3. 计算 centroid 周围真实 training density；
4. 比较 centroid 与 mode center 周围的局部样本密度。

若出现：

$$
density(\mu_c)
<
density(\mu_{c,j})
$$

则说明 single centroid 可能位于低密度区域。

该实验可以成为论文重要 motivation evidence。

---

## 10.7 输出

```text
multi_proto_results.csv
fixed_k_comparison.csv
failure_cases_known.csv
failure_cases_unknown.csv
artificial_region_analysis.csv
prototype_parameters/
figures/
```

---

## 10.8 验收

至少满足一个：

- AUROC 上升；
- FPR95 下降；
- Known FRR 下降；
- Unknown false acceptance 下降；
- Adaptive K 优于多数 Fixed-K；
- failure-case 有明确可解释修复。

---

# 11. 阶段 5：Local Dispersion Audit

> 这是核心问题 2 的证据实验。

## 11.1 目标

验证：

$$
\boxed{
\text{Different local modes have different dispersion.}
}
$$

---

## 11.2 给 validation 样本分配 prototype

对于 Known validation sample：

$$
(\hat c,\hat j)
=
\arg\min_{c,j}D_{c,j}(x)
$$

只保留：

```text
ground-truth class = c
```

的样本用于该 prototype 的边界估计，避免错误 class assignment 污染边界。

---

## 11.3 计算局部半径

$$
\mathcal D_{c,j}
=
\{
D_{c,j}(x_i)
\}
$$

然后：

$$
r_{c,j}^{(q)}
=
Q_q(\mathcal D_{c,j})
$$

尝试：

```text
q = 0.90
q = 0.95
q = 0.975
q = 0.99
```

主方法先固定：

$$
q=0.95
$$

---

## 11.4 分析 boundary heterogeneity

输出：

```text
class
prototype
num_val_samples
Q90
Q95
Q975
Q99
mean_distance
std_distance
```

计算：

$$
CV(r)
=
\frac{\operatorname{std}(r)}
{\operatorname{mean}(r)}
$$

以及：

$$
R_r
=
\frac{r_{max}}{r_{min}}
$$

---

## 11.5 小 prototype 风险

如果某 prototype 的 validation sample 数太少：

```text
< 20
```

不能直接使用经验 Q95。

尝试三种策略：

### Strategy A：合并小 prototype

合并到最近 prototype。

### Strategy B：使用训练 + validation Known 样本估计边界

只用于 boundary estimation，不重新训练 encoder。

### Strategy C：Shrinkage

$$
r_{c,j}
=
\lambda r_{local}
+
(1-\lambda)r_{class}
$$

第一版优先使用 A，保持方法简单。

---

## 11.6 验收

如果：

$$
r_{max}/r_{min}
$$

长期接近 1，而且不同 prototype 的距离分布近似相同，则 prototype-specific boundary 的必要性不足。

如果存在明显差异，则继续下一阶段。

---

# 12. 阶段 6：Prototype-Specific Boundary

## 12.1 目标

把：

```text
all prototypes share one threshold
```

改成：

```text
each prototype has its own valid region
```

---

## 12.2 Normalized Prototype Score

测试样本：

$$
(\hat c,\hat j)
=
\arg\min_{c,j}D_{c,j}(x)
$$

定义：

$$
S_p(x)
=
\frac{
D_{\hat c,\hat j}(x)
}{
r_{\hat c,\hat j}
}
$$

最直接判定：

$$
S_p(x)\le 1
\Rightarrow Known
$$

$$
S_p(x)> 1
\Rightarrow Unknown
$$

---

## 12.3 主消融

严格比较：

| Model | Prototype | Boundary |
|---|---|---|
| M0 | Single | Global |
| M1 | Adaptive Multi | Global |
| M2 | Adaptive Multi | Prototype-Specific |

其中：

- M0 → M1：验证 multimodality；
- M1 → M2：验证 boundary heterogeneity。

---

## 12.4 指标重点

特别关注：

### Known False Rejection Rate

$$
FRR_K
=
\frac{
\# Known\rightarrow Unknown
}{
\# Known
}
$$

### Unknown False Acceptance Rate

$$
FAR_U
=
\frac{
\# Unknown\rightarrow Known
}{
\# Unknown
}
$$

理论上：

Multi-Prototype 主要减少 artificial geometry 问题；

Prototype-specific boundary 进一步平衡：

```text
tight mode
loose mode
```

之间的错误拒绝 / 错误接受。

---

# 13. 阶段 7：Known-Only Final Calibration

## 13.1 目的

保证最终 Open-Set decision 不依赖 Unknown validation。

---

## 13.2 第一版不要复杂化

优先使用 prototype-normalized unknown score：

$$
S_p(x)
$$

仅使用 **Known validation samples** 计算最终阈值。设已知验证集为
$D_{\mathrm{val}}^{\mathrm{Known}}$，则：

$$
\tau
=
Q_{0.95}
\left(
\left\{
S_p(x_i)
\;\middle|\;
x_i \in D_{\mathrm{val}}^{\mathrm{Known}}
\right\}
\right)
$$

其中，$Q_{0.95}(\cdot)$ 表示 95% 分位数。

测试阶段，对任意样本 $x$：

$$
S_p(x) \le \tau
\quad\Longrightarrow\quad
\mathrm{Known}
$$

$$
S_p(x) > \tau
\quad\Longrightarrow\quad
\mathrm{Unknown}
$$

该阈值只能由 Known validation data 确定，不能根据 Unknown test samples 的结果反向调节。

---

## 13.3 比较

```text
Threshold = 1
Known-only Q95
Known-only Q975
Known-only Q99
```

但主方法在开发完成后必须预先固定，不能针对每个 Unknown split 使用 test set 重新选最优。

---

# 14. 阶段 8：核心问题 3 的前置验证——Oracle Unknown Structure Audit

> 这一阶段先绕开 Unknown Detector，只研究 representation 是否能区分不同未知类别。

## 14.1 为什么必须做 Oracle 实验

如果最终 Unknown clustering 差，可能有三种原因：

1. representation 本身没有类别结构；
2. clustering 方法不好；
3. 前面的 detector 漏掉了大量 Unknown 或混入 Known。

Oracle 实验用于隔离问题 1 和 2。

---

## 14.2 数据

直接取 test 中真实 Unknown：

$$
Z_U^{GT}
=
\{
z_i:y_i\in C_U
\}
$$

注意：

真实标签**不能输入 clustering 算法**。

标签只用于最后评估。

---

## 14.3 聚类 Baselines

### Baseline A：K-Means + Oracle K

这里允许：

$$
K=|C_U|
$$

因为这是上限基线，不是最终方法。

---

### Baseline B：GMM + Oracle K

同样作为上限基线。

---

### Baseline C：GMM + BIC

真正 unknown-free：

$$
K_U^*
=
\arg\min_K BIC(GMM_K(Z_U))
$$

搜索：

```text
K = 1 ... Kmax
```

其中：

```text
Kmax = min(20, floor(sqrt(N_unknown)))
```

并做敏感性实验。

---

### Baseline D：HDBSCAN

原因：

- 不需要提前知道 K；
- 支持不同 density；
- 可以输出 noise；
- 适合 unknown buffer。

需要尝试：

```text
min_cluster_size
min_samples
```

但参数不能使用真实 Unknown labels 调最优。

建议依据样本规模设置固定规则，例如：

```text
min_cluster_size = max(10, 0.01 * N_unknown)
```

再做 sensitivity。

---

## 14.4 指标

- ARI；
- AMI；
- NMI；
- Purity；
- Hungarian-matched clustering accuracy；
- estimated number of clusters；
- cluster count error：

$$
|\hat K-K_{GT}|
$$

---

## 14.5 额外分析

对每个 discovered cluster 输出：

```text
cluster_id
sample_num
dominant_gt_class
purity
```

观察：

- 一个真实 unknown class 是否被拆成多个 cluster；
- 多个真实 class 是否被错误合并；
- 哪些 attack family 最难分。

---

# 15. 阶段 9：End-to-End Fine-Grained Unknown Discovery

## 15.1 真正部署流程

```text
All Test Samples
↓
Adaptive Multi-Prototype
↓
Prototype-Specific Boundary
↓
Predicted Unknown Buffer
↓
Unknown Class Discovery
↓
Unknown Class 1 / 2 / ...
```

---

## 15.2 Predicted Unknown Buffer

$$
\hat{\mathcal U}
=
\{
x:S_p(x)> \tau
\}
$$

其中可能包含：

```text
True Unknown
+
False Rejected Known
```

这是现实情况，不能提前清洗。

---

## 15.3 聚类

第一版主方法优先尝试：

```text
GMM + BIC
```

理由：

整篇论文的建模思想统一：

```text
Known:
adaptive mode discovery

Unknown:
adaptive mode discovery
```

同时保留：

```text
HDBSCAN
```

作为重要 baseline。

---

## 15.4 End-to-End 指标

除了：

- ARI；
- AMI；
- cluster accuracy；

还必须报告：

### Unknown Buffer Precision

$$
P_U
=
\frac{
TP_U
}{
TP_U+FP_U
}
$$

### Unknown Buffer Recall

$$
R_U
=
\frac{
TP_U
}{
TP_U+FN_U
}
$$

因为 clustering 的输入质量取决于 detector。

---

## 15.5 两套 Discovery 结果必须同时报告

### Oracle Discovery

```text
GT Unknown
↓
Clustering
```

回答：

> representation 是否能区分未知类别？

### End-to-End Discovery

```text
Predicted Unknown
↓
Clustering
```

回答：

> 完整系统最终能否发现未知类别？

---

# 16. 阶段 10：不同 Openness 正式实验

USTC-TFC2016：

```text
16 Known + 4 Unknown
12 Known + 8 Unknown
8 Known + 12 Unknown
```

如果样本规模允许，再加：

```text
4 Known + 16 Unknown
```

但这组 openness 很高，可作为 stress test，不一定作为主结果。

---

## 16.1 每个 setting

至少：

```text
3 independent class splits
×
3 random seeds
```

即每个 openness 至少 9 次实验。

---

## 16.2 主结果

### Detection

- Known Macro-F1；
- AUROC；
- AUPR-Unknown；
- FPR95；
- Unknown Recall；
- Known FRR；
- Unknown FAR。

### Discovery

- ARI；
- AMI；
- NMI；
- clustering accuracy；
- $\hat K$；
- $|\hat K-K_{GT}|$。

---

# 17. 阶段 11：扩展至 CSE-CIC-IDS2018

这一阶段开始才能重点强调：

$$
\boxed{\text{Unknown Attack Detection}}
$$

---

## 17.1 类别划分原则

例如：

```text
Known:
Benign
DoS-Hulk
FTP-BruteForce
SSH-BruteForce
Bot
...

Unknown:
Heartbleed-like / Web attack / infiltration / selected held-out attacks
```

具体划分必须根据数据集实际 attack class 数量和 PCAP 可用性确定。

要求：

- Unknown attack type 完全 held-out；
- 避免同一攻击 family 的高度重复 variant 同时出现在 Known 与 Unknown；
- 尽量设计 easy / medium / hard 三组 held-out。

---

## 17.2 难度分组建议

### Easy Unknown

与 Known 差异较大的攻击。

### Medium Unknown

与 Known 具有部分相似行为。

### Hard / Near Unknown

Known 攻击的 variant 或行为相近攻击。

这样可以分析：

Multi-Prototype / local boundary 在不同难度下的表现。

---

# 18. 阶段 12：Cross-Dataset Unknown Detection

推荐：

```text
Train Known Space:
CICIDS2017

External Unknown:
CICIDS2018 中不同攻击类型
```

注意 cross-dataset 可能同时包含：

```text
attack shift
+
dataset/domain shift
```

所以不能直接把高 AUROC 全部解释为“未知攻击能力”。

建议增加：

```text
Benign_2018
```

作为 external benign control。

最终测试：

```text
Known attack 2017
Benign 2018
Unknown attack 2018
```

观察模型是否把新的 benign domain shift 错当攻击 Unknown。

---

# 19. 三个核心问题对应的完整证据链

# RQ1：Known multimodality

## Problem

$$
\text{One class}\not\approx\text{one mode}
$$

## Evidence

- BIC-selected K；
- validation NLL；
- bootstrap stability；
- cluster size；
- UMAP。

## Solution

Adaptive Multi-Prototype。

## Performance Evidence

- AUROC ↑；
- FPR95 ↓；
- Known FRR ↓；
- Unknown FAR ↓；
- failure-case 修复。

---

# RQ2：Local boundary heterogeneity

## Problem

$$
r_{c,j}
$$

差异显著。

## Evidence

- radius distribution；
- $r_{max}/r_{min}$；
- CV；
- per-prototype distance KDE。

## Solution

Prototype-Specific Boundary。

## Performance Evidence

M1 → M2：

- Known FRR ↓；
- Unknown FAR ↓；
- AUROC / FPR95 改善。

---

# RQ3：Unknown heterogeneity

## Problem

$$
Unknown\neq one\ homogeneous\ class
$$

## Evidence

Oracle unknown embedding 中真实类别具有可恢复 cluster structure。

## Solution

Fine-Grained Unknown Class Discovery。

## Performance Evidence

- ARI；
- AMI；
- clustering accuracy；
- estimated K；
- End-to-End discovery。

---

# 20. 最终主消融模型

| Model | Adaptive Multi-Prototype | Prototype-Specific Boundary | Fine-Grained Unknown Discovery |
|---|---:|---:|---:|
| M0 | × | × | × |
| M1 | ✓ | × | × |
| M2 | ✓ | ✓ | × |
| M3 | ✓ | ✓ | ✓ |

解释：

### M0

```text
Single Prototype
+
Global Boundary
```

### M1

```text
Adaptive Multi-Prototype
+
Global Boundary
```

### M2

```text
Adaptive Multi-Prototype
+
Prototype-Specific Boundary
```

### M3

```text
M2
+
Fine-Grained Unknown Class Discovery
```

---

# 21. 推荐额外 Baselines

Open-set detection 至少应包含几类思想：

```text
Softmax confidence threshold
Single centroid distance
Single Gaussian Mahalanobis
OpenMax-like baseline
Energy score
One-class / Isolation Forest baseline
RoNeTC-style uncertainty baseline（如代码可复现）
FOSS-style isolation baseline（如可复现）
```

不要求所有 baseline 一开始全部实现。

顺序：

1. 自己的内部 baseline；
2. 简单经典 baseline；
3. 代表性 SOTA。

---

# 22. Optional Extensions：当前先不作为核心任务

以下内容只有三个核心问题跑通后再考虑：

## 22.1 Uncertainty-Aware Dynamic Fusion

```text
TrafficFormer
FIG/TAGCN
↓
sample-wise reliability
↓
dynamic fusion
```

目的：

提升 embedding quality。

---

## 22.2 Cross-View Disagreement

用于 Near-OOD：

$$
JS(p_t||p_g)
$$

但必须先证明：

```text
prototype false-accepted unknown
```

的 disagreement 显著高于 Known。

---

## 22.3 Near-OOD 专项实验

建议最终增加，但暂时不修改主模型。

---

# 23. 学生每一阶段提交格式

每完成一阶段必须提交：

## 23.1 README

写清：

```text
做了什么
为什么做
数据怎么划分
用的参数
运行命令
结果
是否通过验收
存在什么问题
```

---

## 23.2 CSV

所有表格结果必须保存 CSV。

不能只截图。

---

## 23.3 Config

例如：

```yaml
seed: 0
dataset: USTC-TFC2016
known_classes: [...]
unknown_classes: [...]
embedding_dim: ...
gmm_kmax: 5
covariance_type: diag
boundary_quantile: 0.95
```

---

## 23.4 Per-Sample Prediction

后续所有 failure analysis 必须能够回溯到 flow。

---

# 24. 第一批学生任务

目前不要直接让学生做完整方法。

第一批只做：

## Task 1：数据处理

完成：

```text
USTC PCAP
↓
Bidirectional Flow
↓
TrafficFormer Input
+
FIG Input
```

交付：

```text
dataset_statistics.csv
flow_index.csv
data_validation_report.md
```

---

## Task 2：Closed-Set Backbone

完成：

```text
TrafficFormer
FIG/TAGCN
TrafficFormer + FIG/TAGCN
```

交付：

```text
closed_set_results.csv
confusion_matrix/
training_curves/
```

---

## Task 3：保存 Embedding

```text
embedding_train.npy
embedding_val.npy
embedding_test.npy

labels_train.npy
labels_val.npy
labels_test.npy
```

---

## Task 4：Known-Space Distribution Audit

完成：

```text
GMM K=1..5
BIC selection
validation NLL
3 seeds
bootstrap stability
```

交付：

```text
bic_per_class.csv
validation_nll.csv
selected_k_per_seed.csv
bootstrap_k_stability.csv
distribution_audit_summary.md
figures/
```

---

## 第一批验收门槛

只有当：

1. 数据严格对齐；
2. 双分支正常收敛；
3. embedding 可复现；
4. Known-space multimodality 得到客观证据；

才允许学生进入：

```text
Single Prototype
→
Adaptive Multi-Prototype
```

阶段。

---

# 25. 第二批学生任务

## Task 5：Single Prototype Baseline

完成：

```text
16 Known + 4 Unknown
3 class splits
```

---

## Task 6：Adaptive Multi-Prototype

完成：

```text
Single
Fixed K=2
Fixed K=3
Adaptive BIC
```

---

## Task 7：Failure-Case Analysis

必须具体分析：

```text
Known false rejection
Unknown false acceptance
```

---

# 26. 第三批学生任务

## Task 8：Local Dispersion Audit

统计：

```text
prototype radius
distance distribution
rmax/rmin
CV
```

---

## Task 9：Prototype-Specific Boundary

比较：

```text
M0
M1
M2
```

---

# 27. 第四批学生任务

## Task 10：Oracle Unknown Discovery

比较：

```text
K-Means + Oracle K
GMM + Oracle K
GMM + BIC
HDBSCAN
```

---

## Task 11：End-to-End Discovery

```text
Open-Set Detector
↓
Predicted Unknown Buffer
↓
Clustering
```

输出：

```text
ARI
AMI
NMI
Clustering ACC
Unknown Buffer Precision
Unknown Buffer Recall
estimated K
```

---

# 28. 论文结果最终应该形成的结构

## Table 1

Closed-set backbone。

## Table 2

Known distribution audit。

## Figure 1

单一 prototype 错误描述 multimodal class 的 motivation 图。

## Table 3

Single / Fixed-K / Adaptive Multi-Prototype。

## Figure 2

Prototype-specific radius distribution。

## Table 4

Global vs Prototype-Specific Boundary。

## Table 5

不同 openness 下 Unknown Detection。

## Table 6

Oracle Unknown Discovery。

## Table 7

End-to-End Unknown Discovery。

## Figure 3

Unknown discovered cluster visualization。

## Table 8

跨数据集 / 新数据集泛化。

---

# 29. 明确禁止的实验错误

1. **Unknown 进入 validation。**
2. 用 Unknown test AUROC 选择最终 threshold。
3. 用真实 Unknown 类别数选择最终 GMM K。
4. 用真实 Unknown label 调 HDBSCAN 参数。
5. 在 test embedding 上重新 fit scaler。
6. 训练 / test 流存在重复或同一原始 session 泄漏。
7. 只报告最好 seed。
8. 只画 t-SNE 就声称存在多峰。
9. 只比较 Single vs Adaptive，不比较 Fixed-K。
10. clustering 只报告 Oracle，不报告 End-to-End。
11. 只报告 Detection，不分析 Known false rejection。
12. 使用跨数据集结果时忽略 benign domain shift。

---

# 30. 最终论文主线

整篇论文统一为：

$$
\boxed{
\text{Model the heterogeneity of the Known}
}
$$

↓

$$
\boxed{
\text{Construct mode-specific valid regions}
}
$$

↓

$$
\boxed{
\text{Reject samples outside Known regions}
}
$$

↓

$$
\boxed{
\text{Discover the heterogeneity of the Unknown}
}
$$

一句话概括：

> **同一个已知流量类别内部并非单一模式，不同局部模式也具有不同的有效范围；同时，被拒识的未知流量本身也不是一个统一类别。因此，本研究同时建模 Known-side heterogeneity 与 Unknown-side heterogeneity，实现从粗粒度 Unknown Detection 到细粒度 Unknown Class Discovery。**

---

# 31. 当前执行顺序

学生近期只执行：

```text
Stage 0
数据处理
    ↓
Stage 1
Closed-Set Backbone
    ↓
Stage 2
Known-Space Distribution Audit
    ↓
Stage 3
Single Prototype
    ↓
Stage 4
Adaptive Multi-Prototype
```

**暂时不要提前实现 Prototype-Specific Boundary 和 Unknown Discovery。**

等 Stage 4 的结果完成后，再根据：

```text
multimodality evidence
+
Single → Multi improvement
+
failure-case analysis
```

决定是否正式进入第二核心问题实验。
