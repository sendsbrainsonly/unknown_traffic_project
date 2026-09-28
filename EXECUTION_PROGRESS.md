# Unknown Traffic / Unknown Attack Detection 项目过程、实验计划修订与当前进度

## 2026-09-28 — Stage 42/43 代码与轻量结果重新发布

状态：`complete / REMOTE_SHA_VERIFIED`。用户要求重新推送当前代码。起点核验确认 Git 根目录为本项目、分支为 `main`，GitHub `main` 与本地已提交 HEAD 同为 `affc1241968e0cb9a13045c6d9a0c839eac560d8`。Git 可见的本轮内容只有轻量源码、协议、报告与核验元数据；磁盘上的 checkpoint、原始/派生数据、缓存、CSV/逐样本预测和 tmux 日志均未进入暂存区。Stage40/42/43 共 9 个实验包的全部已记录哈希复核 PASS；暂存区 135 个文件、约 842 KiB，35 个 Python 文件语法解析、82 个 JSON 解析、入口链接、敏感模式、权重/数据后缀和 `git diff --check` 均 PASS。内容提交 `c6cf3d6f96a89fb2a8ad4c35201cb774b03b5d43` 已通过 GitHub SSH 443 非强制推送，并与远端 `main` SHA 逐字符一致。本记录作为发布交接收尾；不需要运行新实验。

## 2026-09-28 — Stage43 CIC 混合未知类别与比例鲁棒性实验

状态：`in_progress / INPUT_AUDIT`。目标是在不重训、不重拟合阈值、不读取原始 PCAP 的条件下，复用 Stage42-S 至 Stage42-Y 已保存的 10 个 Unknown 类逐样本分数，检查 Known:Unknown 比例、Unknown 类别组成和 BENIGN:PortScan 组成变化。固定 Known Test 为 Stage40 的 2,272 条 `BENIGN + PortScan`，方法为 MSP、Energy、centroid、DES-v1，阈值继续使用原 Known-Val P95。计划先冻结源文件哈希、重复/对齐审计、5 个 prevalence 设置、7 个 composition 设置、3 个 Known-composition 设置以及 seeds 2022–2041，再生成完整 membership manifest、指标和独立重放。输出目录：`stage43_cic_mixed_prevalence/`；本实验为已暴露候选上的 post-hoc diagnostic，不是 untouched validation。

Stage43 terminal update 2026-09-28 09:43 UTC: `success / COMPOSITION_SENSITIVE`。10 个来源的 2,272 条 Known 分数完全一致；260,220 条 Unknown 无跨类重复，Known/Unknown 无重叠，阈值与保存决策全部重放 PASS。完成 15 settings × 20 seeds = 300 个混合测试、600,000 membership、1,200 方法指标；独立脚本逐行重算最大误差 0。DES-v1 在 5 个 class-balanced prevalence 设置中 AUROC `0.980384–0.981316`、UFAR `0.067250–0.069500`，比例本身稳定；但 all-balanced UFAR `0.067100` 掩盖 Bot `0.495000` 与 DDoS `0.169500`，Bot+DDoS-only AUROC/UFAR 为 `0.947600/0.338350`。Known 组成也改变聚合 FRR：BENIGN:PortScan 3:1 为 `0.091250`，1:3 为 `0.050250`。完整结果见 `stage43_cic_mixed_prevalence/RESULTS.md`。只作事后诊断，不自动调参。

Stage43 completion verification 2026-09-28 09:44 UTC: 独立重放再次 PASS（300 mixtures、1,200 metric rows、600,000 memberships、max absolute error=0）；项目 bundle 哈希校验通过：`status=success artifacts=51 bundle_files=51`。无 Stage43 运行进程，无 GPU 使用。

Stage43 reporting addendum 2026-09-28: 按用户要求将 Binary Accuracy 和 Unknown-positive Binary F1 加入 prevalence、Unknown composition 和 Known composition 三组主表。全均衡 Acc/F1=`0.931125/0.931247`，Authentication/Web=`0.964575/0.965790`，Bot+DDoS=`0.795325/0.763697`。Known:Unknown 从 9:1 改为 1:9 时 Accuracy 仍约 `0.93`，但 F1 从 `0.725275` 升至 `0.960223`，因此明确标注 F1 的 prevalence dependence；模型分数、样本成员和阈值未改动。

## 2026-09-28 — Stage42-Y CIC 四类顺序诊断

状态：`in_progress / FOUR_PROTOCOLS_FROZEN`。在访问这四类新包特征前，已按用户指定顺序固定 SSH-Patator（2,987）、Web Attack - Brute Force（1,364）、Web Attack - XSS（629）、DoS Slowhttptest 全量（5,096）MATCHED 流及其协议；保留 Stage40 `BENIGN + PortScan` Known 模型、支持集、Known-Val P95 阈值。Brute Force/XSS 因 Unknown 数少于 2,272 Known Test，额外预先固定真正 1:1 的 Known 子集。Slowhttptest 与历史 Stage40 Unknown Test 重合 132 流，明确为事后诊断。命名 tmux 队列 `stage42y-sequential-20260928` 按固定顺序执行缓存、冻结推理和独立重放；进度看 `stage42y_cic_remaining_four_open_set/queue_progress.json`。所有候选均需报告，不因指标不佳跳过。

Stage42-Y terminal update 2026-09-28 09:21 UTC: `complete / four independent replays PASS`。队列 exit 0；四类均按冻结顺序完成包缓存、GPU3 冻结推理与 8 行方法×视图指标重算，Unknown/Test fitting=0。DES-v1 自然比例 AUROC/AUPRC/UFAR：SSH-Patator `0.991275/0.989281/0`；Web Brute Force `0.995652/0.991421/0`；Web XSS `0.997489/0.988430/0`；全量 Slowhttptest `0.993400/0.996257/0.001374`。四组 Known FRR 均为 `0.071303`。完整结果见 `stage42y_cic_remaining_four_open_set/RESULTS.md`；只作事后诊断，不再自动启动新候选。

Stage42-Y handoff verification 2026-09-28 09:22 UTC: 首次 bundle 校验因手工使用不被接受的状态值 `complete`、且刷新程序在 bundle 内运行时把自身尚未结束的 tmux 日志/临时文件纳入清单而失败，失败日志保留。已将 manifest 状态修正为规范的 `success`，从项目根目录执行刷新与校验；最终 `validation passed: status=success artifacts=130 bundle_files=130`，128+ 文件哈希重新核对通过。此为归档元数据修正，未改动模型分数、样本划分和阈值。

## 2026-09-28 — Stage42-X CIC 第六个候选：Unknown FTP-Patator

状态：`in_progress / FULL_PACKET_CACHE_RUNNING`。按未完整测试过的攻击标签匹配流数量排序，选择 Tuesday 的 `FTP-Patator`，全部 3,985 条 matched flows、15 个五分钟 group，使用与先前完全相同的 Stage40 `BENIGN + PortScan` 冻结 encoder、Known-Train support 与 Known-Val P95 阈值。候选 manifest SHA256=`f031cbdf6ce8721854ca4f4c55834f60fd703ae664ca9d16b4b0a0c60c9be1b7`，在 Tuesday 包特征访问前冻结，与 Known role ID 无交集。Stage42-X 只新建 Tuesday 缓存路径适配，复用哈希固定的 Stage42-S 评分与独立复核。`stage42x-cache-20260928` 于 08:21 UTC 启动全量包缓存，`stage42x-finish-queue-20260928` 已启动并报告 `WAITING_FOR_CACHE`；缓存成功后顺序进行实时 GPU 选择、冻结推理和独立重放。完成前不得报告开集指标。该第六候选仍属连续查看先前结果后的 post-hoc development diagnostic，不能宣称独立验证。下一步只读检查两个 tmux session 与 `queue_progress.json`，完成后更新 bundle 并做哈希验证。

Stage42-X terminal update 2026-09-28 08:24 UTC: status `complete / PASS` for extraction, frozen evaluation and independent replay; DES-v1 **favorable-screen PASS**. All 3,985 matched Tuesday FTP-Patator flows were evaluated against the unchanged 2,272 Known Test flows; eight method × view metric rows replayed, checkpoint/candidate hashes unchanged, Unknown/Test fit counts zero. DES-v1 natural AUROC/AUPRC/UFAR/Known-FRR = `0.998342/0.998596/0.000000/0.071303`; true 1:1 AUPRC=`0.997596`. It accepts zero FTP-Patator Unknown flows as Known, but MSP/Energy/Centroid also show AUROC 0.984–0.988 and UFAR 0.35–1.51%, so the result is not uniquely evidence of DES-v1. Stage42-X remains a post-hoc diagnostic with Tuesday capture/attack-schedule confounding. See `stage42x_cic_ftp_patator_open_set/RESULTS.md`; no further candidate launched.

## 2026-09-28 — Stage42-W CIC 第五个候选：Unknown Bot

状态：`in_progress / FULL_PACKET_CACHE_RUNNING`。继续使用 Stage40 `BENIGN + PortScan` 冻结模型与 Known-Val P95 阈值。Friday PCAP 中剩余未测攻击标签 `Bot` 的全部 1,228 条 matched flows（37 个五分钟 group）已在包特征访问前冻结，manifest SHA256=`b304556ae0de5043ef58cb1207f234bbb7fe08c7d784c51af51b611695aba24b`。预检发现旧评估器在 Unknown 数小于 Known Test 数时，名义 1:1 视图会保留全部 2,272 个 Known；因此在本实验中另行按固定 SHA256 规则冻结 1,228 个 Known Test ID（SHA256=`6f00684e6d5565c10ce04d9e5c8727c7b49fbefc57bd2d0f635602d715e1124d`），仅隔离修正均衡视图 membership 和派生指标，保持自然比例分数、模型、阈值不变。`stage42w-cache-20260928` 于 07:57 UTC 开始全量 Friday 包缓存；`stage42w-finish-queue-20260928` 已启动并报告 `WAITING_FOR_CACHE`，缓存成功后顺序执行冻结 GPU 推理、均衡视图修正和独立重放。复核必须确认 1,228:1,228、8 行指标重放、Unknown/Test fitting=0。此为连续查看既有结果后的探索性诊断，不能称独立验证。下一步只读查看两个 tmux session 和 `queue_progress.json`；结束后更新 bundle 核心结果并做哈希验证。

Stage42-W terminal update 2026-09-28 07:59 UTC: status `complete / PASS` for extraction, frozen inference, exact 1:1 implementation correction, and independent replay; **favorable-screen FAIL**. All 1,228 Bot Unknown flows were extracted; checkpoint hashes and both frozen membership hashes stayed unchanged, Unknown/Test fit counts were zero, and all eight metric rows replayed. DES-v1 natural AUROC/AUPRC/UFAR/Known-FRR = `0.946433/0.875758/0.508958/0.071303`; 1:1 AUROC/AUPRC = `0.950623/0.936257`. The frozen P95 threshold accepts 625/1,228 Bot flows as Known, so the operating point is poor despite high rank discrimination. See `stage42w_cic_bot_open_set/RESULTS.md`; no further candidate launched.

## 2026-09-28 — Stage42-V CIC 第四个候选：Unknown DDoS

状态：`in_progress / FULL_PACKET_CACHE_RUNNING`。继续使用 Stage40 `BENIGN + PortScan` 冻结三路 encoder、Known-Train support 与 Known-Val P95 阈值。仅按已匹配流元数据选择与 Known PortScan 同属 Friday PCAP、尚未测试且规模最大的攻击标签 `DDoS`；全部 76,613 条 flow、5 个五分钟 group 已冻结，manifest SHA256=`9300363ccdfc786e7524705446455b2a2ac117deced5fc94367c37f5a6f2359a`，无 Known role-ID 交集。原始 packet 特征在冻结前未打开。新目录 `stage42v_cic_ddos_open_set/` 仅增加 Friday 缓存适配，复用哈希固定的 Stage42-S 评分和独立重放。`stage42v-cache-20260928` 已在 07:11 UTC 启动全量缓存，`stage42v-finish-queue-20260928` 已启动并报告 `WAITING_FOR_CACHE`；缓存审计通过后队列会实时选一张符合容量约束的 GPU，顺序做冻结推理与独立重放。该选择发生在前三个候选结果已可见之后，仅属 post-hoc development diagnostic；同 PCAP 也无法排除时间段和端点混杂。下一步只读检查两个 tmux session 与 `queue_progress.json`，完成后更新 bundle 核心结果并做哈希验证。

Stage42-V terminal update 2026-09-28 07:24 UTC: status `complete / PASS` for execution and independent replay, but **favorable-screen FAIL**. Friday extraction emitted 76,613/76,613 DDoS flows; the frozen model on physical GPU0 and the independent score/decision replay both exited 0. All eight method × view metrics were verified, checkpoint/candidate hashes were unchanged, and Unknown/Test fit counts stayed zero. DES-v1 natural AUROC/AUPRC/UFAR/Known-FRR = `0.948990/0.994485/0.166134/0.071303`; balanced AUPRC = `0.882042`. DES-v1 accepted 12,728/76,613 DDoS Unknown flows as Known. Thus this candidate fails the previously used AUROC ≥ 0.95 and UFAR ≤ 0.10 screen despite high prevalence-inflated natural AUPRC. See `stage42v_cic_ddos_open_set/RESULTS.md`; no new candidate should be inferred as pre-registered independent validation.

## 2026-09-28 — Stage42-U CIC 第三个候选：Unknown Hulk

状态：`in_progress / PROTOCOL_FROZEN_BEFORE_FEATURE_ACCESS`。第三个候选使用 Stage40 `BENIGN + PortScan` 冻结模型、Known-Train support 与 Known-Val P95 阈值，Unknown=`DoS Hulk`；Wednesday 映射表中全部 155,168 条 `MATCHED` flow 已冻结，覆盖同一 PCAP 的 6 个五分钟 group，未下采样。候选 manifest SHA256=`b158b2bb751fa5fd61c18f34149411583b8ba3515d4ad0dc5d9d822a573cb37f`。这是尚未在该模型下作为 Unknown 评估的新攻击类；只按标签和数量选择。全量 Unknown Test 包特征缓存正在 `stage42u-cache-20260928` 运行；`stage42u-finish-queue-20260928` 已启动，将在缓存审计 PASS 后依次做实时 GPU 选择、冻结推理和独立重放。进度见 `stage42u_cic_hulk_open_set/queue_progress.json` 与 `progress.json`。

## 2026-09-28 — Stage42-T CIC 第二个候选：Unknown GoldenEye

状态：`complete / INDEPENDENT_REPLAY_PASS`。将 `BENIGN + PortScan` 冻结模型的第二个候选确定为 `DoS GoldenEye`：Wednesday 映射表中全部 7,441 条 `MATCHED` flow，覆盖同一 PCAP 的 2 个五分钟 group；与 Stage40 role IDs 无重叠。候选 manifest SHA256=`630fd0b2ddf41b92670447f7a330023ce7585f2155b908feef9cb5521c05e0bf`，在读取包特征前冻结。复用 Stage40 Known-Val P95 阈值与 Stage42-S 已验证的提取、推理、复核代码。缓存 7,441/7,441、GPU 冻结推理、8 行指标独立重放均 PASS，Unknown/Test fitting=0。DES-v1 自然比例 AUROC/AUPRC/UFAR/Known-FRR=`0.969958/0.977195/0.002419/0.071303`；均衡视图=`0.970014/0.930666/0.002201/0.071303`。结果仅属单 PCAP、2 个时间组的诊断性有利划分；存在 endpoint/time/capture shortcut 风险。详见 `stage42t_cic_goldeneye_open_set/RESULTS.md` 与 `progress.json`。

## 2026-09-28 — Stage42-S CIC favorable protocol：Unknown slowloris

状态：`complete / INDEPENDENT_REPLAY_PASS`。第一个候选固定复用 Stage40 `BENIGN + PortScan` 的三路 encoder、Known-Train support 与 Known-Val P95 阈值，不重新训练/校准。Unknown=`DoS slowloris`；冻结的 5,709 条 MATCHED flow 覆盖 Wednesday 的 8 个五分钟 group；manifest SHA256=`908208f25c3e88a2b4153e538090ec222e24b8d21ce91cc6a08f447df4eca3f4`。缓存构建完成；GPU 冻结推理及 8 行指标独立重放均已完成，`unknown_fit_count=0`、`test_fit_count=0`。DES-v1 在自然比例下 AUROC/AUPRC/UFAR/Known-FRR=`0.993050/0.996724/0.000350/0.071303`；均衡视图对应=`0.992970/0.991799/0.000440/0.071303`。结果仅属单 PCAP、8 个时间组的诊断性有利划分；存在 endpoint/time/capture shortcut 风险。不自动启动第二个候选。详见 `stage42s_cic_favorable_open_set/RESULTS.md` 与 `progress.json`。

## 2026-09-27 — Stage40 CIC 冻结开集实验独立续跑

状态：`complete / BOTH_CIC_ROLES_INDEPENDENTLY_VERIFIED`。v2 顺序队列于 14:32 UTC 报告 `CIC_PILOTS_COMPLETE / all_verified`，tmux 退出码 0；最大并发 GPU workload=1。两套冻结角色、源文件 SHA256 均未变化；Known-Val P95 校准；两个 evaluation audit 均为 PASS，`unknown_fit_count=0`、`test_fit_count=0`、checkpoint hashes unchanged。两个独立重放均 PASS：PortScan 1,400 rows（Known/Unknown Test 264/1,136），Slowhttptest 2,404 rows（2,272/132）。四个分数为 MSP/Energy/centroid/DES-v1。DES-v1：PortScan AUROC/AUPRC/UFAR/Known-FRR=`0.463178/0.761719/0.948063/0.166667`；Slowhttptest=`0.997753/0.955759/0.000000/0.071303`。两项结果明显依赖 Unknown role，不应合并成普遍性能结论；单 seed、两 Known 类，Slowhttptest Unknown Test 仅 132 flows，且 CIC attack/day/capture 混杂。完整报告：`stage40_ustc_cic_open_set/RESULTS.md`；续跑包：`stage40_ustc_cic_open_set/cic_resume_20260927_v2/RESULTS.md`。

结果分布补记（2026-09-28）：`unknown_slowhttptest` 的 Known 类明确为 `BENIGN + PortScan`，Unknown 类为 `Slowhttptest`；Known Train/Validation/Test=`27,698/1,802/2,272`，Unknown Test=`132`。Slowhttptest 未进入训练、支持集构建、归一化或 P95 阈值校准。

### Current handoff

本轮 CIC 开集试运行已完成；没有待启动的训练或评估子任务。可直接查看上述报告和 CSV。若继续研究，应另立预注册协议，不基于这两个 Test 结果回调当前阈值或 detector。

## 2026-09-27 — Stage 23–41 代码发布

状态：`complete`。用户要求将当前主项目代码重新推送到 GitHub。起点核验：仓库根目录为本项目、分支 `main`，原本地与远端均为 `e0a4c77dd4ca5e69ec702562cd5d99ec2ce2185c`；Stage 23–41 此前尚未纳入普通 Git。原有本地改动包括本文档及 `EXPERIMENT_RESULTS.md`，其中后者删去了旧索引条目，故本次发布未暂存该文件，也未覆盖其本地内容。首个发布提交 `b2821a5598e2b3231947dbd587ebbb457d854a81` 纳入 451 个文件、约 3.26 MB；敏感特征扫描、禁止扩展名/子仓库检查均未发现问题；165 个 Python 文件 AST 解析、222 个 JSON 文件解析以及 `git diff --cached --check` 均通过。使用 GitHub SSH 443 非强制推送后，远端 `main` 与本地提交 SHA 完全一致。README、Stage39/40 结果状态及当前进度页面已纠正；权重、数据、缓存、逐样本预测、运行日志与可变队列状态仍留在本地。下一步仅需提交本交接文档收尾并再次核对远端 SHA；`EXPERIMENT_RESULTS.md` 的原有本地改动仍须由其所有者决定如何处理。

Stage40 A-2 Open-Detect-style balanced evaluation (2026-09-27 UTC): `complete / RETROSPECTIVE_NONPAIRED_DIAGNOSTIC`. Starting evidence: Stage40 had one frozen three-view encoder and scores for 4,333 Known + 558 Unknown Test flows; sibling Open-Detect v6 used five different 8:1:1 image splits and 600/600 per A-2 run. Selected all 558 frozen Unknown plus 558 score-blind SHA256-ranked Known IDs (seed 2022), leaving source artifacts unchanged; no new training/inference. Commands in project directory: `python stage40a2_balanced_opendetect_style/run_balanced.py`, then `python stage40a2_balanced_opendetect_style/replay_and_per_class.py`, both via project-local named tmux sessions. DES-v1 AUROC/AUPRC `0.949962/0.944431`; fixed Known-Val P95 Accuracy/Unknown-F1/UFAR/Known-FRR `0.749104/0.682540/0.460573/0.041219`; released-code labeled-Test Youden threshold gives `0.894265/0.901830/0.028674/0.182796` and is non-independent. All 257 P95 Unknown misses are Geodo; oracle reduces misses to 16 with Known false rejects rising 23→102. Frozen source SHA256 unchanged; 8 metric rows and 24 per-class rows independently replayed PASS. Exact shared flow identity with Open-Detect v6 remains NOT_ESTABLISHED; no paired/paper-equivalent superiority claim. See `stage40a2_balanced_opendetect_style/RESULTS.md`. Safe next action if a formal claim is desired: freeze one common A-2 flow-ID protocol and train both models on the same folds.

Stage40 metric supplement (2026-09-27 UTC): `complete / USTC_A2_ONLY`. From 4,891 frozen USTC A-2 per-sample scores, independently replayed four score methods and supplemented the Open-Detect released-code metrics (binary Accuracy, Unknown-positive Precision/Recall/F1; closed-set weighted F1). Formal Known-Val P95 DES-v1: Accuracy 0.898794, Unknown F1 0.548769, AUROC 0.945727, UFAR 0.460573. Retrospective released-code Test-oracle threshold: Accuracy 0.831118, Unknown F1 0.564346, UFAR 0.041219, but Known FRR rises from 0.054927 to 0.185322. It is explicitly non-independent, not the primary result. Frozen inputs' SHA256 values were unchanged. The earlier Stage40 CIC queue subsequently recorded `FAILED` at 2026-09-26 16:16 UTC because the Stage39 training queue failed; no CIC open-set sample scores exist. See `stage40_opendetect_metric_supplement/RESULTS.md` and do not treat the older Stage40 queued handoff below as current live state.

Stage40 handoff (2026-09-26 16:08 UTC, USTC/CIC open-set pilot): `in_progress / USTC_VERIFIED_CIC_QUEUED`. Isolated `stage40_ustc_cic_open_set/` has a PASS pre-Test role/score freeze and three role manifests. Frozen Stage34 USTC A-2 Known-only three-view checkpoints and the historical 10%-per-class Test sampling rule were reused. Its independently replayed open-set metrics (Known Test 4,333; Geodo/Htbot/Tinba Unknown Test 558): MSP AUROC 0.916286/AUPRC 0.629202/UFAR 0.508961; Energy 0.871426/0.584590/0.526882; centroid 0.677617/0.439162/0.537634; DES-v1 0.945727/0.711960/0.460573, all at Known-Val P95. DES-v1 Geodo UFAR is 0.626829, so high AUROC does not yet mean acceptable operating-point rejection. CIC old three-class encoder cannot be used for strict leave-one-attack-out; its two frozen group-disjoint Known-only pilots have Train/Val/Test/Unknown counts 7,818/2,110/264/1,136 for unknown PortScan and 27,698/1,802/2,272/132 for unknown Slowhttptest. Both Known Train/Val caches are PASS; first CIC PortScan graph branch is trained and SHA-verified. New named `stage40_cic_queue_v2_0926` waits for existing Stage39 three-GPU training to finish, then runs remaining CIC branches, Known-Val calibration, gated Test caching, frozen evaluation, and independent replay on one GPU3 workload at a time. Its progress is `stage40_ustc_cic_open_set/cic_queue_progress.json`; no CIC Unknown score exists yet. The first waiting-only queue was closed before any worker launch to add bounded GPU3-capacity retries; its log remains. Stage39 workers and frozen outputs are unchanged. The GPU4 capacity-selector refusal before training is preserved in its tmux log. See `stage40_ustc_cic_open_set/RESULTS.md`; do not infer CIC open-set success from closed-set F1.

Current handoff (2026-09-26 15:04 UTC, Stage39): `in_progress / VNAT_KNOWN_ONLY_TRAINING_RESUMED`. Frozen whole-service roles cover 12 new-training settings plus one Tor P2P-only retrospective control. All four VPN and all four Tor Known-only staged three-view settings completed. The first queue stopped before VNAT training because `build_fold_cache.py` used an uppercase `VNAT` source cache path; the source is lowercase `vnat`. The error occurred while opening the source audit file, before reading VNAT Test/Unknown feature values. The corrected builder produced a PASS Known Train/Val cache audit for VNAT Communication (14,687/1,834 flows); original failure JSON and tmux logs remain. New named queues `stage39_known_training_queue_v2_0926`, `stage39_evaluation_queue_v2_0926`, and `stage39_aggregate_queue_v2_0926` are active; training uses physical GPU lanes 0/1/7, at most three. All eight ISCX settings now have PASS Known-Val-only calibration and audited frozen evaluation-input caches; this does not mean they have final detection scores. VPN Communication remains the only evaluated setting: Known Test Macro-F1 0.995504, DES-v1 primary AUROC 0.405230, P95 UFAR 0.849354, Known FRR 0.058275; deployment AUROC 0.444371, Known FRR 0.136548. Its 14,468 scores and six threshold decisions replayed PASS. Safe next action: inspect `stage39_coarse_open_set_execution/progress.py`; do not treat eight completed training/calibration settings as eight open-set results. Allow VNAT training and the gated evaluation/aggregate queues to finish, then verify all bundle/source hashes and report the full distribution.

Research handoff (2026-09-26, VPN/Tor open-set protocol audit): `complete / READ_ONLY_RESEARCH`. Added `docs/research/2026-09-26_open_set_dataset_protocol_audit.md` and its VPN/Tor literature background note. Rechecked Stage38, the first evaluated Stage39 VPN Communication setting, completed Stage34 USTC and Stage34B CIC results; read original Open-Detect/FEC-OSL/UnDiff papers, INFOCOM 2020 author paper, CCS 2018 Deep Fingerprinting and official UNB label definitions. Findings: closed-set superiority does not exclude representation/support/score limitations; Tor Stage38 AUROC 0.924869 coexists with P95 UFAR 0.634750; VPN whole-Communication pilot remains weak despite Known Macro-F1 0.995504. Directly reviewed VPN/Tor papers differ in packet/site-trace units and auxiliary Unknown exposure. USTC A-2 is a justified candidate for a future frozen Known-only three-view evaluation; CIC's existing three-class checkpoint cannot serve as a strict leave-one-attack-out encoder without Known-only retraining. No training, score fitting, threshold selection, dataset/frozen protocol edit, or queue control was performed by this research task. Stage39 execution state remains governed by its live progress artifacts, not this research note.

> 更新时间：2026-09-21（UTC）  
> 项目目录：`Projects/unknown_traffic_project`  
> 文档性质：持续交接与当前执行依据  
> 当前状态：Stage 18 已完成；最终 Gate=E4_FAIL_NO_FUSION；未启动 E3+E4 融合或新的 detector 设计

## 1. 文档目的与证据边界

本文档把项目从 USTC-TFC2016 数据准备、TrafficFormer 适配、Known-space 分布诊断，到 Open-Detect、DES、VNAT 和 Known-only Hybrid 的全过程汇总到一个可持续维护的交接文件中。它重点回答：

1. 我们实际完成了什么；
2. 原始实验计划为什么以及如何被证据修订；
3. 当前哪些结论成立，哪些结论不能再写；
4. 当前正式方法、数据角色和冻结约束是什么；
5. 项目下一步处于什么状态。

事实和数值优先取自项目内现有 `RESULTS.md`、CSV/JSON、冻结协议和验证文件。聊天记录只用于定位任务，不作为实验结论来源。

原实验计划仍保留在：

- `未知流量_未知攻击检测_三核心问题_学生详细实验计划_公式优化版.md`
- `未知流量_未知攻击检测_三核心问题_学生详细实验计划_公式优化版_before_stage3_revision.md`

当前计划相对 Stage 3 修订前版本增加 405 行、删除 66 行，加入了 2026-09-12 evidence-based Master Plan Revision。但该计划文件的主状态仍停留在早期阶段；Stage 3–15B 的实际进展和最终边界以本文档及各阶段正式结果为准。

## 2. 当前一句话结论

项目已经证明：Known traffic representation 中存在 class-dependent local-support complexity，且 Open-Detect learned prototype、Known-Train empirical support 和 posterior uncertainty 具有真实互补性；但固定 Multi-GMM、DGSB、DES-v0、DES-v1 和现有 percentile-max Hybrid 都没有证明在所有数据集、protocol 和 hard class composition 上稳定优于 Open-Detect。

因此当前正式定位是：

- **Open-Detect Native 仍是正式基线；**
- **H1 = `max(A_OD, A_DES0)` 是唯一推荐继续研究的简单候选，但结论仅为 `HYBRID_PARTIAL`；**
- **DES-v1 不能升级为论文最终主方法；**
- **Adaptive K、semantic mode、Unknown Discovery 均未获证据支持，继续后移。**

## 3. 研究主线如何变化

### 3.1 原始计划

最初路线是：

```text
Known multimodality
→ Adaptive Multi-Prototype
→ Prototype-Specific Boundary
→ Fine-Grained Unknown Discovery
```

其核心假设是每个 Known class 内部包含多个真实且可解释的子模式，可先通过 BIC/GMM 找到 K，再将 component 作为 prototype。

### 3.2 第一次修订：从“真实多模态”转为“local-support sufficiency”

Stage 2.5 发现，在 896 维融合表示 `z_f` 上，PCA + full covariance 的 K=1 已显著强于 diagonal covariance 的 K=4/5；8/8 个代表性比较均显示 diagonal covariance misspecification 解释了 apparent multimodality 的相当部分。

同时，full covariance 下 K>1 的 held-out NLL 仍在 8/8 个代表性比较中改善。因此结论被修订为：

> 一个 global class-conditional support model 不一定足以描述 Known representation，但 residual component 不能直接解释为真实或语义子群。

Stage 2.6 和跨表示审计进一步表明 component 常与 packet count、total bytes、directional statistics 等简单流量统计关联。于是研究术语由 `semantic multimodality` 改为：

- `local-support heterogeneity`；
- `class-conditional support complexity`；
- `traffic-statistics-associated heterogeneity`。

### 3.3 第二次修订：从 density fit 转为 Unknown Detection utility

Stage 3 不再继续扩 K，也不实现 Adaptive K，而是在相同 Open-Detect representation 下严格比较 Single-Full 与固定 Multi-Full-K2。

结果为 `Gate D — MIXED`：

- A-1 Multi−Single UFAR：`+0.101176`，明显恶化；
- A-2：`+0.002689`，基本无改善；
- A-3：`−0.022899`，有所改善。

后续归因表明 A-1 恶化和 A-3 改善由少数 Known/Unknown class pair 主导，Known-validation NLL 改善不能预测 Unknown utility。固定 Multi-GMM 因此不再是主方法。

### 3.4 第三次修订：从 Multi-GMM 转为 representation–support decoupling

CipherSpectrum Stage 9 的 DGSB 在 operating point 上出现 tradeoff，没有稳定降低 UFAR。Stage 10B 的 score decomposition 显示：

- 最大恢复来自 learned prototype 替换为 empirical Known-Train centroid；
- covariance mismatch 是次要机制；
- posterior uncertainty 通常保留有用信号。

于是研究重点由“找更多 Gaussian component”改为：

> 将 representation training prototype 与 detection-time empirical support 解耦，检验数据锚定 support 是否更可靠。

Stage 11A 的 data-anchored prototype 训练失败，Stage 11B 的 decoupled readout 在 USTC 上仍然 scenario-dependent；但 Stage 12 在 ISCX-VPN 和 ISCXTor2016 上，DES-v0 相比 Native Open-Detect 的 mean ΔAUROC 分别为 `+0.023452` 和 `+0.069734`，为 decoupling 提供了外部支持。两套数据使用 `FLOW_DISJOINT_ONLY` fallback，因此不能声称 unseen-capture generalization。

### 3.5 第四次修订：Global + Local support

Stage 13A-2 在 USTC development 上使用固定：

```text
S_GL = 0.5 * Z_centroid + 0.5 * Z_kNN10
```

取得 `GO`：A1/A2/A3 AUROC 分别为 `0.994475/0.953415/0.991972`。但这仍是 USTC development 结果，不是独立外部确认。

VNAT Stage 14D 在同一 Native checkpoint 下比较：

- M0：Open-Detect Native；
- M1：DES-v0 empirical centroid；
- M2：DES-v1 fixed global + local。

总体 AUROC 为 `0.837535/0.851706/0.853085`。M1−M0 只有 `6/15` protocols 提升，bootstrap CI 跨零；M2−M1 仅 `+0.001378`，虽将 UFAR 平均降低 `0.030033`，但 AUROC 增益不稳定。最终为：

```text
DECOUPLING_ONLY_PARTIALLY_CONFIRMED
```

### 3.6 第五次修订：从固定 DES 转向互补性与 Known-only Hybrid

Stage 15A 对 VNAT、USTC、ISCX-VPN、ISCXTor 的冻结结果做机制诊断，确认 Open-Detect 与 DES 存在双向 rescue：

- VNAT P95：OD-only rescue `15,129`，DES-only rescue `11,585`；
- VNAT exact ranking：OD-only 正确 `6,203,271` 对，DES-only 正确 `7,741,572` 对；
- support overlap 与 DES gain：Spearman `rho=-0.343555`，探索性 `p=0.020854`；
- posterior uncertainty 在 21 个 DES-loss 单元中的 19 个保留补充信号。

Stage 15A Gate 为 `COMPLEMENTARITY_CONFIRMED`，但这只允许设计 Hybrid，不等于 Hybrid 已成功。

Stage 15B 仅测试三个预注册 Known-Val percentile-max 公式：

```text
H1 = max(A_OD, A_DES0)
H2 = max(A_OD, A_DES1)
H3 = max(A_proto, A_DES0, A_Vpost)
```

H1/H2/H3 在 VNAT 与 USTC pooled primary 上的 ΔAUROC 为 `+0.021876/+0.027272/+0.012776`，但 negative protocols 分别为 `12/11/14`，都没有充分减少 DES-v0 的 12 个 negative protocols；rsync/scp 失败也没有被修复。因此最终 Gate 为：

```text
HYBRID_PARTIAL
```

H1 因公式最简单、hard-class 行为优于 H2/H3，被保留为唯一后续候选；它不能被描述为已确认的最终方法。

## 4. 数据与表示基础

### 4.1 USTC-TFC2016 Stage 0/1

- Stage 0：构建 `489,101` 条双向流，20 类完整保留；FIG 共 `4,088,199` 节点；420/420 项抽样核验通过。
- TrafficFormer 短流适配：主策略允许至少 1 个真实包，保持官方 bigram、`[SEP]`、64 bytes/packet、最多 5 包和下游 PAD 语义；严格 min3 仅作为敏感性对照。
- Model A TrafficFormer：Test Accuracy `0.9892`，Macro-F1 `0.9920`。
- Model B FIG→TAGCN：Accuracy `0.6998±0.0236`，Macro-F1 `0.7329±0.0132`。
- Model C 融合：Accuracy `0.9893±0.0001`，Macro-F1 `0.9921±0.0001`。

### 4.2 表示术语

必须继续区分：

```text
z_t : TrafficFormer representation
z_g : FIG/TAGCN graph representation
z_f : train-standardized [z_t ; z_g]，896 维
mu_x: Open-Detect deterministic latent mean
```

Stage 2.5/2.6 使用的是 `z_f`，不是纯 `z_t`。Open-Detect 系列使用 `mu_x`。

## 5. 各阶段完成情况

| 阶段 | 状态 | 核心结论 |
|---|---|---|
| Stage 0–1 | 完成 | 20 类数据与 TrafficFormer/FIG/fusion closed-set backbone 验收通过 |
| Stage 2 | 完成 | diagonal GMM 显示 global simple model 描述不足，但不能证明真实 multimodality |
| Stage 2.5 | 完成，结论 B | covariance misspecification 解释相当部分 apparent multimodality |
| Stage 2.6 | 完成 | component 常由简单流统计解释，不应解释为 semantic modes |
| Stage 3 | 完成，Gate D | Fixed K2 utility 在 A1/A2/A3 不一致，Multi superiority 未建立 |
| Stage 4 | 完成，Diagnosis C | Known coverage 需要 local boundary，但 Unknown utility 未确认 |
| Stage 5/5.5 | 冻结后阻断 | CSTNET 协议有 Global Gate 退化和 sparse calibration 风险，`NOT_READY` |
| Stage 6–9 | 完成 | CipherSpectrum DGSB one-shot Test 为 operating-point tradeoff |
| Stage 10B | 完成 | learned-prototype mismatch 为主机制，covariance mismatch 次之 |
| Stage 11A | 完成，NO_GO | data-anchored prototype training 明显退化 |
| Stage 11B | 完成，NO_GO | decoupled support 在 USTC scenario-dependent |
| Stage 11C | 完成，WEAK_SIGNAL | Known-only support complexity selector 证据不足 |
| Stage 12 | 完成，EXTERNAL_CONFIRMED | DES-v0 在 ISCX-VPN/Tor 平均提升，但仅 flow-disjoint claim |
| Stage 13A-1/A-2 | 完成，GO | USTC 上 kNN local 与 centroid fusion 有开发价值 |
| Stage 14A/A.5 | 完成 | VNAT 数据质量和 group feasibility 已审计 |
| Stage 14B/B.5 | 完成并冻结 | 23,449 clean flows，15 protocols，Strict Unknown-Free PASS |
| Stage 14C 系列 | 完成 | Native baseline、特征诊断、训练失败定位与 cleaned 15-run 完成 |
| Stage 14D | 完成，PARTIAL | VNAT decoupling/global-local 平均正向但 protocol 不稳定 |
| Stage 15A | 完成，CONFIRMED | OD/DES 双向互补和 support-overlap failure regime 已确认 |
| Stage 15B | 完成，PARTIAL | H1 平均改善但未消除 negative protocols 或 rsync/scp 失败 |

## 6. VNAT 专项过程与结果

### 6.1 数据审计与协议冻结

- 原始 inventory：172 files，165 PCAP；163/165 完整可读。
- Stage 14A 构建：23,454 flows。
- 永久排除：1 条 duplicate-PCAP flow、4 条 rsync/sftp cross-app duplicate rows；两个 truncated PCAP 整体不进入 freeze。
- 最终 clean pool：`23,454 - 1 - 4 = 23,449`。
- 所有 10 个 application 均保留；VPN/non-VPN 是 metadata，不拆成语义类别。
- Low/Medium/High：Unknown class 数 `2/3/4`，seeds `2022–2026`，共 15 protocols。
- Known flow split：按 application 分层随机 `8:1:1`；不是 capture/group-disjoint。
- Freeze hash：`c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`。

主要有效性风险：

- application label 来自 filename/capture metadata，不是独立 per-flow ground truth；
- class size 从 44 到 13,563，极端不均衡；
- RDP Known Validation/Test 常各只有 4 条；
- VPN/non-VPN 分布严重不均衡；
- flow-random split 允许 capture/group 跨 split，因此不能声称 unseen-capture generalization。

### 6.2 闭集表示训练的失败与修复

初始 F0/F2 闭集结果偏低：F0 Macro-F1 `0.614834`，F2 `0.652969`；released Native Open-Detect 为 `0.830904`。

Stage 14C-3 证明数据和 F0 byte representation 对齐无误，主因是：

- batch 512 导致更新不足；
- patience 5 过早停止；
- seed-sensitive optimization；
- prototype reset 旧实现会断开 optimizer link。

Cleaned pipeline 固定为：

- F2 representation；
- batch 128；
- 100 epochs，无 early stopping；
- Adam LR 0.001；
- MultiStepLR `[50,80]`；
- epoch 51/81 prototype in-place reset，并清理 prototype optimizer state；
- checkpoint 只依据 Known Validation Accuracy/Macro-F1 harmonic mean。

15-run cleaned 结果：

| Scope | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Low | 0.876386 | 0.791160 | 0.867275 |
| Medium | 0.840436 | 0.802217 | 0.821985 |
| High | 0.888714 | 0.869709 | 0.885014 |
| Overall | 0.868512 | 0.821029 | 0.858092 |

相对 Native Open-Detect，mean Δ Accuracy/Macro-F1/Weighted-F1 为 `+0.000451/−0.009875/−0.001386`。因此 cleaned pipeline 达到可用且稳定，但没有证明闭集性能优于 Native。

Medium-2025 的低分主要来自 Known composition：rsync 与 scp 同时为 Known 时，185/191 rsync validation samples 被预测为 scp；当 scp 在 Medium-2026 被 held out 后，rsync 为 191/191。该差距主要不是随机 seed。

## 7. 当前开放集核心结果

### 7.1 Stage 14D VNAT

| Method | AUROC | AUPRC | UFAR | Known FRR |
|---|---:|---:|---:|---:|
| M0 Open-Detect | 0.837535 | 0.874068 | 0.589837 | 0.046577 |
| M1 DES-v0 | 0.851706 | 0.884966 | 0.599280 | 0.046029 |
| M2 DES-v1 | 0.853085 | 0.888365 | 0.569247 | 0.048026 |

解释：

- DES-v0 提高平均 AUROC，但只赢 6/15，并使平均 UFAR 略增；
- DES-v1 相比 DES-v0 主要改善 operating point/UFAR，AUROC 增益很小；
- hard unknown composition 尤其 rsync/scp/sftp 会改变结论；
- 不足以把 DES-v0 或 DES-v1 定义为最终主方法。

### 7.2 Stage 15A failure regimes

- DES-v0 的主要优势类：sftp `+0.202935`、netflix `+0.087576`、rdp `+0.047975`、zoiper `+0.039113`。
- 主要失败类：rsync `−0.161182`、scp `−0.045337`、youtube `−0.038862`。
- rsync/scp empirical support overlap 为 `0.981433/0.958865`。
- sftp 不是平均失败类；它具有 unusually large normalized prototype gap `2.460952`。
- prototype-centroid mismatch 的全局相关性未建立：`rho=0.200659, p=0.186278`。
- 最稳定的可解释信号是 support overlap，而非单独的 prototype mismatch。

### 7.3 Stage 15B Hybrid

| Dataset | H0 OD | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| VNAT | 0.837535 | 0.858118 | 0.859116 | 0.849366 |
| USTC | 0.939528 | 0.962698 | 0.972491 | 0.953250 |
| ISCX-VPN | 0.553024 | 0.561670 | unavailable | 0.521592 |
| ISCXTor | 0.524614 | 0.572999 | unavailable | 0.555676 |

H1 是当前唯一推荐候选，但必须附带以下限定：

- pooled primary ΔAUROC `+0.021876`，CI `[+0.004168,+0.041910]`；
- only `18/30` primary protocols 提升；
- negative protocols 仍为 12 个；
- rsync 平均/最差 ΔAUROC `−0.105989/−0.145269`；
- scp 仍为 `−0.029232/−0.047784`；
- 不能声称解决了 support-overlap failure。

## 8. 当前数据集角色

| 数据集 | 当前角色 | 边界 |
|---|---|---|
| USTC-TFC2016 | Development / mechanism benchmark | 已多次用于设计、诊断和 Gate，不是 untouched validation |
| VNAT | Development / external-dataset benchmark | 已用于 Stage 14–15 开发，不再是 untouched external dataset |
| ISCX-VPN | 已完成外部验证与 retrospective check | flow-disjoint-only，不支持 unseen-capture claim；之后也不再 untouched |
| ISCXTor2016 | 已完成外部验证与 retrospective check | flow-disjoint-only，不支持 unseen-capture claim；之后也不再 untouched |
| CipherSpectrum | 已完成 one-shot external Test | DGSB 为 tradeoff；labels 与 domain/endpoint shortcut 有风险 |
| CSTNET-TLS1.3 | Sparse-calibration stress candidate | Stage 5.5 `NOT_READY`，须另行冻结 protocol v2 后才可继续 |
| CIC-IDS-2017 | Supplementary diagnostic | endpoint/time/capture shortcut 风险，不作为主要 encrypted-traffic benchmark |

若要给 H1 或后续方法做确认性结论，必须使用新的、在公式和 Gate 冻结后才打开的 external protocol；不能继续把 USTC/VNAT 结果称为独立确认。

## 9. 当前冻结方法与不可更改项

### 9.1 正式 baseline

`M0 Open-Detect Native`：released learned-prototype Gaussian KL score，同一 protocol 下 encoder checkpoint 完全共享。

### 9.2 诊断方法

- `M1 DES-v0`：Known-Train empirical centroid minimum squared Euclidean distance。
- `M2 DES-v1`：固定 k=10、Known-Val median/MAD、`0.5 Z_global + 0.5 Z_local`。

### 9.3 当前唯一继续候选

`H1 = max(A_OD, A_DES0)`，其中两个 percentile 都只能由 Known Validation ECDF 得到。

### 9.4 继续实验必须保持

- Unknown 不参与 encoder training、support fitting、normalization、threshold 或权重选择；
- Test 不参与参数或公式选择；
- 阈值只用 Known Validation P95；
- 相同 protocol 的方法共享同一 encoder 和完全相同样本；
- 不搜索连续融合权重；
- 不临时调 k、threshold percentile 或 per-class threshold；
- 不基于开发集结果重新选择 Unknown classes、split 或 seed；
- 保存全部单 run、per-class 和 per-sample 证据。

## 10. 已证实、部分证实与未证实

### 已证实

- diagonal covariance misspecification 会制造 apparent multimodality；
- 一些类别存在 persistent local-support structure；
- component 经常与简单 traffic statistics 关联；
- OD 与 DES 存在真实双向 complementarity；
- empirical support overlap 是 DES failure 的主要可解释 regime；
- VNAT rsync/scp composition 会同时影响闭集和开放集结果；
- 原 F0/F2 闭集低性能主要来自训练机制，而不是数据错位。

### 部分证实

- representation–support decoupling 在平均意义上有收益，但 protocol 稳定性不足；
- global + local support 可改善部分 operating point，但 AUROC 增益较小；
- H1 可提高平均 AUROC/AUPRC/UFAR，但尚未控制 worst-case 与 hard-class failures。

### 未证实或已否定为通用结论

- 每类至少存在多个真实/语义 Gaussian 子群；
- BIC 选择的 K 是真实 mode 数；
- Fixed K2 必然优于 Single support；
- DGSB、DES-v0 或 DES-v1 稳定优于 Open-Detect；
- posterior uncertainty max-fusion 能自动修复 support overlap；
- Adaptive K 有实际 Unknown Detection 收益；
- Prototype-Specific Boundary 已获外部确认；
- Unknown Discovery 已具备启动条件。

## 11. 最终实验计划修订版

### 11.1 当前论文主线

建议论文叙事调整为：

```text
1. 诊断：learned prototype、empirical support 与 uncertainty 的几何失配
2. 机制：representation–support decoupling 与双向 error complementarity
3. 候选：Known-Val calibrated H1 percentile-max fusion
4. 边界：support-overlap hard regime 仍未解决
5. 确认：只有在新的 untouched protocol 上通过稳定性 Gate 后，H1 才能成为主方法
```

不再使用：

```text
真实多模态 → Adaptive K → 每个 component 是语义 prototype → Unknown clustering
```

作为当前已完成结论。

### 11.2 下一阶段若获授权

下一阶段不应继续搜索更多 max/weighted-sum 公式。应先预注册一个边界清晰的 Known-only continuation，目标是减少 H1 的 negative protocols 和 rsync/scp support-overlap failure；公式、输入信号、Gate 和新 external protocol 必须在打开任何 Unknown Test 前冻结。

最低 Gate 应继续包括：

- 新 external dataset mean AUROC/AUPRC 不低于 Open-Detect；
- paired ΔAUROC CI 支持正向结果；
- negative protocols 明显少于 H1/DES-v0；
- hard classes 不出现系统性大幅下降；
- UFAR 改善不能以显著 Known FRR 为代价；
- 全程不使用 Unknown calibration 或 learned test-time gating。

在该 Gate 通过前：

- Open-Detect 保持正式 baseline；
- H1 只称 development candidate；
- 不启动 RQ3 Unknown Discovery；
- 不把 VNAT 再称为 untouched external validation。

## 12. 当前文件与运行状态

### 12.1 Git

- 独立 Git 根目录：当前项目目录。
- Branch：`main`。
- HEAD：`ffcb0dfcaebe3b04b6d3bd42964d5a705303d2cc`。
- 当前无 staged files。
- 用户已有 tracked modifications：`EXPERIMENT_RESULTS.md`、`README.md`、`configs/dataset/ustc_tfc2016.yaml` 和当前详细实验计划；本文档不覆盖这些修改。
- GitHub remote：`sendsbrainsonly/unknown_traffic_project`。
- GitHub 推送已于 2026-09-23 完成；远端 `sendsbrainsonly/unknown_traffic_project` 的 `main` 已更新，发布内容保持为 Stage 22 运行中快照。
- 推荐发布包：代码、文档及小型核心结果约 `9.81 MiB`，预计 Git 压缩约 `3.13 MiB`；模型权重、cache、latent arrays、raw data 和大型逐样本表不得进入普通 Git。

### 12.2 存储清理

2026-09-19 存储清理完成，范围严格限于用户授权的 Python cache 和 75 个已完成 run 的 `latest_checkpoint.pt`：

- 删除前重新核验：Stage 14C.5 `45/45`、Stage 14C Native `15/15`、Stage 14C.6 `15/15` 个 latest/best 对应关系全部 PASS。
- 已删除 75 个 `latest_checkpoint.pt`，共 `24,270,861,345` bytes（`22.604 GiB`）。这些文件是完成 run 的末 epoch/恢复点；删除后不能从项目目录恢复，必要时只能重新训练生成。
- 已删除 93 个 Python cache 目录（89 个 `__pycache__`、4 个 `.pytest_cache`），共 `2,704,414` bytes；未发现额外 `.pyc/.pyo` 文件。
- 总删除量 `24,273,565,759` bytes，约 `22.606 GiB`；项目 apparent size 从约 `91 GiB` 降至约 `68 GiB`，文件系统可用空间由约 `388 GiB` 增至约 `410 GiB`。
- 独立复核：目标目录中 latest checkpoint 剩余 `0`；正式 best checkpoint 保留 Stage 14C.5/Native/Stage 14C.6=`45/15/15`，共 75 个、`24,271,014,094` bytes。
- 未删除任何 best checkpoint、结果表、manifest、frozen protocol、split、score、latent、smoke/interrupted/preliminary 证据或其他实验目录。
- 执行证据：`.tmux-task/cleanup_delete_preflight_20260919_002/`、`.tmux-task/cleanup_delete_execute_20260919_001/`、`.tmux-task/cleanup_delete_verify_20260919_001/`。

同日完成第二轮空间审计与用户授权的清理：

- 已删除另外 33 个历史已完成实验的 `latest/last` checkpoint，共 `10,699,129,825` bytes（`9.964 GiB`）。范围为 OpenDetect USTC audit 1 个、Stage 11A 16 个、Stage 14C-4 4 个、Stage 3 6 个、Stage 7 6 个。
- 删除前逐项验证文件路径、文件大小及对应 `best`；删除后独立复核 `targets_present=0`、`best_missing=0`。上述五个实验目录仍保留 865 个非 latest/last 文件，包括 best 权重、配置、指标、日志、manifest、逐样本/latent 数据及报告。
- 第二轮删除后项目 apparent size 为约 `57.443 GiB`，文件系统可用空间为约 `415.101 GiB`。
- 当前剩余 smoke 产物约 `4.137 GiB`、interrupted 产物约 `0.931 GiB`、Stage 14D preliminary 产物约 `0.291 GiB`、两份 superseded CICIDS shortcut backup 约 `0.056 GiB`。这些文件属于失败、smoke 或修订历史证据，删除前需再次明确授权。
- Stage 14C.5 F1/F3 的 30 个 best checkpoint 共 `9.042 GiB`；若不再做 F1/F3 inference，可作为高收益但会损失消融模型的候选。全部 F1/F2/F3 45 个 best 共 `13.563 GiB`。
- Stage 14C-3/14C-4 诊断 checkpoint 总计 `3.925 GiB`；可由报告和 metrics 支撑结论，但删除会失去复现诊断模型输出的能力。
- 当前正式 Stage 14C Native 15 个 best、Stage 14C.6 15 个 best、Stage 14B frozen protocol 和 Stage 15A/15B 核心证据仍列为必须保留。

### 12.3 进程

当前没有新的训练计划在执行。检测到三个历史 tmux payload shell 仍显示 running，但均为 0% CPU 的旧审计进程：

- `cipher_label_mismatch_audit_v1`
- `s14a-hdfschema-20260917`
- `s15a_manifest_keys_20260918_1604`

本文档任务未停止它们。它们应单独核对日志/exit status 后再决定是否关闭，不能当作正在运行的正式训练。

### 12.4 第三轮存储清理（A+B）

- 目标：按用户授权删除 smoke 模型权重、两份 superseded CICIDS shortcut backup、interrupted 中的大型权重/数组以及 Stage 14D preliminary artifacts；保留正式模型、正式结果、smoke/中断日志、配置和指标。
- 起始状态：项目 apparent size `57.443 GiB`、实际分配空间 `49.024 GiB`；预审计预计 A+B 去重后释放约 `5.288 GiB` 实际磁盘空间。
- 已删除 181 个文件：14 个 smoke 权重、10 个 interrupted 权重/数组、79 个 Stage 14D preliminary 文件、78 个 superseded backup 文件。
- 删除量：`5,752,696,342` logical bytes（约 `5.358 GiB`），实际分配空间 `5,678,365,184` bytes（约 `5.288 GiB`）。删除后项目 apparent size `52.086 GiB`、实际分配空间 `43.736 GiB`。
- 保留证据：338 个非权重 smoke 文件和 33 个非二进制 interrupted 文件仍在；正式 Stage 14C Native/Stage 14C.6/Stage 14C.5/旧 Stage 14C checkpoint 数分别为 `15/15/45/15`，Stage 3 和 Stage 7 正式 best 分别为 `3/3`。
- Git 检查：tracked deletion `0`。Stage 14D formal results/report 和当前 CICIDS component shortcut audit 均保留。
- 第一次独立核验使用了错误的 Stage 14C.6 文件名模式，误报 cleaned count 为 0；修正为实际 `*_best.pt` 命名后，第二次核验完整通过。
- 执行证据：`.tmux-task/cleanup_ab_preflight_20260919_001/`、`.tmux-task/cleanup_ab_delete_20260919_001/`、`.tmux-task/cleanup_ab_verify_20260919_002/`。
- 状态：`complete`。被删除的 smoke/interrupted 权重无法从项目目录恢复，只能重新运行对应实验生成；正式模型和正式结论未受影响。

## 13. 当前 handoff

- 2026-09-23 修订：本节原 Stage 15R handoff 已被后续阶段取代；历史记录保留在下方各阶段条目中。
- 最近完成的科学阶段：`Stage 22 — Pretrained TrafficFormer + E3 Closed-Set Comparison`，状态 `CLOSED_SET_DIAGNOSTIC_COMPLETE`；4/4 runs、99/99 checks PASS。
- 当前活动阶段：无；Stage 22 已完成，未启动后续 open-set 实验。
- 当前正式粗粒度协议：Stage 20 VPN-6 / TOR-7 closed split 与 13 个 LOSO protocols。
- 当前方法边界：Stage 22 的 E3 相对同 run TrafficFormer 在 VPN Macro-F1 低 `0.001115`、TOR 高 `0.000918`，两个数据集都只有 `1/2` positive seeds，不支持稳定优越性结论。
- Storage cleanup：已完成三轮；前两轮删除 108 个 latest/last checkpoint 和全部 Python/pytest cache，第三轮删除 181 个 A+B 候选文件并实际释放约 `5.288 GiB`。当前项目 apparent size `52.086 GiB`、实际分配空间 `43.736 GiB`；正式 checkpoint 与正式结果均保留。
- GitHub publication：Stage 22 完成结果已按同一白名单发布；大模型、数据和运行日志未进入普通 Git。
- 安全恢复点：不得将 Stage 22 闭集结果外推为 Unknown-Free open-set 结论；不得继续批量删除历史 checkpoint。

## 14. 关键证据入口

- 根级结果索引：`EXPERIMENT_RESULTS.md`
- TrafficFormer 短流修复：`docs/TRAFFICFORMER_SHORT_FLOW_ADAPTATION.md`
- Stage 2.5：`outputs/stage2_5_covariance_diagnosis/diagnosis_summary.md`
- Stage 3：`stage3_unknown_utility/README.md`
- CipherSpectrum one-shot：`stage9_cipherspectrum_final_test/RESULTS.md`
- Score decomposition：`stage10b_opendetect_score_decomposition/RESULTS.md`
- Dual external DES-v0：`stage12_dual_external_validation/RESULTS.md`
- USTC Global+Local：`stage13a2_global_local_fusion/RESULTS.md`
- VNAT protocol：`stage14b_vnat_protocol_freeze/RESULTS.md`
- VNAT lineage：`stage14b5_vnat_flow_retention_audit/RESULTS.md`
- Native Open-Detect VNAT：`stage14c_native_opendetect_vnat/RESULTS.md`
- F0/F2 failure diagnosis：`stage14c3_vnat_closed_set_failure_diagnosis/RESULTS.md`
- Cleaned 15-run：`stage14c6_cleaned_pipeline_full15_closed_set/RESULTS.md`
- VNAT open-set：`stage14d_vnat_frozen_open_set_evaluation/RESULTS.md`
- Complementarity：`stage15a_failure_regime_complementarity_diagnosis/RESULTS.md`
- Hybrid：`stage15b_known_only_hybrid_detector/RESULTS.md`

## 15. 四数据集当前结果边界（2026-09-19）

- 当前科学进度停在 `Stage 15B — Known-Only Hybrid Detector Development`；Gate 为 `HYBRID_PARTIAL`，尚未得到跨四数据集稳定胜出的最终 detector。Open-Detect Native 仍是正式 baseline，H1 仅是后续受控验证候选。
- Known-Test 闭集均值（15 个 frozen protocols）：USTC Accuracy/Macro-F1=`0.994324/0.945177`；VNAT Native=`0.858043/0.829606`；ISCX-VPN Native=`0.743350/0.731127`；ISCXTor2016 Native=`0.722965/0.689683`。四者协议和类别组成不同，绝对值不能视为严格同质 benchmark 排名。
- Open-set AUROC（Open-Detect/DES-v0/DES-v1）：USTC=`0.939528/0.970982/0.979954`；VNAT=`0.837535/0.851706/0.853085`；ISCX-VPN=`0.553024/0.576476/NA`；ISCXTor2016=`0.524614/0.594347/NA`。
- DES-v0 相对 Open-Detect 的 mean delta AUROC：USTC `+0.031454`（12/15 positive）、VNAT `+0.014171`（6/15）、ISCX-VPN `+0.023452`（14/15）、ISCXTor2016 `+0.069734`（14/15）。VNAT 提升不稳定；ISCX 两组相对提升稳定，但绝对 AUROC 仍低且 UFAR 分别约 `0.957/0.953`，不能表述为实用检测成功。
- Stage 15A 证明 OD/DES 信号具有双向互补，Gate=`COMPLEMENTARITY_CONFIRMED`；Stage 15B 的 H1/H2/H3 均未通过 worst-case Gate，最终为 `HYBRID_PARTIAL`。H1 被保留为最简单的后续候选，但不是已确认主方法。
- 证据范围：USTC/VNAT 已用于 development；ISCX-VPN/ISCXTor2016 在 Stage 12 提供过冻结外部验证，但进入 Stage 15 retrospective analysis 后不得再称为 untouched external datasets。DES-v1 尚无 ISCX 冻结结果。
- 状态：`complete`（只读汇总，无训练、无重拟合、无新 Test 打开）。

## 16. Stage 15R — Representation Bottleneck Audit（2026-09-19）

- 目标：在不使用 Unknown Test 进行训练、验证、特征选择或调参的前提下，审计 USTC/VNAT/ISCX-VPN/ISCXTor2016 的数据、特征和闭集表示瓶颈，并按 Gate 顺序执行 E0–E3 受控对照。
- 状态：`complete`。Stage 15R-0、20/20 E0–E3 pilots、Gate、报告、冻结资产哈希和 9/9 完成核验均已结束。
- 独立输出目录：`stage15r_representation_bottleneck_audit/`；已用实验留存工具创建 `RESULTS.md` 和 `manifest.json`。
- 冻结边界：Stage 12、14B/14C/14D、15A、15B 原有 checkpoint、score、manifest、protocol 和报告保持不变；Known Test 仅限方案冻结后闭集评价；Unknown Test 用量必须为 0。
- 已核对运行环境：`CONDA_PREFIX` 和 `sys.prefix` 均为固定 DGL Python 3.10 环境；所有 shell 载荷使用项目 tmux helper。
- 起始工作树已有大量历史用户实验文件与未跟踪产物，本阶段必须仅增量写入 Stage 15R 目录和本执行记录，不整理或覆盖其他变更。
- 已验证缓存：USTC `489101/489101`、ISCX-VPN `22142/22142`、ISCXTor `15096/15096` flow 严格闭合；三者均记录 `known_test_or_unknown_test_values_used_for_selection=false`。VNAT development cache 为 `23447/23449`，缺少的两条 ssh flow 在冻结协议中只作为 Known Test，特征值按边界未打开，并非预处理丢失。
- E0 Known-Validation Macro-F1：ISCX-VPN `0.769169`、ISCXTor `0.676900`、VNAT hard `0.739755`、VNAT normal `0.983318`、USTC A-2 `0.977545`；均复用冻结 checkpoint/预测，不重新训练且 Test 用量为 0。
- Pilot Gate：E1/E2/E3 mean ΔMacro-F1=`-0.072320/-0.166239/-0.128971`，均仅 `1/5` protocol 正向，worst Δ=`-0.148221/-0.384359/-0.302445`；clear/diagnostic Gate 全部失败，`full15_status=NOT_RUN_GATE_FAILED`。
- 工程失败留存：E1/E2 首次启动在 epoch 1 前因项目 TMP 路径触发 `AF_UNIX path too long`；原日志保存在 `failed_attempts/afunix_workers4/`。VPN/VNAT 临时用 workers=0 完成；随后实现项目本地 `/proc/self/fd/<fd>` 短路径，USTC/Tor 在不改科学配置的情况下恢复 workers=4。USTC 首次 workers=0 的慢速中断和 ISCXTor tshark partial 也分别完整留存。
- LightGBM `4.6.0` 安装在 Stage 15R 自己的 `vendor/` 中；共享 Conda 环境未修改。
- 最终结论：`DATA_OR_PROTOCOL_BOTTLENECK_IDENTIFIED`。同输入 CE-only 未稳定超过 Native，length+IAT 和简单统计也未形成稳定收益；主要证据指向细粒度 class overlap、Known composition、capture/group 协议与数据规模差异。
- 完整性：Known Test 用于选择=`0`、Unknown Test 使用=`0`、1,797 个受保护文件前后 SHA256 完全一致、20 个新/复用 checkpoint hash 全部验证、9/9 completion checks PASS。
- 下一步：Stage 15R 到此停止。若后续获授权，先做 capture/group-disjoint sensitivity 与 class-pair overlap audit；之后才考虑预训练 byte+temporal hierarchical encoder，不能从本阶段直接修改 DES、H1 或设计 Adaptive Hybrid。

## 17. Stage 15F-0 — Literature-Guided Feature Registry and Feasibility Audit（2026-09-19）

- 状态：`complete / STAGE15F0_PASS`；仅完成文献、特征、数据 lineage、窗口覆盖和加密可见性审计，Stage 15F-1 至 15F-5 均为 `NOT_RUN`。
- 文献：19/19 项一手来源登记完成，每项均记录 16 个输入/表示字段及 `PAPER_CONFIRMED / CODE_CONFIRMED / INFERRED / UNKNOWN` 证据标签；Deep Packet、MT-FlowFormer、FlowLens 未验证到官方代码，MIETT 官方仓库当前仍为 code-coming-soon。
- 特征注册表：冻结 B0--B4、T1--T2、S1--S2、M1--M3、P1 共 13 个配置；P1 在 provenance/license/preprocessing parity/pretraining overlap 审计前保持 blocked。
- Known Train/Val 窗口缓存：USTC `432537`、VNAT `23447`、ISCX-VPN `18121`、ISCXTor `11783`；Known Test/Unknown Test 特征值用量=`0/0`，所有协议 membership unmatched=`0`。
- N=8 协议均值截断率：USTC `31.47%`、VNAT `17.42%`、ISCX-VPN `11.82%`、ISCXTor `28.52%`；N=32 降至 `14.49%/7.59%/5.45%/16.90%`。类别异质性显著，但覆盖率不能替代性能实验或证明因果。
- 完整性：四数据集 cache parity PASS；2,928 条覆盖聚合；19 文献、13 特征、4 数据集、8/16/32/64 四个窗口全部通过 fail-closed verifier；3/3 单元测试通过；192 个初始保护文件和最终扩展范围 193 个文件前后均未变化。
- 失败证据：最初 ISCX-VPN Scapy 大 PCAP 慢路径被停止并完整保存在 `failed_attempts/iscx_vpn_scapy_slow/`；最终使用语义一致的 Stage15R-verified tshark 流并通过逐 flow parity。
- 下一阶段矩阵已写入 `stage15f1_preregistered_matrix.csv`，但没有启动训练；`single_feature_results.csv` 等未来正式结果文件按要求保持不存在。
- 关键入口：`stage15f_literature_guided_feature_benchmark/stage15f_report.md`、`feature_lineage_audit.md`、`packet_window_coverage.csv`、`completion_verification.json`。

## 18. Stage 15F-1A — Packet Window Sufficiency and Class-Conditional Benchmark（2026-09-19）

- 状态：`completed / E4_FAIL_NO_FUSION`。
- 目标：在五个 Stage 15R 预注册 Known-only pilot 上，以同一 T1 特征、同一两层 1D-CNN、同一训练预算和 seed 比较 T8/T16/T32；T8 先与 Stage 15R E2 做 parity，并审计 mask 对卷积与 pooling 的影响。
- 冻结边界：只读取 Known Train/Validation；Known Test/Unknown Test 特征值用量必须为 0；不修改 Stage 12--15R、Stage 15F-0、Open-Detect、DES 或 H1。
- 已核对 Stage 15R E2 配置：signed `log1p(frame_length)`、`log1p(IAT_us)`、valid mask；Known-Train median/IQR、clip `[-4,4]`；3→64→128 两层 Conv1d、masked mean pooling、CE；Adam 0.001、batch 128、100 epochs、MultiStepLR `[50,80]`、seed 2022、按 Known Validation Accuracy 选 checkpoint。
- 当前动作：审计现有缓存最大窗口、预测留存和可复用路径；随后创建独立实验包与 fail-closed 测试。

### Stage 15F-1A terminal record

- Status: `complete`
- Final Gate: `CLASS_CONDITIONAL_WINDOW_BENEFIT`
- Formal runs: `15/15` (T8/T16/T32 x five frozen Known-only pilots).
- T16 mean Delta Macro-F1 vs T8: `+0.015363`; positive protocols `5/5`; worst `+0.005718`.
- Known Test / Unknown Test usage: `0 / 0`; T64 and multiview training: `NOT_RUN`.
- Evidence: `stage15f1a_packet_window_benchmark/RESULTS.md` and `completion_verification.json`.

## 19. Stage 15F-1B — Statistical & Burst Feature Sufficiency Benchmark（2026-09-20）

- 状态：`complete / E4_FAIL_NO_FUSION`。
- 目标：在五个冻结 Known Train/Validation pilot 上，复核 Stage 15R E3 的 S12 parity，并以固定 LightGBM 配置比较 FULL_FLOW 的 S12/S-A/S-AB/S-ABC/S-ABCD/S-Burst 与 EARLY_16 的 S12/S-ABCD/S-Burst。
- 冻结边界：Known Test/Unknown Test 特征值用量必须为 0；不修改 protocol、Open-Detect、DES-v0/v1、H1 或既有 checkpoint/score；Pilot Gate 未通过时不运行 full15。
- 已验证起点：Stage 15F-1A Gate=`CLASS_CONDITIONAL_WINDOW_BENEFIT`；T16 mean ΔMacro-F1=`+0.015363`，T32 无稳定收益；Stage 15R E3 配置为 LightGBM 4.6.0、500 estimators、learning rate 0.05、num_leaves 31、Known-Val logloss early stopping 50、seed 2022。
- 输出目录：`stage15f_literature_guided_feature_benchmark/stage15f1b_statistical_burst_benchmark/`；实验包已初始化为 `stage15f1b-statistical-burst-benchmark-20260920-v1`。
- 当前动作：审计并冻结 feature definition、缺失值边界、FULL_FLOW/EARLY_16 lineage 与 protected hashes；随后先执行 S12 parity。
- 里程碑：S12 parity `5/5 PASS`，精确复用 Stage 15R E3 checkpoint，并复算得到一致的 Known-Validation 指标和 confusion matrix。
- 行为缓存：USTC `389982`、VNAT `21135`、ISCX-VPN `14521`、ISCXTor `9983` 个五协议 Known Train/Validation union flow；四者均与 Stage 15F-0 packet count/bytes/duration/direction 基础统计 parity `PASS`，Known Test/Unknown Test feature values=`0/0`。
- 已保存失败证据：USTC 首次方向编码误读；两次无科学语义变化的性能中断（重复 schema I/O、未合并分位数/重复 Early-16 计算）。修复后缓存完整通过，没有失败缓存进入正式训练。
- 当前动作：运行预注册的 40 个新增 LightGBM 配置；FULL_FLOW S12 的 5 个复用 run 与新增配置合计 45 个正式 Pilot。

### Stage 15F-1B terminal record

- Status: `complete`.
- Conclusions: `STATISTICAL_FEATURE_BENEFIT, BURST_COMPLEMENTARITY_CONFIRMED, CLASS_CONDITIONAL_BEHAVIOR_BENEFIT`.
- Formal runs: `45/45`; S12 parity: `5/5 PASS`; protected hash and completion verification: `PASS`.
- S-ABCD-S12 mean ΔMacro-F1: `+0.019473`; S-Burst-S-ABCD: `+0.006094`.
- Known Test / Unknown Test: `0 / 0`; full15, Stage15F-1C, open-set evaluation, DES/H1 changes, and Byte-Behavior model: `NOT_RUN`.
- Evidence: `stage15f_literature_guided_feature_benchmark/stage15f1b_statistical_burst_benchmark/RESULTS.md`.

## 20. Stage 15F-DQ — Cross-Paper Performance Gap Attribution（2026-09-20）

- 状态：`complete`。
- 当前授权范围：严格依次执行 DQ-0～DQ-3，并提交共享 31 个 ISCX-VPN PCAP 的中期报告；DQ-4～DQ-7、ISCXTor 扩展及跨模型公平训练保持 `NOT_RUN`。
- 数据边界：仅使用开发允许的 Known Train/Validation 与既有冻结 Known Validation 预测；Known Test/Unknown Test 不用于样本定义、标签映射、模型选择或诊断协议设计。
- 起始状态：独立输出目录尚不存在；Stage 12～15F-1B 历史资产保持原位；工作树已有大量历史未跟踪实验文件和 tracked 用户修改，本任务不整理或覆盖它们。
- 执行顺序：先完成跨项目 PCAP/flow 定义与哈希审计，再做 CATE/TrafficFormer 样本资格对账和冻结预测错误归因；只有 flow 对齐、标签映射和支持数 Gate 通过后，才允许执行 DQ-3 匹配的 Fine/Coarse Known-only 训练。
- 关键风险：TrafficFormer 的 service 标签可能由 capture/activity 而不是单一 application 决定；若固定 application→service 映射存在重大歧义，将按任务要求停止相关 F2/F3 分支而不静默改标签。
- 下一步：初始化实验留存 bundle，核对 31 个共享 PCAP、TFE-GNN CATE、TrafficFormer 过滤实现、Stage 12 split/provenance 和 Native Known Validation 预测的可连接键。

### Stage 15F-DQ DQ-0--DQ-3 interim terminal record

- 状态：`complete_with_blocked_branch`；DQ-0=`PASS`，DQ-1=`PASS_DESCRIPTIVE`，DQ-2=`PASS_DIAGNOSTIC`，DQ-3=`BLOCKED_MAPPING_AMBIGUITY`。
- DQ-0：Native/TFE-GNN/TrafficFormer 仅共同覆盖 31 个 VPN PCAP；Native 全量为 137 个 capture、22,142 个选定 session。共享 31 个 PCAP 上完整 flow manifest 23,645 行，其中 Native A 集 4,224 行；TrafficFormer parent-flow multiset 与既有 processing audit 31/31 精确一致。
- DQ-1：A/B/C/D flow 数=`4,224/2,720/1,984/1,804`；A 中 CATE MATCHED/UNMATCHED=`2,720/1,504`。UNMATCHED 只表示未与目标清单唯一匹配，不被解释为错误标签或背景流。
- DQ-2：共享31 Known Validation N=`335`，Accuracy/Macro-F1/Weighted-F1=`0.802985/0.770803/0.817823`；1--2 包流错误率 `15.74%`，>=3 包错误率 `29.00%`，当前子集不支持“短流单位样本更难”；Fine 错误中 22/66 映射后 Coarse 正确。
- DQ-3 Gate：Facebook 在当前 Known 集同时对应 Chat/VoIP，完整 Fine prediction→Service 映射不可识别；映射覆盖仅 `321/335=95.82%`。未引入多数映射或真实 capture 反推，未训练新的 F1/F3；Coarse 正式主任务仍为 `INSUFFICIENT_EVIDENCE`。
- 边界：DQ-4--DQ-7、ISCXTor、新 encoder/分类器训练均为 `NOT_RUN`；Known Test/Unknown Test 用量=`0/0`。
- 完整性：208 个冻结输入 SHA256 前后完全一致；独立完成性核验 PASS，21 个必需文件存在，31 shared PCAP、23,645 manifest rows、4,224 Native A rows 和 NOT_RUN 状态均复核通过。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq_performance_gap_attribution/RESULTS.md`、`performance_gap_attribution.md`、`completion_verification.json`。

## 21. Stage 15F-DQ-3R — Service Label Reconstruction（2026-09-20）

- 状态：`in_progress`。
- 目标：仅在共享 31 个 ISCX-VPN VPN PCAP 的冻结 Known Train/Validation 样本上，重建可审计的 capture/activity-level Service 标签来源，检查 Service 与 capture/group 支持和 Unknown-Free 语义，并预注册 DQ-3F；不训练模型。
- 冻结边界：Known Test/Unknown Test 特征值用量必须为 `0/0`；不修改 Stage 12--15F-DQ、DES、H1、历史 Fine/Open-Set 结果；不启动 Byte--Behavior。
- 独立输出：`stage15f_literature_guided_feature_benchmark/stage15f_dq3r_service_label_reconstruction/`；实验 ID=`stage15f-dq3r-service-label-reconstruction-20260920-v1`。
- 起始证据：DQ-0 已重建 31/31 TrafficFormer parent-flow multiset；共享31 Native A 集 4,224 flows。DQ-3 的完整 F2 因 Facebook application→Chat/VoIP 一对多而阻断，但这不自动否定使用 capture/activity Service 标签直接训练 F3。
- 标签原则：允许同一 application 在不同 capture 对应不同 Service；capture 文件名/项目映射只能作为 capture-level 弱标签，除非存在独立逐流定义，不得宣称为逐流权威真值。
- 工程记录：首次 bundle 初始化命令误用了 `--run-dir/--experiment-id`，exit 2，日志保留于 `.tmux-task/s15fdq3r_init_20260920/`；正确接口重试成功，无科学语义影响。

### Stage 15F-DQ-3R terminal record

- 状态：`complete / SERVICE_TASK_FEASIBLE_WITH_WEAK_LABELS`；本阶段只完成标签来源、支持度、group 可行性、Open-Set 语义与 DQ-3F 预注册，训练次数=`0`。
- 标签重建：31/31 个共享 VPN capture 的 Stage 12 与 TrafficFormer capture/activity 映射一致；冻结 Known Train/Validation 共 `3,065` flows（`2,730/335`），全部获得确定的 capture-derived Service 标签。
- 证据强度：`VERIFIED_DEFINITION/WEAK_CAPTURE_LABEL/AMBIGUOUS/UNAVAILABLE = 0/3065/0/0`。这些标签只说明 capture 的目标实验活动，不能宣称为逐 flow 权威真值，也未利用 Validation 结果反推标签。
- Service 支持：Chat=`132/17`、Email=`145/23`、File-Transfer=`332/43`、P2P=`804/100`、Streaming=`854/104`、VoIP=`463/48`（Train/Validation）。所有 Service 均有流级支持。
- group 边界：P2P 只有 1 个独立 capture，因此共享31子集不支持完整六类 capture-disjoint Train/Validation 泛化结论；当前 flow-level DQ-3F 只能作为弱 capture-label 诊断。
- Open-Set 语义：既有 Fine Unknown application 在 Service 空间与 Known Service 重叠，原 Low/Medium/High split 不可直接改名复用；未来若采用 Service Open-Set，必须独立重建并冻结 Known/Unknown Service。
- DQ-3F：F1/F3 使用完全相同的 3,065 flow IDs 和相同 Train/Validation membership；F2 只作可确定映射样本上的 `PARTIAL` 评价；配置已预注册，状态=`PREREGISTERED_NOT_RUN`。
- 完整性：17 个冻结输入 SHA256 前后一致；独立 verifier PASS；实验保存校验 `status=success artifacts=18 bundle_files=18`；Known Test/Unknown Test 特征值=`0/0`，Byte--Behavior 未启动，DES/H1 未修改，旧实验未覆盖。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq3r_service_label_reconstruction/RESULTS.md`、`dq3r_report.md`、`completion_verification.json`。

## 22. Stage 15F-DQ-3F — Fine vs Service Matched-Sample Training Benchmark（2026-09-20）

- 状态：`in_progress`。
- 目标：在 DQ-3R 冻结的 3,065 条共享31 ISCX-VPN Known Train/Validation flow 上，用相同输入、membership、模型主体和训练预算配对比较 Fine Application（F1）与直接 Service 监督（F3），并生成不训练新模型的 F2-partial。
- 冻结边界：Known Test/Unknown Test 特征值必须保持 `0/0`；不修改 Stage 12--15F-1B、DQ/DQ-3R、Open-Detect、DES-v0/v1、H1；DQ-4--DQ-7、TFE-GNN/TrafficFormer 新训练和 Byte--Behavior 保持 `NOT_RUN`。
- 已核对预注册配置：ResNet18、1 channel、latent 128、100 epochs、batch 128、Adam 0.001、Open-Detect lambda 0.005、MultiStepLR `[50,80]`、prototype reset zero-based `[50,80]`、Known-Validation Accuracy checkpoint selection、early stop patience 10/min epoch 82。
- 重复规则：DQ-3R 固定单次 seed=2022，但未登记 repeat count；按 DQ-3F 任务的缺省规则，在结果打开前冻结配对 seeds=`2022,2023,2024`，F1/F3 获得完全相同训练预算。
- 当前动作：先完成 3,065 条 flow 与 Stage 12 Known Train/Validation image array 的行序、标签、hash、Unknown-free 和输入元数据排除 Gate；不一致则停止训练。
- Preflight Gate：`PASS`。3,065 flow IDs 唯一，Train/Validation=`2730/335`，11 Fine classes、6 Services；Train/Validation flow-ID 与 exact-image overlap 均为 `0`；Stage12 NPZ target/排序/array digest 均通过；Known Test/Unknown Test feature values=`0/0`。
- GPU 运行：六个 paired jobs 使用项目 GPU selector 在单卡串行执行。前两次启动均在 epoch 1 前因当前 PyTorch/CUDA 不支持预初始化 `reset_peak_memory_stats` 而失败；失败目录与 tmux 日志完整保留，移除非科学必需的 reset 后正式队列已正常进入 epoch 循环，未修改模型、数据或训练配置。

### Stage 15F-DQ-3F terminal record

- 状态：`complete / COARSE_TRAINING_BENEFIT`；附加限制=`CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`。
- 正式运行：F1 Fine 与 F3 Service 各 3 seeds（2022/2023/2024），共 `6/6 SUCCESS`；相同 2,730 Train、335 Validation flow、32x32 byte image、Native architecture/loss/optimizer/budget/checkpoint rule，仅监督标签和输出类数不同。
- F1 Fine 三 seed mean±std：Accuracy=`0.784080±0.033625`，Macro-F1=`0.635049±0.046443`，Weighted-F1=`0.775730±0.040095`。
- F2-partial common subset：coverage=`322--324/335`，Accuracy=`0.913223±0.041867`，Macro-F1=`0.871102±0.063815`；它是预测依赖的部分评价，不能替代 F3。
- F3 Service full 335：Accuracy=`0.890547±0.041458`，Macro-F1=`0.856050±0.064200`，Weighted-F1=`0.888036±0.044738`；F3 Macro-F1 在 3/3 seeds 高于 F1，但该差值反映任务粒度而非同任务模型提升。
- 相同 F2-evaluable subset 上，F2 比 F3 mean Accuracy/Macro-F1 高 `+0.017528/+0.008203`；因此“直接 Service 训练优于预测映射”不成立，F3 的优势是覆盖全部335样本和独立监督目标。
- Fine error decomposition：三个 seed 合计 217 个 Fine 错误中，126 个（`58.06%`）在 deterministic Service 语义下兼容；标签粒度是显著因素，但不是全部原因。
- F3 mean Recall：Chat=`0.745098`、Email=`0.855072`、File-Transfer=`0.705426`、P2P=`0.950000`、Streaming=`0.900641`、VoIP=`0.979167`。P2P/Streaming 大类影响 Accuracy，但 Chat/Email 未崩溃；File-Transfer 与 Chat 最困难，seed2023 较弱。
- 边界：P2P 只有一个 capture，不能宣称 capture-disjoint 泛化；所有 Service targets 仍为 weak capture labels。DQ-4 可作为下一步过滤/目标流审计，但 DQ-7 在统一 Service 数据、flow、标签、group split 和指标冻结前尚不公平。
- 完整性：21 个受保护输入前后 SHA256 一致，6 checkpoint SHA256 验证，独立 verifier PASS，实验包 `status=success artifacts=92 bundle_files=92`；Known Test/Unknown Test feature values=`0/0`；DQ-4--DQ-7、TFE-GNN/TrafficFormer 新训练、Byte--Behavior=`NOT_RUN`，DES/H1 未修改。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq3f_fine_service_matched_benchmark/RESULTS.md`、`dq3f_report.md`、`completion_verification.json`。

## 23. Stage 15F-DQ-4 — Flow Selection and Filtering Attribution（2026-09-20）

- 状态：`in_progress`。
- 目标：只在 DQ-3R/DQ-3F 冻结的 3,065 条 Known Train/Validation flow 上，重建 A（全部）、B（CATE matched）、C_PARENT/C_FINAL（TrafficFormer eligible/final Service inclusion）和 D（B∩C），分离 evaluation-population effect 与 training-selection effect。
- 冻结边界：Known Test/Unknown Test 特征值用量保持 `0/0`；不修改 DQ-3F、TrafficFormer、TFE-GNN、DES/H1 或历史结果；不启动 DQ-5～DQ-7、Byte--Behavior、Open-Set。
- 已确认母集：Train/Validation=`2730/335`；旧 DQ-0 A=`4224` 只作来源审计，不能作为本阶段样本。
- 当前动作：对 3,065 条 flow 逐条连接既有 CATE/TrafficFormer lineage，核验 C_PARENT 与 final Service manifest 的关系，先执行六类 Train/Validation 支持 Gate；通过的 B/C/D 才按 DQ-3F F3 冻结配置训练。

### Stage 15F-DQ-4 terminal record

- 状态：`complete`；Gates=`EVALUATION_POPULATION_EFFECT, FILTERING_EFFECT, CLASS_CONDITIONAL_SELECTION_EFFECT`；长期限制=`WEAK_CAPTURE_LABEL, CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`。
- 冻结母集 A=`3,065`（Train/Val=`2,730/335`）；重新求交得到 B=`2,101`、C_PARENT=C_FINAL(Service)=`1,551`、D=`1,440`。旧 DQ-0 的 A/B/C/D=`4,224/2,720/1,984/1,804` 未被错误复用。
- TrafficFormer 实际过滤：17,769 parent rows 中 `<2048 captured bytes` 删除 16,243，成功 1,526；Service final manifest 1,526 与 success multiset 精确一致。Native session 与 TrafficFormer parent 并非一一对应，因此 C 是 parent-derived selection subset。
- 六类 Gate：B/C/D Train/Val 均保留 6 Services，但 B/D VoIP Val 仅 4 条、C 最小 Val=6，所有 per-Service 结论均标记低支持。
- M-A 复用 DQ-3F 3 个 checkpoint 并通过 array/prediction/hash parity；M-B/M-C/M-D 各 3 seeds，共 9/9 新 run SUCCESS，不删除或重跑弱 seed。
- common D_val Macro-F1 mean：M-A/M-B/M-C/M-D=`0.865662/0.673803/0.898537/0.675056`；相对 M-A delta=`−0.191859/+0.032875/−0.190606`。M-C 正向 `2/3` seeds，seed2024 为 `−0.024805`，因此 filtering benefit 是部分且 seed-sensitive，不是稳定全胜。
- M-B/M-D 在 seed2023 出现 optimization collapse（own Macro-F1=`0.200651/0.221315`）；这是冻结配置的真实结果，未调参。全 A stress Macro-F1：M-A/M-B/M-C/M-D=`0.856050/0.431240/0.622356/0.400851`，显示 selection-induced distribution shift 明显。
- Track E 不重训：同一 M-A 在 B/C/D_val 相对 A_val Macro-F1 delta=`−0.019417/+0.003782/+0.009612`，说明仅改变评价人口的影响较小且方向混合，不能解释 M-B/M-D 的大幅退化。
- 完整性：9 个新 checkpoint hashes、3 个 M-A frozen hashes、受保护输入前后 hashes、12 个 verifier checks 全部 PASS；Known Test/Unknown Test feature use=`0/0`；DQ-5～DQ-7、TrafficFormer/TFE-GNN 新训练、Byte--Behavior、Open-Set=`NOT_RUN`，DES/H1 未修改。
- 证据：`stage15f_literature_guided_feature_benchmark/stage15f_dq4_flow_selection_filtering_attribution/RESULTS.md`、`dq4_performance_gap_attribution.md`、`completion_verification.json`。

## 24. Stage 16 — Coarse Service Classification and Open-Detect Benchmark（2026-09-20）

- 状态：`in_progress / method identity gate`。
- 目标：先核验 DQ-3F Service-6 模型与 Open-Detect 的方法身份，并对 A（2730/335）和 C（1367/184）协议做 flow membership、Service 标签及哈希 parity；只有两个方法实体确实独立时才允许新训练。
- 冻结边界：Known Test/Unknown Test 特征值用量保持 `0/0`；不修改 DQ-3F、DQ-4、Open-Detect、DES/H1 或历史 checkpoint/prediction；全部标签仍为 `WEAK_CAPTURE_LABEL`。
- 首轮证据：DQ-3F `train_dq3f.py` 直接从 `Projects/Open-Detect` 导入并实例化 `CorrectedOpenDetectNet`，同时复用 Open-Detect `run_epoch`、`reset_prototypes_in_place` 和 `weight_init`；其冻结配置明确写为 `training_protocol=corrected-paper`。
- 风险处理：不得把 corrected Open-Detect reproduction 重命名为独立“我们的方法”，也不得静默换入未参与 DQ-3F 的 Stage14C-6 F2 模型。若最终身份 Gate 不通过，将保存阻断审计和真实历史结果，但不启动伪两方法训练。

### Stage 16 terminal record

- 状态：`partial / BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`；数据 parity=`PASS`，方法身份 Gate=`FAIL_EXPECTED`。
- 直接证据：DQ-3F trainer 将 Open-Detect `code/` 与 `reproduction/` 加入 import path，实例化 `CorrectedOpenDetectNet(OpenDetectNet)`，并复用 Open-Detect `run_epoch`、in-place prototype reset、`weight_init`、VAE/prototype loss 与 Known-Val Accuracy checkpoint selection；这只能解释为 corrected Open-Detect reproduction，不能作为独立“我们的方法”。
- A 协议复核：Train/Val=`2730/335`；C 协议复核：`1367/184`；6 类标签与 flow IDs 唯一性通过，canonical membership/label hashes 已保存。
- 历史 DQ-3F corrected Open-Detect 三 seed：Accuracy=`0.890547±0.041458`、Macro-F1=`0.856050±0.064200`、Weighted-F1=`0.888036±0.044738`；这些真实结果保留，但不再误记为独立 OURS。
- 新 Open-Detect 训练、两方法 paired delta、paired errors 和两方法 C 对照均标记 `NOT_RUN_IDENTITY_GATE / NOT_IDENTIFIABLE`；训练次数=`0`，GPU用量=`0`，Known Test/Unknown Test=`0/0`，Open-Set/DES/H1/Byte–Behavior 均未启动。
- 完整性：19 个 DQ-3F/DQ-4/Open-Detect 受保护文件前后 SHA256 一致；独立 verifier `PASS`；正式证据位于 `stage16_coarse_service_opendetect_benchmark/`。
- 后续只能二选一重新预注册：准确命名的 corrected-vs-released Open-Detect 变体研究，或选择真正独立的 Stage14C-6 F2 own-method 在 A/C Service 数据上重建公平比较。

## 25. Stage 16S — Service-Level Open-Set Benchmark（2026-09-20）

- 状态：`in_progress / protocol and method reuse audit`。
- 目标：在六类 Service 的 Leave-One-Service-Out flow-level 协议上，为每个留出 Service 与 seed 训练一个严格 Unknown-Free 的 corrected Open-Detect 五类模型，并共享其 encoder/classifier 比较 OD-Native、DES-v0、DES-v1 和 H1。
- 方法冻结：DES-v0 使用 posterior mean 到 Known-Train empirical centroid 的最小平方欧氏距离；DES-v1 固定 k=10、Known-Val median/MAD、global/local=0.5/0.5；H1 固定为 OD 与 DES-v0 Known-Val ECDF percentile 的 max；阈值均为 Known-Val P95 (`method=higher`)。
- 协议设计：保留历史 Known Train 作为新 Known Train；历史 Known Validation 按 Service 和 exact-image group 确定性 1:1 拆成新 Known Validation/Test；被留出 Service 的全部3065-pool样本作为 Unknown Test。这样不把旧 Known Test/Unknown Test 移入训练，也不让 exact image 跨角色。
- 已核验开发池：Train/旧Val=`2730/335`；32 个重复表示行仅形成7个同角色同Service group，旧Train/Val exact-image交集=0，跨Service exact-image group=0；flow-ID交集=0。capture 在角色间重叠，故所有结论限定为 flow-level weak-label，P2P 单capture轮次额外标记 `PROTOCOL_LIMITED`。
- 当前动作：生成并冻结六轮 manifest、方法/资产 hashes 和 preflight Gate；通过后先跑单个 pilot，再执行其余预注册轮次，不按 Unknown 结果删轮次。
- Preflight：`PASS`。六轮 manifest 共18,390行；flow-ID与exact-image跨角色交集均为0；Unknown fitting=0；所有Known类训练支持均满足kNN-10。Unknown样本数 Chat/Email/File-Transfer/P2P/Streaming/VoIP=`149/168/375/904/958/511`。
- Pilot：`loso_chat/seed2022` SUCCESS，GPU0，runtime=`133.83s`，Known-Test closed Macro-F1=`0.956907`。OD/DES-v0/DES-v1/H1 AUROC=`0.678985/0.757718/0.820554/0.735948`；对应UFAR=`0.932886/0.798658/0.798658/0.865772`，说明排序改善未自动解决P95误接收。
- Pilot完整性：一个checkpoint由四方法共享；Unknown train/val/support/normalization/threshold样本均为0；Known-Test selection=0；467条评分样本唯一、四类分数有限、checkpoint/score hashes已保存。
- 正式网格：通过live GPU selector选择物理GPU `0,5,6,7`（均约45.5GiB free、0% util）并启动剩余冻结任务；session=`s16s_full_grid_20260920`，不使用繁忙GPU1--4。

### Stage 16S terminal record

- 状态：`complete / PASS_WITH_PRESERVED_RECOVERY`；正式 checkpoint=`18/18`，四方法指标=`72`行，配对比较=`54`行。
- 原始训练执行：17 个 run 完整 SUCCESS；`Streaming-2023` 完成100 epoch并保存 epoch-79 best checkpoint 后，在后处理 parity 断言处失败。失败目录、日志和 `FAILURE.json` 原样保留，未重训、未删除。
- 失败定位：checkpoint-selection 的 `PIL→ToTensor` 与手写 `float/255` 输入最大差 `5.96e-8`，使塌缩模型中 1/114 个近零 margin Validation 样本翻转；统一原验证预处理后 checkpoint Accuracy 精确恢复为 `0.552632`。18 个冻结 checkpoint 随后在独立 `canonical_evaluation/` 完成纯推理。
- 总体 OD/DES-v0/DES-v1/H1 AUROC=`0.623635/0.652011/0.686584/0.642470`；AUPRC=`0.776025/0.789242/0.806845/0.787600`。
- 相对 OD：DES-v0 ΔAUROC=`+0.028376`（11/18 wins）；DES-v1=`+0.062949`（13/18）；H1=`+0.018835`（8/18）。DES-v1 排名最佳，但 File-Transfer/Streaming Service mean 退化；H1 在 Streaming/VoIP 的 3/3 seeds 均退化。
- P95 工作点仍差：OD/DES-v0/DES-v1/H1 UFAR=`0.927674/0.927872/0.914446/0.915263`；Known FRR=`0.035700/0.039963/0.038198/0.042734`。因此只能支持 Service-conditional ranking benefit，不能宣称未知拒识已解决。
- 训练稳定性：seed2023 在六个 LOSO 协议均出现 optimization collapse，Known-Test Macro-F1=`0.255009--0.377309`；全部保留在正式均值中，没有选择性重跑。
- 边界：六类均完成三 seed，但全部为 `WEAK_CAPTURE_LABEL` flow-level 协议；P2P 只有一个 capture，`CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`，不能作跨capture或外部泛化主张。
- 完整性：Strict Unknown-Free、共享 encoder、Unknown/Test calibration=`0/0`、18 checkpoint hashes、score hashes、72/54 行数、16 个受保护历史资产前后 SHA256 均 PASS；3/3 单元测试通过。
- 证据：`stage16s_service_open_set_benchmark/stage16s_report.md`、`RESULTS.md`、`completion_verification.json`、`service_open_set_six_metrics.csv`、`paired_vs_opendetect.csv`。
## 27. Stage 17-0/17-1 — Historical Encoder Recovery and Open-Set Pilot（2026-09-20）

- 状态：`complete / STRICT_PILOT_COMPLETE_PROMOTION_NOT_YET_SUPPORTED`。
- 目标：恢复历史 TrafficFormer、FIG/TAGCN 与 concat fusion 的真实代码、特征和 checkpoint 血缘，并在 Stage 16S 冻结的 Email/Streaming × seeds 2022--2024 上进行最小 Service-LOSO 编码器比较。
- 已确认：Model A 与 Model B 的实现和 USTC-20 checkpoint 存在；Model C 仅找到可执行的 train-zscore + concat + linear-probe 代码，尚未发现历史成品 checkpoint。历史 checkpoint 不能直接作为当前五类 Known 的 LOSO 模型。
- 暴露边界：TrafficFormer 官方预训练权重可加载，但公开 README 未披露预训练语料组成，记为 `PRETRAINING_EXPOSURE_UNVERIFIED`；严格主轨不把该权重混入 Unknown-Free 比较。
- 输入恢复：Stage 12 provenance 保留原 PCAP 与 packet refs；Stage 17 将复用已经验证的 Stage 12 双向 session 状态机，从 31 个共享 PCAP 为冻结 3,065 flow 恢复 TrafficFormer 前5包字节和 FIG 前30包图，不重划分数据。
- 输出：`stage17_encoder_recovery_and_open_set_pilot/`，实验 ID=`stage17-historical-encoder-recovery-open-set-pilot-20260920-v1`。
- 输入与恢复核验：使用 Stage 12 双向 session 状态机从 31 个共享 PCAP 精确恢复冻结 3,065 flows；缓存 SHA256=`aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff`，TrafficFormer/FIG/fusion forward-backward、确定性、历史 A/B checkpoint 严格加载均 PASS。最初基于 tshark 的恢复因本机不支持 `frame.raw` 失败，失败日志已保留，随后使用相同 flow 构造规则的 Scapy 流式读取成功。
- 正式运行：Email/Streaming × seeds 2022--2024 共 6 个冻结协议全部完成；E0 复用 6 个 Stage 16S checkpoint，E1/E2/E3 新训练 checkpoint 共 18 个；Unknown 用于训练、验证、support、normalization、threshold 的数量均为 0，20,028 条逐样本预测已保存。
- Email mean：E0/E1/E2/E3 Known-Val Macro-F1=`0.698090/0.099522/0.485736/0.907480`，DES-v1 AUROC=`0.718684/0.605146/0.617569/0.801853`，UFAR=`0.948413/1.000000/0.912698/0.805556`。
- Streaming mean：E0/E1/E2/E3 Known-Val Macro-F1=`0.694651/0.119760/0.483253/0.795084`，DES-v1 AUROC=`0.674952/0.504437/0.565708/0.720970`，UFAR=`0.883090/0.990257/0.713640/0.783925`。
- 配对结论：E3 相对 E0 的 AUROC 在 6/6 run 提升；Email/Streaming mean ΔAUROC=`+0.083169/+0.046018`，mean ΔUFAR=`-0.142857/-0.099165`。但绝对 UFAR 仍高，E1 严格随机初始化发生明显塌缩/欠拟合，且 E3 同时改变表示与线性 probe，不能把收益单独归因于图分支。
- 决策：仅支持两个 development Services 上的历史表示恢复与 pilot 证据，四数据集正式升级=`NOT_YET_SUPPORTED`；未启动下一阶段。
- 完整性：6/6 formal runs、18/18 新 checkpoint、6/6 复用 checkpoint、Strict Unknown-Free、17 个受保护资产前后 hash、153 个 bundle 文件及 manifest hash 校验均 PASS。
- 证据：`stage17_encoder_recovery_and_open_set_pilot/stage17_report.md`、`RESULTS.md`、`completion_verification.json`、`encoder_pilot_results.csv`、`manifest.json`。

## 28. Stage 18 — PP-OpenNet-inspired Independent Packet Encoder（2026-09-20）

- 状态：`complete / E4_FAIL_NO_FUSION`。
- 目标：在 Stage 17 冻结的 Email/Streaming × seeds 2022--2024 Service-LOSO 协议上，实现不依赖 Open-Detect 训练循环、模型或原型机制的独立 E4 包级时间—长度编码器，并完成闭集、MSP/Energy/Feature-Distance 开集检测及预注册消融。
- 文献边界：PP-OpenNet 官方公开代码未找到；本阶段只复用论文明确披露的 metadata-only、多尺度、循环融合思想。网络宽度、归一化、优化器细节和拒识实现均标记为项目自主设计，结论为 `approximate / inspired independent reconstruction`，不是作者等价复现。
- 数据边界：复用 Stage 17 SHA256=`aff17d4b3271107852c0f30556627ab5daf693fba6a0a7b18df5f3785f260cff` 的 3,065-flow 缓存；真实字段为方向、caplen 和相对首包时间，可计算 IAT。缓存最多30包，不能冒充论文的1000包/2秒 retroactive slicing。
- 冻结协议：`loso_email`、`loso_streaming`，seeds `2022/2023/2024`；Known Train/Validation 仅用于训练、归一化、checkpoint 选择、支持和阈值；Known Test/Unknown Test 仅最终评价。
- 独立目录：`stage18_pp_opennet_inspired_e4/`；实验 ID=`stage18-pp-opennet-inspired-e4-20260920-v1`。不覆盖 Stage 17 或 Stage 16S 资产。
- 执行结果：6/6 正式 run、42/42 变体和 checkpoint 完成。E4_FULL Known Test Macro-F1 为 Email `0.749697`、Streaming `0.798280`；主 feature-distance AUROC 为 `0.627692/0.560439`，UFAR@Val-P95 为 `0.882937/0.838900`。
- 配对结论：相对 E0，总体 ΔMacro-F1=`+0.077618`，但 ΔAUROC=`-0.056970`，仅 `3/6` AUROC 胜出；相对 E3，总体 ΔAUROC=`-0.167346`，`0/6` 胜出。闭集改善没有转化为稳定开集收益。
- 消融结论：严格的 Time+Length−Length 为 `+0.055788 Macro-F1/-0.035383 AUROC`，Time+Length−Time 为 `+0.176222 Macro-F1/-0.031306 AUROC`；Direction 为 `+0.035479 AUROC`；Multi-scale 为 `+0.011391 Macro-F1/+0.009839 AUROC`；RNN 为 `+0.016056 Macro-F1/-0.007519 AUROC`。当前 30 包单分支不足以降低绝对 UFAR。
- 完整性：42/42 checkpoint 独立重放最大绝对差 `0.0`，19 个受保护 Stage 16S/17 资产前后 SHA256 一致，Unknown 训练/验证/归一化/support/threshold 使用均为 `0`。
- 决策：预注册 Gate 四项失败，融合未启动。详细落地文档为 `CURRENT_PROGRESS_2026-09-21.md`；证据位于 `stage18_pp_opennet_inspired_e4/RESULTS.md`、`stage18_report.md`、`completion_verification.json`。

## 29. Stage 19 — RoNeTC Four-Dataset Fair Comparison（2026-09-21）

- 状态：`in_progress / input-lineage and comparability audit`。
- 目标：将 `Projects/RoNeTC` 中恢复的 RoNeTC 源码接入主项目 Stage 15R 冻结的四数据集五个代表协议，在相同 flow IDs、Known/Unknown 类、Train/Validation/Test membership、seed、训练预算和评价口径下，与既有 E0/Open-Detect 及项目方法结果做配对比较；允许的调参仅使用 Known Train/Validation。
- 冻结协议：ISCX-VPN `medium-2022`、ISCXTor2016 `medium-2022`、VNAT `medium-2025/2026`、USTC-TFC2016 `A-2`。历史协议、checkpoint、预测与结果不得覆盖。
- 已确认 RoNeTC 状态：作者历史源码已恢复并完成结构 smoke；现有真实数据运行仅为 CIC-IDS-2017 三类近似实验，尚无上述四数据集 checkpoint 或结果，不能直接宣称四数据集已完成。
- 输入边界：RoNeTC 需要每流前三视图（IP header、transport header、payload）与前 4/8/12/16 包；Stage 15R 缓存只保存时间、长度、方向，因此必须从已有 Stage-0 PKL/原始 PCAP 按冻结 flow ID 和原 sessionization 规则重建，不允许用 32x32 Native 图像伪装 RoNeTC 三视图。
- 当前动作：完成五协议逐样本 packet-view 可回溯性 Gate，冻结 Known-only 小规模调参方案与计算预算，随后实现独立 Stage 19 adapter、训练、测试、配对汇总及完整性核验。

### Stage 19 user-requested interruption record

- 状态：`paused_by_user / no owned GPU processes`。
- 输入 Gate：`PASS`。冻结协议 manifest 共 523,029 行，Unknown Train/Validation=`0/0`；ISCX-VPN、ISCXTor、VNAT、USTC 三视图缓存分别恢复 `22,142/15,096/23,449/438,893` 条 flow，缺失均为 0。
- VNAT 首次缓存因只处理 `tcp.stream/udp.stream` 而漏掉 12 条 frozen `other:*` IP flow；失败缓存完整保留，解析器改为与 Stage14A/14C 相同的 canonical endpoint+IP protocol identity 后，23,449/23,449 全部通过。
- RoNeTC dense adapter 单卡 parity 和 USTC 两卡 DataParallel smoke 均 PASS；后者保持全局 batch=128，单卡峰值显存约 17.46 GB，未启动 USTC 正式训练。
- 用户要求停止当前用户全部 GPU 进程。共向 7 个属主为 `birkenwald` 的 GPU PID 发送 SIGTERM：Stage19 四个训练，以及三个先前存在的 LLM replay/activation 服务；无需 SIGKILL。用户 `nlp` 的 GPU7 训练未触碰。
- Stage19 中止点：ISCXTor=`35/100`、ISCX-VPN=`24/100`、VNAT-2025=`2/100`、VNAT-2026=`2/100`；四个 session 均 exit 143。checkpoint、history、tmux log 与 `INTERRUPTED.json` 已保留，均不得作为正式结果。
- 完成边界：Known Test/Unknown Test evaluation 未运行，USTC formal 未启动，RoNeTC 与基线的正式四数据集比较尚未完成。
- 当前 GPU 核验：`CURRENT_USER_GPU_PROCESSES=0`；唯一剩余 GPU7 进程属主为 `nlp`。
- 恢复规则：仅在用户明确要求后，把四个 partial run 目录移入 `interrupted_attempts/` 永久保存，再从零启动正式 run；不得从 best-so-far checkpoint 继续并冒充同一预注册训练。

### Stage 19 maximum-three-GPU resume record

- 状态：`running / maximum 3 physical GPUs`。
- 用户授权最多三卡并行且无需持续监控；队列每15秒更新 `stage19_ronetc_four_dataset_comparison/progress.txt`。
- 四个中止 run 已完整移至 `interrupted_attempts/user_stop_20260921/`，正式 run 从零开始，不复用 best-so-far checkpoint。
- USTC 三卡 DataParallel smoke 在物理 GPU `0,4,5` PASS：全局 batch=128，dense adapter parity 最大误差=`0.0`，三卡峰值 allocated memory 约 `11.75/11.74/11.46 GB`。
- 正式调度：首轮最多三个单卡 run；第四个在一个 slot 释放后启动；前四个全部成功后，USTC 独占三卡启动。任何 run 失败均不自动重试，且会阻止 USTC 启动。
- 监控入口：`stage19_ronetc_four_dataset_comparison/progress.txt` 与 `queue_status.json`；正式 Test 指标仅在对应 run 完成100 epoch后生成。

### Stage 19 independent single-GPU scheduling correction

- 状态：`running / independent one-GPU-per-process / maximum 3 physical GPUs`。
- 用户澄清三卡是并发上限，不要求把一个任务的 batch 拆到多卡；调度目标改为每个独立训练进程占一张卡。当前仅剩 VNAT-2026 和 USTC A-2，因此同时使用两张卡，第三张保持空闲。
- 原四卡 takeover 已于 `2026-09-21T10:24:30Z` 通过监督进程 SIGTERM 干净停止；VNAT=`7/100`、USTC=`0/100`，部分产物分别保存在 `stage19_ronetc_four_dataset_comparison/interrupted_attempts/three_gpu_reconfigure_20260921/vnat/medium_seed2026/` 与 `ustc/A-2/`，均有 `INTERRUPTED.json` 且不得作为正式结果。
- USTC 单卡 full-batch smoke：batch=`128`，`data_parallel=false`，峰值 allocated memory=`34,887,070,720` bytes，finite forward/backward/optimizer=`PASS`，dense-adapter parity 最大误差=`0.0`。证据：`stage19_ronetc_four_dataset_comparison/smoke_runs/ustc/A-2/smoke_result.json`。
- 当前正式进程：VNAT-2026 使用物理 GPU0；USTC A-2 使用物理 GPU1，不传 `--data-parallel`。启动后两卡 live memory 均约 `36,953 MiB`；GPU2/3=`0 MiB`，GPU7 的其他用户任务未触碰。
- tmux session：`rontec_stage19_independent_single_gpu`；当前状态入口仍为 `stage19_ronetc_four_dataset_comparison/queue_status.json` 和 `progress.txt`；唯一 takeover 证据为 `takeover_independent_single_gpu_20260921.json`。
- 当前任务仍为 `in_progress`。安全下一步：只读监控两个 PID、epoch history、tmux exit status 与 GPU ownership；任一失败不得自动重试，必须先保存失败目录与日志。

### Stage 19 USTC three-GPU correction

- 状态：`running / USTC DataParallel on exactly 3 physical GPUs`。
- 用户随后明确要求剩余 USTC A-2 使用三卡并行。单卡 USTC 在 `7/100` 轮干净停止，完整保存在 `stage19_ronetc_four_dataset_comparison/interrupted_attempts/single_gpu_to_three_gpu_20260921/ustc/A-2/`，并标记为非正式结果；checkpoint 不含 optimizer/scheduler state，因此三卡 run 从 epoch 1 重启。
- VNAT-2026 已有 `PASS` 的 100/100 正式结果，监督脚本新增 `--reuse-completed-vnat`，只复用完成标记和原日志，不重新启动 VNAT。
- GPU 拓扑核验：物理 GPU0/1/2 位于同一 NUMA 侧，启动前均约 `45,469 MiB` free、0% util；GPU selector 以 `--allowed 0,1,2 --count 3 --min-free-gb 38 --max-utilization 10` 精确选择三卡。
- 首次三卡启动训练路径正常，但状态文件把完成的 VNAT 历史 GPU 计入活动卡数，显示4而实际为3；在 epoch 1 前停止并保存在 `interrupted_attempts/three_gpu_status_metadata_fix_20260921/`，修正后重新启动。
- 当前正式 session=`rontec_stage19_ustc_three_gpu_final`；训练 PID=`1828122`；物理 GPU0/1/2 live memory 约 `12,919/12,899/12,585 MiB`，均有活动负载；`queue_status.json` 断言 `active_physical_gpus=3`、`ustc_data_parallel=true`、`physical_gpu=0,1,2` 全部 PASS，启动日志无 OOM/Traceback/RuntimeError。
- 当前任务仍为 `in_progress`。安全下一步：只读等待首轮 `history.csv` 生成，用实际三卡 epoch 时间更新 ETA；不得把两个中断尝试纳入正式结果。

### Stage 19 user-requested stop and completed-protocol comparison

- 状态：`complete / ABORTED_BY_USER_WITH_FOUR_COMPLETED_PROTOCOLS`。
- 用户要求释放当前训练进程，然后仅比较已完成且协议对应的数据集结果。
- 终止前快照：Stage 19 总进度 `404/500`；ISCXTor/ISCX-VPN/VNAT-2025/VNAT-2026 均为 `100/100 SUCCESS`；USTC A-2 为 `4/100 RUNNING`。
- 目标会话=`rontec_stage19_ustc_three_gpu_final`，目标训练 PID=`1828122`，物理 GPU=`0,1,2`。GPU7 上其他用户训练 PID=`1363550` 不在本次范围内，不得触碰。
- USTC 已完成四轮 history 和 best checkpoint 将作为中断证据保留，不得当作正式 Test 结果。
- 执行结果：项目本地 `STOP_QUEUE` 已完成温和终止；USTC 最终停在 `5/100`，训练子进程返回码 `-15`，未运行 Known Test/Unknown Test，已写入 `runs/ustc/A-2/INTERRUPTED.json` 并排除于正式比较。
- 释放验证：Stage 19 目标进程组和训练脚本匹配数均为 0；物理 GPU0/1/2 均为 `0 MiB, 0%`。GPU7 的非目标训练 PID `1363550` 仍在运行，未受影响。
- 最终 live 快照中 GPU3 于 Stage 19 停止后被另一个 AcMAS 服务 PID `2836757` 使用，其 cwd 在另一项目；它与 GPU7 训练都不属于本任务，均未触碰。
- 中断证据：USTC epoch5 Validation Accuracy/Macro-F1=`0.977083/0.979953`，checkpoint/history SHA256=`d2332dfc...d6e82/f9bca161...4c1ee`；只作中断证据，不作正式 Test 结果。
- 对应协议比较：四个 `100/100 SUCCESS` 协议上，RoNeTC/H1 非加权平均 AUROC=`0.674164/0.689248`，AUPRC=`0.816834/0.821828`，UFAR=`0.886431/0.884747`，Known FRR=`0.055273/0.049348`；AUROC 单协议胜场为 `2/2`。
- 证据入口：`stage19_ronetc_four_dataset_comparison/completed_protocol_comparison.md`、`completed_protocol_comparison.json`、`RESULTS.md`和 `manifest.json`。
- 安全下一步：当前无活动 Stage 19 训练。只有用户明确要求完成四数据集比较时，才从 epoch 1 重新启动 USTC；不得将当前部分 checkpoint 当作正式结果。

## 30. Stage 20 — Dual Coarse Service Task Switch（2026-09-21）

- 状态：`in_progress / protocol freeze`。
- 用户决策：后续 ISCX-VPN 与 ISCXTor2016 主任务由细粒度 Application 分类切换为粗粒度 Service 分类；历史 Stage 12 细粒度协议、模型和结果只降为诊断证据，不删除、不覆盖，也不与新任务作同任务数值比较。
- 已核实历史依据：ISCX-VPN Service-6 的 DQ-3F corrected Open-Detect 三 seed 闭集 Accuracy=`0.890547±0.041458`、Macro-F1=`0.856050±0.064200`，说明标签粒度是显著因素；Stage 16S 已完成 VPN 六类 Service LOSO，但标签为 `WEAK_CAPTURE_LABEL`，P2P 仅一个 capture。
- 新协议目标：VPN 使用 `Chat/Email/File-Transfer/P2P/Streaming/VoIP` 六类；TOR 使用 `Browsing/Chat/Email/File-Transfer/P2P/Streaming/VoIP` 七类，其中官方 `Audio/Video` 归并为 `Streaming`。每个数据集先冻结 Service-level closed split，再为每个 Service 构建 LOSO Unknown-Free 协议。
- 数据复用边界：只复用 Stage 12 已生成的 32×32 流图像与逐流 provenance，按 capture 的官方 category 重标注；不重新解析原始 PCAP，不把旧 Fine Unknown application 直接改名为 Service Unknown。
- 输出目录：`stage20_dual_coarse_service_protocol/`；实验 ID=`stage20-dual-coarse-service-protocol-20260921-v1`。
- 安全下一步：完成 deterministic manifest、flow/image 跨角色隔离审计、独立 verifier 和实验包校验；本阶段不启动 GPU 训练。

### Stage 20 terminal record

- 状态：`complete / PROTOCOL_FROZEN_NOT_TRAINED`。
- 冻结数据：VPN `10,955` flows、6 Services；TOR `11,181` flows、7 Services；总计 `22,136` flows。TOR 官方 `Audio/Video` 归并为 `Streaming`。
- 协议：VPN 6 + TOR 7，共 `13` 个 LOSO；manifest `143,997` 行。每个 held-out Service 仅进入 `unknown_test`，Unknown training/validation/support/normalization/threshold 使用量均为 `0`。
- 完整性：逐协议 flow-ID 跨角色交集=`0`、exact-image 跨角色交集=`0`，独立 verifier `PASS`。闭集 Service split 也按 exact-image group 冻结。
- 失败保留：首轮发现一个 VPN 32×32 image hash 同时属于 File-Transfer 与 VoIP，使两个 LOSO 出现 Known/Unknown 图像交叉；Gate 正确失败。首轮清单、审计和日志已保留，成功版在抽样前整体排除该冲突图像组（3 rows），未放宽检查。
- 科学边界：Service 标签仍为 `WEAK_CAPTURE_LABEL`；并非统一 capture-disjoint，VPN P2P 只有一个 capture；粗粒度分数不能冒充细粒度同任务涨点。
- GPU/训练：本阶段 GPU 使用=`0`，新模型训练=`0`。输出：`stage20_dual_coarse_service_protocol/`。
- 安全下一步：若继续实验，在不改变 membership 的前提下训练选定模型，并同时报告闭集 Accuracy/Macro-F1 与开集 AUROC/AUPRC/UFAR/Known-FRR。

## 31. Stage 21 — OURS-E3-T8 Coarse Service Closed-Set Benchmark（2026-09-22）

- 状态：`in_progress / implementation and cache parity`。
- 用户要求：在 Stage 20 冻结的粗粒度任务上用 `unknown_traffic_project` 的方法重新训练和测试，不得把 corrected Open-Detect 基线冒充为 OURS。
- 方法身份：采用项目 Stage 17 已恢复并获得 6/6 pilot 正向信号的 E3 路线，即 TrafficFormer 分支 + FIG/TAGCN 分支、Known-Train per-branch z-score、768+128 concat、线性分类头；本阶段闭集主指标来自 E3，不使用 Open-Detect 网络、原型或 loss。
- 本地适配：Stage 20 输入由首 8 包冻结，因此 TrafficFormer 使用前 5 包，FIG 使用前 8 包，方法名固定为 `OURS-E3-T8`；这是透明的任务适配，不冒充 Stage 17 的 30 包 E3。
- 正式设计：ISCX-VPN Service-6 与 ISCXTor2016 Service-7，各 seeds `2022/2023/2024`，共 6 个闭集 run；checkpoint 只按 Known Validation Macro-F1 选择，Test 仅最终评价。
- 输出目录：`stage21_coarse_service_ours_e3_benchmark/`；历史 Stage 12/16/17/20 与 TrafficFormer 结果只读。
- 安全下一步：构建并审计 Stage20 exact membership 的 E3-T8 输入缓存，完成单 run smoke 和 GPU 容量核验后，最多三张自动选择的空闲 GPU 并行正式训练。

### Stage 21 terminal record

- 状态：`complete / CLOSED_SET_DIAGNOSTIC_COMPLETE`。
- 用户最终将范围调整为双卡、seeds `2022/2023`；VPN/TOR 共 `4/4` 正式 run 完成。最初 GPU4 上的 seed2024 收到 SIGTERM 后释放，日志和 `INTERRUPTED.md` 保留，正式聚合明确排除。
- 实际训练为一进程一卡：seed2022/2023 分别使用物理 GPU1/2，不使用 DataParallel；每 run 约 `367.18--378.62 s`，峰值 allocated memory=`8.802 GiB`。
- E3 主结果：VPN Accuracy/Macro-F1/Weighted-F1=`0.520128/0.524935/0.512889`；TOR=`0.622202/0.591124/0.617643`。VPN/TOR Macro-F1 seed 标准差=`0.047917/0.014553`。
- 多方位核验：同时保存 Accuracy、Balanced Accuracy、Macro/Weighted P/R/F1、MCC、Kappa、最差类别、逐类 P/R/F1/support、混淆矩阵、seed 稳定性、验证—测试差距和 E3−E1/E2 配对增益。E3 相对 E2 Macro-F1 mean delta 为 VPN `+0.114874`、TOR `+0.212758`，四个 paired run 全为正。
- 主要短板：VPN File-Transfer mean recall=`0.2925`；TOR Chat mean recall/F1=`0.1250/0.174656`。E1 严格随机初始化明显欠拟合，不能把 E3 的融合收益解读为 TrafficFormer 单分支已恢复到高性能。
- 完整性：Stage20 membership、缓存 finite/nonempty、4 个 SUCCESS、run artifact hashes、validation-only best epoch 和逐样本 Test 指标重放共 `98/98` checks PASS；Test 选模样本=`0`。
- 证据：`stage21_coarse_service_ours_e3_benchmark/RESULTS.md`、`multiview_summary.json`、`multiview_confusion_matrices.csv`、`completion_verification.json`、`manifest.json`。
- 安全下一步：若要提升分数，另开实验仅用 Known Train/Validation 调整 TrafficFormer 训练预算/预训练策略，并比较 FIG `T8/T16/T30`；不得覆盖本 Stage21 baseline。

## 32. 精简仓库重新发布（2026-09-23）

- 状态：`complete`。
- 目标：将代码、配置、测试、项目说明、各阶段 `RESULTS.md`、`manifest.json` 和小型核心结果重新整理并推送到 GitHub；模型权重、数据、embedding、缓存、运行日志和大体积中间产物继续保留在本地，不进入普通 Git。
- 起始状态：工作树实际占用约 `54 GiB`、逻辑大小约 `65 GiB`；已跟踪文件约 `0.81 MiB`；现有忽略规则之外仍有约 `3.25 GiB` 未跟踪文件，其中包含大于 GitHub 普通 Git 单文件限制的 Parquet/NPZ 产物。
- 科学状态：Stage 21 已完成并通过 `98/98` checks；Stage 22 正在 `stage22-pretrained-e3-grid-20260923` 会话中运行，当前不得记录为完成，也不得提交其动态 checkpoint、run 目录或日志。
- 既有脏工作树：`README.md`、`EXPERIMENT_RESULTS.md`、`configs/dataset/ustc_tfc2016.yaml` 和研究计划文档已有未提交修改；本次保留这些修改，不回滚。
- 当前动作：补齐忽略规则和公开仓库说明，生成稳定的当前状态文档，按白名单暂存可复现源码与轻量证据，然后审计 staged 文件大小、敏感信息、大文件和 Git 历史后再提交、推送。
- 发布白名单：源码、配置、测试、Markdown/Word 文档、稳定 JSON/manifest/hash，以及小于 2 MiB 的命名汇总 CSV；Stage 22 仅含方法、配置、预检和 `running` 状态，不含动态 run/checkpoint/log。
- 提交前核验：暂存 `1,382` 个文件、`15.24 MiB`，最大文件小于 `1 MiB`；禁止的模型/数组/数据扩展和 `runs/artifacts/checkpoints/cache` 路径均未暂存；敏感模式扫描 PASS；`410` 个新增 Python 文件语法编译 PASS；`532` 个 JSON 解析 PASS；核心 Markdown 相对链接 PASS；USTC YAML 解析 PASS。
- 已知格式边界：全量 `git diff --check` 会报告第三方 UER 源码、历史 CSV 的 CRLF/尾随空白及 Markdown hard-break；这些是原始历史内容，不在本次批量改写，新增的核心发布文档单独检查通过。
- 发布提交链包含 `9422136 Publish reproducible project snapshot through Stage 21` 及其后续鉴权、交接和完成状态记录；完整审计轨迹保留在 Git 历史中。
- 推送状态：已完成。HTTPS push 实际停在 GitHub 用户名提示，SSH 22 端口随后连接超时；复核历史上下文并确认 SSH 身份有效后，改用 GitHub SSH 443 端口执行非强制 push。远端 `main` 已由 `25cd43c` 更新到 `9a4a17e`；最终收尾提交及远端 SHA 核验见本节后续提交历史。
- 当前阻塞：无。GitHub SSH 22 端口在本机网络超时，但官方 SSH 443 通道可用且已完成发布。
- 发布边界：Stage 22 后续生成的最终指标文件仍留在本地工作树且未纳入本次提交；本次远端保持先前约定的 Stage 22 `running` 快照，不含动态 checkpoint、run 目录或日志，也未使用 force push。

## 33. Stage 22 完成结果补充发布（2026-09-23）

- 状态：`in_progress`。
- 目标：将 Stage 22 已完成的核心结果、机器可读清单、项目索引和当前状态补充到 GitHub；继续排除 checkpoint、representation、predictions、运行日志和缓存。
- 起始状态：Stage 22 `4/4` formal runs 已完成，自带核验=`99/99 PASS`；`RESULTS.md` 与核心汇总文件已生成，但根级 README、`CURRENT_PROGRESS.md`、`EXPERIMENT_RESULTS.md` 和本交接文档仍记录为 `running`。
- 已执行：核对四个 JSON 可解析；运行 Stage 22 自带 `verify_completion.py` 得到 `PASS / 99 checks / 0 failures`；发现通用实验包校验因 `manifest.json.artifacts=[]` 失败，随后按保存规范刷新为 80 个 bundle artifact 条目并保留本地证据哈希。
- 科学结论：VPN E1/E3 Macro-F1=`0.860607±0.003961/0.859493±0.007015`，TOR=`0.821259±0.006376/0.822178±0.007643`；E3−E1 mean delta=`-0.001115/+0.000918`，每个数据集均仅 `1/2` positive seeds，不能宣称 E3 稳定优于 TrafficFormer。
- 完成证据：通用 bundle 校验为 `status=success, artifacts=80, bundle_files=80`；公开提交仅含 9 个文件，禁止路径/扩展名、敏感模式、JSON 解析和 diff 检查均 PASS；提交 `f33ae3354243511c9f6753453b17b7d838d9a629` 已通过 GitHub SSH 443 非强制推送，首次远端 SHA 核对一致。
- 发布边界：checkpoint、representation、predictions、运行日志、缓存和其他大型产物继续保留在本地且未提交；普通 Git 仅新增轻量汇总、完成性核验、保护资产哈希、manifest 与同步说明。

## 34. Stage 23 同样本闭集方法横向表（2026-09-23）

- 状态：in_progress；本条为当前任务交接，不改变此前 Stage 20–22 结果。
- 目标：以 Stage 20 冻结的 ISCX-VPN Service-6（10,955 flows）和 ISCXTor Service-7（11,181 flows）为唯一闭集样本集合，逐 flow-ID 比较具备多类分类输出的方法。UnDiff 的 one-class 异常检测不冒充多类闭集模型。
- 已核对的起点：Stage 20 的 exact-image-group 8:1:1 manifest 存在；Stage 22 的预训练 TrafficFormer E1 与 E3 在相同样本、seeds 2022/2023 上完成四个 run，且保存逐样本预测和 checkpoint。Stage 21 的两套特征缓存覆盖 Stage 20 全部 flow，stage20_membership_missing=0。此前跨项目复现结果来自不同样本/标签/单位，不能直接入正式配对表。
- 已执行：通过固定 Conda 环境与命名 tmux 会话，创建独立 stage23_closed_set_method_table/ 实验包；未启动新模型训练，未读取 Unknown Test。
- 保护边界：Stage 20 manifest、Stage 21/22 资产和其他方法项目只读；新适配器、日志及结果仅写入 Stage 23。所有新模型先通过 flow-ID/标签/角色覆盖核查，checkpoint 只用 Known Validation 选择，Known Test 只做最终评估。
- 阶段里程碑：样本级 parity/输入覆盖预检 120/120 PASS；Stage 22 E1/E3 八条 dataset×seed×method 行已逐样本重放。Open-Detect corrected-paper 的 VPN-2022、TOR-2022、VPN-2023 在物理 GPU0/1/2 各启动一个正式 run；TOR-2023 未启动。三 run 均通过 Known Train/Val 输入哈希核查，加载 Known Test feature values=0；新方法最终 Test 指标仍未产生。
- 已新增：stage23_closed_set_method_table/PROTOCOL.md、scripts/build_preflight.py、scripts/train_opendetect_closed.py、scripts/show_progress.py、四个预检/表格 CSV/JSON 文件、RESULTS.md 与 manifest.json；项目级 EXPERIMENT_RESULTS.md 追加 in-progress 索引。全部新文件只在当前主项目内。
- 下一步：一个 GPU slot 释放后启动 TOR-2023；随后验证四个 run 的 checkpoint、逐样本 Test 重放、输入/输出 hash，再更新正式闭集表。不要把别的方法历史高分直接并入配对表。

### Current handoff (2026-09-23)

Stage 23 in_progress。安全续作入口：stage23_closed_set_method_table/RESULTS.md、scripts/show_progress.py 与本节。当前有三个 Stage 23 Open-Detect GPU 训练进程，session 为 codex_stage23_od_vpn2022_20260923、codex_stage23_od_tor2022_20260923、codex_stage23_od_vpn2023_20260923；第四个 TOR-2023 待 GPU slot。不要将预检或旧项目高分误写为八方法同样本比较完成。

### Stage 23 formal plan handoff (2026-09-23)

- 计划名称：Stage 23 — 冻结流样本的多方法闭集公平对照实验（Frozen-Flow Matched Closed-Set Traffic Benchmark）。详细计划已落在 `stage23_closed_set_method_table/EXPERIMENT_PLAN.md`；本次只写计划，未启动或停止训练，也未修改 Stage 20–22。
- 计划落地时的只读进度快照：Open-Detect VPN-2022、VPN-2023、Tor-2022 在进度脚本中为 `SUCCESS`，Tor-2023 为 `NOT_STARTED`；前三项尚待 checkpoint/逐样本指标独立复核。上段“三个进程”的说法是旧快照，不代表当前仍在运行。
- 用户当前许可最多使用六张本用户可用物理 GPU，并要求 2026-09-24 08:00 北京时间前释放至少四张；任何新训练前均重新核查实际 GPU/进程与截止安排。
# 2026-09-23 — Stage 23 live continuation (17:05 UTC)

Selected project: `Projects/unknown_traffic_project`. Stage20 closed-service
manifest SHA256 remains
`6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb`.
The detailed frozen-flow comparison plan is
`stage23_closed_set_method_table/EXPERIMENT_PLAN.md`.

- Stage23 Open-Detect corrected-paper: 4/4 complete; independent replay PASS
  (40,025 checks; 12 total baseline+OD method rows).
- RoNeTC: exact-flow input cache 22,136/22,136 after recovering 137 Tor views;
  four 100-epoch runs live on physical GPUs 0–3, roughly epoch 60.
- YaTC: author-rule MFR Known Train/Val cache PASS on both datasets; two Tor
  200-epoch formal runs live on GPUs 4–5, roughly epoch 170–180.
- YaTC Known Test MFR is deliberately deferred until checkpoint selection.
  Post-selection evaluator and independent final replay scripts are now
  implemented but cannot be declared PASS until real run outputs exist.
- Native ET-BERT strict flow drops at least 15,195 one/two-packet Stage20 flows;
  native TFE-GNN TCP graph cannot cover at least 8,316 UDP Train/Val flows.
  Method admission rationale: `stage23_closed_set_method_table/METHOD_ADMISSION_AUDIT.md`.
- User allowed at most six physical GPUs and requested four released by 08:00
  Asia/Shanghai next morning. Exact RoNeTC-session guard is active for 07:55
  Asia/Shanghai, leaving YaTC GPUs 4–5 untouched. Never infer completion from
  this handoff: inspect live tmux status/logs and per-run SUCCESS/FAILURE.

Progress command (from workspace root, inside the mandatory named tmux helper):
`python -B stage23_closed_set_method_table/scripts/show_progress.py`.
# 2026-09-23 — Stage 23 terminal handoff (17:35 UTC)

Stage 23 is complete. Five methods × ISCX-VPN/ISCXTor × seeds 2022/2023
produce 20/20 flow-ID matched Known Test rows; final independent replay
checks 22,441/22,441 PASS. Stage20 SHA256 is unchanged
(6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb).
Authoritative human report:
stage23_closed_set_method_table/STAGE23_FINAL_REPORT.md.
Machine results: stage23_full_run_results.csv, stage23_full_summary.csv,
stage23_paired_vs_e1_e3.csv, stage23_all_method_per_class.csv,
stage23_all_method_confusion.csv and completion_verification.json.
Bundle summary/index: stage23_closed_set_method_table/RESULTS.md and
EXPERIMENT_RESULTS.md. All new model checkpoints, predictions and failure
evidence are preserved. No Unknown Detection or Test-guided selection ran.
All Stage 23 GPU training and evaluation jobs finished; GPU0–3 were released
before the requested 08:00 Asia/Shanghai deadline. The deadline guard was
closed after the sessions had naturally completed.

# 2026-09-24 — Stage 23B same-flow adapted TFE-GNN / Trident supplement

Status: in_progress. The user clarified that TFE-GNN and Trident must be run
again on the exact Stage20 flow IDs, not compared by borrowing old results.
Stage23's original five-method table is preserved. New evidence bundle:
stage23b_tfe_trident_same_flow_adaptation/ (approximate/adapted claim scope).
Stage20 manifest SHA256 remains
6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb.
Initial source audit confirms TFE native TCP/payload filtering excludes UDP and
empty-payload flows, while the current Trident 86-D USTC cache is from another
population/full-flow observation. Therefore any all-flow Stage20 run must be
explicitly named TFE-8 UDP/short-flow adaptation and Trident early-8 86-D
adaptation, respectively. No new score is yet available. Next: freeze exact
input rules, construct and audit all-flow inputs from frozen packet references,
then train on Known Train/Val and evaluate Known Test after selection only.

Milestone 2026-09-24 02:39 UTC: Stage23B protocol and scripts are in the new
bundle. Exact first-eight packet-ref extraction passed Known Train/Val for
VPN 9,862/9,862 and Tor 10,064/10,064 flows. Trident four Known-Val selections
completed (VPN 2022/2023 Macro-F1 0.633974/0.626922; Tor
0.619967/0.609786). Three TFE jobs run on live-selected physical GPUs 0/1/2;
the fourth Tor-2023 is held by a fail-closed project-local queue until a slot
frees. Queue session: codex_stage23b_completion_queue_20260924. It gates Test
materialization on all four TFE and four Trident selections, then evaluates
and independently replays eight adapted rows. Current state remains
in_progress; no Known Test score yet. Progress command from this project:
python -B stage23b_tfe_trident_same_flow_adaptation/scripts/show_progress.py
(or read stage23b_tfe_trident_same_flow_adaptation/progress.json).

Milestone 2026-09-24 03:35 UTC: Trident Known-Val selection remains 4/4;
TFE VPN-2022/2023 completed their frozen 20 epochs and selected checkpoints.
Tor-2022/2023 are both running their frozen 100-epoch schedules (last progress
32/100 and 10/100). No Known Test input has been materialized. The queue
continues autonomously under the three-GPU ceiling. Its final replay now
checks the original Stage23 results SHA256 and reports TCP/UDP plus packet-count
subgroups without changing selection. A second named tmux task,
codex_stage23b_postverify_20260924, waits for QUEUE_COMPLETE, then verifies
both protected hashes and the preservation bundle. These pending jobs must
not be reported as completed until their actual markers and exit statuses pass.

Terminal update 2026-09-24 05:20 UTC: Stage23B is complete. TFE-GNN and
Trident adapted runs are 8/8, with 8,904 independent checks passing. Known
Test predictions cover 8,840 records; Unknown Test usage is false. Stage20
manifest SHA256 remains
6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb; original
Stage23 baseline SHA256 remains
b10cbf5ccead40312768301c5379027ce983f4bcb8ec3cb31338f5b80ced0bdf. The
experiment bundle validator passes for 178 artifacts. The complete 7-method
comparison and 40 paired candidate-baseline rows were added to the Stage23B
bundle. Macro-F1 mean ±
population std: VPN TFE 0.6585±0.0146, Trident 0.6322±0.0006; Tor TFE
0.5536±0.0138, Trident 0.6053±0.0019. Against paired pretrained Stage23
baselines, adapted methods are lower by 0.20–0.27 Macro-F1 on average; see
the Stage23B paired CSVs and report. This result does not establish native
paper reproduction: TFE uses an 8-packet all-protocol graph adaptation and
Trident uses early-8 reconstructed 86-D features.

Documentation handoff 2026-09-24: the current Stage 23/23B closed-set task
has a project-root record at CURRENT_CLOSED_SET_BENCHMARK_2026-09-24.md. It
summarizes all seven methods on the two frozen Service tasks, marks Stage23B
TFE-GNN/Trident as adaptations, and links the 28 run rows, 40 paired rows,
182 per-class rows and original verification evidence. EXPERIMENT_RESULTS.md
now points to this record without changing the frozen result CSVs.

# 2026-09-24 — Stage 24 dual-branch structural study

Status: `in_progress`. User authorized sustained execution of Stage 24 — 双分支流内结构增强与跨流上下文可行性实验. The independent bundle is `stage24_intraflow_structure_context_feasibility/`; the pre-result protocol and gates are in its `EXPERIMENT_PLAN.md`. Starting evidence: Stage20 has 22,136 frozen VPN/Tor Service flows, Stage22 has four 2022/2023 E1/E3 runs, and earlier E3 did not stably improve Known Test Macro-F1 over E1. Existing uncommitted Stage23/23B assets and project records are preserved. No Stage24 model result is available yet. Next: perform protected-hash/data-role/packet-count audit, then run Known-Train/Validation-only 24A pilot and read-only 24B context feasibility audit. Stage20–23 must remain unchanged.

Terminal update 2026-09-24: Stage 24 pilot and feasibility audit completed. Sixteen of sixteen S1/G1/G2/G2-shuffle runs finished with validated per-run evidence on Known Train/Validation only; queue used at most GPUs 0/1/2. VPN mean Macro-F1 E1/E3/S1/G1/G2=`0.849134/0.851877/0.850469/0.850401/0.849936`; Tor=`0.850669/0.844107/0.846332/0.846216/0.846903`. No candidate passed both-baseline, both-dataset pilot progression. 24B is not feasible for a full-class capture-disjoint claim because VPN P2P has only one capture, and no cross-flow model was trained. Nineteen protected Stage20–22 assets are unchanged by SHA256; Known Test and Unknown Test feature use=`0/0`. Stop at gates: no five-seed confirmation or open-set evaluation. The two-seed progression rule was amended after one negative S1 result was visible and is explicitly disclosed in `DECISION_RULE_AMENDMENT.md`. Full evidence: `stage24_intraflow_structure_context_feasibility/RESULTS.md` and `stage24_report.md`.

# 2026-09-24 — Stage 25 semantic-coarse closed-set development

Status: `in_progress`. User requested sequential performance experiments, prioritizing a coarser label granularity while preserving original model capability. The independent bundle is `stage25_semantic_coarse_closed_set_development/`; its pre-result plan fixes `Chat+Email+VoIP→Communication`, otherwise unchanged Services, full-flow primary and ≥2-packet conditional secondary scopes. Starting evidence: Stage20 has 6/7 original Services and immutable roles; Stage22 has E1/E3 Train/Val/Test representations; Stage23/23B provide seven same-flow prediction sets. Execute no-training label remap first, then frozen-encoder coarse heads, and retain the fine-class result as a capability guardrail. Stage20–24 assets and other users' work remain untouched.

Stage 25 milestone 2026-09-24: phase A completed (`codex_stage25_phase_a_retry1_20260924`, exit 0); 61,964 aligned validation/test prediction records, 224 metric rows, 48 protected source files, 28/28 historical Known Test fine metrics replayed. Hard-remapping our E3 from Fine to Coarse changes VPN Macro-F1 `0.859493→0.896004`, Tor `0.822178→0.852783`; YaTC similarly changes `0.887085→0.923689` and `0.827393→0.876259`. This is label-task simplification, not an improved encoder. First phase-A attempt failed because TFE stores validation predictions by selected epoch; its failure log is preserved and the retry read the already-frozen selection.json rather than reselecting. Next: fixed Known-Train-only frozen-encoder coarse heads, then conditional-scope report.

Stage 25 terminal update 2026-09-24: status `complete`. Phase B fit 8/8 additive E1/E3 coarse heads on Known Train only (no warnings) and preserved original fine checkpoints; phase C reported the pre-fixed ≥2-packet condition without retraining. Full-flow E3 coarse-head Macro-F1 is VPN/Tor `0.897301/0.854317`; YaTC hard-remap remains higher at `0.923689/0.876259`. The ≥2-packet E3 hard-remap Macro-F1 is `0.878272/0.880108`, retaining 807/1,093 VPN and 595/1,117 Tor Known Test flows; filtering hurts VPN and helps only the conditional Tor population. Independent replay passed 61,964 remap predictions, 224 metric rows, 8 heads and 17,704 head decisions; 48/48 source SHA256 unchanged, Unknown use=0 and Test fit/selection=0. No new encoder or open-set experiment was started. Full tables, limits and manifests are in `stage25_semantic_coarse_closed_set_development/`.

# 2026-09-24 — Stage 26 frozen YaTC + E3 score fusion

Status: `in_progress`. User requested an experimental attempt to fuse YaTC with our E3 for closed-set accuracy. `stage26_yatc_e3_frozen_score_fusion/EXPERIMENT_PLAN.md` fixes a single 0.5/0.5 probability average, exact same Stage20 flows and original/fixed-coarse label reporting, with no training or weight search. Existing YaTC/E3 CSVs contain hard predictions only, so frozen checkpoints must be replayed for probabilities; exact Validation/Test argmax parity is mandatory. The previously exposed Stage20 Test limits claims to development. Next: implement extraction/parity, run four dataset×seed pairs sequentially on a live-selected GPU, independently verify metrics and hashes, then stop.

Stage 26 terminal update 2026-09-24: status `complete / FUSION_NOT_CONFIRMED`. Four matched frozen checkpoint pairs completed with exact historical Known Validation/Test argmax parity and unchanged source hashes; no training, Unknown use or weight search. Full-flow coarse Macro-F1 E3/YaTC/Fusion: VPN `0.895838/0.923689/0.906884`; Tor `0.852381/0.875886/0.877392`. Fusion beats YaTC in only 1/4 coarse dataset×seed cells, so it does not satisfy the preregistered improvement gate. Independent verifier replayed 26,556 decisions and 96 metric rows, with 60 source-hash entries unchanged. The Stage20 Test has been exposed, so these are development results, not untouched external confirmation. Detailed evidence: `stage26_yatc_e3_frozen_score_fusion/RESULTS.md`. No follow-on weight tuning was started.

# 2026-09-25 — Stage 27 corrected YaTC feature-level fusion

Status: `in_progress`. The user clarified that the requested YaTC fusion is at the representation layer, not Stage 26's output-probability average. Read-only audit confirmed YaTC's 192-D `forward_features` interface and exact E3/YaTC flow-ID alignment across all 12 dataset×seed×Known-role cells. The independent `stage27_yatc_feature_level_fusion/` bundle and pre-fit `EXPERIMENT_PLAN.md` fix three feature-level variants (linear concat, equal projected fusion, learned per-flow feature gate), with frozen E3/YaTC encoders and Known-Train-only head fitting. Stage20 Test is previously exposed and may only yield development evidence. Stage20–26 assets and sibling YaTC source remain read-only. Next: implement verified feature extraction, train four dataset×seed pairs, independently replay metrics/hashes, and preserve complete results without post-Test design changes.

Stage 27 terminal update 2026-09-25: status `complete / FEATURE_LEVEL_DIAGNOSTIC`. Four frozen E3+YaTC feature-level runs finished; 120/120 metrics independently replayed, all source/checkpoint hashes passed, Unknown usage and encoder updates=0. Fine Macro-F1 VPN E3/YaTC/F1/F2/F3=0.859493/0.887085/0.888900/0.884258/0.880846; Tor=0.822178/0.827393/0.837946/0.842260/0.821191. F3 often collapsed toward TrafficFormer. Stage20 Test was previously exposed; results are development only, not independent validation. Stage26 probability averaging remains separate. Full evidence: `stage27_yatc_feature_level_fusion/RESULTS.md`.

# 2026-09-25 — ER-CMJI paper-grounded Stage 28 fusion plan correction

Status: `in_progress`. The user identified that the previous Stage 28 proposal addressed softmax-gate collapse but did not incorporate the dynamic fusion mechanism from the previously supplied ER-CMJI paper. The source PDF in the sibling `Projects/ER-CMJI(wangyanbin)/paper/` directory is read-only; extraction and any notes remain in this selected project. Starting state: Stage 27 is complete and diagnostic only, with previously exposed Stage20 Test. Next: extract and inspect the paper's actual fusion equations, optimization and ablations, distinguish faithful elements from traffic-specific adaptation, then deliver a corrected controlled experiment plan without training or changing Stage20–27 assets.

Terminal update 2026-09-25: status `complete / PAPER_GROUNDED_PLAN_ONLY`. The sibling PDF `paper/Wang 等 - 2026 - （wangyanbin生成式）Entropy-regulated cross-modal generative fusion for multimodal network intrusion dete.pdf` was read-only; SHA256=`770d501607f075084741d8178c4355ac6b7ea2a1f8d9abd1601a5dbac56bf060`. The workspace PDF wrapper extracted all 16 pages to this project's `.artifacts/pdf/er_cmji_2026.txt`; pages 7–8 were rendered locally with PyMuPDF after `pdftoppm` was unavailable. Equations 12–16, Gaussian entropy, bidirectional generation, KL/classification losses, Tables 3/6/7, and limitations were checked. The paper's per-sample prose conflicts with its batch-mean Eq. 13, and its reported mixing coefficients/weight direction are not copied blindly. The corrected `STAGE28_ER_CMGI_INSPIRED_DYNAMIC_FUSION_PLAN.md` maps existing frozen byte-semantic and FIG behavior views to a clearly labeled probabilistic-adapter adaptation, specifies 2×2 entropy/consistency controls and Known-only selection, and preserves Stage20–27. `git diff --check` and nonempty PDF extraction/plan checks passed. No experiment or GPU workload was run. Safe next step, only on a future execution request: preregister source hashes and implement the Stage28 Known-Train/Validation-only preflight in an independent bundle.

# 2026-09-25 — Stage 28 paper-guided entropy fusion execution

Status: `in_progress`. User authorized execution of the already frozen `STAGE28_ER_CMGI_INSPIRED_DYNAMIC_FUSION_PLAN.md`. New independent bundle `stage28_er_cmgi_entropy_fusion/` was initialized with experiment ID `stage28-er-cmgi-entropy-fusion-20260925-v1`; no source artifacts were overwritten. The selected project remains `Projects/unknown_traffic_project`. Starting evidence: Stage27 matched E3 896-D and YaTC 192-D features, Stage20 Known roles and labels, previously exposed Test; Stage28 must perform Known-Train/Validation-only development before any retrospective Test. Next: hash/flow/label preflight, implement the shared probabilistic adapters and four fixed fusion controls, smoke test, then launch bounded GPU runs with project-local evidence.

Stage 28 execution milestone 2026-09-25: preflight PASS for all four frozen encoder pairs and eight Known-role arrays; Stage20 manifest SHA256 matches `6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb`. P0–P3 plus same-flow YaTC/F1/F2 pilot `iscx_vpn/encoder2022_train2022` completed, seven saved-logit metrics and six model checkpoints independently replayed, and its experiment bundle validated. Pilot Known-Val Macro-F1 P0/P1/P2/P3=`0.872110/0.872110/0.871210/0.871210`, YaTC=`0.892950`; P3 forced-equal=`0.871210`. This is a negative single-unit observation, not the 20-unit gate. The remaining 19 units are running in named tmux queue `codex_stage28_queue2_20260925`, bounded to GPUs 0/1/2 with live capacity selection per run; progress is `stage28_er_cmgi_entropy_fusion/queue_progress.json`. Known Test and Unknown feature usage remain zero. A failed initial queue launch due to `queue.py` shadowing Python's standard-library module was preserved in `.tmux-task/codex_stage28_queue_20260925/`; renamed `run_queue.py` and launched successfully. Next: finish queue, checkpoint replay, aggregate preregistered Known-Val gate, preserve and verify all evidence, stop without Stage28B or Test tuning.

Stage 28 terminal update 2026-09-25: status `complete / PREREGISTERED_KNOWN_VAL_GATE_FAIL`. All 20/20 units trained; each retained A0/A1 and P0–P3 checkpoints, full Known-Val metrics/predictions, entropy weights, curves, counterfactuals and same-flow YaTC/F1/F2 baselines. Twenty run bundles passed saved-logit replay and 120 independent CPU checkpoint replays; the root bundle validated 674 artifacts. Frozen Stage20/22/23/27 input hashes remained unchanged; Known Test and Unknown feature usage=`0/0`. P3 Mean Macro-F1 VPN/Tor=`0.885803/0.871088`, YaTC=`0.889929/0.864355`, P2 equal=`0.885742/0.870925`. P3−P2 mean=`+0.000062/+0.000163`; P3 versus its own forced-equal checkpoint mean=`+0.000382` and positive in only 7/20 units. Gate FAIL: not a stable entropy mechanism, no Stage28B diffusion or Test-based adjustment. The two-view high/low rule algebraically collapses to equal weights whenever λ_b+λ_s=1 (including the initial 0.5/0.5). Full evidence: `stage28_er_cmgi_entropy_fusion/stage28_report.md`, `RESULTS.md`, `completion_verification.json` and 20 run bundles. No follow-on stage started.

# 2026-09-25 — Stage 29 native E3-only dynamic fusion diagnostic

Status: `in_progress`. User requested the entropy-fusion idea be tried on **our model** and asked whether native fusion has three channels. Read-only code audit confirms Stage 22 E3 has two branches, TrafficFormer 768-D and FIG/TAGCN 128-D, each Known-Train standardized then concatenated into a linear head; Stage 27 E3+YaTC alone introduced a third input. New independent bundle `stage29_e3_native_entropy_fusion/` and pre-training `EXPERIMENT_PLAN.md` fix a Known-only two-branch N0 equal-weight versus N1 shared-λ entropy-fusion test on the four frozen Stage22 encoder pairs, five head seeds each. Shared λ is explicitly a Stage28 failure-driven development correction for the two-view cancellation, not the paper's literal rule. No YaTC, Test or Unknown data may be loaded. Next: frozen-source/role preflight, implement and run paired units, independently verify and apply the fixed gate; preserve Stage20–28 untouched.

Stage 29 terminal update 2026-09-25: status `complete / PREREGISTERED_KNOWN_VAL_GATE_FAIL`. Preflight PASS on four exact Stage22 frozen encoder pairs and eight Known-role arrays; an initial parity check that incorrectly compared raw branch features to standardized E3 was corrected before training, with failure log preserved. All 20 E3-only two-branch paired units completed; 40 N0/N1 checkpoints independently replayed on CPU, all run bundles and the 366-artifact root bundle validated, and Stage20/22 hashes remained unchanged. YaTC/Known Test/Unknown usage=`0/0/0`. Known-Val Macro-F1 VPN E3-original/N0/N1=`0.851877/0.837604/0.837617`; Tor=`0.844107/0.849594/0.849822`. N1−N0=`+0.000013/+0.000228`, positive in only 2/10 VPN and 4/10 Tor units; N1−forced-equal average=`+0.000441`, positive 8/20. Learned shared λ mean `0.502877`, average TrafficFormer weight `0.500582`; mechanism can vary per sample but learned nearly equal weighting. Gate FAIL, so no promotion, no Test tuning, no follow-on stage. Full evidence in `stage29_e3_native_entropy_fusion/RESULTS.md`, `stage29_report.md`, `completion_verification.json` and 20 run bundles.

# 2026-09-25 — Stage 30 true three-view entropy-fusion test

Status: `in_progress`. User authorized testing the article-inspired dynamic feature fusion on the **actual three Stage 27 views**: frozen TrafficFormer 768-D, FIG/TAGCN 128-D and YaTC 192-D. Independent bundle `stage30_three_view_entropy_fusion/` and pre-training `EXPERIMENT_PLAN.md` fix a 20-unit Known-only comparison between same-capacity equal-weight T0 and three-view entropy-weighted T1, with separate per-view learned λ and post-selection equal/shuffled counterfactuals. Stage 20/22/27 sources are read-only; no Known Test/Unknown fitting or evaluation, no encoder retraining, and no Stage 28/29 changes. Next: preflight frozen hashes/flow IDs, implement and validate a pilot, run bounded GPU queue, replay checkpoints, apply the fixed Known-Val gate, preserve complete evidence, stop.

Stage 30 terminal update 2026-09-25: status `complete / PREREGISTERED_KNOWN_VAL_GATE_FAIL`. Preflight passed exact Known Train/Validation flow-ID and label alignment for four frozen three-view encoder pairs. Pilot plus 19 remaining paired units finished on at most three live-selected physical GPUs; 20 run bundles saved full 30-epoch curves, three adapter and paired fusion checkpoints, per-class and per-flow predictions, entropy weights, counterfactuals and hashes. Forty T0/T1 checkpoints independently replayed on CPU, all unit bundles validated; frozen sources unchanged, Known Test/Unknown usage=`0/0`. T0/T1 Known-Val Macro-F1 VPN=`0.890366/0.888327`, Tor=`0.887543/0.877244`; T1−T0=`−0.002039/−0.010300`, positive only `2/10` and `0/10`. T1−forced-equal mean=`−0.003166`, positive `8/20`; preregistered Gate FAIL. No Test evaluation, formula retuning or further stage. Evidence: `stage30_three_view_entropy_fusion/RESULTS.md`, `stage30_report.md`, `completion_verification.json` and 20 run bundles.

# 2026-09-25 — Four-dataset three-view equal-fusion retest protocol audit

Status: `in_progress / AWAITING_PROTOCOL_CHOICE`. User requested retesting the selected Stage30 T0 three-view equal-fusion candidate on USTC-TFC2016, VNAT, ISCX-VPN, and ISCXTor2016. Read-only audit confirms Stage30's aligned TrafficFormer 768-D, FIG/TAGCN 128-D, and YaTC 192-D features exist for Stage20 ISCX-VPN/Tor only; `stage30_three_view_entropy_fusion/preflight.py` explicitly supports only those two datasets. USTC has an existing 20-class 391,280/48,910/48,911 split; VNAT has 15 frozen application-held-out protocols with varying Known classes and 8:1:1 Known splits. No same-flow three-view USTC/VNAT feature/checkpoint bundle was found in the project inventory. Two materially different evaluation designs are possible: reuse each dataset's frozen Known protocols, or create a new all-class closed-set task (which would require a new VNAT split). Requested user's protocol choice before training or opening Test. No encoder training, Test read, new split, or historical artifact mutation has occurred. Safe next action: upon protocol choice, freeze a separate experiment plan, audit exact per-view data paths and Unknown-Free boundary, then implement/retrain required dataset-specific encoders and the unchanged T0 head; preserve all outputs in a new independent bundle.

Stage31 milestone 2026-09-25: user chose existing frozen Known Train/Val/Test. Independent bundle `stage31_four_dataset_three_view_equal/` and pre-training `EXPERIMENT_PLAN.md` fix the five previously registered Stage15R representative protocols (VPN medium-2022, Tor medium-2022, VNAT Medium-2025/2026, USTC A-2), Stage30 T0 architecture and Known-only training/selection. Metadata preflight completed: Known Train/Val/Test counts respectively VPN 12,910/1,611/1,621 (13 classes), Tor 8,876/1,107/1,113 (8), VNAT 2025 15,704/1,960/1,960 (7), VNAT 2026 15,328/1,911/1,911 (7), USTC A-2 346,651/43,331/43,332 (17). All five have raw provenance but none has an existing same-protocol aligned three-view feature bundle. No Test/Unknown feature arrays were loaded. Project-local YaTC first-five-packet MFR reconstruction from immutable Stage12 packet refs passed full coverage for VPN (14,521 Known Train/Val flows) and Tor (9,983), with source PCAPs untouched; TrafficFormer/FIG raw-flow reconstruction for VPN is running in named tmux `codex_stage31_tffig_vpn_20260925` and has recovered the first target captures. This is input preparation only: no Stage31 encoder training, Test evaluation, or four-dataset metric exists yet. Next: finish exact three-view input reconstruction for all five protocol cells, verify hashes/flow IDs, then train under the frozen recipe and evaluate Known Test once checkpoints are selected.

Stage31 input-preparation milestone 2026-09-25: all five YaTC MFR Known Train/Val caches now pass exact frozen-flow coverage (VPN 14,521; Tor 9,983; VNAT 2025 17,664; VNAT 2026 17,239; USTC A-2 389,982). Shared-flow byte parity against prior Stage23 MFR is 7,040/7,040 VPN and 6,081/6,081 Tor. VNAT Medium-2025 required a documented input-only IPv6 adaptation for 2 flows/10 selected packets; attempts 1 and 2 failed on `other:` flow lookup and IPv4-only YaTC respectively, with failure evidence retained. VNAT Medium-2026 had 0 adapted IPv6 flows. A one-batch YaTC smoke on physical GPU 0 passed, but no formal Stage31 encoder, adapter, fusion head or Test evaluation has run. Five TrafficFormer/FIG Known Train/Val extractors are running in separate named project-local tmux sessions; their final coverage gates remain pending. Continue from `stage31_four_dataset_three_view_equal/RESULTS.md` and the `.tmux-task/codex_stage31_tffig_*_20260925/` logs. Do not reuse the partial failed caches or call this a four-dataset result.

Stage31 execution handoff 2026-09-25: VNAT Medium-2025, VNAT Medium-2026 and USTC A-2 now have complete exact same-flow three-input caches (17,664, 17,239 and 389,982 Known Train/Val flows); ISCX-VPN/Tor TrafficFormer/FIG extraction continues. Stage31 T0 architecture parity with immutable Stage30 passed at zero numerical difference in state, adapter loss, logits and equal weights. VNAT-2025 graph branch finished 50 epochs (best Known-Val Macro-F1 0.534438 at epoch 30), TF/YaTC branches continue; USTC A-2 graph is running. Bounded queue `codex_stage31_bounded_queue_20260925` is active, fixes lanes to physical GPUs 0/1/2, and writes `stage31_four_dataset_three_view_equal/queue_progress.json`. It will launch remaining same-protocol Known-only branches and T0 heads as gates pass, then stop before Test. A code-level Test lock refuses any Known Test input materialization until all five T0 heads are successful and checkpoint hashes match. No Stage31 Test result exists yet; never present Known-Val metrics as a four-dataset Test result. One-shot progress: run `stage31_four_dataset_three_view_equal/progress.py` through the required tmux helper.

Stage31 continuation 2026-09-25: ISCX-VPN's TrafficFormer/FIG cache now passes exact Known Train/Val flow-ID coverage, raising input readiness to 4/5; ISCXTor recovery remains in progress (last observed 6,809/9,983). Four graph branches have completed; current VNAT-2025 TF and YaTC branches remain active, with 0/5 T0 heads and 0/5 Known Test evaluations. A second named tmux process `codex_stage31_finalize_v2_20260925` waits for the bounded queue to finish all 15 branches and five T0 heads, then applies the all-head checkpoint-hash and source-hash gates before constructing any Known Test inputs. It is configured to fail closed on partial caches/evaluations and to preserve a per-run Test result, replay verification, final bundle and index. The first waiting-only finalizer was closed and replaced by v2 to add queue-liveness protection; neither opened Test input. Live state: `stage31_four_dataset_three_view_equal/finalization_progress.json`. This is an autonomous continuation, not a completed experiment; do not report validation-only values as Test metrics.

Stage31 input gate update 2026-09-25: ISCXTor TrafficFormer/FIG extraction also exited successfully, with cache audit PASS and exact 9,983-flow Known Train/Val coverage. The five protocol cells are now 5/5 input-ready and no Test input has been opened. The bounded queue started ISCXTor's graph branch on GPU 1; at the last check four of 15 branches were complete and both finalizer/queue tmux sessions were live. Continue via the project-local `progress.py`, `queue_progress.json`, and `finalization_progress.json`.

Stage31 training update 2026-09-25: ISCXTor's graph branch finished 50/50 epochs successfully, bringing branch completion to 5/15 (all graph branches). GPU 1 is idle until the queue's five T0 heads can run; TrafficFormer and YaTC branch work remains on GPUs 0/2. All five Known-only input gates are still PASS and Known Test remains locked at 0/5.

Stage31 status check 2026-09-25 08:40 UTC: five inputs remain PASS; branches are 6/15, T0 heads 0/5, Known Test 0/5. Completed: five graph branches plus VNAT Medium-2025 YaTC. Active: VNAT Medium-2025 TrafficFormer at epoch 10/20 (Known-Val Macro-F1 0.921857) on GPU0; ISCX-VPN medium-2022 YaTC at epoch 52/200 (Known-Val Macro-F1 0.901665) on GPU2. GPU1 and GPUs3–7 are currently idle. Queue and finalizer tmux sessions are both live; finalizer remains `WAITING_FOR_ALL_FROZEN_HEADS`, with `test_features_opened=0`. These epoch values are progress only, not selected final metrics.

Stage31 status check 2026-09-25 09:02 UTC: input caches remain PASS 5/5 and formal branch completion remains 6/15; T0 heads and Known Test remain 0/5. The active VNAT Medium-2025 TrafficFormer reached epoch 15/20 (latest validation Macro-F1 0.937039); ISCX-VPN medium-2022 YaTC reached epoch 198/200 (latest validation Macro-F1 0.909220). Queue and finalizer sessions remain live. Stage31 currently occupies GPUs 0 and 2; a separate process is visible on GPU3, so no additional Stage31 workload is started there. The finalizer has not opened Test (`test_features_opened=0`).

Stage31 ETA check 2026-09-25 10:37 UTC: five inputs PASS; 11/15 branches are complete; T0 heads 0/5 and Known Test 0/5. Active runs are ISCXTor medium-2022 TrafficFormer at epoch 2/20 (Known-Val Macro-F1 0.868794) on GPU0 and USTC A-2 YaTC at epoch 8/200 on GPU2. USTC YaTC has 5,416 batches per epoch and reports about 0.045 s/batch; from its 09:59 UTC start this implies roughly 14–16 hours for that run alone, subject to validation/checkpoint overhead. Remaining TrafficFormer branches, five T0 fits, and the locked Known Test feature extraction/evaluation add time; provisional end-to-end estimate is about 17–22 hours from this check (roughly 2026-09-26 03:30–08:30 UTC). Finalizer remains waiting with Test features opened=0.

# 2026-09-25 — Stage 32 three-dataset coarse single-seed plan

Status: `complete / PLAN_ONLY`. User superseded the Stage31 four-dataset fine-application plan: USTC is removed; ISCX-VPN, ISCXTor2016, and VNAT move to coarse closed-set labels with one model-training seed each. The named Stage31 bounded queue, waiting finalizer, and two active branches were stopped; subsequent read-only process check found no Stage31 queue/finalizer/branch-training workers. The completed 11/15 branch artifacts and interrupted-run evidence are preserved; Stage31 has 0/5 T0 heads and 0/5 Known Test results and must not be presented as complete. The new preregistered plan is `STAGE32_THREE_DATASET_COARSE_SINGLE_SEED_PLAN.md`: Stage20/25 VPN4 and Tor5 on exact original service flows; Stage14B VNAT Medium-2025 remapped by the original paper's application-category table to four observed Known coarse classes; one fixed training seed 2022; true Stage30 T0 three-view equal feature fusion, not a single branch or output ensemble. Comparison is paired only for identical flow IDs, split and coarse labels; Known Test was previously exposed and is development evidence. No Stage32 training or Test read occurred. Safe next action on an execution request: audit the three frozen views, historical comparator IDs and checkpoint hashes before any GPU launch.

Stage32 execution start 2026-09-25: status `in_progress`. User authorized execution of the single-seed coarse three-dataset plan and requested a CLI monitor command. Initialized independent bundle `stage32_three_dataset_coarse_single_seed/` via the workspace experiment helper; no model has been trained or Test value opened yet. Read-only source audit found complete VNAT Medium-2025 three-branch Known Train/Val files, Stage20/30 aligned ISCX Known Train/Val features, and existing Stage20 Known Test representations in Stage22/27; VNAT Test requires new input recovery only after all three coarse heads are frozen. Stage25 provides historical same-flow ISCX coarse predictions, while Stage14C6 provides VNAT Known-Val F2 predictions whose flow ordering must be independently verified. Next: implement and run preflight; fail closed on any ID, hash, or coverage mismatch before GPU training.

Stage32 training milestone 2026-09-25: status `in_progress / TEST_INPUT_RECOVERY`. Known-only preflight passed all 3 cells, 19 frozen input hashes, historical comparator ID/truth ordering, and reported Test feature values loaded=0. Sequential single-GPU queue `codex_stage32_three_coarse_train_20260925` completed 3/3 coarse T0 adapter/head runs (30+30 epochs each; no Unknown/Test fitting) and exited 0. Known Validation Macro-F1 VPN/Tor/VNAT=`0.917767/0.909662/1.000000`; all six checkpoint hashes and 19 source hashes passed the pre-Test gate. VNAT's perfect flow-random-split Val score is not capture-disjoint evidence. Gated finalizer `codex_stage32_three_coarse_test_20260925` is recovering only VNAT frozen Known Test packet inputs into the Stage32 bundle; one-shot Test metrics remain pending. Stage31 was separately corrected to `aborted / superseded_by_stage32_scope_change`, preserving 11/15 completed branches and interrupted logs, with 0/5 T0 heads and 0/5 Test evaluations. Safe next action: read `stage32_three_dataset_coarse_single_seed/progress.json` or run its read-only `progress.py`; do not relaunch the finalizer or tune on Test.

Stage32 terminal update 2026-09-25: status `complete / THREE_DATASET_SINGLE_SEED_DIAGNOSTIC_COMPLETE`. iscx_vpn Known Test Macro-F1=0.919350; iscx_tor Known Test Macro-F1=0.899375; vnat Known Test Macro-F1=1.000000; all three 2022-seed coarse T0 heads, one-shot Known Test units, saved-logit independent replay and protected hashes passed. Historical ISCX same-flow comparisons and VNAT Known-Val-only F2 comparison are in `stage32_three_dataset_coarse_single_seed/stage32_report.md`; no VNAT paired Test gain is claimed. Stop Stage32; no additional tuning or next stage launched.

Stage33 start 2026-09-25: status `in_progress / PREFLIGHT`. User clarified that VNAT's four classes are too coarse and requested a finer approximately six-class diagnostic. New isolated bundle `stage33_vnat_six_class_single_seed/` retains Stage14B `medium_seed2025` Known flow membership and frozen three-view branch weights. Fixed taxonomy: `streaming={netflix,youtube}` plus separate `rdp/rsync/scp/skype/ssh`; only a new six-class T0 adapter/head will be trained with Stage32's unchanged seed 2022, 30+30 epochs, batch 256 and Known-Val selection. The choice follows an already exposed four-class Test result, so claim scope is post-hoc exploratory development, not untouched validation. Frozen Known application support has been checked: RDP=36/4/4 Train/Val/Test and the other six applications are present in all splits. Stage32 and Stage14B remain read-only. Next: run `run_six.py preflight`, then if PASS use live GPU selection for train, freeze checkpoint hashes, and evaluate Known Test once without Unknown samples.

Stage33 training milestone 2026-09-25: status `in_progress / TEST_PENDING`. `run_six.py preflight` passed: exact 15,704/1,960/1,960 Known flow membership, six labels populated in all splits, 0 Unknown/Test feature values loaded. Named tmux `codex_stage33_vnat6_train_20260925` selected physical GPU 0 from live capacity, completed 30 adapter + 30 head epochs and exited 0. Known Validation Accuracy/Macro-F1/Weighted-F1=`0.972449/0.956710/0.972422`, adapter/head best epochs `16/6`; selected checkpoint hashes and frozen sources are in the isolated bundle. Next: verify frozen checkpoint hashes, then run `run_six.py evaluate` once on Known Test, preserving sample-level predictions and six-class confusion matrix.

Stage33 terminal update 2026-09-25: status `complete / SIX_CLASS_EXPLORATORY_DIAGNOSTIC_COMPLETE`. One selected six-class T0 adapter/head evaluated on the unchanged VNAT Medium-2025 Known Test (n=1,960): Accuracy/Macro-F1/Weighted-F1=`0.963776/0.943109/0.963750`. All 71 errors were rsync→scp (37) or scp→rsync (34), revealing distinctions hidden by the earlier four-class File Transfer label. `verify_six.py` independently replayed all logits/labels/confusion and verified both selected checkpoint hashes, Stage32 source/Test-cache hashes, and unchanged Stage14B freeze hash `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`. Unknown samples loaded=0; no old experiment or split was altered. This six-class mapping was chosen after four-class Test exposure, so claim scope is post-hoc development only; flow-random capture leakage risk and RDP four-flow Test support remain. Bundle: `stage33_vnat_six_class_single_seed/RESULTS.md`. Stop here; no next experiment automatically started.

### Current handoff (2026-09-25)

Stage33 is complete and its bundle validation/replay passed. Final six-class VNAT report: `stage33_vnat_six_class_single_seed/stage33_report.md`; all run evidence is indexed in that bundle's `manifest.json` and the project `EXPERIMENT_RESULTS.md`. No Stage33 worker remains. Do not interpret four-class versus six-class score differences as a same-task model comparison or claim capture-independent performance. Safe next action only on a new user request: preregister a group/capture-disjoint VNAT sensitivity protocol after checking per-application group feasibility; preserve Stage14B unchanged.

# 2026-09-25 — Stage34 USTC and CIC strict closed-set extension

Status: `in_progress`. The user requested the current T0 three-view equal-fusion method on USTC-TFC2016 and CIC-IDS-2017, allowing 10% stratified sampling. CIC protocol choice was clarified as the prior audit's strict three-class group-disjoint task (`BENIGN`, `DoS Slowhttptest`, `PortScan`); USTC uses the frozen A-2 Known 17 classes. The two tasks have different label granularity and are not score-comparable as equal-difficulty benchmarks. Stage31 USTC lacked complete branch/head/Test results; no prior result is substituted. New isolated bundle: `stage34_ustc_cic_closed_set/`. USTC deterministic SHA256 seed-2022 10%-per-class-and-frozen-role sample is frozen in `ustc_a2_10pct_manifest.csv`: Train=34,665, Val=4,333, Test=4,333. The existing exact USTC Train/Val input caches were subset into the new bundle; cache audit reports 38,998 aligned Train/Val flows, zero Test feature values or Unknown samples read. No USTC branch training, fusion, or Test evaluation has completed. CIC strict group assignments, flow sampling, feature recovery, training and evaluation remain pending. Dataset originals and historical stages are unchanged. Next: run USTC branch training on live-selected physical GPUs (maximum three), then fusion and gated Test; implement the CIC group-disjoint protocol and input path independently. Keep this entry updated with actual metrics and preserved failure evidence.

Stage34 milestone 2026-09-25 12:08 UTC: USTC graph branch finished 50/50 epochs with a frozen best checkpoint; TrafficFormer on physical GPU0 and YaTC on physical GPU2 are still training. CIC metadata-only group split is frozen for BENIGN/Slowhttptest/PortScan. The original 10%-all-classes draft left Slowhttptest Test at 13 flows, so before packet extraction/model results a v2 sample retained all 5,096 Slowhttptest flows while leaving BENIGN/PortScan at fixed 10%. V2 Train/Val/Test sample counts: BENIGN=133,189/16,849/16,793; Slowhttptest=3,909/1,055/132; PortScan=13,849/901/1,136. Five-minute groups remain disjoint; the original draft remains preserved with explanation. CIC Train/Val PCAP extraction is running in `stage34_cic_inputs_trainval`. The fail-closed continuation queue `stage34_bounded_queue` is live, enforces at most three owned physical GPUs, and writes `stage34_ustc_cic_closed_set/queue_progress.json`. It will run USTC fusion and gated Test, then CIC branch/fusion/Test in sequence; no Test feature values have yet been extracted, no Unknown samples used. A separate pre-existing process on GPU3 was untouched. One-shot progress: run `python stage34_ustc_cic_closed_set/progress.py` via the required tmux helper. Do not report branch Validation values as final Test scores.

### Current handoff (2026-09-25): Stage35 joint-vs-staged feature fusion

Status: `in_progress / implementation`. User requested an empirical comparison of three original encoders trained jointly end-to-end versus the current separately trained feature-fusion recipe, and clarified that both must classify from fused TrafficFormer/TAGCN/YaTC **features**, not branch logits. The isolated `stage35_joint_vs_staged_training/` bundle was initialized; its `EXPERIMENT_PLAN.md` freezes USTC A-2 17-class 10% seed-2022 as the first matched comparison against the running Stage34 staged baseline. Stage35 joint training and Test evaluation have not started. Verified Stage34 recipes: TrafficFormer 20e, TAGCN 50e, YaTC 200e, then adapter 30e and fixed-equal fusion head 30e. Joint J will use the same input flows, official TF/YaTC initialization, TAGCN architecture, three 64-D view adapters and one final feature-fusion classifier, with final fused CE backpropagating through all encoders for fixed 20e; cost differences will be reported. Stage34 remains read-only and its branch Validation F1 is not a final result. Next: implement ID/hash-checked Known Train/Val joint loader, memory smoke under the live three-GPU ceiling, then queue joint training and one-shot matched Test after checkpoints freeze. Preserve logs and failures in the Stage35 bundle; do not touch Unknown data or historical stages.

Stage35 milestone 2026-09-25: status `in_progress / BASELINE_DEPENDENT_QUEUE`. USTC A-2 source/official-weight/cache/ID preflight passed with Known Train/Val 34,665/4,333; Test feature values and Unknown values read=0. A 32-flow three-view fusion forward/backward smoke passed at 15.57 GiB allocated peak GPU memory. A separate optimizer smoke verified finite nonzero gradient paths through TrafficFormer, TAGCN, YaTC and the fusion modules and stepped all four optimizers. The initial preflight role-ID mismatch and first smoke launcher path typo are preserved as failures in named tmux logs; both were fixed without changing Stage34 or frozen inputs. `stage35_joint_vs_staged_training/run_queue.py` is prepared to wait for Stage34's full baseline completion before launching 20-epoch joint training, gated Known Test evaluation and independent replay. Do not call the smoke loss a classification metric or report a winner before the matched Test comparison exists.

# 2026-09-25 — Stage34/35 GPU 2 release and two-GPU continuation

Status: `in_progress / GPU2_RELEASED_TWO_GPU_QUEUE_RUNNING`. User requested physical GPU 2 be released and subsequent tasks run on no more than two cards. The exact CIC YaTC process on GPU 2 was terminated at epoch 2/200; its exit code 143, partial best checkpoint, history and log were preserved under `stage34_ustc_cic_closed_set/cicids2017/runs/ustc/A-2/yatc_interrupted_gpu2_20260925/` and `.tmux-task/stage34_cic_yatc/`. A live GPU UUID/PID check found no project process on GPU 2 afterward. The old Stage34 queue/finalizer and waiting Stage35 queue controllers were stopped with exit 143 to prevent their obsolete dependencies from failing or launching further GPU2 work; active CIC TrafficFormer and graph workers on GPUs 0/1 were not stopped. New Stage34 session `stage34_two_gpu_resume` preserves the frozen recipe, restarts YaTC from official initialization on the first available GPU 0/1, and then runs the CIC fusion/Test sequence on GPUs 0/1 only. New finalizer `stage34_two_gpu_finalizer` and dependent Stage35 session `stage35_joint_queue_two_gpu` are live; Stage35 still has no joint training result and is restricted to GPUs 0/1. Stage34 USTC Known Test already passed at n=4,333, Accuracy/Macro-F1/Weighted-F1 `0.984306/0.987649/0.984315`; CIC Test and full bundle verification remain pending. Changed files: additive Stage34 two-GPU queue, Stage34 finalizer queue-session parameter, Stage35 dependency/GPU selector and final metadata, Stage34/35 results/manifests, and this handoff. Next: read `stage34_ustc_cic_closed_set/queue_progress.json`, verify only GPUs 0/1 remain in this project's training process list, then wait for CIC branch/fusion/Test and Stage34/35 independent finalizers. Do not restart the interrupted GPU2 attempt or modify frozen splits.

# 2026-09-25 — Stage36 conditional open-set continuation

Status: `in_progress / CONDITIONAL_QUEUE_WAITING`. User permits GPU2 temporarily until next morning 08:00 and requests the prior open-set plan after the current closed-set work completes. The timezone for 08:00 is unresolved; only a completed short Known-Val smoke used GPU2, and no long queue uses it. Stage34/35 stay on GPUs0/1. Stage36 remains the isolated VNAT Stage14B `medium_seed2025` application-level six-class pilot, not a Service-LOSO or untouched external validation claim. `run_pilot.py preflight` passed frozen Stage14B hash `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`, exact 15,704/1,960/1,960/3,825 role counts, and frozen Stage31/33 checkpoint/source hashes. Added isolated extraction, frozen scoring, independent score replay, conditional queue and finalizer code under `stage36_vnat_sixclass_open_set_pilot/`. The conditional queue waits for Stage34 and Stage35 `completion_verification.json` PASS before opening Unknown and will use only GPUs0/1. Preserve all partial/failure evidence and do not modify Stage14B or historical checkpoints.

Stage36 milestone 2026-09-25 15:42 UTC: Known-Val parity PASS on 11 flows × six arrays = 66 exact comparisons. A short, `timeout 1800`-bounded frozen-head smoke ran on physical GPU2 and exited 0: 1,960 Known Validation logits exactly match Stage33 saved logits, fused embedding dimension=128; Unknown/Known Test feature values read=0. GPU2 has no remaining project process after this smoke. The first smoke launch failed due to a relative selector-path typo (exit 2); its log is retained; retry with the absolute workspace selector path passed. `stage36_conditional_queue` is running but only waiting for Stage34 two-GPU queue, Stage34 finalizer and Stage35 verified completion. It is configured to fail closed on missing/non-PASS dependencies, then run Stage36 Unknown extraction/score evaluation/replay on live-selected GPUs0/1 only. The score-unit smoke and Python syntax checks passed. No Unknown input or score has been opened yet. Next: inspect the three progress JSON files; do not report Stage36 as complete while the conditional queue waits.

### Current handoff (2026-09-25 15:45 UTC)

Stage34 `stage34_two_gpu_resume` is running CIC branches on GPUs0/1; Stage35 `stage35_joint_queue_two_gpu` waits for Stage34 verified completion; Stage36 `stage36_conditional_queue` waits for both. Stage36 Known-only preflight, 66 packet-array parity checks, 1,960-logit frozen-head parity, score-unit check, Python syntax, `git diff --check`, and its in-progress experiment-bundle hash validation all passed. No Stage36 Unknown/Test feature values have been opened. Physical GPU2 is idle and no future Stage36 task selects it, so it will remain released before 08:00 in either timezone. Safe inspection: `tmux_task.sh status/capture` for the three named sessions and the three `queue_progress.json` files; do not restart a running session or modify frozen inputs. If a dependency fails, Stage36 records `queue_failure.json` and stops without opening Unknown.

# 2026-09-26 — Stage37 GPU2 VPN/Tor joint three-view training

Status: `in_progress / GPU2 VPN training`. The user's latest instruction superseded the pending Stage36 open-set launch: Stage36's waiting controller was closed before any Unknown input was opened; its earlier Known-only gates and stopped-controller logs remain preserved. The new isolated Stage37 reuses exact Stage20 frozen ISCX-VPN/Tor flows with coarse VPN4/Tor5 labels, seed2022, and the already verified Stage35 three-view end-to-end model architecture. Stage21 raw TrafficFormer/FIG cache and Stage23 YaTC raw image cache passed per-flow Known Train/Validation alignment; frozen split/weight hashes and counts passed. Physical GPU2 was selected live with >20 GiB free and low utilization. Named session `stage37_gpu2_joint_0926` is running VPN train first, then one-shot Known Test evaluation, then Tor train/evaluation. Its `queue_progress.json`, per-run `progress.json`, `RESULTS.md`, and `manifest.json` are in `stage37_joint_vpn_tor_single_seed/`. No Test/Unknown values are used for fitting or checkpoint selection. This is a one-seed development diagnostic with weak capture-derived service labels; do not call it independent validation. Stage34/35 workers on GPUs0/1 are untouched. Prior 08:00 GPU2 release request should be checked against the current task's runtime before that deadline rather than silently ignored.

Stage37 terminal update 2026-09-26 02:44 UTC: status `complete / SINGLE_SEED_JOINT_DIAGNOSTIC`. Named `stage37_gpu2_joint_0926` exited 0 after VPN and Tor each completed 20 epochs and one-shot Known Test evaluation; queue status PASS. Checkpoints selected at VPN epoch15 and Tor epoch12 by Known Validation Macro-F1 `0.882000/0.852184`. Known Test Macro-F1 is `0.882180/0.853184`, lower than same-flow Stage32 staged fusion `0.919350/0.899375` by `0.037171/0.046192`. Independent saved-prediction replay passed 2,210 exact Test IDs/labels, all headline metrics/confusion matrices, checkpoint SHA256 and unchanged Stage20 hash. `stage37_joint_vpn_tor_single_seed/paired_comparison.csv` and `completion_verification.json` hold paired details; 33 Stage37 bundle artifacts passed SHA256 validation. GPU2 was idle at the final live snapshot. Stage32 used a different, longer staged training budget, so this does not establish inherent joint-training inferiority. No tuning or open-set experiment followed. Stage36 remains stopped/unfinished with no Unknown values opened. Safe next action: read Stage37 `RESULTS.md`; do not rerun or overwrite its checkpoints.

# 2026-09-26 — Stage38 three-view DES open-set transfer plan

Status: `complete / PLAN_ONLY`. At the user's request, the full plan was recorded in `STAGE38_THREE_VIEW_DES_OPEN_SET_TRANSFER_PLAN.md`. It uses the Stage32-style separately trained TrafficFormer/FIG-TAGCN/YaTC equal-feature fusion rather than Stage37 joint training. VNAT uses frozen Stage33 six-class weights and Stage14B `medium_seed2025`; ISCX-VPN/Tor use frozen Stage12 `medium-2022` application-level Unknown protocols and require new Known-only Stage32-style training because Stage32 original all-Service weights saw would-be Unknown classes. It predefines C0/C1 classifier-uncertainty diagnostics and D0/D1 DES-v0/v1 as the primary same-representation support comparison; historical OD/H1 remain separate baselines, with any cross-encoder H1-X explicitly secondary and not renamed as original H1. No model training, Test/Unknown extraction, score fitting, threshold selection, GPU workload or protocol change was started by this planning task. Safe next action only on an execution request: perform Stage38 input/hash/label-mapping preflight, then VNAT frozen evaluation before ISCX Known-only training. Preserve Stage36 and all historical stages read-only.

# 2026-09-26 — Stage34B CIC benign/malicious balanced retest

Status: `in_progress / FROZEN_SAMPLE_AND_TRAINVAL_CACHE_READY`. User requested retesting CIC with approximately equal benign and malicious counts after Stage34 TrafficFormer's late-epoch one-class collapse. New isolated bundle `stage34b_cic_balanced_retest/` retains the Stage34 v2 group-disjoint split and every attack flow. Deterministic seed-2022 SHA256 ranking samples BENIGN to the combined attack count separately inside Train/Validation/Test; class counts are Train 17,758/3,909/13,849, Validation 1,956/1,055/901, Test 1,268/132/1,136 (BENIGN/Slowhttptest/PortScan). New manifest SHA256=`35e99d89ad4eddc828c9d67529ea1b527d1fd511cb39db8c509461b5bf38e25d`; original Stage34 manifest SHA256 remains `9734812c102266e88c58364d499c4f1494c24d0f012ac45e33e6292c93f7bd53`. Train/Val subset caches completed for 39,428 exact flow IDs. The new three-view recipe reuses Stage31/34 code without changing optimizer or checkpoints, and will compare original/balanced models on identical balanced Test IDs. It waits for existing Stage34/35 GPU0/1 sessions before training; no balanced Test feature values or outcome were opened while sampling. Next: start `stage34b_balanced_queue_0926`, inspect `stage34b_cic_balanced_retest/queue_progress.json`, and verify final paired Test metrics plus bundle hashes. Historical Stage34/35 outputs remain untouched.

Stage34B scope correction 2026-09-26 03:27 UTC: the user explicitly stopped the old imbalanced CIC attempt and requested immediate balanced training. Exact project-owned sessions `stage35_joint_queue_two_gpu`, `stage34_two_gpu_finalizer`, `stage34_two_gpu_resume` and `stage34_cic_trafficformer` were closed in dependency-safe order. Live verification showed GPU0/1 at 0 MiB and no surviving target PIDs; `.tmux-task/cic_balance_stop_verify_0926/output.log` preserves evidence. The tmux helper's close command did not create exit.status, so old Stage34 is marked `interrupted_by_user`, and Stage35 is paused before joint training. Old CIC has no fusion/Test model; the balanced experiment cannot claim a paired final-model delta. Stage34B will directly run its unchanged three-view recipe, then extract only its frozen balanced Test packet input after checkpoint selection. Updated scripts and scope are recorded in `stage34b_cic_balanced_retest/`; no historical model/checkpoint was deleted.

Stage34B execution milestone 2026-09-26 03:33 UTC: named queue `stage34b_balanced_queue_0926` passed frozen-manifest/cache preflight; graph worker `stage34b_graph_0926` finished 50/50 epochs on physical GPU1 with best Known-Val Macro-F1≈0.993431 at epoch41. TrafficFormer and YaTC workers started concurrently on physical GPU0/1 via live selectors and are active. No balanced Test packet values have been extracted and no three-view Test metric exists. Monitor `stage34b_cic_balanced_retest/queue_progress.json` and the named `.tmux-task/stage34b_*` logs; queue will gate fusion, raw-Test extraction, single Test evaluation and independent saved-logit/checkpoint/bundle verification.

Stage34B input-parity milestone 2026-09-26 03:40 UTC: a full read-only recheck of all 39,428 selected Known Train/Validation flow IDs found bitwise equality between each balanced cache row and its original Stage34 row for all five TrafficFormer/FIG arrays plus the two YaTC MFR role arrays. `stage34b_cic_balanced_retest/input_parity.json` reports PASS; Unknown/Test feature usage in this check is zero. This supports that sample membership, rather than packet preprocessing, is the changed experimental factor.

Stage38 milestone 2026-09-26 05:21 UTC: VNAT Stage38A finished frozen inference and independent score replay, with 17 protected input hashes unchanged. `vnat/stage38a_verification.json` is PASS. The first queue exited after all calculations because `RESULTS.md` lacked six required experiment-bundle headings; the original `queue_failure.json` and tmux log remain preserved. The headings were added and a separate bundle validation passed for 48 artifacts. VNAT Known Test Macro-F1 is 0.943109; AUROC C0/C1/D0/D1 is 0.888421/0.851933/0.752023/0.897053, while D1 UFAR remains high at 0.623791. Stage38B Known-only metadata and source-lock audits passed for 14,521 VPN and 9,983 Tor Known Train/Val flows. Initial VPN branch launch attempts failed before Python training due to a relative selector path; the logs are retained. Corrected named sessions `stage38b_vpn_graph_gpu2_v2_0926`, `stage38b_vpn_tf_gpu3_v2_0926`, `stage38b_vpn_yatc_gpu4_v2_0926` selected physical GPUs 2/3/4 live and started independently. GPU2 graph command has a timeout that releases it before 07:45 UTC. Stage38B has not loaded Unknown/Test feature values or fitted detection thresholds yet. Next: check all three branch exit markers/checkpoint hashes, then run Known-only equal fusion, extract frozen Test inputs, and evaluate once.

Stage38 automatic handoff 2026-09-26 05:35 UTC: VPN/Tor graph branches finished with exit 0. Physical GPU2 now runs Tor YaTC with a 7,200-second timeout; GPU3 runs VPN TrafficFormer and GPU4 runs VPN YaTC. `stage38b_training_queue_0926` automatically launches Tor TrafficFormer when GPU4 is released, then each dataset's Known-only equal-fusion stage after its branches finish. Per-dataset controllers `stage38b_vpn_detection_queue_0926` and `stage38b_tor_detection_queue_0926` wait for frozen fusion checkpoints, then independently run Known-only score calibration, frozen Test packet extraction, one-shot open-set evaluation, and saved-score replay. `stage38_finalizer_queue_0926` waits for both verified datasets before writing aggregate results and validating the experiment bundle. Live selection for later GPU stages is restricted to physical GPUs3/4/7; the controllers do not reuse GPU2. The metadata-only test packet provenance audit found all 7,621 VPN and 5,113 Tor Test role IDs; no Test packet feature values were opened in that audit. Read-only progress: `python stage38_three_view_des_open_set_transfer/progress.py` from the project root.

Stage38 recovery milestone 2026-09-26 06:42 UTC: all eight VPN/Tor Known-only branch/fusion workers exited 0. Their fusion Known-Val Macro-F1 scores are VPN 0.972701 and Tor 0.944656. Both calibration stages replayed frozen Known-Val predictions and used zero Test/Unknown feature values. The first VPN and Tor Test packet builders failed before creating feature arrays because the new builder accidentally appended `protocol/` to an already protocol-rooted Stage12 path. Original child and controller failure logs remain preserved; the stopped finalizer recorded the dependent failure. The builder path was corrected and a metadata-only frozen Test row check passed for VPN 7,621 and Tor 5,113 IDs. Resumed named controllers `stage38b_vpn_detection_queue_v2_0926`, `stage38b_tor_detection_queue_v2_0926` reuse the verified Known-only calibration without refitting or changing checkpoints, and now stream Test PCAPs into isolated Stage38 caches. `stage38_finalizer_queue_v2_0926` waits for their independent score replay. The old queue failures are retained as provenance; no historical Stage12/31 model or split was modified.

Stage38 terminal update 2026-09-26 08:03 UTC: VNAT, ISCX-VPN and ISCXTor2016 evaluation plus independent frozen-score replay completed. Known Test Macro-F1 is `0.943109/0.975277/0.941121`; best AUROC is `0.897053/0.583954/0.924869`, respectively. Final Gate is `MIXED_OR_INCONCLUSIVE`, driven chiefly by poor VPN open-set separation. The aggregate calculation completed, but its first bundle check rejected the unsupported manifest status `complete`; the status and finalizer source were corrected to schema-supported `success`, original failure evidence was retained, and no model inference or metric was rerun or changed.

# 2026-09-26 — Stage38 GPU2 open-set execution

Status: `in_progress / VNAT 38A prelaunch`。用户允许 GPU2 与正在 GPU0/1 运行的 CIC 平衡重测并行，并明确要求推进开集任务。最新冻结计划是 `STAGE38_THREE_VIEW_DES_OPEN_SET_TRANSFER_PLAN.md`；较早的 Stage36 等待队列已经停止，Stage36 原产物保持只读。Stage38 独立 bundle 已初始化；`run_vnat.py` 在新目录调用经 Known-Val parity 验证的 Stage36 分数实现，复核 Stage14B/31/32/33/36 输入哈希。`run_queue.py` 将按 preflight→Known-Val parity→frozen-head smoke→Unknown 提取→GPU2 冻结推理→独立重放运行。GPU2 推理使用实时容量选择，最迟 07:45 UTC 释放；固定单 seed、Known-Val P95、k=10 和 0.5/0.5，没有训练新 encoder。启动后查看 `stage38_three_view_des_open_set_transfer/queue_progress.json`；完成 VNAT 后再审计 ISCX-VPN/Tor Known-only 迁移。

Stage34B terminal update 2026-09-26T07:03:24.215823+00:00: status `complete / PASS`. Balanced CIC Test n=2536, Accuracy/Macro-F1/Weighted-F1=`0.998028/0.997482/0.998028`; saved logits, per-class metrics, sample IDs, branch/head SHA256 and experiment bundle verified. Old Stage34 CIC fusion/Test had been stopped by user before completion; no paired delta is claimed. See `stage34b_cic_balanced_retest/RESULTS.md`.

# 2026-09-27 — Stage41 matched USTC Scenario A-1/A-2/A-3 comparison

Status: `in_progress / COMMON_PROTOCOL_FROZEN_AND_OD_TRAINING`. User requested a two-method comparison under Open-Detect Scenario A-1/A-2/A-3. `stage41_a123_matched_open_set_comparison/` is isolated from prior Stage34/40 outputs. The frozen seed-2022, 10%-per-class/source-split common flow pool exactly reproduces Stage34 A-2 Known IDs and Stage40 A-2 role IDs. Role counts are A-1 38,448/4,805/4,806/85, A-2 34,665/4,333/4,333/558, A-3 30,884/3,860/3,860/1,031 (Known Train/Validation/Test/Unknown Test). A common flow-ID-based 1:1 balanced Test subset was frozen before reading scores; its SHA256 is `e1e7a982328abf87fde192d5f395ff5bf1b913ddc29ea3146a4fbabec4db100e`. Open-Detect image inputs were prepared from the aligned Stage3 32x32 image arrays and verified against original class labels. Three Open-Detect released-code training jobs started on physical GPUs 0/6/7 with 100 epochs, seed 2022, Known-only Train/Validation; A-1/A-3 three-view input generation and new Known-only three-view training are still pending. Historical A-2 three-view weights may be reused only after frozen flow/role identity and checkpoint hashes pass. No A-1/A-3 comparative Test result exists yet, and historical full-data Open-Detect checkpoints are not used as matched baselines. Read `stage41_a123_matched_open_set_comparison/RESULTS.md` and named `.tmux-task/stage41_*` logs for progress. Do not call the single-seed local 10% subset the paper's original five-fold result.

Stage41 concurrency correction 2026-09-27: user reduced Stage41 to at most two parallel GPUs. Live inspection found the three Stage41 OD jobs on GPUs 0/6/7; A-1 GPU0 had reached 52/100 epochs, A-2 GPU6 51/100, A-3 GPU7 57/100. The waiting-only `stage41_a123_queue_0927` controller was closed first, then the exact A-1 `stage41_od_a1_gpu0_0927` session was closed. The original A-1 `od_run/` config, 52-epoch history, 104 MiB best checkpoint and tmux log were retained. A read-only GPU/PID check immediately afterward showed GPU0 at 0 MiB with no Stage41 process; A-2 and A-3 remained running on GPUs6/7. The new `run_queue_2gpu.py` waits for these two jobs, launches a fresh full A-1 run in `A-1/od_run_2gpu/` on GPU7 after it is free, and schedules all subsequent three-view training/evaluation on GPUs6/7 with a hard maximum of two Stage41 GPU jobs. The interrupted A-1 checkpoint is excluded from final metrics. The new controller `stage41_a123_queue_2gpu_0927` was started and verified `RUNNING / EXISTING_OD_AND_KNOWN_INPUTS` with `max_parallel_stage41_gpus=2`; at that snapshot A-2/A-3 were at epochs 66/74, GPU0 remained at 0 MiB, and only GPUs6/7 had Stage41 GPU processes. Status remains `in_progress`; monitor `stage41_a123_matched_open_set_comparison/progress.py`. Prior frozen split/score artifacts remain unchanged.

Stage41 terminal record 2026-09-27: status `complete / PASS`, single seed 2022. The two-GPU queue and all A-1/A-3 branch, fusion, calibration, frozen Test evaluation, comparison and finalizer sessions exited 0; `completion_verification.json` reports same-flow Test IDs, 3 scenarios × 2 methods, Unknown Train/Validation use 0, and Known Validation P95 as the formal threshold. Balanced Test AUROC OD/three-view is A-1 `0.9907/0.9769`, A-2 `0.9290/0.9406`, A-3 `0.9782/0.9609`; balanced Unknown F1 is `0.9659/0.9605`, `0.5038/0.6741`, `0.9191/0.9231`. In A-2, Geodo accounts for 337/257 OD/three-view false negatives out of the same 410 Geodo Test flows; natural-prevalence F1 is recorded separately in `stage41_a123_matched_open_set_comparison/RESULTS.md`. Final claims are diagnostic for this local 10% matched subset, not the paper's exact five-fold result. Final files: `comparison_run_results.csv`, `per_unknown_class.csv`, `completion_verification.json`, `RESULTS.md`, and `manifest.json`; no Stage41 workers remain on GPUs6/7. Next action: stop at Stage41 and review the recorded comparison before any further experiment.

Stage41 closed/open documentation addendum 2026-09-27: status remains `complete / PASS`. On the frozen Known Test flows, Open-Detect versus three-view Macro-F1 is A-1 `0.977538/0.988065`, A-2 `0.978454/0.987649`, A-3 `0.996973/1.000000`. Open-Detect Macro-F1 was reconstructed from its existing per-flow Known Test predictions; stored Accuracy and Weighted-F1 matched the original `test_metrics.json` exactly. The paired closed/open table and balanced AUPRC values are now in `stage41_a123_matched_open_set_comparison/RESULTS.md`; source prediction files and `.tmux-task/stage41_closed_metrics_0927/output.log` remain available. No model or frozen score was rerun.

Stage42-U terminal update 2026-09-28 03:54 UTC: status `complete / PASS`. The all-flow DoS Hulk cache emitted 155,168 Unknown Test flows, and frozen inference on physical GPU0 evaluated four methods against the unchanged 2,272 Known Test flows. DES-v1 natural AUROC/AUPRC/UFAR = `0.973767/0.998806/0.001134`; balanced 1:1 AUPRC = `0.928132`. Independent replay verified all eight method × view rows, the candidate manifest hash, and zero Unknown/Test fitting. The tmux cache and finish queue both exited 0. Results remain a post-hoc favorable-setting diagnostic; see `stage42u_cic_hulk_open_set/RESULTS.md` and its preserved score, CSV, cache, and log artifacts. No further experiment was launched.
