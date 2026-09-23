# Stage 3 Failure Diagnosis

本目录只做 post-hoc diagnosis。Stage 3 A-1/A-2/A-3 Unknown Test 已经被观察过；这里提出的任何 Adaptive-K 或 class-selection hypothesis 都不是独立验证结果。

## Background

Stage 3 比较冻结的 Single-Full K1 与 Multi-Full K2：A-1 的 Multi UFAR 更差，A-2 接近不变，A-3 更好。本目录定位差异来自哪些 Known class，并审计 Known-only 密度和 component 几何信号。

## Stage3 Gate D

正式 Stage 3 结论为 **Gate D — MIXED**：Multi−Single UFAR 分别为 A-1 `+0.101176`、A-2 `+0.002689`、A-3 `−0.022899`。

## Why This Is Post-Hoc Diagnosis

分析只读取冻结预测、模型和 Known Train/Validation 表征；冻结的 Unknown `mu_x` 仅用于补算缺失的 K2 component 归属，并要求回放 score 与正式冻结 score 逐行一致。不重新 fit scaler/PCA/GMM，不改 threshold/K，也不重跑 Final Test。Unknown Test 只用于归因，所以不能用于优化并再次宣称独立效果。

## A-1 Failure Attribution

全部净恶化来自 Htbot：Single/Multi 分别吸收 444/530 个 Tinba，`DeltaAbsorb=+86`。86 条 Single-reject/Multi-accept 样本全部落入 Htbot K2 component 1。Top-3 中 BitTorrent、Cridex 是 `DeltaAbsorb=0` 的确定性并列对照，不是 failure contributor。

## A-2 Neutral Result

Multi 相对 Single 有 63 条 SM、48 条 MS，仅净增 15/5,579。Known-class 归因是 Miuref `+16`、Shifu `−1`；Geodo→Miuref 的大量共同误接收基本不变，因此总体仅 `Delta UFAR=+0.002689`。

## A-3 Improvement Attribution

全部净改善来自 Virut：Neris→Virut 从 1,198 降为 962，恢复 236 条，`DeltaAbsorb=−236`。这 236 条在 K2 best-class 回放中分配到 Virut component 0/1 的数量为 179/57；A-3 没有 SM regression。

## Known-Only Signals

审计 DeltaNLL2、K2 weights、train/validation TV、欧氏与 pooled-covariance Mahalanobis separation、covariance volume、validation likelihood-tail shift、二值 disagreement 和 margin shift。A-1/Htbot 的 DeltaNLL2=`14.075196`、weights=`0.736541/0.263459`、TV=`0.003103`；A-3/Virut 为 `24.334647`、`0.218307/0.781693`、`0.008581`。51 个单元均无 `<1%` tiny component 或 validation-empty component。

## Density vs Utility

`outputs/known_signal_vs_unknown_failure.csv` 将 Known-only 信号与已观察的 `DeltaAbsorb` 做 post-hoc 合并；`outputs/known_signal_correlations.csv` 报告原始样本量、Spearman 系数和稀疏结局限制。DeltaNLL2 与 DeltaAbsorb 的 pooled Spearman `rho=−0.021545, p=0.880711, n=51`，没有可用单调规律。A-1/Htbot 和 A-2/Miuref 都出现 density fit 更好但 Unknown absorption 更差。

## Candidate Hypotheses

仅在 `outputs/candidate_hypotheses.md` 提出 R1/R2/R3 概念规则。没有在当前 USTC Final Test 上执行 Adaptive detector。

## Final Diagnosis

最终结论是 **Diagnosis A — DENSITY-UTILITY MISMATCH**。A-1 failure class 没有 tiny/empty component，TV 很低且模型收敛，故不支持简单的 degenerate-mixture 解释；Known-only 相关性也未建立可独立复现的 class-selection 规律。

## Validity Boundary

Stage 3 final test has already been observed. Any candidate adaptive rule proposed after this analysis must be validated on a new untouched benchmark. 本目录不声称“Adaptive rule beats Single on USTC”。

## Next Experiment

本任务不执行下一阶段。若未来另行授权，应先冻结候选规则，并优先在未触碰的 CSTNET-TLS1.3、其次 CipherSpectrum 上做第一次正式验证。

运行入口：

```bash
python stage3_failure_diagnosis/scripts/analyze_failure.py \
  --output-dir stage3_failure_diagnosis/outputs
```
