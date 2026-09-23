# Stage 4 — Known-Only Local Boundary Calibration Diagnosis

本目录是 **POST-HOC USTC MECHANISM DIAGNOSIS**，不是新的独立性能验证。A-1/A-2/A-3 Unknown Test 已经被观察；任何候选 boundary rule 必须在新的 untouched CSTNET-TLS1.3 上首次正式验证。

## Motivation

诊断统一 global threshold 是否在异质的 Known class/component score scale 上产生不均衡 acceptance region。

## Why Density Fit Is Not Enough

Stage 3 Failure Diagnosis 已表明 K2 Known-validation density fit 改善不能预测 Unknown utility；Stage 4 固定 K2，只检查 boundary calibration。

## Frozen Stage3 Assets

只读使用冻结 `mu_x`、StandardScaler、PCA64、K1、K2、Stage 3 thresholds 和 final predictions。禁止重新训练或拟合。

## Global Boundary

`max_y log p_GMM(x|y) >= frozen_global_threshold`，与 Stage 3 完全一致。

## Class-Calibrated Boundary

`tau_y` 是 true-class Known Validation K2 class score 的固定 P05；接受条件为 `max_y(s_y-tau_y) >= 0`。

## Local Component Boundary

`tau_yk` 是 true-class posterior-assigned component joint score 的固定 P05；接受条件为 `max_yk(s_yk-tau_yk) >= 0`。若 component validation count `<30`，确定性回退到 `tau_y`。

## Known Validation Coverage

所有阈值先在 Known Validation 冻结，再计算 overall/class/component coverage。Global/Class/Local 的平均 component acceptance gap 分别为 `0.169174/0.234170/0.072024`。Class-P05 将 per-class coverage 显著拉齐，但 component 内部仍不均衡；Component-P05 显著降低 component gap。

## Known Test Generalization

Known Test 只在阈值冻结后用于泛化检查。Global/Class/Local 的平均 component gap 为 `0.171906/0.228309/0.070406`，Component-P05 的均衡趋势延续到 Known Test。

## Htbot–Tinba Case

对 A-1 的 850 个 Tinba、尤其 86 个 Single-reject/Global-K2-accept 样本做诊断。这 86 个样本全部落在 Htbot component 1，validation percentile 中位数为 `0.292899`；Class/Local 均拒绝 `0/86`，所以 local calibration 不能修复该 failure。

## Virut–Neris Case

对 A-3 的 236 个 Single-accept/Global-K2-reject Neris 做相同诊断。原始 Virut component 归属为 179/57，但 Component-P05 将 236 个全部按最大归一化 margin 指向 component 0 并重新接受，消除了 Stage 3 的 rejection 机制。

## Post-Hoc Validity Warning

USTC Unknown 输出必须标记 `POST_HOC_DIAGNOSTIC_ONLY`，不得称为独立提升、验证方法或 SOTA。

## Final Diagnosis

最终为 **Diagnosis C — LOCAL BOUNDARY NECESSARY**，含义仅限于 Known component-coverage consistency：Local 明显优于 Class 均衡 component coverage。它不代表 Unknown Detection 已改善；USTC post-hoc 中 Local UFAR 在 A-1/A-3 反而明显升高。

## Candidate External Rule

已冻结 Component-P05 为 CSTNET 首次独立验证的 proposed candidate，Global 与 Class-P05 为 baseline；没有运行 CSTNET。详见 `outputs/external_validation_protocol_candidate.md`。

## Next Step

本任务不继续下一阶段。新的正式实验必须在访问 CSTNET Unknown Test 前冻结 rule、P05、fallback、score 和 split protocol。
