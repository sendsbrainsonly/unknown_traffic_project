#!/usr/bin/env python3
"""Write the evidence-backed Stage 14C-5 Markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[1]


def table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include="number").columns:
        selected[column] = selected[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    return selected.to_markdown(index=False)


def main() -> None:
    summary = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    protocols = pd.read_csv(OUT / "protocol_summary.csv")
    composition = pd.read_csv(OUT / "protocol_class_composition.csv")
    attribution = pd.read_csv(OUT / "protocol_difficulty_attribution.csv")
    per_class = pd.read_csv(OUT / "per_class_comparison.csv")
    errors = pd.read_csv(OUT / "nonzero_confusion_pairs.csv")
    dip_summary = pd.read_csv(OUT / "validation_dip_summary.csv")
    windows = pd.read_csv(OUT / "replay_reset_window.csv")
    drop_class = pd.read_csv(OUT / "per_class_drop_recall.csv")

    count_table = composition[composition.class_role == "known"][
        ["protocol_id", "application", "train_samples", "validation_samples"]
    ].sort_values(["protocol_id", "application"])

    hardest = drop_class.sort_values("delta_recall").groupby("protocol_id", as_index=False).head(8)
    report = f"""# Stage 14C-5 — VNAT Protocol Difficulty & Validation Dip Audit

## 审计结论

**`{summary['conclusion']}`**

Medium-2025 与 Medium-2026 的闭集差距主要由 **Known 类组成难度**导致，而不是训练随机种子。Medium-2025 同时把 `rsync` 与 `scp` 作为 Known；其最佳验证 checkpoint 中，`rsync` 的 191 个样本有 185 个被预测成 `scp`。Medium-2026 将 `scp` 整类留作 Unknown，`rsync` 达到 191/191 正确。

观测回放与原 Stage 14C-4 逐 epoch 核心指标严格一致，最大绝对差异为 `{summary['maximum_replay_absolute_difference']:.3e}`。大幅 validation drop 没有与 epoch 51/81 的 scheduler/reset 同步；reset 会造成 prototype norm 的预期离散跳变，但 reset 当轮 Macro-F1 变化小且方向不一致，随后恢复。因此属于已知、可恢复的验证波动，不构成新的实现错误。

## 1. 边界与完整性

- Stage 14B freeze hash：`{summary['freeze_hash']}`
- Known Test 使用量：`{summary['known_test_samples_used']}`
- Unknown Test 使用量：`{summary['unknown_test_samples_used']}`
- DES 执行：`{summary['des_executed']}`
- 原训练代码/配置修改：`{summary['training_code_modified']}`
- 两个 telemetry replay parity：`{summary['observational_replays_parity_pass']}`
- 本回放只增加只读观测；没有参与 checkpoint 选择，也没有改变训练随机状态。

## 2. Frozen protocol 组成

| protocol | Known | Unknown | Train | Val |
|---|---|---|---:|---:|
| Medium-2025 | {protocols.iloc[0].known_classes} | {protocols.iloc[0].unknown_classes} | {int(protocols.iloc[0].known_train_samples)} | {int(protocols.iloc[0].known_val_samples)} |
| Medium-2026 | {protocols.iloc[1].known_classes} | {protocols.iloc[1].unknown_classes} | {int(protocols.iloc[1].known_train_samples)} | {int(protocols.iloc[1].known_val_samples)} |

### 每个 Known 类 Train / Validation 数量

{table(count_table, ['protocol_id', 'application', 'train_samples', 'validation_samples'], 0)}

`rdp` 在两组中均仅有 36/4 个 Train/Val 样本，但两个 selected checkpoint 的 rdp F1 分别为 0.8571 和 1.0000；它会增加方差，却不是两组约 0.25 Macro-F1 差距的主因。`sftp` 在两组均为 Unknown，不参与训练/验证，也不是该差距来源。

## 3. 闭集指标与 seed 对照

{table(attribution, ['protocol_id', 'protocol_training_seed', 'primary_val_accuracy', 'primary_val_macro_f1', 'primary_val_weighted_f1', 'fixed2022_val_macro_f1', 'within_protocol_seed_delta_macro_f1', 'best_epoch', 'epochs_completed'])}

- 主 runs 的 Medium-2026 − Medium-2025 Macro-F1：`{summary['primary_medium2026_minus_medium2025_macro_f1']:.6f}`。
- 固定相同训练 seed=2022 后差距仍为：`{summary['fixed2022_medium2026_minus_medium2025_macro_f1']:.6f}`。
- 相同 seed 下仍保留主差距的 `{summary['same_seed_gap_fraction_of_primary_gap']:.2%}`，所以协议类别组成是主因，随机 seed 是次要因素。

## 4. Selected checkpoint 类别级结果

{table(per_class, ['protocol_id', 'class', 'train_support', 'val_support', 'precision', 'recall', 'f1'])}

### 非零混淆项

{table(errors, ['protocol_id', 'true_class', 'predicted_class', 'count'], 0)}

Medium-2025 的性能损失高度集中在 `rsync ↔ scp`，尤其是 `rsync → scp = 185`；不是所有类别都普遍退化。Medium-2026 没有 `scp` 参与 Known 决策，rsync/scp 的语义边界冲突不再出现。

## 5. Validation dip 与 reset/scheduler 对应关系

实现顺序经原代码确认：每轮先训练；epoch 51/81 在验证前原位重置 prototype 并清除该参数的 optimizer state；验证完成后才调用 `scheduler.step()`。因此 epoch 51/81 使用前一轮 milestone 后已经降低的 LR，同时也发生 reset。

### 显著下降汇总

“显著下降”预注册为：相邻轮 Macro-F1 或 Accuracy 下降至少 0.15，或 validation loss 至少增至 3 倍且绝对增加至少 0.10。

{table(dip_summary, ['protocol_id', 'obvious_drop_count', 'obvious_drops_at_reset_epochs', 'obvious_drops_before_first_lr_reduction', 'obvious_drops_after_first_lr_reduction', 'recovered_within_3_epochs_count', 'reset_epoch_51_delta_macro_f1', 'reset_epoch_81_delta_macro_f1', 'reset_epoch_51_prototype_norm_jump', 'reset_epoch_81_prototype_norm_jump', 'max_val_macro_f1_after_epoch_82'])}

两个 exact-config primary replay 中，显著下降全部发生在首次 LR 降低前，epoch 51/81 均没有命中显著下降规则。reset 当轮 Macro-F1 有正有负，且负变化仅约 0.03–0.04；这与“reset 导致大幅崩塌”的假设不符。

对 Stage 14C-4 的四条完整历史轨迹（两条 primary + 两条 fixed-2022 control）使用同一规则复核，共发现 24 个显著下降事件：Medium-2025 为 4/6 个（primary/control），Medium-2026 为 7/7 个；其中 reset epoch 为 0 个，epoch 51 以后为 0 个。该结果不是单一训练 seed 的偶然现象。

### epoch 49–52 与 79–82 原始观测

{table(windows, ['protocol_id', 'epoch', 'learning_rate_used', 'prototype_reset', 'val_loss', 'val_accuracy', 'val_macro_f1', 'delta_val_macro_f1', 'prototype_norm_after_train', 'prototype_norm_after_reset', 'prototype_norm_reset_delta', 'prototype_gradient_norm_mean', 'encoder_gradient_norm_mean'])}

prototype norm 在 reset 后明显跳变，这是重算 prototype 的直接结果；但梯度范数没有在 reset 当轮爆炸，且下一轮/随后数轮验证指标恢复。scheduler 降低 LR 后，大幅下降事件反而消失。

### 显著下降时类别 Recall 的最大变化

{table(hardest, ['protocol_id', 'drop_epoch', 'class', 'support', 'previous_recall', 'drop_epoch_recall', 'delta_recall'])}

下降主要表现为少数易混类别的决策边界短暂切换，而不是所有类别同步失效。训练 loss、梯度范数与数值检查均未显示 NaN 或系统性爆炸；这更符合高 LR 阶段、类别不平衡和难类边界共同造成的验证瞬态。

## 6. 必须回答的问题

1. **Medium-2025 与 Medium-2026 差距的主要原因**：Known class composition。Medium-2025 同时包含高度混淆的 rsync/scp；Medium-2026 把 scp 留作 Unknown。相同 seed 对照仍保留 84.8% 的主差距，随机 seed 不是主因。
2. **validation dip 是否由 scheduler/prototype reset 引起**：没有支持“大幅 dip 由它们触发”的证据。显著 dip 没有发生在 reset epoch；LR 降低后显著 dip 消失。reset 与 LR 下降在同一轮共现，不能用本观察审计完全分离二者的微小即时效应，但可以排除其造成已观察大幅崩塌。
3. **dip 是正常动态还是新实现问题**：属于可恢复的已知训练瞬态。exact replay parity 通过、没有 NaN/崩溃、没有 reset/梯度异常对应，且后续达到更高验证分数。
4. **清理后 pipeline 能否冻结**：可以，需把高 LR 阶段 validation 波动作为已知 transient 记录，不应因单轮 dip 提前停止或修改协议。
5. **能否进入完整 15-run 重跑**：从本审计的训练完整性和稳定性证据看可以；本阶段本身没有启动该重跑。

## 最终结论

```text
{summary['conclusion']}
```

该结论不是声称训练轨迹无波动，而是确认波动可恢复、没有与 prototype reset 构成稳定因果对应，并且主要闭集差距可由冻结协议中的类别组合严格解释。
"""
    (OUT / "stage14c5_report.md").write_text(report, encoding="utf-8")
    print(OUT / "stage14c5_report.md")


if __name__ == "__main__":
    main()
