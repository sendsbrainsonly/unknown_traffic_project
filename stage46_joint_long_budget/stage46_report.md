# Stage46 — 联合训练100epoch预算结果

Status: complete / INDEPENDENT_REPLAY_PASS

同一100epoch调度轨迹的20/50/100预算；Known-Val选模，六选点冻结后才Test。单seed/已暴露Test，不能宣称等FLOPs或纯epoch因果效应。

| Dataset | Budget | Best epoch | Role | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---|---:|---:|---:|
| iscx_vpn | 20 | 6 | known_validation | 0.877049 | 0.880098 | 0.875882 |
| iscx_vpn | 20 | 6 | known_test | 0.878317 | 0.884922 | 0.877149 |
| iscx_vpn | 50 | 6 | known_validation | 0.877049 | 0.880098 | 0.875882 |
| iscx_vpn | 50 | 6 | known_test | 0.878317 | 0.884922 | 0.877149 |
| iscx_vpn | 100 | 6 | known_validation | 0.877049 | 0.880098 | 0.875882 |
| iscx_vpn | 100 | 6 | known_test | 0.878317 | 0.884922 | 0.877149 |
| iscx_tor | 20 | 18 | known_validation | 0.858676 | 0.862831 | 0.859516 |
| iscx_tor | 20 | 18 | known_test | 0.847807 | 0.851394 | 0.848633 |
| iscx_tor | 50 | 36 | known_validation | 0.865832 | 0.867322 | 0.867253 |
| iscx_tor | 50 | 36 | known_test | 0.847807 | 0.851864 | 0.849235 |
| iscx_tor | 100 | 36 | known_validation | 0.865832 | 0.867322 | 0.867253 |
| iscx_tor | 100 | 36 | known_test | 0.847807 | 0.851864 | 0.849235 |

## 历史同流参照（训练预算/调度不同）

- iscx_vpn Stage37_original_J20: Test Accuracy/Macro/Weighted-F1=0.878317/0.882180/0.875811
- iscx_vpn Stage32_staged: Test Accuracy/Macro/Weighted-F1=0.915828/0.919350/0.914487
- iscx_tor Stage37_original_J20: Test Accuracy/Macro/Weighted-F1=0.847807/0.853184/0.850408
- iscx_tor Stage32_staged: Test Accuracy/Macro/Weighted-F1=0.896150/0.899375/0.896717

## 限制与结论

100epoch warmup/decay比历史20epoch拉长，且使用修复后的原子GradScaler；不能将历史差异全部归因于轮数。Train在线指标含训练模式dropout，不与eval指标直接等同。

## 训练曲线来源说明

首次Stage46尝试曾在用户要求释放GPU时中断；原始`runs/*/training_history.jsonl`保留了该次尝试的不完整末段。正式100epoch结果已从epoch75完整恢复并完成，但本报告生成器的默认曲线读取原始历史文件。为避免把中断尝试的曲线误当成完整100epoch轨迹，请使用`resume_attempt1/figures/`及对应`resume_attempt1/combined/*_training_history.jsonl`作为完整曲线；它们由原始epoch1–75与精确恢复的epoch76–100拼接，并逐epoch核对数量。根目录自动生成的两张曲线不作为最终曲线引用。

结论：`LONGER_BUDGET_NOT_CONSISTENTLY_BETTER`。只回答当前固定配方；无多seed稳定性结论。不自动启动下一实验。
