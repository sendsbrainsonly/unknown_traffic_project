# Stage 33 — VNAT 六类闭集诊断（单 seed）

Stage 32 的四类 Known Test 满分已被查看；本六类标签决定因此是**事后开发诊断**，不是 untouched Test 或预注册独立验证。保留 Stage 14B `medium_seed2025` 的 15,704/1,960/1,960 个 Known Train/Val/Test flow、七个 Known application、三个 frozen branch checkpoint 和 seed 2022，不读取 Unknown 特征。三路 T0 训练配置直接复用 Stage 32：Train-only 标准化，30 epoch adapter + 30 epoch head，Known-Val Macro-F1 选最佳权重。

| 六类标签 | 原 application |
|---|---|
| streaming | netflix + youtube |
| rdp | rdp |
| rsync | rsync |
| scp | scp |
| skype | skype |
| ssh | ssh |

这是六类而不是将任何 Unknown application 加入 Known；`zoiper/vimeo/sftp` 仍完全排除。仅合并两种流媒体应用，显式暴露 `rsync`–`scp` 和 `rdp`–`ssh` 的应用级混淆。RDP 在 Val/Test 各只有 4 个 flow，须报告逐类支持数和不稳定性。不能将本轮六类 Macro-F1 与旧四类 Macro-F1 的数值差写成方法性能变化；两者任务不同，且 VNAT 按 flow 随机划分，可能共享 capture。

按 `preflight → train → evaluate` 顺序运行 `run_six.py`。Test 特征仅在训练 checkpoint 经 Known Val 选定和 SHA256 校验后读取；Test 用 Stage 32 已冻结的 Known Test input cache 重新执行分支推理，不修改其文件。
