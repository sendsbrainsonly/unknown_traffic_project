# Stage 7 CipherSpectrum Known-only Encoder Training — Formal Results

## Outcome

Low、Medium、High 三个 Open-Detect encoder 已按 Stage 6 冻结协议独立从零训练完成。三档均只读取 `KNOWN_TRAIN` 与 `KNOWN_VALIDATION`；未打开或使用 Known Test/Unknown PCAP，未执行 Unknown inference。

## Core results

| Setting | Known / excluded Unknown classes | Known Train | Known Validation | Best / stop epoch | Validation Accuracy | Validation Macro-F1 | Harmonic composite |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 38 / 2 | 75,692 | 16,242 | 26 / 31 | 0.769179 | 0.769144 | 0.769161 |
| Medium | 34 / 6 | 56,641 | 12,168 | 22 / 27 | 0.715319 | 0.704509 | 0.709873 |
| High | 30 / 10 | 49,976 | 10,740 | 22 / 27 | 0.746741 | 0.729661 | 0.738102 |

最佳 checkpoint 由 Known Validation Accuracy 与 Macro-F1 的调和平均数选择，连续 5 轮无提升后早停。三个正式运行的数值稳定性保护均触发 0 次。

## Frozen checkpoints

| Setting | Checkpoint | SHA-256 |
|---|---|---|
| Low | `runs/formal-low-20260913/artifacts/training_best_checkpoint.pt` | `78e23b0d9795a7829c03d740c67c5a823ca9f6ea36bda4cd0e24338a168b4d92` |
| Medium | `runs/formal-medium-20260913/artifacts/training_best_checkpoint.pt` | `59858ee8d9dcc0d36e64a8a396fcf0f401b720dbed8d2aa2f783761be352ac48` |
| High | `runs/formal-high-20260913/artifacts/training_best_checkpoint.pt` | `588c9eeec5d352f28b5598163aed193e01e53b8a408cdbf6ddbbaf58dbaae213` |

## Reproduction and verification

正式输入构建、训练和归档的会话名分别记录在每个 run bundle 的 `RESULTS.md` 与 `manifest.json`。整体复核命令：

```bash
python stage7_cipherspectrum_known_training/scripts/verify_stage7.py
```

该验证器不会读取 PCAP；它校验 16 个 Stage 6 协议哈希、冻结数量、输入数组和标签域、训练指标、早停轨迹、checkpoint 哈希及 `(num_known_classes, 128)` 原型形状。

## Interpretation boundary

这些结果只证明三档 Known-only encoder 训练成功并通过 Known Validation checkpoint selection。它们不是 Known Test 结果，也不是 CipherSpectrum Unknown Detection 结果，不能据此声明外部 open-set 性能。进入下一阶段前应继续冻结这三个 checkpoint 和后续比较/校准规则。
