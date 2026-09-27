# Stage 31 — Four-dataset frozen-protocol three-view equal-fusion retest

Frozen on 2026-09-25 before any Stage31 encoder training or Test evaluation. User selected **existing frozen Known Train/Validation/Test**, not a new all-class split. This first pass is a four-dataset **representative-protocol pilot** using the five previously registered Stage15R protocols; it is not the full 15-protocol VNAT or USTC result. Historical Stage30 and all source protocols remain read-only.

## Frozen pilot cells and target

| Dataset | Existing protocol | Rationale |
|---|---|---|
| ISCX-VPN | medium-2022 | Previously registered mid-openness task |
| ISCXTor2016 | medium-2022 | Previously registered mid-openness task |
| VNAT | Medium-2025 | Previously identified hard Known composition |
| VNAT | Medium-2026 | Previously identified normal Known composition |
| USTC-TFC2016 | A-2, seed 2022 | Previously registered middle-openness positive control |

Take sample IDs, class lists, labels, and split roles *exactly* from each frozen protocol. Unknown classes must not enter any encoder/adaptor/head fitting, feature normalization, checkpoint selection, or threshold setting. Train and select only on Known Train/Validation. Open Known Test only once the entire training/selection recipe and all five checkpoints are frozen; no Unknown Test is needed for this closed-set experiment. Report Test results as development evidence because prior project experiments have exposed some Test data.

The unchanged **method identity** is Stage30 `T0_equal`: TrafficFormer 768-D, FIG/TAGCN 128-D, YaTC 192-D; one 64-D Gaussian adapter per view (mean used for deterministic inference); fixed `(1/3,1/3,1/3)` weight on the three 64-D means; weighted concatenation; same width-128 two-layer classifier. Adapter and head budgets/selection remain Stage30's 30 epochs, batch 256, Adam 1e-3, Known-Val Macro-F1. Do not substitute Open-Detect, image features, packet-length CNN, LightGBM, or Stage27's 128-D projected-and-summed F2. Do not retune weights, architecture, losses or split after Test.

For protocols lacking frozen aligned three-view embeddings, recover the exact flow-level packet inputs from existing project-local source/caches, and train each branch on that protocol's Known Train only. Reuse Stage22's official-pretrained TrafficFormer recipe (20 epochs, batch 64, LR 6e-5, Val Macro-F1 selection) and FIG/TAGCN recipe (50 epochs, batch 64, LR 1e-3, Val Macro-F1 selection), and Stage23's YaTC official-pretrained recipe (200 epochs, batch 64, AdamW with author schedule, Known-Val selection). If a protocol's raw flow cannot be reconstructed with exact frozen flow-ID alignment, mark that cell blocked with a count and reason; **do not silently drop flows, relabel, replace views, use an incompatible historical checkpoint, or report a partial sample set as matched**. Hash source inputs and record all config/weight lineage.

## Required gates before GPU work

1. Audit five frozen protocol manifests and the available raw flow/packet cache for each of the three views. Exact flow-ID and class-order alignment, uniqueness, and Train/Val/Test disjointness must pass.
2. Quantify missing per-view samples and existing checkpoints. Training starts only after all three views can cover the same Known flows within each cell without changing the frozen split.
3. Record the cost and device estimate. At most three physical GPUs may be occupied by Stage31, each selected from live capacity immediately before its workload.

## Outputs and reporting

Preserve failed and successful attempts, training curves, selected checkpoints and SHA256, per-sample ID/label/prediction, class metrics, run metrics (Accuracy, Macro-F1, Weighted-F1), configuration, source hashes, independent replay, `RESULTS.md` and `manifest.json`. Summarize the five pilot cells individually. Do not average fundamentally different label spaces into one four-dataset accuracy. A four-dataset result requires all four datasets to have completed matched-sample Test evaluations; otherwise report the exact partial status. No open-set conclusion follows from closed-set performance.

## Input-recovery incident and preregistered compatibility handling

Before Stage31 GPU training or Test evaluation, VNAT Medium-2025 raw-input recovery exposed two cases not accepted by the initial extractor: Stage14A `other:<hash>` IP flows (not TCP/UDP streams), and selected IPv6 packets rejected by the IPv4-only YaTC MFR converter. The first failed attempt and second failed attempt remain in separate cache directories and tmux logs. Stage31 computes `other:` IDs using the original Stage14A canonical endpoint hash; it does not drop these flows. For IPv6 only, the Stage31 MFR adapter applies the same fixed 80-byte network-header and 240-byte Raw-payload truncation/padding rule to the IPv6 layer, retaining the 40×40 tensor. Each adapted flow and packet is counted and the result is explicitly **YaTC-input-adapted**, not strict official YaTC. The architecture, Known split, and all labels stay frozen. Any remaining unrepresentable flow fails the cell; it is never silently removed. This exception is fixed before any downstream Stage31 fitting and is not selected on Validation/Test performance.
