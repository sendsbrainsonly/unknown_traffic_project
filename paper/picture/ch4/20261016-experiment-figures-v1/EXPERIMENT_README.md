# Chapter 4 experiment figures

- Artifact ID: `paper-figures-20261016-v1`.
- Status: `complete`; accepted outputs: `render-v4/`.
- Display compatibility fix (2026-10-16): manuscript image references now use paths relative to `paper/md/ch4/1.md`, resolving the reproduced VS Code broken-image issue. See [repair and verification](display-fix-20261016/README.md). The original `ARTIFACT_MANIFEST.json` records the earlier generation snapshot; subsequent changes to this README and the verifier are documented in the repair manifest.
- Created: 2026-10-16 21:11 Asia/Shanghai.
- Owner/route: Codex, chat `01a144b7-de8c-7521-a7ae-d8575754129d`.
- Objective: Visualize already reported experiments and integrate seven figures into `paper/md/ch4/1.md`.
- Claim scope: Derived visualization only; no new model training, inference, threshold fitting, dataset sampling, or experiment result is generated.
- Parent evidence: existing Stage23B, Stage32, Stage41, Stage42S–Y, Stage43, Stage44 and Stage44B results. Their source identities and hashes will be recorded in each render's source manifest.

## Figures and files

七幅图已嵌入[第4章正文](../../../md/ch4/1.md)，含中文图注与正文分析。论文排版优先使用 PDF；SVG 保留可编辑文字及矢量图形；PNG 为 300 dpi。下表仅链接最终采用的 render-v4。

| 图号 | 内容 | 预览 | 矢量 PDF | 可编辑 SVG |
| --- | --- | --- | --- | --- |
| 4-1 | 七方法闭集 Macro-F1 与双种子标准差 | [PNG](render-v4/fig4_1_closed_baselines.png) | [PDF](render-v4/fig4_1_closed_baselines.pdf) | [SVG](render-v4/fig4_1_closed_baselines.svg) |
| 4-2 | 当前三视图与 E3／YaTC 同流分类对照 | [PNG](render-v4/fig4_2_current_closed.png) | [PDF](render-v4/fig4_2_current_closed.pdf) | [SVG](render-v4/fig4_2_current_closed.svg) |
| 4-3 | VNAT 四个服务留出任务的系统对照 | [PNG](render-v4/fig4_3_vnat_matched.png) | [PDF](render-v4/fig4_3_vnat_matched.pdf) | [SVG](render-v4/fig4_3_vnat_matched.svg) |
| 4-4 | VNAT 同表示拒识分数消融热图 | [PNG](render-v4/fig4_4_vnat_ablation.png) | [PDF](render-v4/fig4_4_vnat_ablation.pdf) | [SVG](render-v4/fig4_4_vnat_ablation.svg) |
| 4-5 | CIC 十类未知攻击识别差异 | [PNG](render-v4/fig4_5_cic_attack_profiles.png) | [PDF](render-v4/fig4_5_cic_attack_profiles.pdf) | [SVG](render-v4/fig4_5_cic_attack_profiles.svg) |
| 4-6 | CIC 未知比例与组成敏感性 | [PNG](render-v4/fig4_6_cic_mixture_sensitivity.png) | [PDF](render-v4/fig4_6_cic_mixture_sensitivity.pdf) | [SVG](render-v4/fig4_6_cic_mixture_sensitivity.svg) |
| 4-7 | USTC 三配置的匹配未知识别对照 | [PNG](render-v4/fig4_7_ustc_matched.png) | [PDF](render-v4/fig4_7_ustc_matched.pdf) | [SVG](render-v4/fig4_7_ustc_matched.svg) |

[全部图件缩略预览](render-v4/overview.png)；[绘图数据 CSV](render-v4/plot_data.csv)；[完整数据及来源 JSON](render-v4/plot_data.json)；[绘图脚本](generate_figures.py)；[核验脚本](verify_figures.py)；[数值核验回执](render-v4/verification.json)；[视觉核验记录](render-v4/VISUAL_QA.md)。

## Evaluation scope

1. Seven-method closed-set Macro-F1 on VPN6/Tor7: mean ± population SD across seeds 2022/2023.
2. Current three-view model versus matched E3/YaTC configurations on VPN4/Tor5, seed 2022.
3. VNAT four-fold matched Open-Detect versus DES-v1: AUROC, Unknown F1, UFAR and Known FRR at Known-Val P95.
4. VNAT same-representation score ablation by held-out service.
5. CIC ten Unknown attacks: MSP versus DES-v1 AUROC and DES-v1 UFAR; include all ten classes.
6. CIC fixed-score prevalence and composition diagnosis: means over 20 deterministic sample draws, not 20 training seeds.
7. USTC A-1/A-2/A-3 matched 1:1 comparison at Known-Val P95.

## Artifact and recovery policy

- The reusable Python script lives here. Every render uses a new, create-only subdirectory and retains a script snapshot, plot data, input paths/hashes and PNG/PDF/SVG exports. The accepted render additionally has numeric and visual verification receipts. Earlier attempts remain labeled and preserved.
- Values are extracted from original CSVs where available and from retained report tables otherwise. Do not silently increase reported precision or synthesize missing per-sample scores.
- Local environment: Python 3.12, matplotlib 3.11.2, NumPy, Pillow. No GPU or remote operation.
- Figure labels use English for consistent portable typesetting; manuscript captions explain protocol and interpretation in Chinese.
- Source datasets/checkpoints remain in their original experiment directories; they are not required for this summary-level visualization.
- Completed: seven figures, 21 final exports, 173 metric records, 14 reported SDs, 13 source files, 128 manuscript numeric checks, seven valid image references and eleven unchanged table identities. No requested figure work remains.

## Reproduction

From the project root, generate a new directory using the frozen plot data. The script refuses to overwrite an existing directory:

```bash
python3 paper/picture/ch4/20261016-experiment-figures-v1/generate_figures.py --data paper/picture/ch4/20261016-experiment-figures-v1/render-v4/plot_data.json --out paper/picture/ch4/20261016-experiment-figures-v1/render-reproduction-01
```

Omit `--data` to extract the existing source reports and CSVs again. The source manifest hashes must still match for frozen-data rendering. No experiment models or GPUs are needed. To recheck the currently embedded figures without overwriting the prior receipt:

```bash
python3 paper/picture/ch4/20261016-experiment-figures-v1/verify_figures.py paper/picture/ch4/20261016-experiment-figures-v1/render-v4 --output paper/picture/ch4/20261016-experiment-figures-v1/render-v4/verification-repeat-01.json
```

`plot_data.json` retains fractions at their source precision, protocol context and hashes. Figure numbers display percentages rounded half up to two decimals. Values taken from reports retain report precision; no extra precision is inferred.

## Render and verification history

- `render-v1`: completed, superseded. Visual review found mean labels intersecting SD lines and a binary-float midpoint giving 0.03 instead of the manuscript's 0.04 for slowloris UFAR. Source value remained 0.000350.
- `render-v2`: completed, superseded. Corrected label spacing and decimal rounding; all seven plots individually viewed. CIC left/right panels had different automatic vertical margins.
- `render-v3`: completed, superseded. Fixed CIC row alignment. SVG inspection identified embedded raster cells/colorbars in the heatmap, so a further vector export revision was made.
- `render-v4`: accepted. Replaced heatmap with vector meshes and disabled colorbar rasterization. All seven SVGs have editable text and zero image elements; all seven PDFs have zero image objects. Five PNGs are byte-identical to the individually reviewed v2 plots; final heatmap and CIC plot were viewed again directly.
- `verification-v1.log` / `verification-v2.log`: independent checker stopped because two historical reports label the field `Natural AUROC` rather than `AUROC`. Checker snapshots are preserved as `verify_figures_attempt1.py` / `verify_figures_attempt2.py`. These failures did not change plotted data.
- `verification-v3.log`: PASS after header normalization. All 173 metrics and 14 SDs match sources, 128 manuscript numeric checks pass, and exports/link/numbering checks pass.
- Each render's `manifest.json` preserves the original rendering-time status (`RENDERED_AWAITING_VISUAL_REVIEW`). Final acceptance is established by the later `verification.json`, `VISUAL_QA.md` and this README, rather than rewriting earlier evidence.

## Known interpretation limits

- Old VPN6/Tor7 baselines and current VPN4/Tor5 model are different label tasks; show them in separate figures.
- Population SD across two training seeds is not a confidence interval. Single-seed figures have no fabricated uncertainty bars.
- VNAT folds are service scenarios, not independent random seed repetitions. CIC sampling repeats quantify frozen-pool mixture variability only.
- Some original prediction/score CSVs are absent locally. Summary figures cannot reconstruct ROC/PR curves, feature-space scatter plots or confusion matrices without the corresponding underlying data.
- Existing model/pretraining/adaptation and development-set limitations remain as stated in the manuscript.
