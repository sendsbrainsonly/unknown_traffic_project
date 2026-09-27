# Stage 26 — Frozen YaTC + E3 probability ensemble

Recorded before Stage 26 probability extraction and metrics. This is development work on previously exposed Stage20 Test, not untouched validation.

## Inputs and integrity

- Same Stage20 frozen `known_validation`/`known_test` flows, service labels and roles; datasets ISCX-VPN and ISCXTor2016, seeds 2022/2023.
- Frozen Stage22 E3 896-D representations and `E3_model_best.pt`; frozen Stage23 YaTC `model_best.pt` plus exact Stage23 MFR cache and author-consistent CUDA autocast inference.
- No encoder or classifier weight update. Before/after SHA256 checks on the Stage20 manifest, four E3 checkpoints/representations, four YaTC checkpoints, cache metadata and source prediction files.
- Extract fine-class logits/probabilities for both models. Exact argmax parity with the existing Known Validation **and** Known Test prediction CSV is mandatory. If parity fails, stop that run and preserve the failure; do not adapt an inference path to the Test result.

## Single fixed fusion

For each exact same flow, fine Service probability vector:

`p_fusion = 0.5 * softmax(logits_E3) + 0.5 * softmax(logits_YaTC)`.

No weight, temperature, confidence gate, class-dependent rule or threshold search. Fine prediction = argmax of this vector. Coarse probability for `Communication` is the sum of the Chat, Email and VoIP coordinates; other Service probabilities are unchanged. Coarse prediction = argmax of the coarse vector. The original fine model remains available and unmodified.

Evaluate E3, YaTC and fusion on the same flows, separately for original 6/7 fine and Stage25 4/5 coarse labels. Also report the preregistered ≥2-packet conditional subset, but the all-flow task is primary. Known Validation metrics are diagnostic only; do not select a candidate or alter fusion based on them. Known Test is used once for a development evaluation, not for fitting.

## Decision and limitations

Report Accuracy, Macro-F1, Weighted-F1, per-class F1, individual seeds, mean ± population std, prediction-level rescue and degradation, runtime and model-storage cost. Compare fusion against **both** component models. The Stage25 **full-flow coarse task is primary**; original fine task is an ability guardrail. Call it a useful coarse precision improvement only if full-flow coarse Test Macro-F1 is higher than YaTC on both datasets and positive on at least 3/4 dataset-seed cells, without a material fine-task regression; with only two seeds, this remains exploratory. Otherwise retain YaTC as the stronger baseline and label fusion as partial/failed. Never claim an architecture-level causal effect from this ensemble.

No open-set evaluation, Unknown data, new pretraining, hyperparameter search, sample removal or modification of Stage20–25 assets.
