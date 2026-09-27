# Stage 27 — Frozen YaTC feature-level fusion with E3

Status: preregistered before Stage 27 model fitting or new Test evaluation. This is a corrected interpretation of the user's intended fusion; Stage 26 probability averaging remains a separate, immutable diagnostic.

## Scope and data

- Exact frozen Stage 20 ISCX-VPN and ISCXTor2016 flow IDs, original 6/7 Service labels, seeds 2022 and 2023.
- Preserve the Stage 22 E1/E2/E3 representations and Stage 23 YaTC best checkpoints. No encoder weight update.
- Extract YaTC's `forward_features` output (192-dimensional penultimate flow vector), not its logits/probabilities. For every role, assert exact flow-ID order and labels against Stage 22.
- Fit each YaTC feature coordinate's z-score on Known Train only; Stage 22 E3 coordinates already have branch-wise Known-Train z-score. Apply the same statistics to Known Validation/Test.
- Train heads on Known Train only. Select each head's best epoch by Known Validation fine-class Macro-F1. Open Known Test only after selection. Unknown data use = 0.

## Fixed variants

1. **F1-LinearConcat:** concatenate frozen E3 (768 TrafficFormer + 128 FIG/TAGCN) and standardized YaTC (192); train one linear classifier on 1088 features. This tests added YaTC information with the same head family as E3.
2. **F2-EqualProjected:** project each of E1 (768), E2 (128), and YaTC (192) through its own Linear→ReLU layer to 128 dimensions, average the three projected vectors equally, then use one linear classifier. This controls for nonlinear projection capacity.
3. **F3-FeatureGate:** use the same three projectors and classifier; a 384→64→3 ReLU MLP and softmax computes per-flow nonnegative weights summing to one before the weighted feature sum. No logits or model probabilities enter the gate.

F2 and F3 have the same branch features, projection width, loss, optimizer, epochs and checkpoint rule; only the data-dependent gate differs. The F3 gate is an independent feature-level adaptation, not a literal ER-CMGI entropy/diffusion reproduction.

All heads: cross-entropy; Adam learning rate 1e-3, zero weight decay; batch size 256; 30 complete epochs; torch/NumPy seed equals protocol seed; no early stopping, class reweighting, scheduler, or architecture search. Record each epoch, predictions, per-class metrics, gate weights and checkpoint SHA256.

## Evaluation

- Primary: full-flow Known Test original fine-label Macro-F1; secondary: fixed Stage 25 Communication coarse remap and Accuracy/Weighted-F1. Report both seeds individually and mean±std. Retain E3 and YaTC single-model baselines plus Stage 26 probability ensemble only as separate context.
- Paired differences: F1−E3, F1−YaTC, F2−F1, F3−F2, F3−YaTC, with class-level error rescue and worst-run cost. No candidate chosen or redesigned from Test results.
- Check original E3 and YaTC checkpoint/hash identity before and after, exact YaTC `head(forward_features(x))` parity with frozen classifier inference, finite features, and no role/sample mismatch.

Stage 20 Test has already been viewed in earlier stages. These Stage 27 Test measurements are development evidence only; they cannot establish independent confirmation or justify another formula/width search on the same Test.
