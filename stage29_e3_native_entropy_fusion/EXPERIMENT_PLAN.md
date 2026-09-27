# Stage 29 — Native E3 two-branch entropy-fusion diagnostic

Frozen before Stage 29 training, 2026-09-25. Claim scope: **Known-only development diagnostic**, not ER-CMGI author-equivalent reproduction or untouched external validation.

## Question and input

The original Stage 22 E3 is **two branches**: pretrained TrafficFormer `z_t∈R^768` and FIG/TAGCN `z_g∈R^128`, Known-Train z-scored, concatenated into 896 dimensions and classified by a linear head. Three projected branches occur only in later Stage 27 E3+YaTC; they are not the native E3.

This experiment uses **only** frozen Stage 22 E3 `known_train`/`known_validation` branch representations and original Stage 20 Service labels for ISCX-VPN/ISCXTor, encoder seeds 2022/2023. YaTC feature, checkpoint and logits usage = **0**. Frozen source hashes and exact flow IDs/class order are checked before and after. Known Test and Unknown usage = **0**. The existing E3 linear head is a read-only reference, not refitted.

## Fixed method and controls

Fit two small 64-D Gaussian adapters to `z_t` and `z_g` using identical Known-Train CE + 0.05 per-dimension KL, 30 epochs, Adam 1e-3, batch 256. Select their common checkpoint by Known-Val mean of branch Macro-F1. Gaussian entropy is `0.5 log(2πe) + 0.5 mean(clamp(log σ²,[-8,4]))`; adapter inference uses deterministic `μ`. Fit per-view entropy median/MAD only on Known Train. Use the same two-layer 128-D ReLU/Dropout-0.1 MLP on weighted concatenation of two latent means, 30 epochs, Adam 1e-3, batch 256, selected by Known-Val fine Service Macro-F1.

Compare:

- `N0`: same adapters/MLP, fixed `w_t=w_g=0.5`.
- `N1`: same adapters/MLP, sample-level high/low entropy mixture, but **one shared learned** `λ∈[0.1,0.9]`, initialized at 0.5. Positive entropy transform and high/low definitions are exactly the Stage 28 adaptation. Sharing `λ` is an explicit correction for the two-view cancellation line found in Stage 28; it is not claimed to be the paper's original formula.
- `E3_original`: previously frozen linear-head Known-Val predictions on the same flows, read only.

No cross-modal generator is added here: this isolates dynamic fusion from the Stage 28 A1 consistency proxy. `N0/N1` are new heads on original E3 encoders, **not an end-to-end retraining of E3**.

Use five new head seeds 2022–2026 for each of four frozen dataset×encoder pairs (20 paired units). All runs are retained; no hyperparameter search, no Test-guided revisions. Batch-order invariance and `N1` forced-equal/shuffled-entropy counterfactuals are mandatory. Record losses, epochs, class metrics, predictions, weights, λ and hashes.

## Decision gate

`N1` supports the dynamic-fusion mechanism only if, on **both** datasets: paired `N1−N0` mean Known-Val Macro-F1 ≥ +0.005, at least 7/10 units positive, worst unit decline > −0.02, and mean Accuracy/Weighted-F1 do not drop. `N1` must also be at least as good as frozen `E3_original` mean and beat its own forced-equal checkpoint on average with ≥14/20 positive units overall. If not, report failure and stop. Repeated head seeds share validation flows and are not independent datasets.

Stage 20 Test was exposed in earlier work. This task does not read it, does not tune on it, and makes no independent-generalization or open-set claim. No YaTC or Stage 28 source/result is modified.
