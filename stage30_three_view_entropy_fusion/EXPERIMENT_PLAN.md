# Stage 30 — True three-view entropy-fusion diagnostic

Frozen before training, 2026-09-25. This is a **Known-only development experiment** inspired by the ER-CMGI entropy rule, not an author-equivalent reproduction.

## Question and fixed data

Does per-sample high-/low-entropy feature weighting help when the Stage 27 inputs are kept as **three independent views**, rather than combining TrafficFormer and YaTC as Stage 28 did?

Views are frozen TrafficFormer `z_t∈R^768`, FIG/TAGCN `z_g∈R^128`, and YaTC `z_y∈R^192`. Use exact Stage 20 ISCX-VPN/ISCXTor known-flow Train/Validation IDs and fine Service labels, with the four Stage 22/27 encoder pairs (2022/2023). YaTC is a frozen feature source, not a trainable model in this task. The Stage 20 Known Test was previously exposed and is **not opened** here; Unknown usage is 0. No encoder fine-tuning, no changed split, no feature filtering. Check source SHA256 and exact flow-ID order before/after.

## Pre-registered models

Each view has its own 64-D Gaussian adapter producing `μ_m, log σ²_m`, clamp `[-8,4]`, trained together with per-view CE (mean) + 0.05 per-dimension KL. Use stochastic reparameterization for Train and deterministic `μ` for Validation. Train 30 full epochs, batch 256, Adam 1e-3; select adapter checkpoint by the mean of the three Known-Val unimodal Macro-F1 values. This is the same diagnostic probability adapter idea as Stage 28; no diffusion or cross-view generator is added.

After freezing adapters, fit each entropy median/MAD on **Known Train only**. Per-sample Gaussian entropy is `H_m=.5 log(2πe)+.5 mean_d(log σ²_m)`. Define `u_m=softplus((H_m−median_m)/(MAD_m+1e-6))+0.1`, `high_m=u_m/Σu`, `low_m=(1/u_m)/Σ(1/u)`. For the dynamic head, each of the **three** views gets a separate learned `λ_m=.1+.8 sigmoid(a_m)`, initialized at 0.5, and `r_m=λ_m high_m+(1−λ_m)low_m`, `w_m=r_m/Σr`. Unlike the two-view case, the 3-view rule does not algebraically reduce to sample-independent equal weights at the initialization.

Compare exactly two new heads on the **same** adapter features:

- `T0_equal`: fixed weights `(1/3,1/3,1/3)`.
- `T1_three_entropy`: above sample-level entropy weights.

Both use weighted **concatenation** of the three 64-D `μ` vectors, followed by the same two-layer width-128 ReLU/Dropout-0.1 MLP and linear classifier. Train 30 full epochs, batch 256, Adam 1e-3; select by Known-Val fine Macro-F1. Use new head seeds 2022–2026 for each frozen encoder pair: **20 paired units**. Preserve every run and failed attempt.

Read the existing Stage 27 three-branch `F1/F2/F3` Known-Val scores only as frozen historical references on the same flows. They are not matched five-seed retrains and are not the primary causal control. Primary comparison is `T1−T0` per exact run.

## Mechanism checks and gate

Record every epoch loss, Macro-F1, λ, weight mean/quantiles and collapse rate, and every validation flow's prediction, entropy and three weights. For each selected T1 checkpoint, evaluate the **same** head with forced equal weights and entropy shuffled between validation flows. Verify batch-size/permutation invariance. Save checkpoints, hashes and independently replay metrics/checkpoint predictions.

T1 supports the entropy mechanism only if **both datasets** have mean paired `T1−T0` Macro-F1 ≥ +0.005, at least 7/10 positive units, worst decline > −0.02, no negative mean Accuracy/Weighted-F1 change, and T1 beats its own forced-equal checkpoint on average with ≥14/20 positive units overall. Report comparison with Stage 27 F1/F2/F3, but do not use their previously exposed Test scores or tune this gate after seeing outcomes. If the gate fails, stop without formula/weight search or Test evaluation.

The five head seeds reuse two encoder pairs and validation flow pools per dataset; do not treat ten units as ten independent captures. This experiment answers feature-fusion behavior only, not open-set effectiveness or true mode count.
