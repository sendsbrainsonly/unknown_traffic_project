# Stage 28 paper-to-code boundary

This experiment tests selected ideas from *Entropy-regulated cross-modal generative fusion for multimodal network intrusion detection*. It is **not** an author-equivalent reproduction.

| Paper element | Stage 28 implementation | Boundary |
|---|---|---|
| Raw byte and packet-length branches | Frozen TrafficFormer+YaTC content view (960-D) and frozen FIG behavior/structure view (128-D), with Train-only z-score | Different inputs and upstream encoders; FIG also contains time, direction and burst cues |
| Gaussian latent `μ, log σ²`, KL and reparameterization | `AdapterPair` in `run_one.py`: two 64-D adapters, clamp `[-8,4]`, CE + `0.05 KL`; stochastic training, deterministic `μ` inference | Small adapter to frozen features, not a raw-input probabilistic encoder |
| Cross-modal generation | A1 adds two small MLPs predicting Train-only PCA-64 targets; `0.1` normalized MSE | Diagnostic consistency proxy, **not** the paper's diffusion model |
| Differential entropy | Per-sample Gaussian entropy `0.5 log(2πe) + 0.5 mean(log σ²)` after adapter freeze | Operational entropy of our adapters, not measured epistemic uncertainty |
| High-/low-entropy mixing | Positive softplus of Train-only median/MAD standardized entropy; reciprocal low term; learned per-view `λ∈[0.1,0.9]` | Explicit adaptation of the paper's Eq. 12 idea; Eq. 13 batch-mean ambiguity is not silently copied |
| Feature fusion | Weighted latent **concatenation**, two-layer 128-D ReLU MLP and class head | Unlike Stage 27's projected feature sum/gate; fixed equal-weight control shares classifier capacity |
| Selection and testing | Known Train fitting, Known Validation Macro-F1 checkpoint selection; no Test/Unknown access | Only Known-Val development evidence; the Stage 20 Test was already exposed in earlier work |

For exactly two views, `high_b + low_b = 1` and `high_s + low_s = 1`. In this formula, `λ_b + λ_s = 1` makes normalized fusion weights exactly `0.5/0.5` for every sample, regardless of its entropy. The pre-registered initialization is on that cancellation line. The experiment therefore records learned coefficients, weights, forced-equal and shuffled-entropy counterfactuals, and does **not** equate a small P3 gain with proof of entropy-guided reliability.

The source PDF SHA256 recorded in the frozen plan is `770d501607f075084741d8178c4355ac6b7ea2a1f8d9abd1601a5dbac56bf060`; the extracted text and page-render audit are project-local under `.artifacts/pdf/`.
