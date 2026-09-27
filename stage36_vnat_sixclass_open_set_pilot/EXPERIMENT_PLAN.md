# Stage 36 — Frozen Three-View VNAT Application-Level Open-Set Pilot

## Question and frozen task

Does the completed Stage 33 three-feature-branch fusion model reject applications
that were entirely absent from its Known-only training? This is a development
pilot, not a new Service-LOSO experiment or untouched external validation.

Use exactly Stage 14B `medium_seed2025` and Stage 33 six-class checkpoints:

- Known applications: netflix, youtube, rdp, rsync, scp, skype, ssh.
- Known Train/Validation/Test: 15,704 / 1,960 / 1,960 frozen flows.
- Unknown Test only: sftp 1,668; vimeo 1,218; zoiper 939; total 3,825.
- Stage 33 class labels: streaming={netflix,youtube}, with rdp, rsync, scp,
  skype and ssh separate. No taxonomy or membership change.

`vimeo` shares the Streaming service with Known netflix/youtube and `sftp`
is related to Known file-transfer applications. Thus the question is
application-level novelty under the frozen six-class head, **not** whether a
new service is unknown. Treat the resulting difficulty as part of the task.

## Frozen inputs and role boundaries

Read Stage 14B protocol and split manifests and verify freeze hash
`c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
Verify all three Stage 31 branch checkpoint SHA256 values and both Stage 33
adapter/head checkpoint SHA256 values before and after inference. Reuse the
exact Stage 31 packet/graph/MFR definitions and Stage 33 Known-Train feature
scalers; validate the new extraction path against existing Known Validation
cache values before opening Unknown packet values. Do not train, fine-tune,
reselect or modify any checkpoint.

Known Train embeddings may fit empirical centroids and class-conditional kNN
support. Known Validation alone may fit robust score normalization and P95
thresholds. Known Test and Unknown Test are used once for final evaluation.
Unknown may not enter fitting, normalization, checkpoint selection, score
selection or threshold calibration.

## Preregistered scores (larger means more anomalous)

All scores use the frozen Stage 33 fused classifier. Let `h(x)` be its 128-D
feature **after** the final fusion MLP and before the class logits.

1. `MSP`: `1 - max softmax(logits)`.
2. `Energy`: `-logsumexp(logits)` with fixed temperature 1.
3. `Centroid`: `min_y ||h(x)-c_y||^2`, with `c_y` the Known-Train class mean.
4. `DES-v1-like`: nearest empirical centroid class `y*`, global
   `d_g=min_y ||h-c_y||^2`, local `d_l=mean distance to the 10 nearest
   Known-Train h of class y*`. Known-Val median/MAD calibrates each distance,
   `Z=(d-median)/(MAD+1e-8)`; score is fixed `0.5 Z_g+0.5 Z_l`.

These are frozen feature-based diagnostic scores, not the historical
Open-Detect-native KL/prototype score. Do not choose a winner or alter formulas
after viewing Unknown outcomes.

For each score, the operating threshold is **Known Validation P95** with
NumPy `method="higher"`; classify unknown iff score is strictly above it.
Unknown is the positive class. Report AUROC, AUPRC, UFAR, Known FRR, Known
Acceptance, and per-Unknown-application values. Preserve all sample IDs,
truth, scores, thresholds and decisions.

## Execution and claim limits

Preflight and extractor parity read no Unknown packet values. Open Unknown
only after source/checkpoint/formula guards pass. The Stage 14B freeze and all
historical Stage 31–33 assets remain read-only. GPU inference launches through
the live selector and waits for both Stage 34 and Stage 35 terminal verification.
The current conditional launcher uses physical GPUs 0/1 only; GPU2 remains
released before either interpretation of the user's next-morning 08:00 deadline.

Stage 33's six-class taxonomy followed an exposed four-class Test result, and
VNAT's flow-random split can share captures between roles. These findings are
development evidence only. This single protocol cannot establish multi-seed,
cross-capture or multi-dataset generalization. Stage 20 ISCX Service-LOSO
requires new Unknown-Free encoder training; do not reuse the all-Service
Stage 32 representations for that task.
