# DQ-3F Preregistered Protocol

Status: `PREREGISTERED_NOT_RUN`  
Claim scope: Known-only closed-set diagnostic on weak capture labels.

## Frozen population

- Dataset: shared 31 ISCX-VPN VPN PCAPs.
- Eligibility: all and only the 3065 frozen `medium_seed2022` Known Train/Validation flows from those PCAPs.
- Membership: Train=2730, Validation=335.
- F1 and F3 must use exactly the same flow IDs and membership; no filtering by Service, packet count, CATE status or TrafficFormer eligibility.
- Known Test and Unknown Test feature usage: 0.

## Tasks

- **F1:** train Native on Fine Application labels; evaluate all Validation flows with Accuracy, Macro-F1, Weighted-F1 and per-class F1.
- **F2-PARTIAL:** map F1 predictions only where both true and predicted Application have a singleton Service mapping. Exclude ambiguous outputs explicitly and report coverage. F2 is diagnostic and cannot substitute for F3.
- **F3:** train Native directly on capture-activity Service labels for all identical F1 flows; evaluate Accuracy, Macro-F1, Weighted-F1, per-Service Recall and confusion matrix.

## Frozen Native configuration

- architecture=`resnet18`, channels=1, latent_dim=128
- epochs=100, batch_size=128, eval_batch_size=256
- optimizer=Adam, learning_rate=0.001, betas=[0.9, 0.999]
- Open-Detect lambda=0.005
- MultiStepLR milestones=[50, 80], gamma=0.1
- prototype reset zero-based epochs=[50, 80]
- checkpoint criterion=`Known Validation accuracy`
- early stop patience=10, min_epoch=82, min_delta=0.0001
- seed=2022

Only output dimension/label encoding changes between F1 and F3. Architecture, input bytes, loss family, optimizer, budget, seed and membership remain fixed.

## Gates and interpretation

- DQ-3F may quantify whether direct Service training is easier under the frozen flow-level split.
- It may **not** establish capture-generalized performance because P2P has one capture and the existing Train/Validation groups overlap.
- It may **not** establish an Open-Set method or justify changing DES/H1.
- A later formal Service benchmark requires a separately frozen held-out-Service protocol and group feasibility resolution.
