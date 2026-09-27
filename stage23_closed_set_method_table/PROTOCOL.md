# Stage 23 — matched closed-set method table

This is a new diagnostic benchmark, not a revision of Stage 20–22 or an
author-exact reproduction of the other methods.

## Frozen comparison population

- ISCX-VPN: Stage 20 Service-6, 10,955 flows, Known Train/Val/Test 8,764/1,098/1,093.
- ISCXTor2016: Stage 20 Service-7, 11,181 flows, 8,946/1,118/1,117.
- Membership, labels, exact-image grouping and roles come only from
  `../stage20_dual_coarse_service_protocol/closed_service_manifest.csv`.
- Seeds 2022 and 2023 are the existing Stage 22 paired scope. Do not change
  sample selection for a method. If an input cannot be reconstructed for a
  frozen flow, report its coverage and do not silently remove that flow.
- Checkpoint selection and any fitted preprocessing use Known Train/Validation
  only. Known Test is loaded for the final evaluation after selection.
- Report Accuracy, Macro-F1, Weighted-F1, per-class metrics, all seed results
  and method-specific coverage. Scores from historical, nonmatching datasets
  are context only and never enter the paired table.

## Method identity

- Stage 22 E1 is the official-pretrained TrafficFormer branch.
- Stage 22 E3 is the project-specific pretrained TrafficFormer + FIG/TAGCN
  fusion, not the independent Open-Detect detector.
- The first new baseline is `Open-Detect corrected-paper`, reusing Stage 16S
  architecture, loss, augmentation, optimizer, scheduler, prototype reset and
  Known-Validation-accuracy checkpoint rule. It is not mislabeled as an
  unmodified released-code reproduction.
- YaTC, ET-BERT, TFE-GNN, RoNeTC and Trident require exact Stage 20 input
  adapters and parity audits before formal training. UnDiff is a one-class
  anomaly detector and is not natively eligible for a 6/7-way closed-set table.

## Limits

Stage 20 labels are capture-derived weak labels and its roles are not
uniformly capture-disjoint. Public pretrained corpora may overlap target
traffic; provenance is not fully established. These results cannot be called
unseen-capture or strict pretraining-Unknown-Free validation.
