# Stage 23B — frozen-flow adapted TFE-GNN and Trident

Pre-registered before any new Known Test evaluation, 2026-09-24.

## Population and decision rule

Use **every** Stage20 flow ID, Service label and Known Train/Validation/Test
membership: ISCX-VPN Service-6 (10,955) and ISCXTor Service-7 (11,181), seeds
2022 and 2023. Stage20 manifest SHA256 is
6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb.
Source PCAP, Stage12 provenance, and the earlier Stage23 table are read-only.
No flow may be dropped to make a method look native. Report Accuracy, Macro-F1,
Weighted-F1, per-class F1, full prediction files and paired differences to the
same-flow Stage23 TrafficFormer/E3 baseline. Select a checkpoint using Known
Validation Macro-F1 only; materialize/evaluate Known Test only afterwards.
Unknown data is never used.

## TFE-GNN-8-UDP-short (adapted, **not author-native**)

Retain the released two packet-byte graphs (header 40 bytes, payload 150 bytes,
byte PAD=256), four-layer GraphSAGE branch, cross-gated filter, bidirectional
LSTM and multiclass head. Construct both graphs from the *same* first at most
eight frozen packet references. Keep empty-payload UDP/TCP packets and UDP
packets; represent an empty payload/padding packet as a one-node PAD graph with
self-loop so all frozen flows remain defined. Anonymize IP addresses and
transport ports in the header branch. Zero- to eight-packet PAD is explicit.
The window differs from the author's 50 payload-bearing TCP packets, so any
score is an **8-packet all-protocol adaptation**, not a TFE-GNN paper result.
Keep released model dimensions and dataset-specific optimizer/training defaults
(VPN 20 epochs; Tor 100 epochs; Adam LR 0.01, cosine/warmup), without test-driven
tuning. Seeds replace author seed32 solely for paired Stage23 repeats.

## Trident-early8-86D (adapted, **not author-native**)

Use the existing project's reconstructed 86-D FlowStats definition. Retain
released per-class 256-128-64-32 autoencoders with MSE, Adam LR 0.001,
weight decay 5e-4, batch10, ten epochs. Compute the same directional counts,
bytes, IAT, TTL, TCP-window, flag and fragmentation statistics **only over the
frozen first up to eight packets**; absent TCP fields are zero. Keep UDP,
short and non-TCP/UDP flows. Fit StandardScaler on Known Train alone. Use the
local Trident v3 known-only loss-space RandomForest calibration (300 trees,
balanced, min leaf2), trained on a fixed class-stratified subset of Known
Train; the disjoint remaining Known Train fits the AEs. Known Validation
selects no other hyperparameters. This is a local reconstructed Trident
classification pipeline, not an official USTC author preprocessing claim.

## Integrity

The adapter records every flow ID, role, label and packet count, verifies
Stage20 and source-ref hashes, and fails if a reference is missing. Partial or
failed runs are retained. Prior Stage23 20 rows remain untouched. No result
from the old TFE/Trident datasets may enter this matched-flow table.
