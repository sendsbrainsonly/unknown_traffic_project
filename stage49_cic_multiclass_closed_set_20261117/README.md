# Stage49 — CIC-IDS-2017 eleven-class closed-set result

Completed, seed2022. Current three-view method: TrafficFormer, FIG/TAGCN and
YaTC encoders, existing adapters and equal feature fusion. Train/Validation/Test
contain99,922/8,438/11,688 flows, with disjoint five-minute source-PCAP groups.

| Test metric | Value |
|---|---:|
| Accuracy |98.8193%|
| Macro-Precision |98.2042%|
| Macro-Recall |99.0148%|
| Macro-F1 |98.5877%|
| Weighted-Precision |98.8421%|
| Weighted-Recall |98.8193%|
| Weighted-F1 |98.8232%|

- [Full report and per-class results](RESULTS.md)
- [Run-level CSV](run_results.csv), [per-class CSV](per_class_results.csv)
- [Class eligibility/count audit](class_audit.csv), [split counts](split_counts.csv)
- [Confusion matrix](confusion_matrix.png), [PDF](confusion_matrix.pdf), [counts](confusion_matrix.csv)
- [Frozen protocol](protocol.json), [original plan](PLAN.md)
- [User-selected TrafficFormer epoch1 amendment](epoch1_selection_20261117/PLAN.md)
- [Independent numerical completion receipt](completion_verification.json)
- [Finalization repair and preserved failure provenance](finalize_recovery_attempt1/README.md)

The selected TrafficFormer checkpoint reproduces epoch1 metrics exactly while
retaining the original20-epoch scheduler horizon. FIG/YaTC trained50/200epochs;
adapter/head training used30/30epochs. The original finalizer exited1 only at
bundle schema validation: external logs used absolute paths. The metadata-only
recovery retained the original failure evidence and verified the full local
bundle without rerunning training, extraction or inference.

This is a single-seed development result on previously exposed CIC data. Attack
day/endpoint confounding remains; XSS Test has17flows. The benign/attack balance
does not imply equal support among attacks. This task does not provide a new
baseline comparison or an open-set result.

## GitHub publication boundary

Ordinary Git contains source, documentation, protocol definitions, aggregate
tables, confusion-matrix figures, hash inventories and verification receipts.
`manifest.json` describes the complete local experiment, including assets that
are intentionally absent from Git. It is an evidence index, not a claim that a
fresh clone includes the complete experiment bundle.

Weights/checkpoints, raw data, packet caches, embeddings, scalers, flow-level
split/quarantine tables, sample predictions/logits, per-run output directories,
tmux logs and live queue state remain local. In particular, `split_manifest.csv`
and `group_assignments.csv` are represented by frozen hashes and aggregate
counts. Reproducing exact members requires the authorized local source data.

The source keeps historical internal `ustc/A-2` path aliases; actual data and
labels are CIC-IDS-2017 eleven-class, as recorded in
[the alias audit](cicids2017/COMPATIBILITY_ALIAS.json). The original locked
training/finalizer sources are preserved for provenance. Use the independent
`finalize_recovery_attempt1` to understand the recorded schema repair; restarting
the old queue is unnecessary for reviewing completed results.
