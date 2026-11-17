# Stage49 — CIC-IDS-2017 Eleven-Class Closed-Set Retest

Registered 2026-11-17, before any training or evaluation of this task.

## Task and labels

Use the current method only: independently trained TrafficFormer, FIG/TAGCN and YaTC branches, followed by their existing adapters and equal feature fusion classifier. This is a new closed-set development task, seed2022. Previous exposed CIC results mean this is not untouched validation.

Keep BENIGN, Bot, DDoS, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, PortScan, SSH-Patator, Web Attack - Brute Force, Web Attack - XSS.

Exclude Heartbleed(11), Infiltration(34), SQL Injection(12) for insufficient independent sample support and GoldenEye because only two five-minute capture groups cannot populate three disjoint partitions. These exclusions are determined from the prior inventory, not model outcomes.

## Label provenance and sampling

Read existing matched-flow Parquets without modifying them. Require MATCHED-prefixed status, non-null label, and consistency with the previous independent LABEL_MATCH audit. Quarantine the783 mismatch IDs; verify their counts and labels against the audit. Preserve counts for all fifteen classes and every removal reason.

Assign entire `(source_pcap, floor(flow_start_epoch_utc/300))` groups to Train/Validation/Test. Approximate8:1:1 per-class whole-group counts using a fixed deterministic metadata-only integer program; require every kept class in all three partitions. Report actual ratios and group counts, as whole groups make exact ratios impossible. Do not retry splits based on model results.

For attack classes with at least10000 verified flows, keep10% per assigned class/split by seed2022 SHA256 flow-ID rank. Retain all flows from smaller attack classes. In each split, select BENIGN by the same deterministic ranking to equal the total selected attack count. Do not undersample difficult classes based on performance. Freeze all members and source hashes before packet extraction and training.

## Fixed method

- TrafficFormer: official pretrained initialization,20epochs,batch64,LR6e-5.
- FIG: existing30-node7-feature TAGCN,K2,50epochs,batch64,LR1e-3.
- YaTC: official pretrained initialization,200epochs,batch64,existing AdamW/warmup/smoothing recipe.
- Existing adapters30epochs and equal-feature classifier30epochs; existing normalization and checkpoint rules.
- TrafficFormer/FIG checkpoints: Validation Macro-F1; YaTC: existing Validation Weighted-F1 rule; adapters/head: existing Validation rules.
- No architecture, loss, early-stopping, scheduler or optimizer changes. No new hyperparameter search.
- At most two GPU workers, selected immediately before launch using live free memory/utilization. Queue on lack of capacity rather than occupy another job's GPU memory. Inherited conservative30GiB minimum, utilization<=30%.

## Execution and evaluation

Reuse Stage34/31 implementations through an isolated adapter; record actual dataset/class list despite their legacy internal `ustc/A-2` path alias. Extract Train/Val packet inputs first. Materialize Test inputs only after all branch/adapter/head weights are frozen. Evaluate the final three-view classifier once and independently replay saved logits/predictions.

Report Accuracy, macro/weighted Precision, Recall, F1; per-class Precision/Recall/F1/support; confusion matrix; Train/Val counts, selection epochs and checkpoint hashes. Include Stage34B as historical context, not a matched three-versus-eleven-class gain estimate.

## Limitations

Attack/day/endpoint confounding remains, even with disjoint time groups. Bot/SSH schedule-sanity warnings are retained as uncertainty rather than replacing official labels. No claimed label correction from attack schedules. Groups are time blocks, not independent physical captures. Class balance is benign-versus-total-attacks, not equal across all eleven classes.

## Outputs

`protocol.json`, `split_manifest.csv`, `group_assignments.csv`, `class_audit.csv`, `split_counts.csv`, packet-cache audits, histories/checkpoints, `run_results.csv`, `per_class_results.csv`, `confusion_matrix.csv/.png`, saved Test logits/predictions, `completion_verification.json`, `RESULTS.md`, `manifest.json`, and `progress.py`.
