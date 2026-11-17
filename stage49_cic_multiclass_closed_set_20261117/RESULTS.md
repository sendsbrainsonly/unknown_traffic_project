# Stage49 — CIC eleven-class closed-set result

- Status: success / independently replayed PASS / finalize metadata recovery PASS
- Seed:2022; current three-view method, no baseline retraining.

## Data and split

Frozen eleven-class flow task. Train/Validation/Test={'train': 99922, 'validation': 8438, 'test': 11688}. Entire five-minute source-PCAP groups are disjoint. Original matched pool2087440; label-disagreement quarantine783. Large attack classes10% per split, smaller attacks all retained; BENIGN equals total attacks in each role. Full counts and exclusions are in class_audit.csv and split_counts.csv.

## Configuration and execution

User-selected replayed TF epoch1 (original20-epoch scheduler horizon); unchanged FIG50/YaTC200/adapter30/head30 recipe. Original first-epoch metrics independently reproduced before selection. Full-attempt history preserved. Official TF/YaTC pretraining; Validation-only selection and Train-only normalization. Branch best epochs={'trafficformer': 1, 'graph': 38, 'yatc': 13}. Internal ustc/A-2 keys are documented path aliases, not USTC data. At most two capacity-selected GPU workers.

## Core results

Accuracy=98.819302%; Macro-F1=98.587682%; Weighted-F1=98.823214%.

|Class|Test support|Precision|Recall|F1|
|---|---:|---:|---:|---:|
|BENIGN|5844|99.1922%|98.7509%|98.9710%|
|Bot|119|95.2000%|100.0000%|97.5410%|
|DDoS|1323|95.7060%|99.3953%|97.5158%|
|DoS Hulk|742|98.9056%|97.4394%|98.1670%|
|DoS Slowhttptest|1055|100.0000%|99.4313%|99.7148%|
|DoS slowloris|711|99.2968%|99.2968%|99.2968%|
|FTP-Patator|336|99.6923%|96.4286%|98.0333%|
|PortScan|1130|100.0000%|98.7611%|99.3767%|
|SSH-Patator|294|98.6532%|99.6599%|99.1540%|
|Web Attack - Brute Force|117|93.6000%|100.0000%|96.6942%|
|Web Attack - XSS|17|100.0000%|100.0000%|100.0000%|

## Preserved evidence

Frozen protocol/members/group assignments, label quarantine, packet-cache audits, branch/adapter/head checkpoints and histories, per-sample Test predictions/logits, confusion matrices, independent replay, source/checkpoint hashes, task logs and manifest. Logs/statuses remain under project-local .tmux-task/stage49_cic_*.

The original `stage49_selected_finalize_20261117` exit1 is preserved. Numerical completion and `completion_verification.json` had already passed; the failure was limited to 21 absolute external-log paths rejected by the bundle validator. `finalize_recovery_attempt1` normalized and deduplicated those references, added the original failure/controller evidence, confirmed no Stage49 worker remained, and passed two complete artifact/hash validations. Training, feature extraction, GPU inference, predictions, and metrics were not rerun.

## Limitations

Single-seed development experiment on previously exposed CIC data; not untouched validation. Attack classes still share capture-day/endpoint shortcuts. BENIGN-versus-attacks balancing does not equalize individual attacks. Excluded tiny classes and GoldenEye are explicitly listed. Historical Stage34B99.75% Macro-F1 describes a different three-class task, not a matched performance comparator.

## Conclusion and next step

Completed the preregistered eleven-class retest and metadata-only finalization recovery. Inspect per-class errors; no automatic class removal, split reselection, baseline run or open-set experiment.
