# Stage34B — CIC benign/malicious 1:1 retest

- Status: `complete / PASS`
- Experiment type: `benchmark`; claim scope: `development diagnostic`
- Objective: retest the same three-view closed-set model after BENIGN is sampled to the total attack count.

## Data and split

- Original Stage34 five-minute group-disjoint split retained. Original manifest SHA256: `9734812c102266e88c58364d499c4f1494c24d0f012ac45e33e6292c93f7bd53`. Balanced subset manifest SHA256: `35e99d89ad4eddc828c9d67529ea1b527d1fd511cb39db8c509461b5bf38e25d`.
- All Slowhttptest and PortScan flows retained. BENIGN chosen by seed-2022 SHA256 rank separately in each split, before Test outcomes existed.
- Train BENIGN/Slowhttptest/PortScan: 17758/3909/13849 (35,516 total).
- Validation: 1956/1055/901 (3,912 total).
- Test: 1268/132/1136 (2,536 total).

## Configuration and execution

- Same Stage31/34 TrafficFormer, FIG/TAGCN, YaTC and equal-feature-fusion recipe, seeds, optimizer, epochs and Known-Validation selection; only sample membership changed.
- Physical GPUs 0/1, at most two concurrent branch jobs. Named `stage34b_balanced_queue_0926` orchestrated all project-local tmux worker sessions; see `queue_progress.json` and `.tmux-task/stage34b_*/`.
- Unknown use=0. Test used only for one final evaluation after all branch and fusion checkpoint hashes were fixed.

## Core results

| Known Test samples | Accuracy | Macro-F1 | Weighted-F1 | Fusion Known-Val Macro-F1 |
|---:|---:|---:|---:|---:|
| 2,536 | 0.998028 | 0.997482 | 0.998028 | 0.999730 |

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| BENIGN | 1268 | 0.996855 | 1.000000 | 0.998425 |
| DoS Slowhttptest | 132 | 0.992481 | 1.000000 | 0.996226 |
| PortScan | 1136 | 1.000000 | 0.995599 | 0.997794 |

## Preserved evidence

- `balanced_manifest.csv`, `balanced_manifest_audit.json`, exact-ID Train/Val caches, raw-PCAP extracted balanced Test cache, all three branch histories/checkpoints/features, fusion history/checkpoints, Test predictions/logits, `completion_verification.json` and `manifest.json`. Execution logs and exit statuses reside under project-local `.tmux-task/`.

## Limitations

- The user stopped the original CIC training before its fusion/Test result. Thus there is no completed old model on these exact Test flows and no valid paired performance delta.
- Attack classes are not individually balanced; Slowhttptest Test has only 132 flows. Slowhttptest and PortScan each remain tied to one capture day/PCAP, so high scores may exploit capture/endpoint artifacts.
- Changing the Test class prior changes Accuracy and Weighted-F1 interpretation. This is a one-seed development diagnostic, not independent validation or Unknown detection.

## Conclusion and next step

- Balanced Test Macro-F1 is `0.997482`. Inspect branch training curves and per-class recall before deciding whether any optimization change is warranted. Stage34 historical assets remain preserved as interrupted evidence.
