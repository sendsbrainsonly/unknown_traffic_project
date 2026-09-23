# RESULTS — Stage 15F-DQ-3F

## Conclusion and next step

Primary Gate: `COARSE_TRAINING_BENEFIT`  
Capture limitation: `CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`

Direct six-Service training was evaluated on exactly the same 2,730 Train and 335 Validation flows as Fine Application training. The result is a Known-only diagnostic under weak capture-derived labels, not a capture-disjoint or authoritative per-flow-label claim.

DQ-3F stops here. DQ-4--DQ-7, TFE-GNN/TrafficFormer training, Byte--Behavior, DES and H1 changes remain `NOT_RUN`.

## Configuration and execution

- Paired seeds: `[2022, 2023, 2024]`; 6/6 F1/F3 runs completed.
- Native architecture/training: ResNet18, 1 channel, latent 128, 100 epoch budget, batch 128, Adam 0.001, lambda 0.005, MultiStepLR [50,80], prototype reset [50,80].
- Checkpoint selection: Known Validation Accuracy only.
- F1/F3 differences: supervision labels and output class count only.
- Known Test/Unknown Test feature values used: 0/0.
- Protected input hashes: `PASS` (21 files).

## Core results

| Task | Scope | Accuracy mean ± std | Macro-F1 mean ± std | Weighted-F1 mean ± std |
|---|---|---:|---:|---:|
| F1 Fine | Full 335 | 0.784080 ± 0.033625 | 0.635049 ± 0.046443 | 0.775730 ± 0.040095 |
| F2 Partial Service | Common subset | 0.913223 ± 0.041867 | 0.871102 ± 0.063815 | 0.911535 ± 0.043950 |
| F3 Service | Full 335 | 0.890547 ± 0.041458 | 0.856050 ± 0.064200 | 0.888036 ± 0.044738 |
| F3 Service | F2 common subset | 0.895695 | 0.862898 | 0.893537 |

- F3 Macro-F1 exceeds F1 on 3/3 paired seeds. This cross-task difference indicates task-granularity difficulty, not a same-task model improvement.
- Across three seeds, 126/217 (`58.06%`) Fine errors are compatible with the correct capture-derived Service after deterministic mapping (counts include repeated seed decisions).
- On each seed's identical F2-evaluable subset, F2 exceeds F3 by mean Accuracy `+0.017528` and mean Macro-F1 `+0.008203`. Direct Service training therefore does **not** outperform prediction mapping on this restricted subset, although F3 covers all 335 samples and F2 does not.
- F3 mean Service recalls: Chat=0.745098, Email=0.855072, File-Transfer=0.705426, P2P=0.950000, Streaming=0.900641, VoIP=0.979167.
- P2P and Streaming contribute 204/335 Validation samples, but F3 is not supported only by those large classes: Chat and Email retain mean recalls `0.745098` and `0.855072`. File-Transfer is the hardest Service (`0.705426` mean recall), and seed 2023 remains visibly weaker.

## Required research questions

1. **Fine performance:** Accuracy/Macro-F1/Weighted-F1 = `0.784080/0.635049/0.775730` (three-seed means).
2. **Direct Service performance:** `0.890547/0.856050/0.888036` on all 335 Validation flows.
3. **F3 versus F2-partial:** F3 does not beat F2 on the common subset on average; F2 has the above `+0.017528` Accuracy and `+0.008203` Macro-F1 advantages. F3's distinct advantage is complete 335-flow coverage and a directly trained Service objective.
4. **Fine errors explained by granularity:** `126/217` repeated-seed Fine errors (`58.06%`) become Service-compatible, supporting label granularity as a substantial contributor but not the sole cause.
5. **Easy/hard Services:** VoIP/P2P/Streaming are strongest on mean recall; File-Transfer and Chat are hardest. Actual correct/error counts for every seed are retained in `per_service_metrics.csv`.
6. **Large-class dependence:** P2P/Streaming materially influence Accuracy, but all six Services have non-zero recall and small Chat/Email do not collapse. The conclusion is therefore not solely a majority-class artifact.
7. **Service as a research task:** supported as a development diagnostic with weak capture labels; not yet supported as an authoritative or capture-generalized final benchmark.
8. **Remaining cross-paper mismatches:** PCAP scope, flow/session construction, sample filtering, target/background separation, label provenance, split/group policy, training-pool size, inputs and evaluation metrics remain unaligned.
9. **Next stage:** DQ-4 is feasible as a controlled data-selection/filtering audit. DQ-7 is not yet fair without a newly frozen unified Service dataset/protocol and a solution to the P2P single-capture limitation.

## Data and split

- Dataset: 31 shared VPN captures; Train/Validation=2,730/335.
- Fine classes: 11; Service classes: 6.
- All 3,065 labels are `WEAK_CAPTURE_LABEL`.
- Flow-ID overlap and exact-image overlap across Train/Validation: 0/0.
- F2 uses only deterministic singleton Application→Service predictions; ambiguous true or predicted Applications are excluded and explicitly counted.

## Limitations

- F1 and F3 have different label spaces, so their Macro-F1 difference is not a same-task algorithmic gain.
- P2P has one capture; full six-Service capture-disjoint generalization is not identifiable.
- Capture-derived weak labels may include incidental/background flows.
- Current data/flow definition, filtering, split and weak labels remain different from TFE-GNN and TrafficFormer paper protocols; no direct superiority claim is allowed.
- A Service Open-Set protocol would require independently frozen Known/Unknown Services and cannot reuse the existing Application-level Unknown split.

## Preserved evidence

- All six checkpoints, SHA256 files, per-run logs/configs/predictions and metrics are retained under `runs/`.
- Requested aggregate CSVs, capture audit, protected hashes and completion record are stored in this bundle.
- Two pre-epoch CUDA peak-memory API failures are retained under `failed_attempts/`; neither consumed training data or updated a model.
