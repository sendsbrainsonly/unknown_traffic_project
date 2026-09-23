# Stage 15F-0 — Literature-Guided Feature Registry and Feasibility Audit

## Final status

**STAGE15F0_PASS — audit complete; Stage 15F-1 through Stage 15F-5 are NOT_RUN.**

No encoder was trained, no checkpoint was selected, no detector was fit, and neither Known Test nor Unknown Test feature values were read into the audit aggregates. Existing Stage 12--15R assets remain frozen.

## 1. Primary-source literature audit

Nineteen requested works were audited from primary papers and, where verifiable, official repositories: ET-BERT, YaTC, TFE-GNN, MIETT, MH-Net, AN-Net, MM4flow, TrafficFormer, DecETT, MT-FlowFormer, Tracegram, Deep Packet, FlowPic, FS-Net, FlowLens, Rosetta, Deep Fingerprinting, The Sweet Danger of Sugar, and SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers.

For every work, the registry records title, year/venue, paper and code status, raw input, granularity, packet/byte budget, header/payload boundary, direction, IAT, length, statistics, burst, pretraining, encoder, objective, tasks/datasets, and leakage risk. Every claim is tagged `PAPER_CONFIRMED`, `CODE_CONFIRMED`, `INFERRED`, or `UNKNOWN`; paper/code disagreements are retained rather than silently resolved.

Important boundaries:

- no official implementation was verified for Deep Packet, MT-FlowFormer, or FlowLens;
- the MIETT repository currently states that code is coming, so it is not an executable reproduction source;
- exact TrafficFormer packet/time/start-index fields, some MM4flow task/preprocessing details, Rosetta sequence cap, MH-Net byte cap, and several group-split guarantees remain unresolved;
- the local B/T/S candidates are **literature-guided ablations**, not claimed reproductions of those papers.

The detailed evidence is in `literature_primary_source_notes.md`; the machine-readable registry is `literature_feature_registry.csv`.

## 2. Feature registry

Thirteen concrete configurations are frozen in `feature_group_definitions.json`:

| Family | Definition | Observation regime |
|---|---|---|
| B0 | Native eight-packet 32x32 raw-byte image | EARLY_8 |
| B1 | Packet-structured bytes with byte/packet masks | EARLY_N |
| B2 | Header-only bytes with identifier masking | EARLY_N |
| B3 | Visible transport-payload-only bytes | EARLY_N |
| B4 | Explicit header/payload segments and masks | EARLY_N |
| T1 | Direction-signed frame length + IAT + mask | EARLY_N |
| T2 | Directional burst count/bytes/duration/gap | EARLY_N or FULL_FLOW, kept separate |
| S1 | Existing 12 audited flow statistics | FULL_FLOW |
| S2 | Expanded bidirectional statistics | FULL_FLOW |
| M1 | Byte + temporal fusion | EARLY_N |
| M2 | Byte + statistics fusion | FULL_FLOW |
| M3 | Byte + temporal + statistics fusion | FULL_FLOW |
| P1 | Public pretrained representation | blocked pending provenance/overlap/parity audit |

All future fitted transforms are Known-Train-only. Missing fields require explicit masks/indicators and may not be replaced by observational zero.

## 3. Actual dataset availability

| Dataset | Raw captures/files | Raw size | Frozen Known Train/Val cache | Packet lineage | Status |
|---|---:|---:|---:|---|---|
| USTC-TFC2016 | 56 | 4.270 GB | 432,537 | Stored raw frames in Stage-0 flow PKLs | PASS |
| VNAT | 165 | 36.137 GB | 23,447 | Capture-local `tcp.stream`/`udp.stream`; official connection H5 also present | PASS |
| ISCX-VPN | 140 | 27.682 GB | 18,121 | Stage 12 SHA session ID, 60 s timeout, TCP boundaries | PASS |
| ISCXTor2016 | 95 | 23.227 GB | 11,783 | Same Stage 12 session semantics | PASS |

All four can support packet order, timestamp, endpoint-relative direction, captured length, first-N byte/time coverage, visible header/payload parsing, and explicit masks. VPN/Tor only expose the outer tunnel view. VNAT's 23,447 count is the Known Train/Val union; the frozen clean pool remains 23,449 and the two omitted rows are Test-only.

## 4. Packet-window coverage

The table below averages protocol-level `__ALL__` rows equally. Byte coverage is the per-flow mean of first-N captured-frame bytes/full-flow bytes; time coverage is the per-flow mean of first-N elapsed/full duration.

| Dataset | N | Complete by N | Truncated after N | Padded before N | Byte coverage | Time coverage |
|---|---:|---:|---:|---:|---:|---:|
| USTC | 8 | 68.53% | 31.47% | 67.16% | 83.47% | 80.17% |
| USTC | 16 | 79.63% | 20.37% | 79.44% | 90.99% | 87.81% |
| USTC | 32 | 85.51% | 14.49% | 85.37% | 99.06% | 94.82% |
| USTC | 64 | 99.56% | 0.44% | 99.56% | 99.78% | 99.79% |
| VNAT | 8 | 82.58% | 17.42% | 82.39% | 87.03% | 83.68% |
| VNAT | 16 | 85.13% | 14.87% | 84.61% | 92.88% | 87.50% |
| VNAT | 32 | 92.41% | 7.59% | 92.26% | 95.68% | 94.10% |
| VNAT | 64 | 94.94% | 5.06% | 94.90% | 96.98% | 96.51% |
| ISCX-VPN | 8 | 88.18% | 11.82% | 87.47% | 92.05% | 91.24% |
| ISCX-VPN | 16 | 91.77% | 8.23% | 91.37% | 94.77% | 93.72% |
| ISCX-VPN | 32 | 94.55% | 5.45% | 94.42% | 96.24% | 96.25% |
| ISCX-VPN | 64 | 96.12% | 3.88% | 96.11% | 97.03% | 97.23% |
| ISCXTor | 8 | 71.48% | 28.52% | 70.46% | 80.71% | 77.67% |
| ISCXTor | 16 | 78.84% | 21.16% | 78.17% | 84.47% | 80.99% |
| ISCXTor | 32 | 83.10% | 16.90% | 82.82% | 87.56% | 85.66% |
| ISCXTor | 64 | 86.89% | 13.11% | 86.84% | 89.79% | 88.36% |

The aggregate hides strong class differences. At N=8, USTC Weibo and WorldOfWarcraft truncate 95.94% and 98.77% of flows; VNAT vimeo truncates 97.46%, youtube 83.62%, and netflix 84.47%; ISCXTor FTP truncates 50.67%. N=32 materially reduces but does not eliminate truncation: VNAT rdp remains 54.32%, youtube 63.10%, and ISCXTor FTP 27.28%.

Therefore the E2 eight-packet representation has a real and class-dependent information horizon. However, many flows are genuinely short, and this audit contains no classification experiment. Coverage alone does not establish that N=8 caused the prior Macro-F1 gap.

Full dataset/protocol/class/role statistics occupy 2,928 rows in `packet_window_coverage.csv`.

## 5. Encryption visibility

`encryption_visibility_audit.csv` separates application/transport encryption from tunnel encryption. It records TLS/QUIC, SSH, visible plaintext, undetermined traffic, VPN outer tunnels, and Tor outer tunnels. Non-VPN is never converted to plaintext. USTC remains undetermined without reliable frozen per-flow evidence. These categories are descriptive metadata only and are not registered detector inputs.

## 6. Stage 15R boundary

Stage 15R showed that the existing E2 eight-packet T1 and E3 twelve-statistic implementations did not stably exceed Native Open-Detect. Stage 15F-0 does not reinterpret those failures as evidence against all temporal or statistical representations. It identifies three untested factors: a longer controlled temporal horizon, packet/header/payload structure, and richer bidirectional/burst statistics. Whether they improve classification remains an experimental question.

## 7. Preregistered next-stage matrix

`stage15f1_preregistered_matrix.csv` freezes five pilot protocols: USTC A-2, VNAT medium-2025 and medium-2026, ISCX-VPN medium, and ISCXTor medium. It holds data, architecture/training policy, and selection roles fixed while changing one feature family at a time. T8/T16/T32 are mandatory; T64 is retained as a preregistered coverage endpoint because N=32 still truncates 5.45--16.90% at dataset level and substantially more in some classes. Full-flow S/burst results remain separately labeled. P1 is blocked until provenance and overlap checks pass. No row has been run.

## 8. Answers to the 15 final research questions at Stage 15F-0

1. **Does E2's first eight packets show obvious information insufficiency?** Descriptively yes for a nontrivial, highly class-dependent subset: dataset-level truncation is 11.82--31.47%, with several classes above 80--98%. This is evidence of limited observation coverage, not yet causal performance evidence.
2. **Do T16/T32 improve hard ISCX/VNAT cases?** `NOT_RUN`.
3. **Do expanded statistics outperform E3?** `NOT_RUN`.
4. **Does burst provide independent gain?** `NOT_RUN`.
5. **Is header/payload structure better than continuous bytes?** `NOT_RUN`.
6. **Does Native over-rely on protocol headers or capture identifiers?** `NOT_RUN`; registry identifies the leakage risk but does not measure reliance.
7. **Are byte, temporal, and statistical features usefully complementary by class?** `NOT_RUN`.
8. **Which feature improves rsync/scp?** `NOT_RUN`.
9. **Which feature improves ISCX-VPN and ISCXTor2016?** `NOT_RUN`.
10. **Which features work only on one protocol?** `NOT_RUN`.
11. **Is an ET-BERT/YaTC/MIETT-style advanced encoder needed?** Undetermined; P1 remains blocked and simple controlled families must be tested first.
12. **Does a new representation stably improve Known classification?** `NOT_RUN`.
13. **Does better Known representation improve Unknown detection?** `NOT_RUN`.
14. **Do DES/H1 retain gain on the new representation?** `NOT_RUN`.
15. **Which final features and encoder should be retained?** Undetermined; no feature family has passed the preregistered pilot gate.

## 9. Reproducibility and completion checks

- literature works: 19/19;
- registered feature configurations: 13;
- datasets audited: 4/4;
- window sizes: 8/16/32/64;
- protocol counts: USTC 3, VNAT 15, ISCX-VPN 3, ISCXTor 3;
- unmatched frozen membership rows: 0;
- Known Test feature values used: 0;
- Unknown Test feature values used: 0;
- window-cache parity: PASS;
- model training/checkpoints created: 0;
- formal Stage 12--15R hash comparison: recorded in `completion_verification.json`;
- future result files (`single_feature_results.csv`, `per_class_feature_results.csv`, `feature_complementarity.csv`, `multiview_ablation.csv`, `paired_vs_native.csv`, `open_set_six_metrics.csv`) are intentionally absent because their stages are `NOT_RUN`.
