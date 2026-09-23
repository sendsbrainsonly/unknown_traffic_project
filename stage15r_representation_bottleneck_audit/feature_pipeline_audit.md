# Stage 15R-0 — Current feature pipeline audit

## Verified Open-Detect input semantics

- The active raw-PCAP pipeline uses bidirectional IPv4 TCP/UDP flow/session keys, a 60-second idle timeout, and TCP SYN/FIN/RST session boundaries.
- Each flow image contains at most the first 8 packets. Each packet contributes 128 bytes: 80 bytes from the zero-addressed IPv4/header serialization and 48 bytes from Scapy `Raw` payload. Eight packet blocks are concatenated and reshaped row-major to `32×32` uint8.
- Ethernet bytes are not included in the 128-byte packet block because serialization starts at the IPv4 layer. IPv4 source/destination addresses are zeroed. Other header fields, including transport ports when present, remain visible.
- Packet boundaries are implicit fixed 128-byte slots. Direction and IAT are not represented in E0. Missing packet slots are zero padded; packets after packet 8 are truncated.
- The encoder receives `ToTensor()` values in `[0,1]`. No dataset-level mean/std scaler is fitted for E0. Training uses random crop and horizontal flip; validation is deterministic `ToTensor()`.
- Stage 12 and VNAT Native use the same 32×32 representation semantics, but their flow construction differs: Stage 12 replays aggregate PCAPs with a 60-second timeout/TCP boundaries, while VNAT uses tshark stream IDs from its own clean-pool construction. USTC uses preserved Stage-0 five-tuple flow PKLs.

## Lineage and alignment

- USTC: `input_alignment_manifest.csv` maps every historical Stage-1 flow ID to exactly one Stage-0 PKL flow and one image row; the existing alignment gate reports all inputs valid.
- VNAT: Stage 14C input manifests map `flow_uid → cache_index → image/label`; Stage 14B supplies frozen application and VPN metadata. Known Train/Validation manifests are used by pilot development.
- VNAT's feature cache intentionally contains the union of flows that appear as Known Train/Validation in at least one frozen protocol: 23,447/23,449. The two omitted rows occur only as Known Test in every protocol in which they are Known; their feature values were therefore not opened. This is a development-visibility boundary, not a preprocessing loss.
- ISCX-VPN/ISCXTor: each image has a SHA-256 flow ID, image hash, source capture and frozen role. Exact-image groups were kept disjoint by the Stage 12 protocol logic.
- Unknown Test values are not loaded by the pilot input builders. Stage 15R-0 may count frozen manifest rows, but does not use Unknown feature values or outcomes for feature/model selection.

## Recoverability audit

| Dataset | Exact full-flow statistics cache | First-8 length/IAT/direction | Payload bytes | Encryption evidence |
|---|---|---|---|---|
| USTC | ready | recoverable from Stage-0 PKLs | not separately audited in Stage-0 tuple | currently undetermined |
| VNAT | ready from Stage 14C.5 cache | ready | unavailable in existing cache | VPN metadata; SSH application inference; otherwise undetermined |
| ISCX-VPN | ready | replayed from source PCAP with tshark | tshark TCP/UDP payload length | actual `frame.protocols` plus VPN metadata |
| ISCXTor | ready | replayed from source PCAP with Scapy `PcapReader` | Scapy TCP/UDP payload length | actual Scapy packet-layer tokens plus Tor metadata |

## Data and task-definition differences from published baselines

- Current ISCX-VPN is a 16-application pool and its Medium pilot has 13 Known applications. The local YaTC reproduction reports a 7-class `ISCXVPN2016_MFR` task, while the local ET-BERT official processed run is a balanced 12-class packet-level service task. Their reported Accuracy cannot be compared directly with the current 13/15-class application-level flow task.
- Current ISCXTor has 10 eligible canonical classes and the Medium pilot has 8 Known classes. YaTC uses its own MFR preprocessing and large-scale masked-autoencoder pretraining; its published 99.72% result is not an architecture-only comparison to the present raw-byte Open-Detect flow image.
- Current USTC mixes ten benign applications with ten malware families and has a much larger aligned training pool. Its high Accuracy does not by itself prove that the same representation is adequate for fine-grained encrypted application pairs.

## Dependency gate

- `lightgbm` available in the fixed environment: **True**.
- E3 uses the real project-local LightGBM implementation when the availability flag is true. The shared Conda environment remains unchanged; a scikit-learn estimator is never relabeled as LightGBM.
