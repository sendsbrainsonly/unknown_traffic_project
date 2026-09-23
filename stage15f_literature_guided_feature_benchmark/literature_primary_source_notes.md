# Stage 15F-0 — Literature Primary-Source Notes

## Scope and evidence policy

This document is a literature-only evidence registry for the Stage 15F feature benchmark. It does **not** define the final benchmark matrix, run training, or change any Stage 12–15R artifact. Only original papers, publisher/author paper pages, and author-maintained official repositories are used.

Evidence tags:

- `PAPER_CONFIRMED`: stated explicitly in the original paper.
- `CODE_CONFIRMED`: stated or implemented explicitly in an author-maintained official repository.
- `INFERRED`: a narrow consequence of confirmed evidence, but not stated verbatim by the source.
- `UNKNOWN`: not verified from the inspected primary sources. It must not be silently filled from a survey, blog, or third-party implementation.

`bytes/packet` below means the byte budget or byte-tokenization applied to one packet. A token limit is not silently converted to a byte limit. “No raw bytes” means that only metadata such as length, timing, or direction is used.

## 1. ET-BERT

- Accurate title: **ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification**
- Year / venue: **2022, The Web Conference (WWW '22)** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/2202.06335>
- Official code: <https://github.com/linwhitehat/ET-BERT>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` Datagram raw bytes encoded as hexadecimal adjacent-byte bigrams and then BPE tokens. |
| granularity | `PAPER_CONFIRMED` Burst/datagram context for pre-training; packet- and flow-level downstream classification. |
| packet window | `PAPER_CONFIRMED` Flow classification concatenates `M=5` consecutive packets; Transformer maximum is 512 tokens. `CODE_CONFIRMED` The released fine-tuning examples commonly set sequence length 128. |
| bytes/packet | `PAPER_CONFIRMED` No single fixed byte cap is established by the paper-level token limit; byte bigrams are BPE-tokenized with vocabulary up to 65,536. `CODE_CONFIRMED` Fine-tuning serialization slices hexadecimal input at `[76:]`, i.e. skips 38 bytes before later truncation/tokenization. |
| header / payload | `CODE_CONFIRMED` The 38-byte prefix skip is not sufficient evidence to call the remainder “payload only”; transport/header bytes can remain. |
| boundary | `PAPER_CONFIRMED` Five-tuple/session flow boundaries, same-direction burst boundaries, and `[CLS]/[SEP]` sequence structure are represented. |
| direction | `PAPER_CONFIRMED` Encoded implicitly by grouping consecutive same-direction packets into a burst; no separate numeric direction channel was verified. |
| IAT | `UNKNOWN` No explicit IAT channel verified. |
| length | `UNKNOWN` No independent packet-length channel verified; truncation/padding may indirectly expose length. |
| statistics | `PAPER_CONFIRMED` No explicit hand-crafted flow-statistics vector. |
| burst | `PAPER_CONFIRMED` Yes: consecutive same-direction packets within a session flow. |
| pretraining | `PAPER_CONFIRMED` Masked BURST Model and Same-origin BURST Prediction. |
| encoder | `PAPER_CONFIRMED` BERT-style Transformer, 12 layers, 12 heads, hidden size 768. |
| objective | `PAPER_CONFIRMED` Masked-token recovery with 15% masking plus same-origin binary prediction; supervised downstream classification. |
| tasks / datasets | `PAPER_CONFIRMED` Encrypted application/service/traffic classification on Cross-Platform, USTC-TFC2016, ISCX-VPN, ISCX-Tor and CSTNET-TLS1.3; pre-training traffic also includes CICIDS2017/ISCX-VPN and CSTNET sources. |
| leakage risks | `CODE_CONFIRMED` Only a fixed prefix is removed; remaining protocol fields and content can retain identifiers. `UNKNOWN` No comprehensive identifier-masking guarantee was verified. |

Paper/code difference: `CODE_CONFIRMED` the released fine-tuning generator adds operational constraints not fully conveyed by the paper summary, including dropping flows below three packets, selecting at most five packets, and skipping a fixed 38-byte prefix.

## 2. YaTC

- Accurate title: **Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation**
- Year / venue: **2023, AAAI-23** (`PAPER_CONFIRMED`)
- Paper: <https://ojs.aaai.org/index.php/AAAI/article/view/25674>
- Official code: <https://github.com/NSSL-SJTU/YaTC>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED + CODE_CONFIRMED` PCAP-derived IP-packet bytes arranged into a Multi-level Flow Representation (MFR) image. |
| granularity | `PAPER_CONFIRMED` Flow. |
| packet window | `PAPER_CONFIRMED + CODE_CONFIRMED` First five adjacent packets. |
| bytes/packet | `CODE_CONFIRMED` 320 bytes per packet: 80 header bytes plus 240 payload bytes, independently truncated/zero-padded; five packets form a `40 x 40 = 1600` byte image. |
| header / payload | `PAPER_CONFIRMED + CODE_CONFIRMED` Explicit, fixed header and payload regions. |
| boundary | `CODE_CONFIRMED` Fixed 320-byte packet slots preserve packet and header/payload boundaries and packet order. |
| direction | `PAPER_CONFIRMED` No independent direction channel. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `PAPER_CONFIRMED` No explicit length channel; zero padding can indirectly encode observed length. |
| statistics | `PAPER_CONFIRMED` No separate flow-statistics vector. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` Masked autoencoding. `CODE_CONFIRMED` Default mask ratio is 0.9. |
| encoder | `CODE_CONFIRMED` ViT-like Traffic Transformer; released default uses image size 40, patch size 2, embedding 192, depth 4, and 16 heads. |
| objective | `PAPER_CONFIRMED` Masked-patch reconstruction followed by supervised classification. |
| tasks / datasets | `PAPER_CONFIRMED + CODE_CONFIRMED` Encrypted traffic classification; released processed data include ISCXVPN2016, ISCXTor2016, USTC-TFC2016 and CICIoT2022, while the paper reports five downstream datasets. |
| leakage risks | `PAPER_CONFIRMED` The described preprocessing zeroes ports and randomizes IP addresses. `CODE_CONFIRMED` The inspected released `data_process.py` path does not visibly implement the same explicit IP/port masking, so raw header identifiers and fixed-padding length shortcuts require audit before reuse. |

Paper/code difference: the paper describes IP randomization and port zeroing, whereas the inspected released preprocessing path does not make that operation explicit. The benchmark must verify the exact path used rather than assume the paper policy is automatically enforced.

## 3. TFE-GNN

- Accurate title: **TFE-GNN: A Temporal Fusion Encoder Using Graph Neural Networks for Fine-grained Encrypted Traffic Classification**
- Year / venue: **2023, The Web Conference (WWW '23)** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/2307.16713>, <https://doi.org/10.1145/3543507.3583227>
- Official code: <https://github.com/ViktorAxelsen/TFE-GNN>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED + CODE_CONFIRMED` PCAP split by SplitCap into bidirectional flows; packet header bytes and payload bytes become separate byte graphs. |
| granularity | `PAPER_CONFIRMED` Packet graph followed by flow/segment sequence. |
| packet window | `PAPER_CONFIRMED` Up to the first 50 packets; Tor data use non-overlapping 60-second segments. |
| bytes/packet | `PAPER_CONFIRMED + CODE_CONFIRMED` Header capped at 40 bytes and payload at 150 bytes; byte-value graph has at most 256 byte nodes and PMI context window 5. |
| header / payload | `PAPER_CONFIRMED` Separate header and payload graphs; Ethernet header, IP addresses and ports are removed. |
| boundary | `PAPER_CONFIRMED` Packet order/boundaries are retained inside bidirectional flow/segment sequences. |
| direction | `UNKNOWN` No independent direction channel verified. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `UNKNOWN` No independent packet-length channel verified. |
| statistics | `PAPER_CONFIRMED` No explicit hand-crafted flow-statistics vector. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` No independent pre-training phase. |
| encoder | `PAPER_CONFIRMED` Header/payload byte embeddings, GraphSAGE traffic-graph encoders, mean pooling, cross-gated fusion, and a two-layer BiLSTM; a Transformer temporal variant is also evaluated. |
| objective | `PAPER_CONFIRMED` Supervised cross-entropy. |
| tasks / datasets | `PAPER_CONFIRMED` Fine-grained encrypted traffic/user-behavior classification on WWT (WeChat/WhatsApp/Telegram), ISCX VPN/non-VPN and ISCX Tor/non-Tor. |
| leakage risks | `PAPER_CONFIRMED` IP addresses and ports are removed. `INFERRED` The reported stratified 9:1 sample split is not evidence of capture/group-disjoint isolation, so same-origin capture/session leakage remains possible. |

Paper/code difference: none established; the 40/150-byte and 50-packet limits are implementation-critical details that must be retained if claiming code-equivalent reproduction.

## 4. MIETT

- Accurate title: **MIETT: Multi-Instance Encrypted Traffic Transformer for Encrypted Traffic Classification**
- Year / venue: **2025, AAAI-25** (`PAPER_CONFIRMED`)
- Paper: <https://ojs.aaai.org/index.php/AAAI/article/download/33748/35903>, <https://arxiv.org/abs/2412.15306>
- Official code/project page: <https://github.com/Secilia-Cxy/MIETT>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` PCAP to session flow to complete packet header plus encrypted payload bytes, encoded as hexadecimal bigrams and BPE. |
| granularity | `PAPER_CONFIRMED` Packet instances arranged as a flow-level multi-instance matrix. |
| packet window | `PAPER_CONFIRMED` Default first five packets; pre-training samples five among the first ten. Each packet is standardized to 128 tokens. |
| bytes/packet | `PAPER_CONFIRMED` No fixed byte cap can be equated to the 128-token cap; byte bigrams use BPE with vocabulary up to 65,536. |
| header / payload | `PAPER_CONFIRMED` Complete packet content; source/destination IP addresses and ports are zeroed. |
| boundary | `PAPER_CONFIRMED` Packet boundary and order are explicit. |
| direction | `UNKNOWN` No independent direction channel verified. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `UNKNOWN` No independent length channel verified. |
| statistics | `PAPER_CONFIRMED` No explicit hand-crafted statistics. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` ET-BERT initializes the packet encoder; MIETT freezes that encoder while training flow-level attention, then fine-tunes the full model. |
| encoder | `PAPER_CONFIRMED` Two-level/intra-packet and inter-packet attention Transformer, default hidden size 768. |
| objective | `PAPER_CONFIRMED` Masked Flow Prediction, Packet Relative Position Prediction and Flow Contrastive Learning; downstream cross-entropy. |
| tasks / datasets | `PAPER_CONFIRMED` VPN/Tor service, mobile-app and IoT-attack classification on ISCXVPN2016, ISCXTor2016, CrossPlatform Android/iOS and CICIoT2023. |
| leakage risks | `PAPER_CONFIRMED` IPs and ports are masked. `UNKNOWN` Capture/group-disjoint splitting and exact preprocessing cannot be code-audited because implementation is unavailable. |

Paper/code difference: the linked official repository currently states **“Code is coming soon”**. Paper claims therefore cannot yet be independently checked against an implementation.

## 5. MH-Net

- Accurate title: **Revolutionizing Encrypted Traffic Classification with MH-Net: A Multi-View Heterogeneous Graph Model**
- Year / venue: **2025, AAAI-25** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/2501.03279>, <https://ojs.aaai.org/index.php/AAAI/article/download/32091/34246>
- Official code: <https://github.com/ViktorAxelsen/MH-Net>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED + CODE_CONFIRMED` PCAP via SplitCap to bidirectional flows; header/payload bytes form heterogeneous multi-view graphs. |
| granularity | `PAPER_CONFIRMED` Packet heterogeneous graph followed by flow sequence. |
| packet window | `PAPER_CONFIRMED` First 15 packets; Tor uses non-overlapping 60-second blocks. |
| bytes/packet | `PAPER_CONFIRMED` 4-bit and 8-bit graph views, PMI window 5; an exact per-packet byte cap was not established in the inspected paper text (`UNKNOWN`). |
| header / payload | `PAPER_CONFIRMED` Header-header, payload-payload and header-payload edge types; Ethernet header, IP addresses and ports are removed. |
| boundary | `PAPER_CONFIRMED` Packet boundary/order are retained. |
| direction | `UNKNOWN` No independent direction feature verified. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `UNKNOWN` No independent packet-length channel verified. |
| statistics | `PAPER_CONFIRMED` No explicit hand-crafted flow-statistics vector. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` No independent unsupervised pre-training; supervised contrastive terms are part of training. |
| encoder | `PAPER_CONFIRMED` Non-shared heterogeneous GraphSAGE encoders for each view plus LSTM/RNN flow encoder. |
| objective | `PAPER_CONFIRMED` Packet and flow cross-entropy plus packet-level and flow-level supervised contrastive losses. |
| tasks / datasets | `PAPER_CONFIRMED` Packet/flow encrypted traffic classification on CIC-IoT, ISCX VPN/non-VPN, ISCX Tor/non-Tor and supplementary CrossPlatform Android. |
| leakage risks | `PAPER_CONFIRMED` Packets from one flow stay together during splitting. `INFERRED` The flow-level stratified 9:1 split is not confirmed capture/group-disjoint. |

Paper/code difference: none established from the inspected sources.

## 6. AN-Net

- Accurate title: **AN-Net: an Anti-Noise Network for Anonymous Traffic Classification**
- Year / venue: **2024, The Web Conference (WWW '24)** (`PAPER_CONFIRMED`)
- Paper: <https://openreview.net/pdf?id=aV21vc0OD8>, <https://doi.org/10.1145/3589334.3645691>
- Official code: <https://github.com/SJTU-dxw/AN-Net>

| Field | Primary-source finding |
|---|---|
| raw input | `CODE_CONFIRMED` Scapy reads PCAP and retains TCP packets only. |
| granularity | `CODE_CONFIRMED` Consecutive packets from a capture/file, not a standard reconstructed five-tuple flow. |
| packet window | `CODE_CONFIRMED` Each sample contains 100 consecutive TCP packets, subdivided into ten short-term blocks of ten packets. |
| bytes/packet | `CODE_CONFIRMED` 64 bytes of TCP-layer serialized content per packet. |
| header / payload | `CODE_CONFIRMED` Repository comments describe partial TCP header plus payload while excluding ports and sequence number; exact semantics depend on the byte slice and remain partly `UNKNOWN`. |
| boundary | `CODE_CONFIRMED` Packet order/boundaries and ten-packet short-term blocks are explicit. |
| direction | `CODE_CONFIRMED` No explicit direction channel. |
| IAT | `CODE_CONFIRMED` Per-packet relative IAT, with first packet set to zero. |
| length | `CODE_CONFIRMED` TCP payload length. |
| statistics | `CODE_CONFIRMED` TTL, IP flags, TCP flags; short-term mean, standard deviation, skewness, kurtosis, median, min, max and FFT-derived features. |
| burst | `CODE_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED + CODE_CONFIRMED` None. |
| encoder | `PAPER_CONFIRMED + CODE_CONFIRMED` Payload-byte GRU, statistical/modal encoders, high-temperature self-attention and multimodal fusion. |
| objective | `PAPER_CONFIRMED` Supervised classification. |
| tasks / datasets | `PAPER_CONFIRMED` Anonymous/VPN traffic classification and noise robustness on SJTU-AN21, ISCX-Tor and ISCX-VPN; CIC-IoT contributes to noise-related construction. |
| leakage risks | `INFERRED` Adjacent 100-packet blocks from one capture can cross a non-grouped split; TTL/flags can encode capture environment. Capture-disjoint splitting was not verified. |

Paper/code difference: repository comments and the actual TCP-byte slice do not make the exact semantic boundary fully unambiguous; the benchmark must treat this as an implementation audit item rather than call the 64 bytes “payload only.”

## 7. MM4flow

- Accurate title: **MM4flow: A Pre-trained Multi-modal Model for Versatile Network Traffic Analysis**
- Year / venue: **2025, ACM CCS 2025** (`PAPER_CONFIRMED`)
- Paper: <https://doi.org/10.1145/3719027.3744804>
- Official code: <https://github.com/Shangshu-LAB/MM4flow>

| Field | Primary-source finding |
|---|---|
| raw input | `CODE_CONFIRMED` Zeek-derived raw payload stream plus packet-size/transmission-pattern stream, joined using Zeek flow UID. |
| granularity | `PAPER_CONFIRMED + CODE_CONFIRMED` Flow. |
| packet window | `CODE_CONFIRMED` Packet-size/burst token stream is capped at 256 tokens; this is not necessarily 256 original packets because burst notation can be expanded. |
| bytes/packet | `CODE_CONFIRMED` Forward/backward raw payload byte stream is capped at 512 tokens; no fixed per-packet byte slot is used. |
| header / payload | `CODE_CONFIRMED` Model modalities are payload bytes and transmission pattern. Connection metadata retain IP/port/protocol/service, but entry into the formal model was not verified (`UNKNOWN`). |
| boundary | `CODE_CONFIRMED` Raw byte stream does not preserve a fixed packet slot; transmission-pattern modality preserves packet/burst structure. |
| direction | `CODE_CONFIRMED` Forward/backward payload streams are distinct; exact direction encoding in the packet-size stream is `UNKNOWN`. |
| IAT | `UNKNOWN` No model IAT input verified. |
| length | `CODE_CONFIRMED` Packet-length tokens. |
| statistics | `CODE_CONFIRMED` Duration/packet/byte counts exist in the dataframe; their use by the formal model is `UNKNOWN`. |
| burst | `CODE_CONFIRMED` `packet_length:packet_count` transmission/burst patterns are represented and can be expanded to length tokens. |
| pretraining | `PAPER_CONFIRMED + CODE_CONFIRMED` Separate byte-BERT and packet-size-BERT masked-language pre-training; code mask probability 0.2. |
| encoder | `PAPER_CONFIRMED + CODE_CONFIRMED` Two BERT encoders followed by bidirectional cross-attention fusion. |
| objective | `PAPER_CONFIRMED + CODE_CONFIRMED` MLM pre-training and supervised downstream classification. |
| tasks / datasets | `PAPER_CONFIRMED` Six network-traffic-analysis task families are claimed. `UNKNOWN` The exact six-task/dataset list was not safely recoverable from the inspected accessible primary material. |
| leakage risks | `CODE_CONFIRMED` Extracted tables retain endpoint/service metadata and labels originate from directory structure. `UNKNOWN` Whether all such metadata are excluded from model input and whether splits are capture/group-disjoint. |

Paper/code difference: the repository contains a `min_pkts=5`-related statistic/filter path, but inspected evidence does not prove that every released training pipeline applies it; it must not be reported as a universal effective filter without a code-path trace.

## 8. TrafficFormer

- Accurate title: **TrafficFormer: An Efficient Pre-trained Model for Traffic Data**
- Year / venue: **2025, IEEE Symposium on Security and Privacy** (`PAPER_CONFIRMED`)
- Paper: <https://doi.org/10.1109/SP61157.2025.00102>
- Official code: <https://github.com/IDP-code/TrafficFormer>

| Field | Primary-source finding |
|---|---|
| raw input | `CODE_CONFIRMED` PCAP to flow/burst; packet hexadecimal bytes are bigram/BPE-tokenized, with datagram, length, time, direction and message-type feature streams. |
| granularity | `PAPER_CONFIRMED + CODE_CONFIRMED` Burst pre-training and flow-level downstream classification. |
| packet window | `CODE_CONFIRMED` Pre-training sequence length 512 and fine-tuning sequence length 320. `UNKNOWN` The exact universal packet-count rule was not established from the inspected official paths. |
| bytes/packet | `CODE_CONFIRMED` Data-generation option `select_packet_len=64`; exact interpretation relative to the `start_index=28` slicing path must be traced before calling this “64 payload bytes.” |
| header / payload | `CODE_CONFIRMED` Packet/datagram bytes with a configurable start index. `UNKNOWN` The start-index unit and precise protocol-layer boundary are not safely inferred from the option name alone. |
| boundary | `CODE_CONFIRMED` Flow/burst structure is retained; pre-training text uses `||` for new streams and blank lines for adjacent bursts. |
| direction | `CODE_CONFIRMED` Explicit direction feature. |
| IAT | `CODE_CONFIRMED` Explicit time feature. `UNKNOWN` Exact formula was not verified as adjacent-packet IAT. |
| length | `CODE_CONFIRMED` Explicit length feature. |
| statistics | `UNKNOWN` No independent flow-statistics vector verified. |
| burst | `PAPER_CONFIRMED + CODE_CONFIRMED` Yes. |
| pretraining | `PAPER_CONFIRMED` Masked BURST Modeling and Same-Origin-Direction-Flow. |
| encoder | `PAPER_CONFIRMED` BERT-style Traffic Transformer; exact layer count should be taken from the selected official configuration, not assumed. |
| objective | `PAPER_CONFIRMED` Masked-token prediction and SODF, followed by supervised classification; RIFA augments fine-tuning. |
| tasks / datasets | `PAPER_CONFIRMED` Traffic classification and protocol-understanding tasks on six datasets. `UNKNOWN` Complete six-dataset names were not safely extracted from the accessible official text. |
| leakage risks | `PAPER_CONFIRMED` RIFA randomizes initializable fields to reduce shortcut learning. `CODE_CONFIRMED` Header-derived inputs remain, so the exact randomized field set must be audited in the selected path. |

Paper/code difference: the paper-level representation and released preprocessing contain multiple data streams and configurable slicing; the Stage 15F benchmark must not reduce this to an unverified shorthand such as “five packets of pure payload.”

## 9. DecETT

- Accurate title: **DecETT: Accurate App Fingerprinting Under Encrypted Tunnels via Dual Decouple-based Semantic Enhancement**
- Year / venue: **2025, The Web Conference (WWW '25)** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/2504.15565>
- Official code: <https://github.com/DecETT/DecETT>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED + CODE_CONFIRMED` Paired original TLS flow and encrypted-tunnel flow represented by packet payload-length sequences. |
| granularity | `PAPER_CONFIRMED` Flow; privileged TLS/tunnel pairs are used in training, tunnel flow alone at inference. |
| packet window | `PAPER_CONFIRMED` Maximum 200 packets in the formal setting. |
| bytes/packet | `PAPER_CONFIRMED` No raw packet bytes. |
| header / payload | `PAPER_CONFIRMED` No header/payload content; payload length only. |
| boundary | `PAPER_CONFIRMED` One signed length token per packet. |
| direction | `CODE_CONFIRMED` Sign of payload length encodes direction. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `PAPER_CONFIRMED + CODE_CONFIRMED` Core signal. |
| statistics | `PAPER_CONFIRMED` No separate statistics vector. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` No generic pre-training phase. |
| encoder | `PAPER_CONFIRMED` Partly shared Siamese architecture; protocol-view and app-view encoder/decoders use two-layer stacked Bi-GRUs. |
| objective | `PAPER_CONFIRMED` Self-reconstruction, gradient-reversal protocol-semantic minimization, cross-protocol reconstruction/decoupling, TLS-tunnel semantic alignment and app classification. |
| tasks / datasets | `PAPER_CONFIRMED` App fingerprinting for 54 apps under Shadowsocks, ShadowsocksR, V2Ray, Trojan and OpenVPN using self-collected paired flows. |
| leakage risks | `PAPER_CONFIRMED` Training requires privileged TLS/tunnel pairing derived from client socket mapping and start time. `INFERRED` Released sample-level random stratification is not confirmation of capture/pair-group isolation. |

Paper/code difference: code permits configurable sequence limits, while the registered paper-equivalent setting is 200 packets.

## 10. MT-FlowFormer

- Accurate title: **MT-FlowFormer: A Semi-Supervised Flow Transformer for Encrypted Traffic Classification**
- Year / venue: **2022, KDD '22, pp. 2576–2584** (`PAPER_CONFIRMED`)
- Paper: <https://doi.org/10.1145/3534678.3539314>
- Official code: `UNKNOWN` — no author-verified official repository was found.

| Field | Primary-source finding |
|---|---|
| raw input | `UNKNOWN` Accessible primary abstract did not expose the exact feature tuple. |
| granularity | `PAPER_CONFIRMED` Flow sequence. |
| packet window | `UNKNOWN` |
| bytes/packet | `UNKNOWN` |
| header / payload | `UNKNOWN` |
| boundary | `UNKNOWN` |
| direction | `UNKNOWN` |
| IAT | `UNKNOWN` |
| length | `UNKNOWN` |
| statistics | `UNKNOWN` |
| burst | `UNKNOWN` |
| pretraining | `PAPER_CONFIRMED` Mean Teacher semi-supervised learning rather than a generic offline traffic-pretraining stage. |
| encoder | `PAPER_CONFIRMED` Lightweight attention-based Flow Transformer in a teacher–student framework. |
| objective | `PAPER_CONFIRMED` Supervised classification plus teacher–student consistency; exact loss form remains `UNKNOWN`. |
| tasks / datasets | `PAPER_CONFIRMED` Low-label encrypted traffic classification on two real-world datasets. Dataset names remain `UNKNOWN` from accessible primary text. |
| leakage risks | `UNKNOWN` Preprocessing, capture/group isolation and code-level split semantics cannot be audited without paper full text or official code. |

Paper/code difference: not assessable because no official code was verified. This work must not be used to justify an exact Stage 15F feature vector until the primary full text is obtained.

## 11. Tracegram

- Accurate title: **Tracegram: Framing Trace-Level Traffic Analysis with Temporally-Aware Multiple Instance Learning**
- Year / venue: **2026, 35th USENIX Security Symposium, pp. 6127–6146** (`PAPER_CONFIRMED`)
- Paper: <https://www.usenix.org/conference/usenixsecurity26/presentation/qu>, <https://www.usenix.org/system/files/usenixsecurity26-qu.pdf>
- Official code: <https://github.com/YuchenZhang-Academic/Tracegram>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` PCAP to five-tuple flows; complete packet header fields and application payload bytes become tokens. |
| granularity | `PAPER_CONFIRMED` Packet token to flow instance to trace bag (multiple-instance learning). |
| packet window | `PAPER_CONFIRMED` Maximum flow-token length `Lmax=12,032`; UAV uses non-overlapping five-second traces, while other tasks use interaction/startup/session trace boundaries. |
| bytes/packet | `PAPER_CONFIRMED` Complete packet serialized as hexadecimal-character tokens; no fixed per-packet byte cap was established. |
| header / payload | `PAPER_CONFIRMED` Full headers and application payload, plus link-layer protocol token. |
| boundary | `PAPER_CONFIRMED` Per-packet `[Start]`, flow boundary, flow order and trace boundary are retained. |
| direction | `UNKNOWN` No independent direction channel verified; full headers can implicitly encode endpoints. |
| IAT | `PAPER_CONFIRMED` Packet IAT is quantized into timestamp tokens; the temporal trace encoder also uses inter-flow time gaps. |
| length | `UNKNOWN` No independent size token verified; serialized token-block length can implicitly expose size. |
| statistics | `PAPER_CONFIRMED` No hand-crafted flow-statistics vector. |
| burst | `PAPER_CONFIRMED` No explicit burst input; local attention is intended to learn micro-burst patterns. |
| pretraining | `PAPER_CONFIRMED` Two-step supervised tuning: universal flow encoder tuning on public data, then trace-label weak supervision for task adaptation. |
| encoder | `PAPER_CONFIRMED` CNN sequence compression, linear/local attention backbone with reversible network/token shift, and temporal BiLSTM/attention/weighted pooling. |
| objective | `PAPER_CONFIRMED` Flow-level supervised tuning, temporary trace-to-flow weak labels, then trace-level classification/MIL training. |
| tasks / datasets | `PAPER_CONFIRMED` UAV user activity, KWS encrypted keyword search, IDI IoT-device identification, ISD intrusion detection, plus DAPT attribution case study. |
| leakage risks | `PAPER_CONFIRMED` Full headers retain five-tuple identifiers. `UNKNOWN` Capture/entity-disjoint guarantees and pre-training/downstream source overlap were not established for every task. |

Paper/code difference: none established from the inspected sources.

## 12. Deep Packet

- Accurate title: **Deep Packet: A Novel Approach for Encrypted Traffic Classification Using Deep Learning**
- Year / venue: **2020, Soft Computing 24(3), 1999–2012; preprint 2017** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/1709.02656>
- Official code: `UNKNOWN` — no author-maintained official repository was verified; third-party reproductions are intentionally excluded.

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` Raw bytes of one IP packet. |
| granularity | `PAPER_CONFIRMED` Packet. |
| packet window | `PAPER_CONFIRMED` One packet per example. |
| bytes/packet | `PAPER_CONFIRMED` 1,500-byte vector: IP header plus up to 1,480 bytes of IP payload; byte values divided by 255 and short packets zero-padded. |
| header / payload | `PAPER_CONFIRMED` Ethernet header removed; IP header and transport/application content retained, with source/destination IP addresses masked. |
| boundary | `PAPER_CONFIRMED` Single-packet sample; no multi-packet boundary. |
| direction | `PAPER_CONFIRMED` Not used. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `PAPER_CONFIRMED` No explicit channel; zero padding indirectly exposes length. |
| statistics | `PAPER_CONFIRMED` None. |
| burst | `PAPER_CONFIRMED` None. |
| pretraining | `PAPER_CONFIRMED` CNN has none; stacked autoencoder uses layer-wise reconstruction pre-training before supervised fine-tuning. |
| encoder | `PAPER_CONFIRMED` 1D CNN or stacked autoencoder. |
| objective | `PAPER_CONFIRMED` SAE reconstruction plus supervised classification, or supervised CNN classification. |
| tasks / datasets | `PAPER_CONFIRMED` Application identification and traffic characterization on ISCX VPN-nonVPN 2016. |
| leakage risks | `PAPER_CONFIRMED` Authors found IP/application correlation and mask source/destination IPs. `UNKNOWN` Comprehensive masking of remaining header shortcuts. |

Additional paper-confirmed filtering: payload-free TCP SYN/ACK/FIN packets and DNS packets are discarded. Paper/code comparison is unavailable because no official code was verified.

## 13. FlowPic

- Accurate title: **FlowPic: Encrypted Internet Traffic Classification is as Easy as Image Recognition**
- Year / venue: **2019, IEEE INFOCOM Workshops** (`PAPER_CONFIRMED`)
- Paper/author page: <https://talshapira.github.io/publication/flowpic_ni19>
- Official code: <https://github.com/talshapira/FlowPic>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` Packet arrival time and IP packet size. |
| granularity | `PAPER_CONFIRMED` Unidirectional flow/session image. |
| packet window | `PAPER_CONFIRMED` Fixed-duration flow chunks; experiments use 15-second or 60-second windows. |
| bytes/packet | `PAPER_CONFIRMED` No packet bytes. |
| header / payload | `PAPER_CONFIRMED` No header/payload content. |
| boundary | `PAPER_CONFIRMED` Packets contribute points/counts to a 2-D histogram; original packet slots are not retained. |
| direction | `PAPER_CONFIRMED` No independent direction channel because each representation is unidirectional. |
| IAT | `PAPER_CONFIRMED` Arrival-time/time axis is used, not a separate adjacent-IAT vector. |
| length | `PAPER_CONFIRMED` Packet size is one histogram axis. |
| statistics | `PAPER_CONFIRMED` Histogram counts are aggregated statistics; no extra hand-crafted vector. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` None. |
| encoder | `PAPER_CONFIRMED` LeNet-5-like CNN. |
| objective | `PAPER_CONFIRMED` Supervised image classification. |
| tasks / datasets | `PAPER_CONFIRMED` Traffic category, application, encryption technique, and unknown application/category transfer on ISCX VPN-nonVPN, ISCX Tor-nonTor and TAU captures. |
| leakage risks | `PAPER_CONFIRMED + CODE_CONFIRMED` A labeled capture can contain STUN/HTTPS auxiliary sessions; inheriting the capture filename label at session level can mislabel flows. |

Paper/code difference: the official repository also supports the later 2021 extension; this entry records the original 2019 method, so configuration from the extension must not be silently mixed in.

## 14. FS-Net

- Accurate title: **FS-Net: A Flow Sequence Network For Encrypted Traffic Classification**
- Year / venue: **2019, IEEE INFOCOM** (`PAPER_CONFIRMED`)
- Paper: <https://doi.org/10.1109/INFOCOM.2019.8737507>
- Official code: <https://github.com/WSPTTH/FS-Net>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED + CODE_CONFIRMED` Flow packet-length sequence; released records also contain an encoded status sequence. |
| granularity | `PAPER_CONFIRMED` Flow. |
| packet window | `CODE_CONFIRMED` Released defaults accept sequence length 2–200 for train and up to 2,000 at test; over-limit flows are discarded. |
| bytes/packet | `PAPER_CONFIRMED` No raw bytes. |
| header / payload | `PAPER_CONFIRMED` No header/payload content. |
| boundary | `PAPER_CONFIRMED` One sequence element per packet position. |
| direction | `UNKNOWN` Inspected primary sources did not establish whether direction is an independent channel or encoded in status/length. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `PAPER_CONFIRMED` Core input. |
| statistics | `PAPER_CONFIRMED` No extra hand-crafted statistics. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` No separate unsupervised pre-training; reconstruction and classification are jointly trained. |
| encoder | `PAPER_CONFIRMED + CODE_CONFIRMED` GRU-style encoder-decoder flow sequence network. |
| objective | `PAPER_CONFIRMED` Sequence reconstruction plus supervised classification. `CODE_CONFIRMED` Default reconstruction-loss coefficient 0.5. |
| tasks / datasets | `PAPER_CONFIRMED` Encrypted application classification. `CODE_CONFIRMED` Released setup defaults to 18 classes; the formal dataset name remains `UNKNOWN`. |
| leakage risks | `INFERRED` Excluding payload/IP/port reduces direct identifier shortcuts. `UNKNOWN` Capture/group-disjoint split guarantees. |

Paper/code difference: released train and test maximum lengths differ substantially (200 vs. 2,000), so preprocessing equivalence must record split-specific filtering.

## 15. FlowLens

- Accurate title: **FlowLens: Enabling Efficient Flow Classification for ML-based Network Security Applications**
- Year / venue: **2021, NDSS Symposium** (`PAPER_CONFIRMED`)
- Paper: <https://www.ndss-symposium.org/ndss-paper/flowlens-enabling-efficient-flow-classification-for-ml-based-network-security-applications/>, <https://www.ndss-symposium.org/wp-content/uploads/ndss2021_7C-2_24067_paper.pdf>
- Official code: `UNKNOWN` — no author-verified public repository was found.

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` Online packet lengths and, where required, inter-packet times; five-tuple is used to index flows. |
| granularity | `PAPER_CONFIRMED` Flow-level frequency-distribution sketch (“flow marker”). |
| packet window | `PAPER_CONFIRMED` Ordinarily accumulated over a flow; botnet conversation aggregation uses `flowgap=3600 s`. No universal first-N packet cap was stated. |
| bytes/packet | `PAPER_CONFIRMED` No raw bytes. |
| header / payload | `PAPER_CONFIRMED` No content bytes used for classification. |
| boundary | `PAPER_CONFIRMED` Five-tuple flow boundary. |
| direction | `PAPER_CONFIRMED` Website fingerprinting uses separate incoming/outgoing packet-length distributions; other tasks are configuration-dependent. |
| IAT | `PAPER_CONFIRMED` Botnet detection uses an inter-packet-time distribution and stores the prior timestamp per flow. |
| length | `PAPER_CONFIRMED` Packet-length distribution is a core signal. |
| statistics | `PAPER_CONFIRMED` Quantized and truncated frequency histograms/markers; quantization and top-bin selection are application-profiled. |
| burst | `PAPER_CONFIRMED` No explicit burst segmentation. |
| pretraining | `PAPER_CONFIRMED` No representation pre-training. A profiling pass trains a task classifier to select bins; that is not foundation-model pre-training. |
| encoder | `PAPER_CONFIRMED` No universal neural encoder: XGBoost for covert channels, Multinomial Naive Bayes for website fingerprinting, and Random Forest for botnet chatter. |
| objective | `PAPER_CONFIRMED` Task-specific supervised classification plus Bayesian optimization of marker size/accuracy. |
| tasks / datasets | `PAPER_CONFIRMED` Skype Facet/DeltaShaper covert channels, Herrmann et al. OpenSSH website traces, and PeerShark/P2P botnet chatter. |
| leakage risks | `UNKNOWN` Capture/host/site group isolation was not verified. `INFERRED` Application-specific feature selection must be nested inside training data to avoid feature-selection leakage; the inspected text does not establish that as a general guarantee. |

Paper/code difference: not assessable because no official code was verified.

## 16. Rosetta

- Accurate title: **Rosetta: Enabling Robust TLS Encrypted Traffic Classification in Diverse Network Environments with TCP-Aware Traffic Augmentation**
- Year / venue: **2023, 32nd USENIX Security Symposium, pp. 625–642** (`PAPER_CONFIRMED`)
- Paper: <https://www.usenix.org/conference/usenixsecurity23/presentation/xie>, <https://www.usenix.org/system/files/usenixsecurity23-xie.pdf>
- Official code: <https://github.com/sunskyXX/Rosetta>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` TLS-flow packet-length sequence. |
| granularity | `PAPER_CONFIRMED` Flow. |
| packet window | `UNKNOWN` A fixed first-N or maximum sequence length was not verified in the inspected primary material. |
| bytes/packet | `PAPER_CONFIRMED` No raw bytes. |
| header / payload | `PAPER_CONFIRMED` No content bytes. |
| boundary | `PAPER_CONFIRMED` Packet-by-packet length sequence. |
| direction | `UNKNOWN` Independent signed-direction encoding was not verified. |
| IAT | `PAPER_CONFIRMED` Not a model input; RTT is used only to simulate TCP-aware augmentation. |
| length | `PAPER_CONFIRMED` Core input. |
| statistics | `PAPER_CONFIRMED` No explicit hand-crafted flow-statistics vector. |
| burst | `PAPER_CONFIRMED` No explicit burst input. |
| pretraining | `PAPER_CONFIRMED` BYOL-style self-supervised Traffic Invariant Extractor trained on a third-party TLS dataset disjoint from the two principal classification datasets. |
| encoder | `PAPER_CONFIRMED` BYOL online/target encoder, projector and predictor; learned invariant is attached to CNN, SDAE, LSTM, DF, FS-Net and Transformer classifiers. |
| objective | `PAPER_CONFIRMED` Symmetric normalized-MSE agreement between two TCP-aware variants of the same flow. |
| tasks / datasets | `PAPER_CONFIRMED` Robust TLS website/application classification using CIRA-CIC-DoHBrw-2020, ISCX-VPN, a disjoint TLS pre-training set, and real website/application captures. |
| leakage risks | `PAPER_CONFIRMED` Pre-training data are disjoint from the two main classification datasets. `UNKNOWN` Downstream capture/group-disjoint guarantees; replayed variants measure controlled domain shift, not fully independent-source generalization. |

Paper/code difference: none established. Augmentation parameters are method mechanics rather than input features: loss rate 0–10%, duplicated subsequence length ranges, RTT 0–200 ms, and MSS 500–1,500 B.

## 17. Deep Fingerprinting

- Accurate title: **Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning**
- Year / venue: **2018, ACM CCS** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/1801.02265>, <https://mjuarezm.github.io/assets/pdf/ccs18.pdf>
- Official code: <https://github.com/deep-fingerprinting/df>

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` Tor packet/cell direction sequence with `+1/-1` values. |
| granularity | `PAPER_CONFIRMED` Website-visit/session trace. |
| packet window | `CODE_CONFIRMED` Fixed 5,000 direction elements, truncated or zero-padded. |
| bytes/packet | `PAPER_CONFIRMED` No raw bytes. |
| header / payload | `PAPER_CONFIRMED` No content. |
| boundary | `PAPER_CONFIRMED` Sequence position preserves packet order. |
| direction | `PAPER_CONFIRMED` Core and sole per-packet input signal. |
| IAT | `PAPER_CONFIRMED` Not used. |
| length | `PAPER_CONFIRMED` Packet size is not used. |
| statistics | `PAPER_CONFIRMED` No hand-crafted statistics. |
| burst | `PAPER_CONFIRMED` No explicit burst feature. |
| pretraining | `PAPER_CONFIRMED` None. |
| encoder | `PAPER_CONFIRMED` Deep 1D CNN. |
| objective | `PAPER_CONFIRMED` Supervised website classification. |
| tasks / datasets | `PAPER_CONFIRMED` Closed- and open-world Tor website fingerprinting on undefended, WTF-PAD and Walkie-Talkie traces, including 95 monitored websites. |
| leakage risks | `PAPER_CONFIRMED` Payload, addresses and ports are absent. `UNKNOWN` Collection/session-template independence beyond the reported protocol. |

Paper/code difference: none established from inspected sources.

## 18. The Sweet Danger of Sugar

- Accurate title: **The Sweet Danger of Sugar: Debunking Representation Learning for Encrypted Traffic Classification**
- Year / venue: **2025, ACM SIGCOMM** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/2507.16438>
- Official code: <https://github.com/SmartData-Polito/Debunk_Traffic_Representation>

This work is a representation-learning diagnosis and introduces Pcap-Encoder; it is not one ordinary fixed classifier. The fields below describe Pcap-Encoder where applicable and the paper's audit conclusions otherwise.

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` Protocol-header bytes represented as two-byte hexadecimal words and tokenized with a T5 tokenizer. |
| granularity | `PAPER_CONFIRMED` Pcap-Encoder is packet-level; the broader audit covers packet- and flow-level representations. |
| packet window | `PAPER_CONFIRMED` One packet. Exact byte cap is `UNKNOWN`. |
| bytes/packet | `PAPER_CONFIRMED` Header byte words; exact cap `UNKNOWN`. |
| header / payload | `PAPER_CONFIRMED` Header retained, payload content ignored; payload length is one question/answer semantic. |
| boundary | `PAPER_CONFIRMED` Single packet; field semantics are learned through question answering rather than a separate hand-crafted boundary channel. |
| direction | `PAPER_CONFIRMED` No explicit direction channel. |
| IAT | `PAPER_CONFIRMED` Not used by Pcap-Encoder. |
| length | `PAPER_CONFIRMED` Payload length is a Q&A target, not an ordinary appended statistic. |
| statistics | `PAPER_CONFIRMED` No separate flow-statistics vector. |
| burst | `PAPER_CONFIRMED` No burst input. |
| pretraining | `PAPER_CONFIRMED` T5-base initialization, bottleneck autoencoding, then eight categories of packet-header Q&A. |
| encoder | `PAPER_CONFIRMED` T5-base encoder producing a 768-dimensional representation. |
| objective | `PAPER_CONFIRMED` Packet reconstruction cross-entropy plus header retrieval/computation Q&A. |
| tasks / datasets | `PAPER_CONFIRMED` Representation benchmark and downstream traffic classification; Pcap-Encoder training uses MAWI, UNSW-NB15 and campus traffic, with six downstream tasks including VPN-app and TLS-120. |
| leakage risks | `PAPER_CONFIRMED` Random packet splitting can place packets from one flow in train and test; full fine-tuning can erase pre-trained structure; header/IP strong-identifying information can act as shortcuts. |

Paper/code difference: no specific contradiction established; the key scope warning is that the paper's diagnostic variants and Pcap-Encoder must not be collapsed into one generic feature configuration.

## 19. SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers

- Accurate title: **SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers**
- Year / venue: **2025, IEEE Symposium on Security and Privacy** (`PAPER_CONFIRMED`)
- Paper: <https://arxiv.org/abs/2503.20093>
- Official code: <https://github.com/nime-sha256/ntc-enigma>

This is a systematization and occlusion audit, not a new encoder with one fixed input.

| Field | Primary-source finding |
|---|---|
| raw input | `PAPER_CONFIRMED` Taxonomy covers raw packet bytes, five-tuples, packet sizes, timestamps and other protocol fields across audited methods. |
| granularity | `PAPER_CONFIRMED` Packet, burst, flow and session are all analyzed; there is no unique SoK granularity. |
| packet window | `PAPER_CONFIRMED` Compares first-`m` bytes, first-`m` bytes of `n` packets, per-packet prefixes and consecutive-packet strategies; no unique window. |
| bytes/packet | `PAPER_CONFIRMED` Depends on the audited model/occlusion variant. |
| header / payload | `PAPER_CONFIRMED` Separately occludes and analyzes header, payload and protocol-field groups. |
| boundary | `PAPER_CONFIRMED` Compares data units and packet-selection strategies; no unique boundary encoding. |
| direction | `PAPER_CONFIRMED` Direction is an audited traffic attribute; no unique SoK input channel. |
| IAT | `PAPER_CONFIRMED` Timestamp/timing information is audited; no unique IAT input. |
| length | `PAPER_CONFIRMED` Packet size is a core audited property. |
| statistics | `UNKNOWN` No unified statistics vector applies. |
| burst | `PAPER_CONFIRMED` Burst is part of the taxonomy. |
| pretraining | `PAPER_CONFIRMED` Not applicable to the SoK itself; it evaluates pre-trained models including ET-BERT and YaTC. |
| encoder | `PAPER_CONFIRMED` No new unified encoder; existing classifiers are reproduced and occluded. |
| objective | `PAPER_CONFIRMED` Systematization, feature occlusion, shortcut diagnosis and generalization evaluation. |
| tasks / datasets | `PAPER_CONFIRMED` Encrypted-network-traffic classifier audit across legacy public datasets and the introduced CipherSpectrum TLS 1.3 corpus. Complete table transcription remains `UNKNOWN` here. |
| leakage risks | `PAPER_CONFIRMED` MAC/IP/port strong-identifying information, IP ID/checksum, TCP sequence/acknowledgment, SNI, random/per-packet splits and padding/truncation can form shortcuts; encrypted payload often exposes length rather than stable semantic content. |

Paper/code difference: not applicable as a single model. The SoK taxonomy and its 348 occlusion experiments must not be registered as one fixed feature configuration.

## Cross-paper implementation discrepancies and audit cautions

1. **MIETT:** the paper is public, but the official repository still says “Code is coming soon”; no code-equivalent preprocessing claim is currently defensible.
2. **ET-BERT:** released fine-tuning applies packet-count and fixed-prefix conditions not obvious from the paper overview.
3. **YaTC:** the paper's IP/port masking description is not visibly mirrored by the inspected released preprocessing path.
4. **TrafficFormer:** configurable byte slicing, token lengths and multiple auxiliary streams must be traced in the selected config; `time` must not automatically be renamed IAT, nor the selected byte segment called pure payload.
5. **AN-Net:** comments and byte slicing do not fully resolve the semantic boundary of the 64-byte TCP-layer segment.
6. **MM4flow:** endpoint/service metadata and a possible minimum-packet path exist in preprocessing, but neither may be asserted as formal model input/effective filtering without an execution-path trace.
7. **FS-Net:** official defaults use different train/test maximum sequence lengths, creating split-dependent filtering.
8. **FlowPic:** repository material spans the 2019 method and later extension; configurations must be versioned.
9. **Deep Packet, MT-FlowFormer, FlowLens:** no author-verified official implementation was found, so paper-only and code-equivalent claims must remain separate.
10. **Sugar and SoK:** both are principally diagnostic works. Their taxonomies/ablations are evidence for feature design and leakage control, not ready-made single encoders.

## Feature families supported by the primary evidence

The literature supports several distinct families; it does **not** support treating one as universally best:

- `PAPER_CONFIRMED` raw byte/content families: ET-BERT, YaTC, TFE-GNN, MIETT, MH-Net, TrafficFormer, Tracegram, Deep Packet and Pcap-Encoder.
- `PAPER_CONFIRMED` packet-length/direction sequence families: FS-Net, DecETT, Rosetta and Deep Fingerprinting; MM4flow adds a packet-size/transmission-pattern modality.
- `PAPER_CONFIRMED` timing families: AN-Net uses packet IAT; FlowPic uses arrival-time bins; FlowLens can use IPT distributions; Tracegram tokenizes packet IAT and inter-flow gaps.
- `PAPER_CONFIRMED` aggregate/statistical families: AN-Net uses moments/FFT plus flags/TTL; FlowPic uses 2-D histograms; FlowLens uses quantized/truncated frequency markers.
- `PAPER_CONFIRMED` burst-aware families: ET-BERT and TrafficFormer explicitly construct bursts; MM4flow represents transmission/burst patterns. Other models may learn local patterns without a declared burst input.

These findings justify feature **families** for a later registry, not a final benchmark choice. Exact Stage 15F candidates must additionally pass local availability, leakage-control, split, and reproducibility checks.

## Unresolved primary-source items

- **MT-FlowFormer:** exact raw input, feature tuple, packet window, datasets, and author-maintained official code.
- **TrafficFormer:** exact packet-count rule for each official fine-tuning path, exact `time` formula, exact meaning of `start_index`, selected RIFA fields, and complete six-dataset table.
- **MM4flow:** complete task/dataset list, whether connection-level metadata enter the trained model, and whether `min_pkts=5` is effective in the formal run path.
- **Rosetta:** fixed packet/sequence cap and explicit direction encoding, if any.
- **MH-Net:** exact per-packet byte cap.
- **FlowLens:** official implementation availability and group/capture split semantics.
- **Deep Packet:** author-maintained official implementation availability.
- **FS-Net:** formal dataset name and exact semantics of the released status sequence.
- **Tracegram:** task-by-task capture/entity-disjoint guarantees and pre-training/downstream source overlap.
- **YaTC:** which official executable preprocessing path, if any, enforces the paper-described IP/port anonymization.

No unresolved item above should be filled from a secondary survey without preserving the `UNKNOWN` status and citing the secondary source separately in a later, explicitly non-primary appendix.
