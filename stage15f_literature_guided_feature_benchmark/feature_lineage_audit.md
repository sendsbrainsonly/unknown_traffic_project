# Stage 15F-0 Feature Lineage Audit

## Scope and audit boundary

This is a read-only feature recoverability and packet-window audit. It did not train an encoder, fit a detector, evaluate Known Test or Unknown Test, or alter any frozen Stage 12--15R artifact. The only sample-level values materialized here are the union of **Known Train and Known Validation** samples in each frozen protocol.

Missing and unrecoverable fields remain explicit. No missing value is represented as an observed zero.

## Dataset lineage

| Dataset | Frozen sample identity | Flow/session construction | Raw packet source used here | Known Train/Val cache | Test values stored |
|---|---|---|---|---:|---:|
| USTC-TFC2016 | Stage 3 A-1/A-2/A-3 manifests joined to Stage-0 `flow_id` | Bidirectional IPv4 TCP/UDP five-tuple; no timeout/session split in the retained Stage-0 PKL | Stored packet tuples `(timestamp, caplen, wirelen, direction, raw_frame)` | 432,537 | 0 |
| ISCX-VPN | Stage 12 SHA256 session ID | Bidirectional IPv4 TCP/UDP; 60 s timeout; TCP SYN/FIN/RST boundaries | Frozen capture list replayed with `tshark`; non-target packets were used only to preserve session state and discarded | 18,121 | 0 |
| ISCXTor2016 | Stage 12 SHA256 session ID | Same Stage 12 state machine as ISCX-VPN | Frozen capture list replayed with `tshark` | 11,783 | 0 |
| VNAT | Stage 14B `flow_uid` mapped to `capture_id` plus `tcp.stream`/`udp.stream` | `tshark` stream inside capture; application is metadata, VPN/non-VPN is not a semantic class | 162 valid, non-duplicate captures from the Stage 14 lineage | 23,447 | 0 |

The VNAT clean pool contains 23,449 frozen flows. The two rows absent from this audit cache are Test-only rows, not extraction losses. Previously isolated corrupted captures, the exact duplicate capture contribution, and four cross-application duplicate flows remain excluded exactly as frozen in Stage 14B.

## Recoverable fields

All four datasets preserve or can deterministically recover packet order, timestamps, endpoint-relative direction, captured frame length, packet count, duration, and first-N cumulative frame bytes/time. Header bytes and visible TCP/UDP payload bytes can be recovered by replaying the raw packet/frame through a layer parser. A valid-packet mask is derived from the observed packet count and the declared window.

Important non-equivalences:

- USTC uses previously materialized Stage-0 raw frames and its no-timeout five-tuple flow definition.
- ISCX-VPN and ISCXTor use the Stage 12 60-second session state machine and TCP boundary rules.
- VNAT uses capture-local `tshark` stream identity, not the Stage 12 session hash.
- A captured frame length is not interchangeable with IPv4 total length or transport-payload length.
- For VPN and Tor captures, visible headers/payload describe the **outer tunnel**. They do not reveal inner application headers or plaintext.

The full field-level matrix is in `dataset_feature_availability.csv`.

## Packet-window definitions

For each flow and `N in {8,16,32,64}`:

- `complete_coverage_ratio`: fraction with packet count `<= N`;
- `truncation_ratio`: fraction with packet count `> N`;
- `padded_flow_ratio`: fraction with packet count `< N`;
- byte coverage: cumulative captured-frame bytes in the first N packets divided by full-flow captured-frame bytes;
- time coverage: elapsed time from the first through Nth retained packet divided by full-flow duration;
- zero-duration truncated flows have undefined time coverage and are counted separately rather than imputed.

Every aggregate is labeled `EARLY_N`, `FULL_FLOW`, or `MIXED_EARLY_N_AND_FULL_FLOW`. Current dataset/protocol aggregates are mixed because they contain both flows completed by N and longer flows truncated at N. Later experiments must not compare a full-flow statistic branch with an early-N detector without declaring the latency difference.

## Cache integrity

The new caches were checked against frozen Stage 15R full-flow aggregates on sample ID, packet count, captured bytes, and duration. USTC, ISCX-VPN, ISCXTor, and VNAT all passed with zero missing target flows and zero unmatched protocol membership rows. The machine-readable evidence is `window_cache_parity.json` and `stage15f0_aggregation_audit.json`.

One initial Scapy implementation for ISCX-VPN was stopped because replaying a multi-gigabyte capture was operationally too slow. Its log and partial artifacts are preserved under `failed_attempts/iscx_vpn_scapy_slow/`. The final cache uses the semantically equivalent Stage 15R-verified `tshark` field stream and passed exact parity checks.

## Encryption and visibility interpretation

The audit keeps application/transport encryption separate from tunnel encryption:

- `TLS_QUIC`
- `SSH`
- `VISIBLE_PLAINTEXT`
- `UNDETERMINED`
- `UNDETERMINED_WITHIN_TUNNEL`
- tunnel type `VPN`, `TOR`, or `NONE_OBSERVED`

`non-VPN` does **not** mean plaintext. USTC is left `UNDETERMINED` because the frozen lineage has no reliable per-flow encryption annotation. VNAT VPN and ISCX-VPN VPN samples expose outer tunnel traffic; ISCXTor Tor samples expose outer Tor traffic. Counts in `encryption_visibility_audit.csv` are protocol-membership counts and therefore must not be summed across overlapping protocols as unique dataset flow counts.

## Feature-family lineage decision

- B0 is exactly the existing eight-packet 32x32 byte image baseline.
- B1--B4 require raw packet/frame replay and explicit byte/packet/segment masks.
- T1 is recoverable on all four datasets from length, direction, and IAT.
- T2 burst features are recoverable but EARLY_N-derived and FULL_FLOW variants are distinct configurations.
- S1/S2 are full-flow statistics and carry capture-duration/volume leakage risk.
- M1--M3 inherit the visibility, latency, and normalization constraints of their component branches.
- P1 remains blocked pending checkpoint-license, preprocessing-parity, and pretraining-overlap audits; the presence of local weights is not sufficient evidence of a valid benchmark.

## Strict Unknown-Free result

`unknown_test_feature_values_used = 0` and `known_test_feature_values_used = 0` in the aggregation audit. All normalization rules for later stages are preregistered as Known-Train-only, while model/checkpoint selection is Known-Validation-only.
