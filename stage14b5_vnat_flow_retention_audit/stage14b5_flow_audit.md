# Stage 14B.5 — VNAT Flow Retention & Filtering Audit

## Audit boundary and verdict

- **Verdict: `PASS_WITH_EXPLAINED_DIFFERENCE`.** The frozen Stage 14B clean pool closes exactly as `23,454 - 1 - 4 = 23,449` flows.
- This audit did not regenerate Stage 14B, did not train a model, and did not run Open-Detect/DES.
- Frozen protocol identity before the audit: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`; protocol JSON `5177bf7276812bbb417f085043f8518c8573ea1851de2f1d733deb77c9939ced`; split manifest `66bf155e6906231b316c384a52f4b46dabd9d3aee76d5f66d86a3941eb172c4e`.
- The raw-to-clean story has one important boundary: a PCAP has no unique "raw flow count" until a sessionization definition is chosen. The two truncated PCAPs were rejected before Stage 14A flow construction, so their exact counterfactual Stage 14A full-file flow count is unknowable. We report both a read-only readable-prefix diagnostic and the official connection dataframe count, without treating either as interchangeable with the frozen local flow definition.

## Exact local quantity conservation

```text
Stage 14A flows from 163 fully readable PCAPs = 23,454
- exact duplicate PCAP contribution            =      1
- cross-application duplicate flow rows        =      4
- all other flow filters                       =      0
--------------------------------------------------------
Stage 14B clean pool                           = 23,449
```

The five successfully constructed flows that did not enter the clean pool are fully identified: one UDP flow from duplicate `vpn_skype-chat_capture4`, plus four non-TCP/UDP rows representing two packet-identical sequences present under both `rsync` and `sftp`. There is no unresolved gap inside the accepted Stage 14A flow pool.

For the broader raw-file audit, tshark emitted all complete packet records before each truncation error and the diagnostic applied the same grouping semantics:

```text
Locally groupable flows over all complete packet rows = 34,009
- flows in two whole rejected truncated PCAPs          = 10,555
- exact duplicate PCAP contribution                    =      1
- cross-application duplicate flow rows                =      4
----------------------------------------------------------------
Stage 14B clean pool                                    = 23,449
```

Thus **10,560 locally groupable flows are absent from the final pool when the corrupt-PCAP gate is included**, while only five of those existed as Stage 14A flow rows and were then deleted. This second ledger is diagnostic-only; it does not mutate or repopulate Stage 14B.

## PCAP and packet path

- Source inventory: 165 PCAPs.
- Fully passed `capinfos` and complete `tshark` traversal: 163; rejected as truncated: 2.
- Packets in the 163 successful PCAPs: **24,901,394**; IP-eligible packets: **24,859,370**; non-IP packets excluded before flow grouping: **42,024**.
- Resulting flows: **23,454** = TCP 3,795 + UDP 19,643 + other-IP 16.
- Single-packet flows retained: **30**. Flows with fewer than five packets retained: **20,027**.

### Two truncated PCAPs

| capture | capinfos packets before error | tshark prefix rows | prefix flows under local grouping | official H5 connections | Stage 14B contribution |
|---|---:|---:|---:|---:|---:|
| nonvpn_scp_long_capture1 | 13,474,841 | 13,474,841 | 10,554 | 10,555 | 0 |
| vpn_skype-chat_capture6 | 10,284 | 10,284 | 1 | 1 | 0 |

The current pipeline loses the entire contribution of both captures because `scan_pcap()` calls `capinfos()` first, and any non-zero return jumps to the exception handler before `scan_flows()` is invoked. Thus the **pipeline-observed contribution is exactly zero**. The official H5 contains **10,556** connections for those two filenames; the read-only diagnostic found **10,555** locally grouped flows in bytes emitted before the truncation error. These are loss indicators, not additions to the frozen protocol, and the unknown missing tail prevents claiming an exact full-file counterfactual.

## Actual construction and filtering rules

1. **PCAP parsing gate** — `capinfos -Tm -c -a -e -u -H`; any non-zero status rejects the whole PCAP (`audit_vnat.py:134-149,247-268`).
2. **Packet parsing** — `tshark -n -o frame.generate_md5_hash:TRUE -r ... -T fields`; no display/capture filter (`audit_vnat.py:152-158`). Runtime: `TShark (Wireshark) 3.6.2 (Git v3.6.2 packaged as 3.6.2-2)`.
3. **Packet eligibility** — only packets with an IPv4 or IPv6 source and destination enter flow construction (`audit_vnat.py:184-189`). This is a non-IP exclusion, not a TCP/UDP-only filter.
4. **Session construction** — TCP uses `tcp.stream`; UDP uses `udp.stream`; other IP protocols are retained and grouped bidirectionally by canonical endpoints plus IP protocol (`audit_vnat.py:190-203`). No explicit timeout is configured. Therefore TCP/UDP session boundaries depend on the installed tshark defaults; other-IP grouping has no time split.
5. **Flow summary** — packet count, captured byte count, first/last timestamp, tuple hash and ordered packet-MD5 content hash are recorded (`audit_vnat.py:205-243`). This is not the official 129-feature H5 extractor.
6. **No general flow filter** — no single/short-flow threshold, payload-length threshold, flow-duration threshold, TCP-completion requirement, port allowlist, or malformed-5-tuple rejection exists. Empty payload cannot be tested because payload fields are not extracted.
7. **Duplicate PCAP** — Stage 14A detects identical full-file SHA256; Stage 14B lexicographically retains `vpn_skype-chat_capture3` and drops `capture4` (`freeze_vnat_protocol.py:77-85,133-135`). Both contain one identical UDP flow, so exactly one flow is removed.
8. **Cross-application duplicate flows** — ordered packet hashes form `content_sha256`; all four rows spanning `nonvpn_rsync_newcapture1` and `nonvpn_sftp_newcapture2` are quarantined (`freeze_vnat_protocol.py:65-75,136-140`).
9. **Labels** — filename grammar maps `voip -> zoiper`, `skype-chat -> skype`, and all other keywords to the same application name; each flow inherits its PCAP's application and VPN status (`audit_vnat.py:29-45,100-124`). VPN and non-VPN use exactly the same extractor and filters.

The complete machine-readable rule table, including explicitly absent filters, is `stage14b5_filter_breakdown.csv`.

## Per-application retention

Here `raw_flow_count` means flows successfully constructed by Stage 14A from fully readable PCAPs; it does not silently substitute official H5 connections for failed-capture flows.

| application | raw local | removed after construction | final | retention | VPN raw/final | non-VPN raw/final | official all-PCAP connections | main reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| netflix | 205 | 0 | 205 | 100.0000% | 1/1 | 204/204 | 205 | none after successful flow construction |
| rdp | 44 | 0 | 44 | 100.0000% | 6/6 | 38/38 | 36 | none after successful flow construction |
| rsync | 1914 | 2 | 1912 | 99.8955% | 3/3 | 1911/1909 | 1915 | cross-app duplicate rows=2 |
| scp | 2290 | 0 | 2290 | 100.0000% | 2/2 | 2288/2288 | 12845 | failed PCAP excluded before local flow construction=nonvpn_scp_long_capture1 |
| sftp | 1670 | 2 | 1668 | 99.8802% | 4/4 | 1666/1664 | 1670 | cross-app duplicate rows=2 |
| skype | 1270 | 1 | 1269 | 99.9213% | 56/55 | 1214/1214 | 1301 | duplicate PCAP flow=1; failed PCAP excluded before local flow construction=vpn_skype-chat_capture6 |
| ssh | 13563 | 0 | 13563 | 100.0000% | 5/5 | 13558/13558 | 13563 | none after successful flow construction |
| vimeo | 1218 | 0 | 1218 | 100.0000% | 1/1 | 1217/1217 | 1218 | none after successful flow construction |
| youtube | 341 | 0 | 341 | 100.0000% | 3/3 | 338/338 | 341 | none after successful flow construction |
| zoiper | 939 | 0 | 939 | 100.0000% | 301/301 | 638/638 | 617 | none after successful flow construction |

`rdp`, `netflix`, `youtube`, and `ssh` have **no post-construction filtering at all**. Their apparent size differences are source/capture and sessionization properties, not preprocessing removals. `scp` is affected by whole-PCAP rejection of `nonvpn_scp_long_capture1`; `skype` is affected by one truncated PCAP and one duplicate-PCAP flow; `rsync` and `sftp` each lose two cross-application duplicate rows.

## Official H5 comparison

- Official `VNAT_Dataframe_release_1.h5`: **33,711** connection rows across all 165 filenames.
- Official rows belonging to the two failed local PCAPs: **10,556**; official rows for the 163 locally readable PCAPs: **23,155**.
- Local Stage 14A flow rows for those readable PCAPs: **23,454**. Difference (official minus local): **-299**.
- Applying the same one duplicate-PCAP and four cross-app row exclusions gives an indicative official count of **23,150**, versus local **23,449**, still **-299**.

This difference is **not a filter loss**: the official artifact stores prebuilt `connection` objects, while this pipeline uses tshark stream IDs and a custom rule for other IP protocols. No one-to-one connection matching or official extractor code exists in the audited local pipeline. Per-capture, per-application and VPN/non-VPN counts are in `artifacts/official_connection_comparison.csv` and `stage14b5_class_retention.csv`.

## Fairness assessment

- No class-dependent packet/flow filter and no VPN-specific filter was found.
- The main fairness risk is whole-capture exclusion: `scp` loses a large non-VPN capture and `skype` loses one VPN capture. This can alter class/domain composition, even though the exclusion is integrity-driven and applied before any model result.
- Filename-derived application labels and tshark default sessionization are reproducible in this environment but are not proven equivalent to the official connection dataframe definition.
- Duplicate removal is appropriate for leakage control: the duplicate Skype capture is not double-counted, and cross-application packet-identical rows cannot straddle Known/Unknown semantics.

## Required answers

1. **Why 23,449?** Exactly `23,454 successful Stage 14A flows - 1 duplicate-PCAP flow - 4 cross-application duplicate rows`.
2. **How many were deleted/filtered?** Five after successful Stage 14A flow construction. Counting all complete packet rows present in the two truncated files, 10,555 additional locally groupable flows were excluded at the whole-PCAP gate, so the diagnostic raw-file-to-clean total absent is **10,560**. The unavailable intended tail remains unknowable. Official H5 reports 10,556 connections for the two files.
3. **Rules?** Non-IP packets are excluded before grouping; two corrupted PCAPs fail the parser gate; one exact duplicate PCAP and four cross-app duplicate rows are removed. All queried short/payload/TCP/port/duration filters are absent.
4. **Code/functions?** `audit_vnat.py::{capinfos,scan_pcap,scan_flows,canonical_tuple,parse_pcap_name,duplicate_annotations}` and `freeze_vnat_protocol.py::{integrity_exclusions,build_clean_flows}`; exact locations are in the CSVs above.
5. **Key parameters?** `capinfos -Tm -c -a -e -u -H`; tshark `-n`, `frame.generate_md5_hash:TRUE`, listed fields, no display filter, no explicit timeout, and no flow-size/duration/payload thresholds.
6. **Unexpected filtering?** No hidden post-construction filter was found. Whole-PCAP rejection before partial salvage is consequential but explicit in control flow; official/local sessionization differs.
7. **Can 23,449 be strictly explained?** Yes, from the successful Stage 14A pool. The intended full-capture counterfactual for truncated tails cannot be reconstructed exactly.
8. **Potential external-validation fairness impact?** Yes: integrity exclusion changes `scp` non-VPN and `skype` VPN coverage, and official/local connection definitions differ. There is no evidence of outcome-driven or class-specific filtering.
9. **Freeze hash before/after?** Verified below and machine-recorded in `artifacts/freeze_hash_before.json` / `freeze_hash_after.json`.
10. **Final conclusion:** `PASS_WITH_EXPLAINED_DIFFERENCE`.

Freeze hash after audit: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` — **PASS, unchanged**. Protocol JSON and split manifest SHA256 also match their before-audit values.
