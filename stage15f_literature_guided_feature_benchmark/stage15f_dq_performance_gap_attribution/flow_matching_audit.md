# DQ-0/DQ-1 Flow Matching Audit

## PCAP scope

- Native inventory: 137 PCAP/PCAPNG files.
- TFE-GNN scope: 31 VPN PCAPs.
- TrafficFormer scope: 31 VPN PCAPs.
- Exact three-way shared scope: 31 PCAPs.
- Live SHA256 values for the shared PCAPs matched the prior verified capture audit.

Therefore, the projects do **not** use the same complete PCAP population. TFE-GNN and TrafficFormer share 31 VPN captures; Native additionally uses 106 other VPN/nonVPN captures.

## Flow definitions and counts

| Pipeline | Definition / filter | Count |
|---|---|---:|
| Native, all 137 captures | bidirectional IPv4 TCP/UDP sessions; 60s idle; TCP SYN/FIN/RST boundaries; no packet/byte minimum | 22142 |
| Native A, shared31 only | same Native definition | 4224 |
| TFE-GNN CATE raw targets | TCP target five-tuples | 1902 |
| TFE-GNN paper-text valid | CATE plus empty-payload/anomalous-length exclusions | 1662 |
| TrafficFormer parent flows | capture-wide bidirectional IPv4 TCP/UDP five-tuples | 17769 |
| TrafficFormer eligible | first `captured frame bytes >= 2048`, then `packets >= 3` | 1526 |

Native session rows cannot be equated one-to-one with TrafficFormer parent flows because one capture-wide five-tuple can contain multiple Native sessions. This is why set C contains Native sessions whose own packet count is 1 or 2: eligibility belongs to their TrafficFormer parent, not the Native child session.

## Reconstruction validation

- TrafficFormer processing audit exact multiset parity: 31/31 captures.
- Native reconstruction explicitly requires outer IPv4 protocol TCP/UDP and ignores non-first fragments. This prevents inner TCP/UDP headers encapsulated in ICMP and fragment-carried ports from being falsely treated as outer flows.
- Two failed parser probes are preserved under `failed_attempts/`; they are diagnostic evidence and not part of the formal counts.

## CATE matching

- A (Native shared31): 4224 flows.
- B (A with unique packet-count-consistent CATE parent match): 2720 flows.
- C (A linked to an eligible TrafficFormer parent): 1984 flows.
- D (B intersect C): 1804 flows.
- CATE statuses in A: MATCHED=2720, UNMATCHED=1504, AMBIGUOUS=0, MATCH_FAILED=0.

`UNMATCHED` means only that no unique verified CATE parent relation was found. It is not interpreted as background traffic or label error.
