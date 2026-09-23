# Stage 14A — VNAT Data Audit

## Scope and grain

- Dataset (read-only): `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT`
- Dataset files: **172**, including **165 PCAPs** and **7 metadata/HDF/script files**.
- PCAP-derived flow grain: tshark TCP/UDP stream within one source PCAP; other IP traffic is grouped by canonical bidirectional protocol/endpoints. Total: **23,454 flows**.
- `capture_id` is the source PCAP stem. `source_group_id` reproduces existing local metadata. `strict_base_group_id` conservatively shares VPN/non-VPN files with the same application/variant/number; final `group_id` additionally unions exact duplicate PCAPs.
- Exact flow duplication uses an ordered digest of tshark `frame.md5_hash` values, so it represents identical captured packet bytes, not merely a repeated 5-tuple.

## Application statistics

| application | PCAPs usable/total | flows | effective groups | VPN flows/groups | non-VPN flows/groups | duplicate PCAP rows | exact duplicate flow rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| netflix | 3/3 | 205 | 2 | 1/1 | 204/2 | 0 | 0 |
| rdp | 8/8 | 44 | 5 | 6/3 | 38/5 | 0 | 0 |
| rsync | 5/5 | 1914 | 4 | 3/3 | 1911/2 | 0 | 2 |
| scp | 4/5 | 2290 | 3 | 2/2 | 2288/2 | 0 | 0 |
| sftp | 9/9 | 1670 | 7 | 4/4 | 1666/5 | 0 | 2 |
| skype | 109/110 | 1270 | 56 | 56/55 | 1214/53 | 2 | 2 |
| ssh | 10/10 | 13563 | 5 | 5/5 | 13558/5 | 0 | 0 |
| vimeo | 2/2 | 1218 | 1 | 1/1 | 1217/1 | 0 | 0 |
| youtube | 7/7 | 341 | 4 | 3/3 | 338/4 | 0 | 0 |
| zoiper | 6/6 | 939 | 3 | 301/3 | 638/3 | 0 | 0 |

## Data-quality findings

- Readability: **163/165 PCAPs passed** both capinfos and full tshark traversal; failures: **2**.
- Failed/truncated PCAPs: `nonvpn_scp_long_capture1.pcap` (12833968128 bytes; RuntimeError: capinfos exit 1: capinfos: An error occurred after reading 13474841 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT/VNAT_release_1/nonvpn_scp_long_capture1.pcap".
capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT/VNAT_release_1/nonvpn_scp_long_capture1.pcap" appears to have been cut short in the middle of a packet.
  (will continue anyway, checksums might be incorrect)); `vpn_skype-chat_capture6.pcap` (3182592 bytes; RuntimeError: capinfos exit 1: capinfos: An error occurred after reading 10284 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT/VNAT_release_1/vpn_skype-chat_capture6.pcap".
capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT/VNAT_release_1/vpn_skype-chat_capture6.pcap" appears to have been cut short in the middle of a packet.
  (will continue anyway, checksums might be incorrect)).
- Filename/metadata consistency failures: **0**.
- Exact duplicate PCAP hash groups: **1**.
- Exact duplicate PCAP pair: `vpn_skype-chat_capture3` and `vpn_skype-chat_capture4`; their final `group_id` is unified.
- Flow rows with an identical packet-byte sequence in another capture: **6** (detailed rows: 6).
- Exact duplicate flow rows crossing application labels: **4**. These cannot remain in a future Known/Unknown protocol.
- Cross-application exact flow duplication occurs between `nonvpn_rsync_newcapture1` and `nonvpn_sftp_newcapture2` (two packet sequences, represented by four peer rows).
- Flow rows whose canonical 5-tuple also appears in another capture: **1263**. Reused endpoint tuples alone are not treated as exact duplicates.
- All ten applications contain both VPN and non-VPN captures; application, not tunnel status, must be the semantic class.
- Existing labels are filename-derived capture labels. They do not independently prove that every background/auxiliary flow inside a PCAP belongs to the named application.
- The raw HDF has 33,711 rows. The supplied feature HDF has 15,093 rows and only five coarse category labels; it has no capture/group column. It cannot by itself support a strict group-aware ten-application split.
- Capture and flow volume are highly imbalanced: Skype has 110 PCAPs, while the smallest applications have only 2–3 PCAPs; RDP has 44 parsed flows versus SSH with 13,563. RDP is retained because no empirical flow threshold was preset and it has five usable groups.

## Eligibility decision

No empirical minimum flow/group threshold was preregistered. After reporting the full distribution, eligibility uses only the logical requirement that a three-way group-disjoint split needs at least three non-empty usable groups. Corrupt captures are quarantined rather than automatically disqualifying an otherwise usable application.

- Eligible: **rdp, rsync, scp, sftp, skype, ssh, youtube, zoiper**.
- Excluded: **netflix, vimeo**.
  - `netflix`: only 2 usable duplicate-aware groups; cannot allocate non-empty Train/Val/Test without group overlap.
  - `vimeo`: only 1 usable duplicate-aware groups; cannot allocate non-empty Train/Val/Test without group overlap.

## Required final answers

1. 总文件数 / 成功解析数 / 失败数：**172 / 170 / 2**；其中 PCAP **165 / 163 / 2**。
2. application 总数：**10**。
3. 每类 flow 与 group 数：见上方 Application statistics 表；CSV 中同时保留 source-group 与 conservative strict-group 计数。
4. VPN/non-VPN 分布：所有 application 两种状态均存在；逐类 flow/group 数见表。
5. group-aware split 是否可行：**对 eligible 子集有条件可行，对全部 10 类不可行**。Train/Val/Test 必须按 duplicate-aware `group_id` 整组分配，禁止同一组内 flow 随机拆分。
6. eligible / excluded classes：eligible = **['rdp', 'rsync', 'scp', 'sftp', 'skype', 'ssh', 'youtube', 'zoiper']**；excluded = **['netflix', 'vimeo']**。
7. 发现的数据质量问题：见 Data-quality findings；核心风险是极端类别/组数不平衡、少数组无法三分、重复证据，以及标签仅来自文件名而非逐 flow 权威标注。
8. 是否可以进入 Stage 14B Protocol Freeze：**CONDITIONAL YES**。只能使用 eligible 类；先隔离损坏 PCAP、去除/隔离跨 application 的 exact duplicate flows、合并 exact duplicate PCAP group，再冻结 group-aware split。把同一 application 的 VPN/non-VPN 整体置于同一语义侧；不得直接对无 group 字段的 feature HDF 随机切分。

No Unknown class or Low/Medium/High setting was selected in this audit.
