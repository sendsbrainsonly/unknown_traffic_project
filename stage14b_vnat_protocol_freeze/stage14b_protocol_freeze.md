# Stage 14B — VNAT Open-Set Protocol Freeze

- Freeze status: **FROZEN_READY_FOR_STAGE_14C**
- Freeze hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`
- This stage generated data manifests only. No model, embedding, Open-Detect, DES, threshold score, or test metric was produced or consulted.
- Semantic class is `application`; VPN/non-VPN is metadata only.
- Known split is deterministic class-stratified flow-random 8:1:1. Capture/group metadata is retained but is not a hard split boundary, exactly as requested.
- Claim boundary: Strict Unknown-Free is enforced, but Known Train/Validation/Test is **not capture-disjoint** and must not be reported as capture-generalization evidence.

## 1–2. Final class audit and clean flow counts

| application | status | raw readable | cross-app removed | duplicate-PCAP removed | clean flows | VPN/non-VPN | Known 8:1:1 if known | failed PCAP |
|---|---|---:|---:|---:|---:|---:|---:|---|
| netflix | retained | 205 | 0 | 0 | 205 | 1/204 | 165/20/20 | none |
| rdp | retained | 44 | 0 | 0 | 44 | 6/38 | 36/4/4 | none |
| rsync | retained | 1914 | 2 | 0 | 1912 | 3/1909 | 1530/191/191 | none |
| scp | retained | 2290 | 0 | 0 | 2290 | 2/2288 | 1832/229/229 | nonvpn_scp_long_capture1 |
| sftp | retained | 1670 | 2 | 0 | 1668 | 4/1664 | 1336/166/166 | none |
| skype | retained | 1270 | 0 | 1 | 1269 | 55/1214 | 1017/126/126 | vpn_skype-chat_capture6 |
| ssh | retained | 13563 | 0 | 0 | 13563 | 5/13558 | 10851/1356/1356 | none |
| vimeo | retained | 1218 | 0 | 0 | 1218 | 1/1217 | 976/121/121 | none |
| youtube | retained | 341 | 0 | 0 | 341 | 3/338 | 273/34/34 | none |
| zoiper | retained | 939 | 0 | 0 | 939 | 301/638 | 753/93/93 | none |

- Retained: **['netflix', 'rdp', 'rsync', 'scp', 'sftp', 'skype', 'ssh', 'vimeo', 'youtube', 'zoiper']**.
- Excluded: **none**. All ten classes have at least 44 clean flows and support non-empty 8:1:1 when Known.
- Corrupt PCAP flow contribution is zero because each failed PCAP is excluded before constructing the flow manifest. Its unparsed flow count is not inferred.

## 3. Frozen setting sizes

| setting | Known classes | Unknown classes | Unknown share | seeds |
|---|---:|---:|---:|---|
| Low | 8 | 2 | 20% | 2022–2026 |
| Medium | 7 | 3 | 30% | 2022–2026 |
| High | 6 | 4 | 40% | 2022–2026 |

The 2/3/4 design is based on ten retained classes: it avoids a single-class Low setting, preserves at least six Known classes at High, and is not the previous 1/3/5 schedule. Unknown sets are nested within each seed and balanced across five seeds.

## 4–6. Frozen Unknown applications, Known split sizes, and VPN/non-VPN

| protocol | Unknown applications | Known Train/Val/Test | Train VPN/non-VPN | Val VPN/non-VPN | Test VPN/non-VPN | Unknown VPN/non-VPN |
|---|---|---:|---:|---:|---:|---:|
| low_seed2022 | scp, rdp | 16901/2107/2107 | 296/16605 | 41/2066 | 36/2071 | 8/2326 |
| low_seed2023 | skype, netflix | 17587/2194/2194 | 261/17326 | 31/2163 | 33/2161 | 56/1418 |
| low_seed2024 | ssh, rsync | 6388/793/793 | 303/6085 | 33/760 | 37/756 | 8/15467 |
| low_seed2025 | zoiper, vimeo | 17040/2126/2126 | 60/16980 | 12/2114 | 7/2119 | 302/1855 |
| low_seed2026 | sftp, youtube | 17160/2140/2140 | 306/16854 | 35/2105 | 33/2107 | 7/2002 |
| medium_seed2022 | scp, rdp, skype | 15884/1981/1981 | 255/15629 | 33/1948 | 30/1951 | 63/3540 |
| medium_seed2023 | skype, netflix, ssh | 6736/838/838 | 257/6479 | 31/807 | 32/806 | 61/14976 |
| medium_seed2024 | ssh, rsync, zoiper | 5635/700/700 | 61/5574 | 4/696 | 7/693 | 309/16105 |
| medium_seed2025 | zoiper, vimeo, sftp | 15704/1960/1960 | 56/15648 | 12/1948 | 7/1953 | 306/3519 |
| medium_seed2026 | sftp, youtube, scp | 15328/1911/1911 | 304/15024 | 35/1876 | 33/1878 | 9/4290 |
| high_seed2022 | scp, rdp, skype, netflix | 15719/1961/1961 | 255/15464 | 32/1929 | 30/1931 | 64/3744 |
| high_seed2023 | skype, netflix, ssh, rsync | 5206/647/647 | 255/4951 | 31/616 | 31/616 | 64/16885 |
| high_seed2024 | ssh, rsync, zoiper, vimeo | 4659/579/579 | 60/4599 | 4/575 | 7/572 | 310/17322 |
| high_seed2025 | zoiper, vimeo, sftp, youtube | 15431/1926/1926 | 54/15377 | 11/1915 | 7/1919 | 309/3857 |
| high_seed2026 | sftp, youtube, scp, rdp | 15292/1907/1907 | 298/14994 | 35/1872 | 33/1874 | 15/4328 |

## 7. Empty, small, and imbalanced classes

- Empty classes: **0**. Classes unable to form non-empty 8:1:1: **0**.
- Class-size imbalance remains substantial: minimum **44** flows (RDP), maximum **13563** (SSH), ratio **308.2×**.
- RDP is retained under the requested flow-count rule but remains statistically small: when Known it contributes only 36/4/4 Train/Validation/Test flows.
- VPN domain imbalance is severe for most classes: only Zoiper has a substantial VPN fraction; several classes have fewer than 1% VPN flows. The exact per-protocol split counts are frozen above and in the JSON.
- Flow-random splitting actually places **18–74 groups** and **21–77 captures** across multiple Known splits, depending on protocol. This is permitted by the requested Stage 14B rule, but creates a capture-leakage risk for Known generalization and must be tested later as sensitivity analysis.
- Vimeo has only one conservative group; its Known Train/Validation/Test samples necessarily share that group. This does not violate class-held-out Unknown-Free, but it prevents a capture-generalization claim.

## 8. Strict Unknown-Free verification

- Status: **PASS** across all 15 setting × seed protocols.
- Every Unknown application is absent from Known Train, Validation, Test, scaler fitting, prototype/support fitting, threshold calibration, and parameter selection.
- Unknown flows occur only in `unknown_test`; all VPN and non-VPN flows of an Unknown application move together.
- Frozen downstream roles: encoder/scaler/prototype/support fit on Known Train only; threshold is global Known Validation P95 only; Known Test and Unknown Test are evaluation-only.

## 9. Stage 14C readiness

**READY_FOR_STAGE_14C under the frozen flow-random protocol.** Stage 14C must verify the freeze hash before use. Unknown classes, seeds, 8:1:1 assignments, exclusions, and the global Known-Validation-P95 threshold rule are immutable and may not be changed after inspecting VNAT Test results.

No Stage 14C training was started.
