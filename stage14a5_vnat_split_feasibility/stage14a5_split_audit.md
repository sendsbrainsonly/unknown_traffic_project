# Stage 14A.5 — VNAT Split Feasibility Audit

## Scope and decision rules

- This is a data/split audit only. No model was trained; no Unknown class or Low/Medium/High setting was generated; Open-Detect and DES were not run.
- Four cross-application duplicate flow rows are quarantined before every count: two from `nonvpn_rsync_newcapture1` and two from `nonvpn_sftp_newcapture2`.
- Input `group_id` is frozen from Stage 14A: VPN/non-VPN same-number conservative linkage plus exact-PCAP duplicate union.
- No Train/Val/Test ratio was specified. Therefore the audit uses a maximin witness: maximize the smallest split flow count, then minimize split range. This tests whether any defensible three-way allocation exists; it is not a frozen Stage 14B split.
- Up to 10 groups are exhaustively enumerated. Skype (56 groups) uses deterministic longest-processing-time allocation plus local moves/swaps, so its result is a feasible lower bound rather than a global optimality proof.
- Decision rule derived after profiling: fewer than three usable groups or best maximin split below 10 flows is `excluded`; technically feasible classes with exactly three groups, fewer than 100 total flows, at most four usable groups, or a single group containing at least 80% are `borderline`; the rest are `eligible`.
- P95 support labels are sample-count diagnostics, not a theorem about score stability: fewer than 5 expected upper-tail samples is sparse, 5–19 limited, and at least 20 count-adequate. Actual threshold stability cannot be established without validation scores.

## Split feasibility by application

| application | groups | flows | group min/max | max group share | train/val/test flows | min split | train k=10 | val P95 tail | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| netflix | 2 | 205 | 65/140 | 68.3% | 0/0/0 | 0 | False | 0.0 | **excluded** |
| rdp | 5 | 44 | 5/13 | 29.5% | 16/15/13 | 13 | True | 0.75 | **borderline** |
| rsync | 4 | 1912 | 1/1011 | 52.9% | 1011/899/2 | 2 | True | 44.95 | **excluded** |
| scp | 3 | 2290 | 1/1214 | 53.0% | 1214/1075/1 | 1 | True | 53.75 | **excluded** |
| sftp | 7 | 1668 | 1/709 | 42.5% | 709/646/313 | 313 | True | 32.3 | **eligible** |
| skype | 56 | 1270 | 1/87 | 6.9% | 424/423/423 | 423 | True | 21.15 | **eligible** |
| ssh | 5 | 13563 | 118/11369 | 83.8% | 11369/1601/593 | 593 | True | 80.05 | **borderline** |
| vimeo | 1 | 1218 | 1218/1218 | 100.0% | 0/0/0 | 0 | False | 0.0 | **excluded** |
| youtube | 4 | 341 | 5/137 | 40.2% | 137/125/79 | 79 | True | 6.25 | **borderline** |
| zoiper | 3 | 939 | 276/348 | 37.1% | 348/315/276 | 276 | True | 15.75 | **borderline** |

## Group imbalance findings

- The CSV marks severe imbalance when the largest group contributes at least 80% of class flows or the observed max/min group ratio is at least 100. This is a diagnostic flag, not an automatic exclusion rule.
- Severe raw group imbalance is present for `rsync`, `scp`, `sftp`, `ssh`, and the single-group `vimeo`. For Rsync/SCP it produces unusably tiny held-out splits; for SSH it creates dominant-capture dependence.
- SFTP is retained despite a 709:1 raw group range because seven groups permit five smaller groups to be combined into a 313-flow held-out split; its maximin support is therefore materially better than Rsync/SCP.
- `vnat_group_statistics.csv` records every `group_id`, its VPN/non-VPN flow counts, capture membership, failed-PCAP count, quarantine count, and class-level imbalance metrics.

## Targeted checks

- **RDP / DES-v1 k=10:** PASS at the count level: the maximin witness has 16 training flows. It remains borderline because only 44 total flows produce 15/13 held-out flows and fewer than one expected P95-tail validation sample.
- **SCP:** too fragile and excluded. Its three usable groups yield 1214/1075/1; the best possible minimum is only 1 flow.
- **Zoiper:** balanced in flow volume (348/315/276) but borderline because exactly three groups force one group per split; there is no alternative grouping robustness.
- **SSH:** extreme concentration is confirmed: the largest group holds 83.8% of flows. A count-rich split exists (11369/1601/593), but its semantics depend strongly on assigning the dominant capture to Train, so SSH is borderline.
- **Known Validation P95:** retaining eligible plus borderline classes gives 3,125 validation flows and about 156.2 expected upper-tail observations. This is count-adequate for one global P95, but class composition is highly uneven and score-level/bootstrap stability remains untested.

## Classification

- `eligible`: **sftp, skype**.
- `borderline`: **rdp, ssh, youtube, zoiper**.
- `excluded`: **netflix, rsync, scp, vimeo**.
  - `netflix`: only 2 usable groups; non-empty group-disjoint Train/Val/Test is impossible.
  - `rdp`: only 44 total flows despite a technically feasible split; DES-v1 k=10 training requirement is satisfied (16 train flows).
  - `rsync`: best possible minimum split has only 2 flows.
  - `scp`: best possible minimum split has only 1 flows.
  - `sftp`: adequate group count and maximin split support under this audit.
  - `skype`: adequate group count and maximin split support under this audit.
  - `ssh`: largest group contains 83.8% of class flows.
  - `vimeo`: only 1 usable groups; non-empty group-disjoint Train/Val/Test is impossible.
  - `youtube`: only 4 usable groups limits split robustness.
  - `zoiper`: exactly three groups gives one group per split and no reassignment robustness.

## Required final conclusion

1. 最终推荐保留哪些 application：核心 eligible 为 **['sftp', 'skype']**；为维持可用类别规模，可将 **['rdp', 'ssh', 'youtube', 'zoiper']** 作为带显式限制的候选一并带入 Stage 14B，但不得在该阶段掩盖其脆弱性。
2. 哪些需要排除及原因：**['netflix', 'rsync', 'scp', 'vimeo']**。Netflix/Vimeo 无法形成三路非空 group split；Rsync/SCP 即使采用最优 maximin 分组，最小 held-out split 也只有 2/1 条 flow。
3. 是否可以正式进入 Stage 14B Protocol Freeze：**CONDITIONAL YES**。Stage 14B 只能从上述 eligible + 明确标记的 borderline 集合中冻结协议，必须保持 4 条跨 application duplicate flow 永久隔离；不得重新纳入 excluded 类，也不得把本次 maximin witness 误称为已经冻结的最终协议。

Quarantined rows: 4. No Unknown setting was generated.
