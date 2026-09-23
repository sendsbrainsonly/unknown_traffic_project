# CIC-IDS-2017 Group-aware Split Feasibility

- Strict reliable matched flows: **2,087,440**.
- class ↔ day NMI: **0.274164**.
- class ↔ PCAP NMI: **0.274164**.
- Fixed thresholds: TOO_SMALL < 100; PRIMARY minimum 1000; severe endpoint if dst IP and dst port top-1 shares are each >= 0.95; minimum 3 five-minute groups.
- PRIMARY_ELIGIBLE classes: **BENIGN, DoS Slowhttptest, PortScan**.
- Group-aware Known Train/Validation/Test feasibility: **FEASIBLE**.
- Candidate group: `(source_pcap, floor(flow_start_epoch_utc / 300))`; all flows in a five-minute block must remain in one partition.
- This is a feasibility audit only: no Unknown classes or split assignments were frozen.
- The high NMI and one-day-per-attack structure must be reported as temporal/capture confounding; a random flow split is forbidden.
