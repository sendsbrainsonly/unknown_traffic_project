# CIC-IDS-2017 Open-Detect Input Audit

- Fixed sample seed: **20260913**; labels: **10**; sampled strict flows: **200**.
- Stable flow reconstruction: **200/200**.
- Valid actual 32x32 inputs: **200/200**.
- IP mask check: **1300/1300** encoded packets masked.
- Port visibility check: **1300/1300** eligible transport headers retain ports.
- Nonzero header region: **200/200** valid flows.
- Nonzero retained payload region: **155/200** valid flows.

The source Parquet first/last packet boundaries and canonical bidirectional five-tuple were used to stream the original per-day PCAPs. No flow-level random split, model training, or inference was performed.
