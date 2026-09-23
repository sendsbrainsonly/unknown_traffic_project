# Four-Dataset Final Data Readiness Audit

## Goal

Audit only local data completeness, label provenance, grouping, duplicate/provenance risk, representation leakage, calibration capacity, and experiment readiness. No encoder training, embedding extraction, density fitting, thresholding, Unknown inference, or formal metric evaluation was performed.

## Dataset Paths

- USTC-TFC2016: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ustc-tfc2016`
- CipherSpectrum: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ipherSpectrum(sok)`
- CSTNET-TLS1.3: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CSTNET-TLS1.3`
- CIC-IDS-2017: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CIC-IDS-2017`

## USTC

The existing logical map remains **20 classes / 489,101 flows**. SMB-1 and SMB-2 are merged into SMB; Weibo-1..4 are merged into Weibo. All 24 canonical extracted PCAPs are readable/non-empty. Existing train/validation/test manifests, Open-Detect reconstructed images, and Stage3/4/5 protocol/hash assets are present. Data readiness is **READY**, but independent-validation status is **DEVELOPMENT_ONLY**.

## CipherSpectrum

The existing official40 manifest has **160,200 PCAPs**, exactly 4,005 per class. The conservative primary candidate excludes MIX and has **120,000 PCAPs**, 3,000 per class. `getpocket.com` remains a **NON_PRIMARY_EXTRA_CLASS**. The physical tree has **164,207 PCAP-suffixed files**; the two non-manifest files are macOS resource forks: `aes-128-gcm/__MACOSX/aes-128-gcm/yahoo.co.jp/._traffic_2024-01-22_yahoo.co.jp_aes-128_firefox_1.pcap.TCP_10-0-2-15_38184_183-79-248-252_443.pcap; chacha20-poly1305/__MACOSX/chacha20-poly1305/hubspot.com/._traffic_2024-01-13_adtelligent.com_chacha20_chromium_2.pcap.TCP_10-0-2-15_34206_104-19-154-83_443.pcap`.

## CSTNET-TLS1.3

Frozen protocol SHA-256 verification is **PASS (10/10)**; the broader Stage 5.5 frozen record remains 26/26 PASS. The source has 120 classes and 46,372 readable per-flow PCAPs; 119 classes are eligible and `chia.net` is excluded.

## CIC-IDS-2017

Strict reliable official flow-CSV matching yields **2,087,440 flows / 15 labels**. All attack labels occur in exactly one day and one PCAP. The sample-weighted class↔day and class↔PCAP NMI are both **0.274164**; this moderate aggregate number is suppressed by the dominant multi-day BENIGN class and does not negate the 100% per-attack day binding.

| CICIDS label | Strict flows | Eligibility |
|---|---:|---|
| BENIGN | 1,668,300 | PRIMARY_ELIGIBLE |
| Bot | 1,228 | MAPPING_RISK |
| DDoS | 76,613 | SEVERE_ENDPOINT_SHORTCUT |
| DoS GoldenEye | 7,441 | SEVERE_CAPTURE_CONFOUNDING |
| DoS Hulk | 155,168 | SEVERE_ENDPOINT_SHORTCUT |
| DoS Slowhttptest | 5,096 | PRIMARY_ELIGIBLE |
| DoS slowloris | 5,709 | SEVERE_ENDPOINT_SHORTCUT |
| FTP-Patator | 3,985 | SEVERE_ENDPOINT_SHORTCUT |
| Heartbleed | 11 | TOO_SMALL |
| Infiltration | 34 | TOO_SMALL |
| PortScan | 158,863 | PRIMARY_ELIGIBLE |
| SSH-Patator | 2,987 | MAPPING_RISK |
| Web Attack - Brute Force | 1,364 | SEVERE_ENDPOINT_SHORTCUT |
| Web Attack - Sql Injection | 12 | TOO_SMALL |
| Web Attack - XSS | 629 | SEVERE_ENDPOINT_SHORTCUT |

## Duplicate/Provenance Risks

MIX_STATUS is **PARTIALLY_OVERLAPPING**. Exact SHA duplicates are **0** and normalized packet-fingerprint overlaps are **0** among 40,200 MIX PCAPs. MIX shares **16,596/18,260** collection groups with canonical sources. Primary experiments should **exclude MIX**; no file was deleted.

## Leakage Risks

CipherSpectrum actual Open-Detect inputs: full domain **17/400**, parsed SNI **0/400**, class token **18/400**, union **18/400**; direct level **LOW**. CSTNET remains 0/600 for direct domain/SNI/token, with IP masked and ports visible. CICIDS endpoint concentration is substantial:

| Label | dst IP top-1 | dst port top-1 |
|---|---:|---:|
| Web Attack - Brute Force | 100.00% | 100.00% |
| Web Attack - Sql Injection | 100.00% | 100.00% |
| Web Attack - XSS | 100.00% | 100.00% |
| DoS slowloris | 99.96% | 99.96% |
| Bot | 99.84% | 100.00% |
| SSH-Patator | 99.73% | 99.73% |
| FTP-Patator | 99.72% | 99.70% |
| DDoS | 99.25% | 99.25% |
| DoS GoldenEye | 98.67% | 98.67% |
| DoS Hulk | 98.40% | 98.40% |

## Calibration Capacity

CipherSpectrum canonical-three view gives 300 validation samples/class under 80/10/10 and 450/class under 70/15/15. A hypothetical K=2 95/5 small component receives only 15 or 22 samples/class, below the n=30 fallback boundary. CSTNET Known-validation medians are 44/42/39 for Low/Medium/High; K=2 fallback stress is severe, so its correct role is **SPARSE_CALIBRATION_STRESS_BENCHMARK**, not the primary component-boundary benchmark.

## Group Split Feasibility

CipherSpectrum `split_group_id` is available. In official40 there are **28,312 groups**, including **12,096 cross-class and 16,629 cross-source groups**; every shared group must remain in one partition. CSTNET has a frozen group-aware protocol. CICIDS can use `(source_pcap, five-minute flow-start block)` and is count-feasible for the three current primary classes, but no Unknown policy or split was frozen.

## Four-Dataset Comparison

| Dataset | Role | Samples | Usable classes | Final readiness |
|---|---|---:|---:|---|
| USTC-TFC2016 | Development / Mechanism | 489,101 | 20 | READY |
| CipherSpectrum | Primary Independent External Validation | 120,000 | 40 | READY |
| CSTNET-TLS1.3 | Secondary Sparse-Calibration Stress Test | 46,372 | 119 | CONDITIONALLY_READY |
| CIC-IDS-2017 | Independent NIDS / Unknown-Attack Validation | 2,087,440 | 3 | CONDITIONALLY_READY |

## Final Readiness

- USTC: **READY** for development/mechanism only.
- CipherSpectrum: **READY**, using the canonical-three 120k corpus and a future group-aware split.
- CSTNET: **CONDITIONALLY_READY** because component calibration is sparse.
- CIC-IDS-2017: **CONDITIONALLY_READY**; only 3 classes currently pass the fixed primary gates.

## Recommended Experimental Roles

- USTC = Development / Mechanism: supported.
- CipherSpectrum = Primary Independent External Validation: supported only with MIX excluded and `split_group_id` enforced.
- CSTNET = Secondary Sparse-Calibration Stress Test: supported conditionally.
- CIC-IDS-2017 = Independent NIDS / Unknown-Attack Validation: conditionally supported; not yet a clean 15-class benchmark.

## Blocking Issues

- CipherSpectrum: freeze the canonical-three group-aware split; do not allow `split_group_id` overlap.
- CSTNET: freeze a calibration-aware protocol revision before method execution; do not change the existing audit protocol silently.
- CICIDS: resolve/freeze the class-eligibility and Unknown policy; retain schedule-risk labels as risk, not silently relabel them; use grouped time blocks; report endpoint/capture confounding.
- Excluded from CICIDS primary under fixed gates: Bot (MAPPING_RISK), DDoS (SEVERE_ENDPOINT_SHORTCUT), DoS GoldenEye (SEVERE_CAPTURE_CONFOUNDING), DoS Hulk (SEVERE_ENDPOINT_SHORTCUT), DoS slowloris (SEVERE_ENDPOINT_SHORTCUT), FTP-Patator (SEVERE_ENDPOINT_SHORTCUT), Heartbleed (TOO_SMALL), Infiltration (TOO_SMALL), SSH-Patator (MAPPING_RISK), Web Attack - Brute Force (SEVERE_ENDPOINT_SHORTCUT), Web Attack - Sql Injection (TOO_SMALL), Web Attack - XSS (SEVERE_ENDPOINT_SHORTCUT).

## Next Step

The data-readiness audit is complete. The project is **not yet ready to launch one unified formal four-dataset experiment** because three downstream protocol freezes remain. The next authorized step would be protocol design/freeze only; this audit did not start training or evaluation.
