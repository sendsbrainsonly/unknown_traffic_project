# Three-Dataset Open-Set Suitability Audit

## Scope

Read-only inventory/schema/label/protocol audit. No training, extraction, PCA/GMM, Unknown Detection, label rewriting, split creation, archive extraction, or bulk PCAP parsing was performed.

## Core inventory

| dataset | files | logical GiB | allocated GiB | valid PCAP | PCAP GiB | primary candidate classes | balance min / median / max |
|---|---:|---:|---:|---:|---:|---:|---|
| CipherSpectrum | 164,223 | 14.348 | 14.847 | 164,205 | 14.348 | 85 | 1 / 1049 / 9653 |
| CSTNET-TLS1.3 | 46,418 | 12.577 | 10.427 | 46,372 | 10.141 | 120 | 16 / 474.5 / 500 |
| CICDDoS2019 | 2,951 | 35.367 | 8.415 | 0 | 0.000 | 19 | 439 / 3134645 / 20082580 |

## Dataset decisions

### CSTNET-TLS1.3

- Best first external dataset after USTC-TFC2016.
- 120 domain/service classes, valid per-flow PCAPs, packet order and bytes available.
- Local readme identifies both flow-level and packet-level 120-class arrays. The numeric ID-to-domain map was not found.
- The very small tail class and possible endpoint/domain shortcut require predeclared eligibility and grouped/session-aware evaluation.
- Claim that every capture is TLS 1.3 is supported by dataset documentation and only sampled packet inspection here; it is not a full-PCAP proof.

### CipherSpectrum

- Raw per-flow PCAPs are directly usable; on-disk capture groups and filenames expose cipher and website/domain levels.
- Domain is the scientifically useful held-out level; the four capture groups alone are too few.
- LABEL_PROVENANCE_UNCLEAR: no standalone authoritative label map/readme was found; website labels are strongly entangled with endpoint/domain identity, so this is secondary rather than primary validation.
- Unparsed valid-PCAP filenames: 0.

### CICDDoS2019

- Current disk contains authoritative official CSV labels and completed derived aggregate flow Parquets, but no PCAP/PCAPNG/archive containing PCAP.
- Therefore raw-byte Open-Detect cannot be constructed from the current directory. Feature-level held-out detection remains possible.
- Official labels are row-level; filenames are not label rules. Nineteen labels include BENIGN and fine-grained DDoS types.
- Training/testing-day and attack/capture binding is a high scientific shortcut risk.

## Ranking

| rank | dataset | score / 20 | tier |
|---:|---|---:|---|
| 1 | CSTNET-TLS1.3 | 18 | Tier A — PRIMARY EXTERNAL DATASET |
| 2 | CipherSpectrum | 16 | Tier B — SECONDARY VALIDATION |
| 3 | CICDDoS2019 | 9 | Tier C — DIAGNOSTIC ONLY |

## Recommended Experimental Role

1. **CSTNET-TLS1.3** — first full encrypted-traffic external validation candidate.
2. **CipherSpectrum** — secondary raw-byte/domain-held-out validation with explicit endpoint-shortcut controls.
3. **CICDDoS2019** — diagnostic/feature-level validation unless original PCAPs are restored; even then use day/capture-aware protocols.

Technical runnability is not treated as scientific suitability. Exact evidence tables are stored in the CSV outputs.
