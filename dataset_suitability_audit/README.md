# Three-Dataset Open-Set Suitability Audit

## Audit Objects

- CipherSpectrum: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ipherSpectrum(sok)`
- CSTNET-TLS1.3: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CSTNET-TLS1.3`
- CICDDoS2019: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CICDDoS2019`

The first path intentionally preserves the directory spelling found on disk.

## Audit Purpose

This read-only audit evaluates whether each dataset is suitable for:

1. Open-Detect preprocessing;
2. class-held-out Unknown Detection;
3. a strict Unknown-Free protocol;
4. Single Gaussian versus Multi local-support comparison;
5. cross-dataset validation after USTC-TFC2016.

No model training, encoder extraction, PCA/GMM fitting, Unknown Detection,
label rewriting, flow reconstruction, archive extraction, or bulk PCAP parsing
is performed.

## Completed Read-Only Audit

The formal audit completed without modifying the three source datasets. The
machine-readable evidence is in [`outputs/`](outputs/) and the consolidated
result is [`outputs/audit_summary.md`](outputs/audit_summary.md).

| Priority | Dataset | Score | Tier | Current role |
|---:|---|---:|---|---|
| 1 | CSTNET-TLS1.3 | 18/20 | Tier A | Primary external encrypted-traffic validation |
| 2 | CipherSpectrum | 16/20 | Tier B | Secondary domain-held-out raw-byte validation |
| 3 | CICDDoS2019 | 9/20 | Tier C | Feature-level diagnostic only on the current disk |

Key boundary: CICDDoS2019 currently has authoritative flow CSV labels and
derived aggregate-flow Parquets, but no raw PCAP/PCAPNG. It cannot currently
reproduce the raw-byte Open-Detect input route. CSTNET-TLS1.3 and
CipherSpectrum have per-flow PCAPs, but both require explicit controls against
domain/endpoint/capture shortcuts.

## CipherSpectrum

- Data format: 164,205 readable non-sidecar per-flow PCAPs; no tabular feature dataset found.
- Size: 15,405,783,497 logical bytes (14.348 GiB); valid PCAPs occupy 15,405,690,161 bytes.
- Classes: 85 website/domain classes at the preferred label level; four capture/cipher groups are also present. Counts are 1 / 1,049 / 9,653 (min/median/max), so 11 classes have fewer than 100 captures.
- Label provenance: `LABEL_PROVENANCE_UNCLEAR` — directory and filename organization consistently expose capture group and domain, but no standalone authoritative label map/readme was found.
- Raw packet availability: yes; packet order and bytes remain in the per-flow PCAPs.
- Open-Detect compatibility: directly constructable technically; no external flow-to-PCAP join is needed.
- Unknown-free suitability: technically feasible with 85 candidate domains, but held-out domains can be detected through endpoint novelty rather than traffic structure.
- Leakage risks: high domain/endpoint and directory/capture shortcut risk; class counts also have a severe long tail.
- Final tier: **Tier B — SECONDARY VALIDATION**.

## CSTNET-TLS1.3

- Data format: 46,372 readable per-flow PCAPs, 24 NumPy arrays, and two valid small TSV archives among 11 `.zip` paths (the remainder are AppleDouble sidecars).
- Size: 13,504,861,723 logical bytes (12.577 GiB); PCAPs occupy 10,888,582,594 bytes and NumPy arrays 2,545,747,166 bytes.
- Classes: 120 website/service-domain classes. Raw-PCAP counts are 16 / 474.5 / 500 (min/median/max); one class has fewer than 100 and 117 are below 500.
- Label provenance: domain directory organization plus local readmes declaring 120 flow/packet classes; the NumPy numeric-ID-to-domain map was not found.
- Raw packet availability: yes. Local documentation calls the corpus TLS 1.3 and bounded packet samples contain TLS 1.3 indicators, but this audit did not prove the version for every PCAP.
- Open-Detect compatibility: directly constructable from the per-flow labeled PCAPs; packet order and raw bytes are retained.
- Unknown-free suitability: feasible for held-out website/service classes, subject to a predeclared minimum-support rule and group/session-aware evaluation.
- Leakage risks: medium-high; labels are directory/domain bound, capture dates are unavailable in a manifest, and endpoint entanglement was not established without bulk packet parsing.
- Final tier: **Tier A — PRIMARY EXTERNAL DATASET**.

## CICDDoS2019

- Data format: 21 CSVs and 974 Parquets; 18 CSVs are official flow-label tables and three are local audit/smoke tables. No HDF5 or raw capture is present.
- Size: 37,974,510,488 logical bytes (35.367 GiB); CSVs occupy 31,055,815,744 bytes and Parquets 6,714,729,380 bytes.
- Classes: 19 row-level labels (BENIGN plus fine-grained DDoS families/subtypes), totaling 70,427,637 official rows. Counts are 439 / 3,134,645 / 20,082,580 (min/median/max).
- Label provenance: the official CSV ` Label` column with Flow ID, 5-tuple, timestamp and duration; filenames are not used as label rules.
- Raw packet availability: no PCAP/PCAPNG is present on the current disk. Derived Parquets contain aggregate flow fields, not packet byte sequences.
- Open-Detect compatibility: not directly compatible with the raw-byte route. Restoring original captures would still require the existing time-offset-aware flow mapping to be revalidated.
- Unknown-free suitability: class-heldout evaluation is possible only at feature level now; strict encoder/scaler/density isolation is conceptually possible, but raw-byte Open-Detect is not.
- Leakage risks: high. Seventeen of 19 labels are single-day and/or limited-source risks; attack, day and source capture are strongly coupled.
- Final tier: **Tier C — DIAGNOSTIC ONLY**.

## Recommended Experimental Role

1. **CSTNET-TLS1.3** — first external dataset after USTC-TFC2016 and the primary encrypted-traffic validation candidate.
2. **CipherSpectrum** — secondary raw-byte/domain-held-out validation after adding explicit endpoint and capture shortcut controls.
3. **CICDDoS2019** — feature-level diagnostic only with the current files; do not present it as a raw-byte Open-Detect benchmark unless original PCAPs are restored and mapping is revalidated.

## Reproduction Commands

All payloads were run in project-local named tmux sessions with the workspace
helper and the required fixed Conda environment:

```bash
TMUX_HELPER=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/tmux-task-execution/scripts/tmux_task.sh
AUDIT_DIR=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project/dataset_suitability_audit

"$TMUX_HELPER" run dataset_suitability_formal_v3 "$AUDIT_DIR" -- python scripts/run_audit.py
"$TMUX_HELPER" run dataset_audit_unit_tests_v3 "$AUDIT_DIR" -- python -m pytest -q tests
"$TMUX_HELPER" run dataset_audit_verify_v3 "$AUDIT_DIR" -- python scripts/verify_audit.py
```

The audit reads table headers, Parquet/NumPy metadata and deterministic samples;
it checks PCAP magic for inventory and uses packet tools only on a bounded
sample. It does not extract archives or bulk-parse complete captures.
