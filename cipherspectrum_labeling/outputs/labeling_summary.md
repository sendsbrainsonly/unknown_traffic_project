# CipherSpectrum Labeling Summary

Generated: 2026-09-12T13:49:01.552147+00:00

## Outcome

- Raw PCAP files scanned read-only: **164,205** (15,405,690,161 bytes).
- Official-compatible view: **40 classes / 160,200 flows**.
- Complete local audit view: **41 normalized classes / 164,205 candidate flows**.
- Rows requiring manual/release-compatibility review: **7,924**.
- Filename parse errors: **0**; invalid PCAP magic values: **0**.
- Capture-group/cipher mismatches: **5,919**.
- Of those mismatches, **3,919** retain an
  official-compatible class label; the mismatch concerns cipher packaging,
  not the SNI/directory class assignment.
- Filename `visited_domain` differs from the canonical class label for
  **97,947** flows.  This is expected when a page
  contacts third-party TLS services and is why that field is not the label.

## Label policy

The class label is derived from the immediate class directory, which is the
release's SNI-oriented grouping.  The visited/page domain inside a PCAP filename
is retained as `visited_domain` metadata and is **not** used as the class label.

The official-compatible 40-class manifest contains only labels appearing as
class directories in every capture group.  This avoids silently treating a
local release discrepancy as an official class.

The full manifest additionally preserves `getpocket.com`.  The literal
`aes-256-gcm/aes-256-gcm/chacha20/` directory is normalized to
`getpocket.com` only in this audit view; those rows remain marked for review.
The `chacha20-poly1305/getpocket.com` material also reports an observed
`aes-256` cipher, so both cross-group packaging mismatches stay explicit.

## Primary references

- Official CipherSpectrum dataset page: https://cgi.cse.unsw.edu.au/~cspectrum/
- CipherSpectrum paper, including the traffic-labeling description:
  https://arxiv.org/html/2503.20093v4

## Leakage safeguard

`capture_id` and the identical `split_group_id` are exported for every flow.
Future train/validation/test creation must group on this field.  Flow-level
random splitting would place related split sessions from the same browser
capture in different partitions because capture IDs overlap across cipher/mix
groups.

## Recommended use

1. Use `cipherspectrum_official40_manifest.csv` for the conservative first
   class-label experiment.
2. If an experiment also assumes cipher-group purity, filter rows with
   `manual_review_required=true`; some official-class rows have a packaging
   mismatch even though their class label remains usable.
3. Treat the 41-class view as a local-release extension until its extra class
   and two capture-group anomalies receive a separate SNI/package audit.
4. Do not infer authoritative semantics beyond website/SNI class membership;
   these labels do not identify attack families or independent endpoints.
