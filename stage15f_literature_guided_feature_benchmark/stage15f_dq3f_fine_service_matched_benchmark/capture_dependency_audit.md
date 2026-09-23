# DQ-3F Capture Dependency Audit

## Conclusion

`CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE`

The benchmark is flow-disjoint but not capture-disjoint. P2P has one capture, so its Validation score cannot demonstrate generalization to a new P2P capture.

## Per-Service capture support

| Service | Captures | Train | Validation | Train/Validation overlapping captures |
|---|---:|---:|---:|---:|
| Chat | 6 | 132 | 17 | 4 |
| Email | 2 | 145 | 23 | 2 |
| File-Transfer | 4 | 332 | 43 | 4 |
| P2P | 1 | 804 | 100 | 1 |
| Streaming | 4 | 854 | 104 | 4 |
| VoIP | 3 | 463 | 48 | 3 |

## Integrity

- Duplicate flow IDs: 0.
- Exact image hashes shared across Train/Validation: 0 (preflight verified).
- Capture identifiers, filenames and activities were audit metadata only and never model inputs.
- No single P2P capture was split into fictional independent capture groups.
- An independent non-shared P2P capture would be required for a separate cross-capture claim.
