# DQ-3F Label and Membership Audit

Status: `PASS`

- Frozen flows: 3,065; Known Train/Validation: 2,730/335.
- Fine classes: 11 (`AIM | BitTorrent | Email | FTPS | Facebook | ICQ | Netflix | SFTP | Spotify | Vimeo | VoIPBuster`).
- Service classes: 6 (`Chat | Email | File-Transfer | P2P | Streaming | VoIP`).
- Flow IDs are unique; Train and Validation flow IDs are disjoint.
- Stage12 NPZ row order was reconstructed from `(canonical_class, flow_id_sha256)` and both target and array digests passed.
- Exact-image hashes shared across Train/Validation: 0.
- Every Service label is `WEAK_CAPTURE_LABEL`; no Validation prediction is used to define labels.
- Model input is only the frozen 32x32 uint8 traffic image. Service, capture, filename, activity and split metadata are not input features.
- Known Test and Unknown Test arrays were not loaded.

## Frozen Service support

| Service | Train | Validation | Captures |
|---|---:|---:|---:|
| Chat | 132 | 17 | 6 |
| Email | 145 | 23 | 2 |
| File-Transfer | 332 | 43 | 4 |
| P2P | 804 | 100 | 1 |
| Streaming | 854 | 104 | 4 |
| VoIP | 463 | 48 | 3 |

## Interpretation boundary

P2P has one independent capture. DQ-3F is a flow-level matched-sample diagnostic and cannot establish capture-disjoint generalization.
