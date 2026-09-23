# Service Label Mapping Audit

## Evidence hierarchy

1. **Primary dataset definition:** the [UNB ISCXVPN2016 page](https://www.unb.ca/cic/datasets/vpn.html) defines Chat, Email, Streaming, File Transfer, VoIP and P2P and explicitly places Facebook/Hangouts/Skype in multiple service types according to the performed activity. It states that the objective application was the only application deliberately executed and packets were filtered to the local client IP. It also warns that incidental browsing flows can be captured during another task.
2. **Capture identity:** all 31 filenames encode an activity or single-service application (`chat`, `audio`, `files`, FTPS/SFTP, BitTorrent, streaming application, and so on).
3. **Local independent implementations:** Stage12 `_vpn_service`, TrafficFormer `service_label`, and the TFE-GNN CATE category agree on every capture for which CATE matches exist. The 31/31 Stage12/TrafficFormer capture mappings agree.
4. **Flow-level limit:** none of these sources provides an independent semantic annotation for every reconstructed Native flow. Consequently, a capture-derived label is not upgraded to per-flow ground truth.

## Multi-service applications

- `Facebook` → Chat, VoIP
- `Hangouts` → Chat, VoIP
- `Skype` → Chat, File-Transfer, VoIP

The ambiguity is resolved at the **capture activity** level: Facebook chat and Facebook voice-call captures receive different Service labels. It is not resolved by a global application→service map.

## Flow provenance result

- Allowed population: frozen `medium_seed2022` Known Train/Validation rows whose source is one of the shared 31 VPN PCAPs.
- Rows: 3065 = 2730 Train + 335 Validation.
- `VERIFIED_DEFINITION`: 0.
- `WEAK_CAPTURE_LABEL`: 3065.
- `AMBIGUOUS`: 0.
- `UNAVAILABLE`: 0.

The Service schema and capture activity are auditable, but all flow labels remain `WEAK_CAPTURE_LABEL` because background/non-target flows cannot be excluded without independent per-flow truth. Validation outcomes were not used to assign or alter any label.
