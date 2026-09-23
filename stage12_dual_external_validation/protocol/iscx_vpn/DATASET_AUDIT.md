# iscx_vpn Dataset Audit

## Status before preprocessing

- Total files: `140`
- Total size: `27682407809` bytes
- PCAP/PCAPNG: `140` files / `27682407809` bytes
- Readable PCAP: `140`; unreadable/corrupt: `0`
- Flow CSV: `0`; archive: `0`
- Filename-label mapping errors: `0`
- Final class eligibility remains pending until Open-Detect flow conversion counts are available.

## Canonical label rule

The canonical class is the application/service identity derived from the official capture filename and official application list. The official coarse traffic category and tunnel state are retained separately. Tunnel state is metadata, not a class.

## Readable source groups before flow eligibility

| Canonical class | Readable source groups | Domain states | Packet count |
|---|---:|---|---:|
| AIM | 6 | NonVPN, VPN | 7324 |
| BitTorrent | 1 | VPN | 422098 |
| Email | 6 | NonVPN, VPN | 96660 |
| FTPS | 6 | NonVPN, VPN | 8064237 |
| Facebook | 18 | NonVPN, VPN | 2745210 |
| Gmail | 3 | NonVPN | 12272 |
| Hangouts | 15 | NonVPN, VPN | 4933869 |
| ICQ | 6 | NonVPN, VPN | 12120 |
| Netflix | 5 | NonVPN, VPN | 1169303 |
| SCP | 12 | NonVPN | 841496 |
| SFTP | 10 | NonVPN, VPN | 976132 |
| Skype | 26 | NonVPN, VPN | 4821309 |
| Spotify | 5 | NonVPN, VPN | 157193 |
| Vimeo | 6 | NonVPN, VPN | 466377 |
| VoIPBuster | 7 | NonVPN, VPN | 1569019 |
| YouTube | 8 | NonVPN, VPN | 460470 |

## Unreadable or corrupt captures

None detected by `capinfos`.

## Leakage boundary

The same canonical application across VPN/non-VPN or Tor/non-Tor will be assigned wholly to Known or wholly to Unknown. Source-PCAP groups will not cross Known Train/Validation/Test.

## Final pre-training eligibility and protocol freeze

- Eligible canonical classes: `16` — AIM, BitTorrent, Email, FTPS, Facebook, Gmail, Hangouts, ICQ, Netflix, SCP, SFTP, Skype, Spotify, Vimeo, VoIPBuster, YouTube
- Split policy: `FLOW_DISJOINT_ONLY`
- GROUP_AWARE_NOT_FEASIBLE classes: `BitTorrent`
- Unknown protocol SHA-256: `a088787beba0ca5a7c85607d816e2301d22e493d0334e709b7e9e56d255e0999`
- Split manifest SHA-256: `618670af28797d9ceffc7138dcb9d2622552cdfc4712c38bd5da0886ec95c793`
- Unknown-free pre-training check: `PASS`
