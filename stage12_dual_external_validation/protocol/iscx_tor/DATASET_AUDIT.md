# iscx_tor Dataset Audit

## Status before preprocessing

- Total files: `151`
- Total size: `44935414579` bytes
- PCAP/PCAPNG: `95` files / `23227417524` bytes
- Readable PCAP: `85`; unreadable/corrupt: `10`
- Flow CSV: `2`; archive: `2`
- Filename-label mapping errors: `0`
- Final class eligibility remains pending until Open-Detect flow conversion counts are available.

## Canonical label rule

The canonical class is the application/service identity derived from the official capture filename and official application list. The official coarse traffic category and tunnel state are retained separately. Tunnel state is metadata, not a class.

## Readable source groups before flow eligibility

| Canonical class | Readable source groups | Domain states | Packet count |
|---|---:|---|---:|
| AIM | 4 | NonTor, Tor | 1523 |
| Browsing | 12 | NonTor, Tor | 1467414 |
| Email | 6 | NonTor, Tor | 550971 |
| FTP | 2 | NonTor, Tor | 4662932 |
| Facebook | 9 | NonTor, Tor | 1048917 |
| Google | 1 | Tor | 10678 |
| Hangouts | 8 | NonTor, Tor | 1098806 |
| ICQ | 4 | NonTor, Tor | 1869 |
| P2P | 9 | NonTor, Tor | 10568521 |
| SFTP | 2 | NonTor, Tor | 1456750 |
| Skype | 10 | NonTor, Tor | 1100455 |
| Spotify | 5 | NonTor, Tor | 684265 |
| Twitter | 1 | Tor | 14654 |
| Vimeo | 5 | NonTor, Tor | 2533463 |
| YouTube | 7 | NonTor, Tor | 1619058 |

## Unreadable or corrupt captures

- `NonTor/SSL_Browsing.pcap`: capinfos: An error occurred after reading 97926 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/SSL_Browsing.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/SSL_Browsing.pcap" appears to have been cut short in the middle of a packet. |   (will continue anyway, checksums might be incorrect)
- `NonTor/Workstation_Thunderbird_Imap.pcap`: capinfos: An error occurred after reading 81653 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/Workstation_Thunderbird_Imap.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/Workstation_Thunderbird_Imap.pcap" appears to be damaged or corrupt. | (pcap: File has 3351244096-byte packet, bigger than maximum of 262144)
- `NonTor/spotify.pcap`: capinfos: An error occurred after reading 42947 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/spotify.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/spotify.pcap" appears to have been cut short in the middle of a packet. |   (will continue anyway, checksums might be incorrect)
- `NonTor/spotify2-1.pcap`: capinfos: An error occurred after reading 42947 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/spotify2-1.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/spotify2-1.pcap" appears to have been cut short in the middle of a packet. |   (will continue anyway, checksums might be incorrect)
- `NonTor/spotifyAndrew.pcap`: capinfos: An error occurred after reading 34480 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/spotifyAndrew.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/spotifyAndrew.pcap" appears to have been cut short in the middle of a packet. |   (will continue anyway, checksums might be incorrect)
- `NonTor/ssl.pcap`: capinfos: An error occurred after reading 52855 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/ssl.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/NonTor/ssl.pcap" appears to have been cut short in the middle of a packet. |   (will continue anyway, checksums might be incorrect)
- `Tor/Tor/AUDIO_tor_spotify.pcap`: capinfos: An error occurred after reading 35493 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/AUDIO_tor_spotify.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/AUDIO_tor_spotify.pcap" appears to have been cut short in the middle of a packet. |   (will continue anyway, checksums might be incorrect)
- `Tor/Tor/BROWSING_tor_browsing_ara.pcap`: capinfos: An error occurred after reading 99878 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/BROWSING_tor_browsing_ara.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/BROWSING_tor_browsing_ara.pcap" appears to be damaged or corrupt. | (pcap: File has 1347703880-byte packet, bigger than maximum of 262144)
- `Tor/Tor/MAIL_Gateway_Thunderbird_POP.pcap`: capinfos: An error occurred after reading 84118 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/MAIL_Gateway_Thunderbird_POP.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/MAIL_Gateway_Thunderbird_POP.pcap" appears to be damaged or corrupt. | (pcap: File has 280378437-byte packet, bigger than maximum of 262144)
- `Tor/Tor/tor_spotify2-1.pcap`: capinfos: An error occurred after reading 35493 packets from "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/tor_spotify2-1.pcap". | capinfos: The file "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw/pcap/Tor/Tor/tor_spotify2-1.pcap" appears to have been cut short in the middle of a packet. |   (will continue anyway, checksums might be incorrect)

## Leakage boundary

The same canonical application across VPN/non-VPN or Tor/non-Tor will be assigned wholly to Known or wholly to Unknown. Source-PCAP groups will not cross Known Train/Validation/Test.

## Final pre-training eligibility and protocol freeze

- Eligible canonical classes: `10` — Browsing, Email, FTP, Facebook, Hangouts, P2P, Skype, Spotify, Vimeo, YouTube
- Split policy: `FLOW_DISJOINT_ONLY`
- GROUP_AWARE_NOT_FEASIBLE classes: `FTP`
- Unknown protocol SHA-256: `2b8ad9dfe176190f88bce1126d7905e85fab9ebdc953aa50448328f3fd705dbb`
- Split manifest SHA-256: `7de5289ef75966fb360dc743778ba90a31f72c2eaf6694dc3a33291a794bc0d7`
- Unknown-free pre-training check: `PASS`
