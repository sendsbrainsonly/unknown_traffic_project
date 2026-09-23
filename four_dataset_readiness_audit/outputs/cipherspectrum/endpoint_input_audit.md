# CipherSpectrum Open-Detect Input Audit

- Fixed sample: seed **20260913**, 20 classes x up to 20 PCAPs = **400**; valid inputs **400**.
- Raw PCAP full class-domain visible: **396/400 (99.00%)**.
- Actual 32x32 input full class-domain visible: **17/400 (4.25%)**.
- Actual input parsed SNI visible: **0/400 (0.00%)**.
- Actual input class token visible: **18/400 (4.50%)**.
- Actual input domain/SNI/token union: **18/400 (4.50%)**.
- IP masking: **3200/3200** checked packet fields masked.
- Port retention: **3200/3200** checked transport headers preserve ports.
- CIPHERSPECTRUM_DIRECT_DOMAIN_LEAKAGE = **LOW**.

Raw visibility and model-input visibility are reported separately. Port visibility is a potential endpoint shortcut, not proof that a classifier uses it.
