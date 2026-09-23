# Endpoint Input Audit

This audit reuses `opendetect_ustc_encoder_audit/adapters/opendetect_preprocessing.py`: first 8 packets, each represented by 80 masked-IPv4/header bytes plus 48 Scapy Raw payload bytes, then reshaped to 32x32.

- IP: **MASKED** (4800/4800 encoded packet IP source/destination fields were zero).
- Port: **VISIBLE** (4800/4800 eligible transport headers retained exact source/destination ports).
- SNI/domain: **0.0000%** direct visibility among valid model inputs.
- Full class-domain visibility: **0.0000%**.
- Parsed-SNI visibility: **0.0000%**.
- TLS record/header prefix visible: **600/600** valid inputs.

The port result is a representation-level visibility finding, not a demonstrated classifier shortcut.
