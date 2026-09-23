# CipherSpectrum Frozen Input Leakage Summary

- Reused pre-protocol audit sample: **400/400 valid actual inputs**.
- Complete class domain visible: **17/400 (4.25%)**.
- Parsed SNI visible: **0/400 (0.00%)**.
- Class token visible: **18/400 (4.50%)**.
- Combined domain/SNI/token: **18/400 (4.50%)**.
- IP mask: **PASS (3200/3200)**.
- Port retained: **YES (3200/3200)**.
- Conclusion: **LOW_DIRECT_DOMAIN_LEAKAGE**.
- `PORT_SHORTCUT_RISK = PRESENT`; Stage 6 does not mask ports because preprocessing is frozen to the existing Open-Detect pipeline.
