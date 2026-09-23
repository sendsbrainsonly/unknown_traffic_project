# CSTNET Input Leakage Summary

- Reused fixed audit sample: **600 PCAPs; 600 valid 32x32 inputs**.
- Full class-domain visible: **0/600 (0.00%)**.
- Parsed SNI visible: **0/600 (0.00%)**.
- Domain/token union visible: **0/600 (0.00%)**.
- IP fields: **MASKED (4,800/4,800 checked packet fields)**.
- Transport ports: **VISIBLE (4,800/4,800 checked packet headers)**.
- Direct domain leakage level: **LOW**. Port visibility remains a representation-level shortcut risk.
