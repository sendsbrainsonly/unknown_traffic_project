# DQ-4 Six-Service Feasibility Gate

Gate: `PASS_WITH_LOW_VALIDATION_SUPPORT`; training allowed: `True`.

- M-B / B: Train=1864, Val=237, minimum Train/Val support=61/4, status=PASS_WITH_LOW_VALIDATION_SUPPORT
- M-C / C_FINAL: Train=1367, Val=184, minimum Train/Val support=44/6, status=PASS_WITH_LOW_VALIDATION_SUPPORT
- M-D / D: Train=1269, Val=171, minimum Train/Val support=44/4, status=PASS_WITH_LOW_VALIDATION_SUPPORT

All six Services are non-empty in Train and Validation. B and D have only four VoIP Validation flows, so six-class metrics are computable but fragile. Capture-disjoint generalization remains unidentifiable.
