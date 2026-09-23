# Coarse Open-Set Semantics

| Existing Fine setting | Known Service union | Fine-Unknown Service union | Overlap | Reusable as strict Service open set? |
|---|---|---|---|---|
| low | Chat|Email|File-Transfer|P2P|Streaming|VoIP | Chat|VoIP | Chat|VoIP | False |
| medium | Chat|Email|File-Transfer|P2P|Streaming|VoIP | Chat|File-Transfer|Streaming|VoIP | Chat|File-Transfer|Streaming|VoIP | False |
| high | Chat|Email|File-Transfer|P2P|Streaming|VoIP | Chat|File-Transfer|Streaming|VoIP | Chat|File-Transfer|Streaming|VoIP | False |

The Fine protocols hold out entire **applications**, not Services. An Unknown application that belongs to a Known Service is not Service-level unknown. Accordingly, none of the existing Fine Open-Set settings may be relabeled and reused as a Service Open-Set protocol.

A future Service protocol must:

1. choose Unknown at the Service-class level before training;
2. remove every flow/capture carrying the held-out Service from encoder training, validation, normalization, support/prototype construction, threshold calibration and parameter selection;
3. freeze Known/Unknown Service classes and all sample IDs before model execution;
4. preserve historical Fine protocols and DES/H1 results unchanged;
5. explicitly audit multi-service applications that occur in both Known and Unknown Service captures as a separate application-identity sensitivity issue.

No new Service Open-Set split is created in DQ-3R.
