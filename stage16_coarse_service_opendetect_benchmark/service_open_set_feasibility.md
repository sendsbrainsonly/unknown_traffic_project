# Service-level open-set feasibility

## Decision

`SERVICE_OPEN_SET_PROTOCOL_BLOCKED`

A six-fold Leave-One-Service-Out design is conceptually valid only after a new
Service-level protocol is frozen. The historical Fine-Application Unknown sets
cannot be relabeled as Service Unknowns. Current evidence is insufficient for a
formal next-stage protocol because:

1. every label is a capture-activity `WEAK_CAPTURE_LABEL`, not authoritative
   per-flow truth;
2. P2P comes from one independent capture, so its held-out result would confound
   Service novelty with a single capture/source;
3. the present artifact has only frozen Known Train/Validation membership; no
   independent Service-level Test membership has been defined;
4. Service/capture overlap and minimum group support must be frozen before any
   detector, support, normalization, or threshold fitting.

No Unknown feature, open-set detector, AUROC, AUPRC, UFAR, DES, or H1 experiment
was run in Stage 16.
