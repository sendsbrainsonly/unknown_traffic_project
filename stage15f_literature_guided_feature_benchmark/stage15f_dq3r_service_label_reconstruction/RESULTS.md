# RESULTS — Stage 15F-DQ-3R

## Conclusion and next step

`SERVICE_TASK_FEASIBLE_WITH_WEAK_LABELS`

The earlier global application→service mapping failure is resolved for direct F3 training by assigning Service from the capture's experimental activity, not from Application alone. All 3065 eligible Known Train/Validation flows receive a deterministic capture-derived label, including Facebook Chat and Facebook VoIP captures.

This does **not** create authoritative per-flow truth. All 3065 labels are `WEAK_CAPTURE_LABEL`; target captures can include incidental or background flows. No label was changed using Validation behavior.

Next step: DQ-3F may run the preregistered same-flow F1/F2-partial/F3 comparison. It must retain the weak-label qualifier and must not reuse the Fine Open-Set protocols as Service Open-Set protocols.

## Configuration and execution

- Scope: shared 31 VPN PCAPs and their frozen Known Train/Validation membership.
- Comparison prepared: F1 Fine Application, F2 partial deterministic mapping, and F3 direct capture-activity Service training.
- F1 and F3 use exactly the same 3065 flow IDs and split membership.
- No model was trained in DQ-3R; the DQ-3F configuration is preregistered only.
- Frozen input assets checked before/after: 17; SHA256 comparison: PASS.

## Core results

- Shared PCAPs audited: 31/31.
- Project capture-level mapping agreement: 31/31.
- Train/Validation flows: 2730/335.
- Services: Chat|Email|File-Transfer|P2P|Streaming|VoIP.
- Every Service has non-zero Train and Validation support.
- Strict six-Service capture-group Train/Validation split: **not feasible** because P2P has one capture.
- Existing Fine Open-Set protocols reusable as Service Open-Set protocols: **NO**; Fine Unknown applications overlap Known Service semantics.
- DQ-3F same-flow flow-level diagnostic: **runnable under the preregistered weak-label scope**.

## Data and split

- Eligible Known Train flows: 2730.
- Eligible Known Validation flows: 335.
- No Known Test or Unknown Test feature values were accessed.
- Every Service has non-zero flow-level Train and Validation support.
- P2P has only one independent capture, so a strict six-Service capture-disjoint Train/Validation evaluation is unavailable on this subset.

## Limitations

- Solved: multi-service applications can be represented correctly at capture activity level for direct F3 labels.
- Not solved: authoritative target/background separation within a capture.
- Not solved: full six-Service group-generalization on the shared31 subset.
- Not solved: Service-level Unknown protocol; it must be redesigned and frozen independently.

## Preserved evidence

- `service_label_provenance.csv`: row-level label provenance and status.
- `service_train_val_support.csv`: Service support and capture coverage.
- `capture_service_mapping_audit.csv`: 31-capture source agreement audit.
- `service_open_set_overlap.csv`: Fine-to-Service semantic-overlap evidence.
- `dq3f_preregistered_protocol.md` and `dq3f_preregistered_config.json`: frozen next-stage plan.
- `completion_verification.json`: machine-readable boundary and completion checks.

Execution boundaries:

- New model training: 0.
- Known Test feature values read: 0.
- Unknown Test feature values read: 0.
- Byte--Behavior, DES and H1 changes: 0.
- Historical Stage 15F-DQ outputs modified: 0.
