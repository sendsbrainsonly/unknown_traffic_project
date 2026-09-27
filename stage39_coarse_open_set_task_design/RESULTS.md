# Experiment results: stage39-coarse-open-set-task-design-20260926

- Status: `success / TASK_DESIGN_AND_METADATA_AUDIT_ONLY`
- Experiment type: `metadata-audit-and-protocol-design`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-26T08:14:05Z`
- Objective: Align coarse Unknown semantics with the current strong Known classifier and audit service-holdout feasibility without retraining

## Data and split

- Metadata only: Stage12 ISCX-VPN/Tor medium manifests and Stage14B VNAT medium_seed2025. Their source CSV SHA256 values are unchanged before/after the audit. No features or model scores were loaded by the audit script.

## Configuration and execution

- `python stage39_coarse_open_set_task_design/audit_semantics.py`, fixed DGL Python environment, tmux `stage39_design_audit_0926`, exit 0. Metadata-only proposed training seed 2022; no models trained and no formal split generated.

## Core results

- Historical Unknown samples in services already present in Known Train: VPN 6,000/6,000 (100%), Tor 2,000/4,000 (50%), VNAT 3,825/3,825 (100% under the proposed four-family taxonomy).
- All 13 candidate service-holdout settings pass metadata count conservation and nonempty class checks; the minimum per-class Known Train support is at least fixed k=10. These are 12 candidate new-training settings and one Tor P2P-only reinterpretation that can reuse Stage38 weights. No candidate is a frozen or executed new experiment yet.
- VPN has four services; Tor five, with P2P absent from old Known Train; VNAT has four proposed families while retaining the remaining original six-class labels for Known classification.
- Previously Unknown samples belonging to still-Known services are retained for auxiliary unseen-application / Known-service evaluation, never moved into training. Every source flow has a proposed role; source Known Test samples are not moved to training.

## Preserved evidence

- `manifest.json`
- `PLAN.md`, `semantic_overlap_audit.csv`, `service_support.csv`, `service_holdout_feasibility.csv`, `audit_semantics.py`, `completion_verification.json`.
- Original audit stdout and exit status: project `.tmux-task/stage39_design_audit_0926/`.

## Limitations

- This is a task redesign, not evidence of detection improvement. Stage38 was correctly an application-level Unknown task; its coarse-service overlap is not itself a labeling or implementation bug.
- ISCX labels are capture-derived; VNAT grouping is a proposed semantic taxonomy. VPN P2P has one capture and only VPN samples; domain/capture confounding must be reported. Flow-level splits may share groups.
- Historical development Test results have been exposed. Tiny VNAT rdp validation support (4) remains a limitation; no independent cross-capture or multi-seed claim is made.

## Conclusion and next step

- `COARSE_TASK_DESIGN_READY`: see PLAN.md for Known-preserving service holdout, fixed three-view training, fixed DES-v1/MSP, Known-Val P95 and proposed acceptance targets. Freeze actual sample-role derivations and source configs before future Known-only training. No new GPU experiment was launched in this audit.
