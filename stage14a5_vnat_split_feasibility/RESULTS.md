# Experiment results: stage14a5-vnat-split-feasibility-20260917-v1

- Status: `success`
- Experiment type: `dataset-audit`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-17T09:37:48Z`
- Completed (UTC): `2026-09-17T09:46:50Z`
- Objective: Audit VNAT duplicate-quarantined group-disjoint Train/Validation/Test split feasibility without model execution or Unknown-setting generation

## Data and split

- Frozen inputs: Stage 14A `vnat_manifest.csv`, per-PCAP scan records, and `outputs/duplicate_flows.csv`.
- Four cross-application exact duplicate flow rows were logically quarantined: two rows from `nonvpn_rsync_newcapture1` and two from `nonvpn_sftp_newcapture2`.
- Flow population changed from 23,454 to 23,450 after quarantine. Source PCAPs and Stage 14A outputs were not modified.
- Group identity is inherited unchanged from Stage 14A: conservative VPN/non-VPN capture linkage plus exact-PCAP duplicate union.
- This audit did not freeze a final split. It generated a maximin feasibility witness that maximizes the smallest of Train/Validation/Test flow counts.

## Configuration and execution

- Exact enumeration was used for applications with at most 10 usable groups. Skype's 56 groups used deterministic LPT allocation followed by local moves/swaps.
- Split labels are assigned by partition size: largest partition = Train, middle = Validation, smallest = Test.
- No random process or seed was used.
- Command: `python stage14a5_vnat_split_feasibility/scripts/run_stage14a5.py`.
- Runtime: fixed TrafficClassifier Conda environment; CPU only; no GPU.
- Checkpoint/model-selection rule: not applicable.

## Core results

| application | usable groups | flows | train/val/test | minimum split | decision |
|---|---:|---:|---:|---:|---|
| netflix | 2 | 205 | 0/0/0 | 0 | excluded |
| rdp | 5 | 44 | 16/15/13 | 13 | borderline |
| rsync | 4 | 1,912 | 1,011/899/2 | 2 | excluded |
| scp | 3 | 2,290 | 1,214/1,075/1 | 1 | excluded |
| sftp | 7 | 1,668 | 709/646/313 | 313 | eligible |
| skype | 56 | 1,270 | 424/423/423 | 423 | eligible |
| ssh | 5 | 13,563 | 11,369/1,601/593 | 593 | borderline |
| vimeo | 1 | 1,218 | 0/0/0 | 0 | excluded |
| youtube | 4 | 341 | 137/125/79 | 79 | borderline |
| zoiper | 3 | 939 | 348/315/276 | 276 | borderline |

- Eligible: `sftp`, `skype`.
- Borderline: `rdp`, `ssh`, `youtube`, `zoiper`.
- Excluded: `netflix`, `rsync`, `scp`, `vimeo`.
- RDP satisfies DES-v1 `k=10` at the count level with 16 Train flows, but has only 44 total flows and sparse per-class P95 tail support.
- SCP is not merely fragile: its exact maximin optimum leaves one split with one flow, so it is excluded.
- Zoiper is count-balanced but has exactly one group per split and therefore no reassignment robustness.
- SSH's largest group contains 83.8% of the class flows; it is count-rich but capture-dependent.
- Retaining eligible plus borderline classes yields 3,125 Validation flows, corresponding to 156.25 expected observations above a global P95 threshold. Count support is adequate, but score-level threshold stability was not measured.

## Preserved evidence

- Required outputs: `vnat_group_statistics.csv`, `vnat_split_feasibility.csv`, `stage14a5_split_audit.md`.
- Quarantine evidence: `outputs/quarantined_duplicate_flows.csv`.
- Audit metadata and frozen-input hashes: `outputs/audit_details.json`, `outputs/input_hashes.json`.
- Implementation and regression checks: `scripts/`, `tests/`.
- Machine-readable evidence inventory: `manifest.json`.

## Limitations

- The maximin witness evaluates feasibility; it is not a frozen 60/20/20 or other final protocol.
- Skype's partition is a deterministic feasible lower bound, not a global optimum proof.
- P95 assessment is based on validation sample counts and expected tail size only. No score distribution or bootstrap stability was available because model execution was prohibited.
- Filename-derived application labels and the conservative Stage 14A group assumptions remain inherited limitations.
- Group dependence, not just flow count, limits effective statistical sample size.

## Conclusion and next step

Stage 14B is `CONDITIONAL YES`: proceed only with `sftp`, `skype`, and explicitly flagged borderline candidates `rdp`, `ssh`, `youtube`, `zoiper`; permanently quarantine the four duplicate flow rows and exclude `netflix`, `rsync`, `scp`, `vimeo`. No Stage 14B protocol or Unknown setting was generated in this run.
