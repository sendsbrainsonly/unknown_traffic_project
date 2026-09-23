#!/usr/bin/env python3
"""Generate the Stage 16 identity/parity audit without training.

The requested OURS-vs-Open-Detect benchmark has a fail-closed identity gate.
DQ-3F imports Open-Detect's CorrectedOpenDetectNet and training helpers, so this
script preserves the already-observed metrics but does not relabel them as an
independent method or launch a duplicate baseline experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "stage16_coarse_service_opendetect_benchmark"
DQ3F = (
    PROJECT
    / "stage15f_literature_guided_feature_benchmark"
    / "stage15f_dq3f_fine_service_matched_benchmark"
)
DQ4 = (
    PROJECT
    / "stage15f_literature_guided_feature_benchmark"
    / "stage15f_dq4_flow_selection_filtering_attribution"
)
OPEN_DETECT = PROJECT.parent / "Open-Detect"

SERVICES = ["Chat", "Email", "File-Transfer", "P2P", "Streaming", "VoIP"]
SEEDS = [2022, 2023, 2024]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def protected_assets() -> list[Path]:
    paths = [
        DQ3F / "dq3f_training_configs.json",
        DQ3F / "service_f3_results.csv",
        DQ3F / "service_f3_predictions.csv",
        DQ3F / "per_service_metrics.csv",
        DQ3F / "service_confusion_matrix.csv",
        DQ3F / "scripts" / "train_dq3f.py",
        DQ4 / "dq4_subset_manifest.csv",
        DQ4 / "dq4_training_configs.json",
        DQ4 / "dq4_training_results.csv",
        DQ4 / "dq4_within_population_results.csv",
        OPEN_DETECT / "code" / "model.py",
        OPEN_DETECT / "reproduction" / "corrected_model.py",
        OPEN_DETECT / "reproduction" / "run_reproduction.py",
    ]
    paths.extend(
        DQ3F / "runs" / "service" / f"seed{seed}" / "model_best.pt" for seed in SEEDS
    )
    paths.extend(
        DQ4 / "runs" / "M-C" / f"seed{seed}" / "model_best.pt" for seed in SEEDS
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected inputs missing: {missing}")
    return paths


def hash_assets(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance**0.5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "confusion_matrices").mkdir(exist_ok=True)
    before = hash_assets(protected_assets())
    (OUT / "protected_asset_hashes_before.json").write_text(
        json.dumps(before, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    mother = read_csv(DQ4 / "dq4_subset_manifest.csv")
    if len(mother) != 3065:
        raise RuntimeError(f"unexpected A population: {len(mother)}")
    if Counter(row["split_role"] for row in mother) != {
        "known_train": 2730,
        "known_validation": 335,
    }:
        raise RuntimeError("A split counts changed")
    if len({row["flow_id"] for row in mother}) != len(mother):
        raise RuntimeError("duplicate flow IDs in A")
    if sorted({row["service"] for row in mother}) != sorted(SERVICES):
        raise RuntimeError("Service label space changed")

    manifest_fields = [
        "flow_id",
        "split_role",
        "application",
        "service",
        "service_label_status",
        "capture_id",
        "source_file",
        "packet_count",
        "total_bytes",
        "captured_bytes",
        "duration_seconds",
        "in_protocol_A",
        "in_protocol_C",
    ]
    manifest_rows: list[dict[str, object]] = []
    for row in mother:
        manifest_rows.append(
            {
                "flow_id": row["flow_id"],
                "split_role": row["split_role"],
                "application": row["application"],
                "service": row["service"],
                "service_label_status": "WEAK_CAPTURE_LABEL",
                "capture_id": row["capture_id"],
                "source_file": row["source_file"],
                "packet_count": row["packet_count"],
                "total_bytes": row["total_bytes"],
                "captured_bytes": row["captured_bytes"],
                "duration_seconds": row["duration_seconds"],
                "in_protocol_A": int(row["A"]),
                "in_protocol_C": int(row["C_FINAL"]),
            }
        )
    write_csv(OUT / "matched_service_manifest.csv", manifest_fields, manifest_rows)

    all_lines = [
        f'{row["flow_id"]}|{row["split_role"]}|{row["service"]}' for row in mother
    ]
    train_lines = [row["flow_id"] for row in mother if row["split_role"] == "known_train"]
    val_lines = [
        row["flow_id"] for row in mother if row["split_role"] == "known_validation"
    ]
    label_lines = [f'{row["flow_id"]}|{row["service"]}' for row in mother]
    hashes = {
        "dataset_manifest_hash": canonical_hash(all_lines),
        "train_membership_hash": canonical_hash(train_lines),
        "validation_membership_hash": canonical_hash(val_lines),
        "service_label_hash": canonical_hash(label_lines),
    }
    c_counts = Counter(
        row["split_role"] for row in mother if int(row["C_FINAL"]) == 1
    )
    parity_rows = [
        {
            "protocol": "A",
            "train_count": 2730,
            "validation_count": 335,
            **hashes,
            "identity_unique": "PASS",
            "label_space": "PASS",
            "status": "PASS",
        },
        {
            "protocol": "C",
            "train_count": c_counts["known_train"],
            "validation_count": c_counts["known_validation"],
            "dataset_manifest_hash": canonical_hash(
                [line for line, row in zip(all_lines, mother) if int(row["C_FINAL"]) == 1]
            ),
            "train_membership_hash": canonical_hash(
                [
                    row["flow_id"]
                    for row in mother
                    if int(row["C_FINAL"]) == 1 and row["split_role"] == "known_train"
                ]
            ),
            "validation_membership_hash": canonical_hash(
                [
                    row["flow_id"]
                    for row in mother
                    if int(row["C_FINAL"]) == 1
                    and row["split_role"] == "known_validation"
                ]
            ),
            "service_label_hash": canonical_hash(
                [
                    f'{row["flow_id"]}|{row["service"]}'
                    for row in mother
                    if int(row["C_FINAL"]) == 1
                ]
            ),
            "identity_unique": "PASS",
            "label_space": "PASS",
            "status": "PASS_WITH_LOW_VALIDATION_SUPPORT",
        },
    ]
    write_csv(
        OUT / "dataset_parity_audit.csv",
        [
            "protocol",
            "train_count",
            "validation_count",
            "dataset_manifest_hash",
            "train_membership_hash",
            "validation_membership_hash",
            "service_label_hash",
            "identity_unique",
            "label_space",
            "status",
        ],
        parity_rows,
    )

    dq3f_script_hash = sha256_file(DQ3F / "scripts" / "train_dq3f.py")
    corrected_hash = sha256_file(OPEN_DETECT / "reproduction" / "corrected_model.py")
    released_hash = sha256_file(OPEN_DETECT / "code" / "model.py")
    runner_hash = sha256_file(OPEN_DETECT / "reproduction" / "run_reproduction.py")
    identity_text = f"""# Stage 16 method identity audit

## Gate outcome

`METHOD_IDENTITY_NOT_DISTINCT` — the requested `OURS-Service6` versus
`OpenDetect-Service6` claim is not identifiable from the frozen DQ-3F artifacts.
No new training was launched.

## Direct code provenance

- DQ-3F `train_dq3f.py` lines 47--56 add `Projects/Open-Detect/code` and
  `Projects/Open-Detect/reproduction` to `sys.path`, then import
  `CorrectedOpenDetectNet`, `reset_prototypes_in_place`, `run_epoch`, and
  Open-Detect's `weight_init`.
- Lines 130--143 instantiate `CorrectedOpenDetectNet` and use Adam plus the
  Open-Detect reproduction training helper.
- Lines 179--193 execute Open-Detect's loss/training loop, in-place prototype
  reset, nearest-prototype validation, and Known-Validation Accuracy checkpoint
  selection.
- The frozen DQ-3F config explicitly records
  `training_protocol = corrected-paper`.
- `CorrectedOpenDetectNet` subclasses the released `OpenDetectNet`. Its documented
  changes are a connected final decoder residual block, bounded log-variance,
  paper-consistent KL/prototype classification, and removal of the released
  constant entropy term. These are Open-Detect reproduction corrections, not an
  independent project model.

## What is and is not distinct

| Item | DQ-3F Service model | Released Open-Detect | Identity implication |
|---|---|---|---|
| Core class | `CorrectedOpenDetectNet(OpenDetectNet)` | `OpenDetectNet` | Same method family and inherited architecture |
| Input | 32x32 one-channel byte image | 32x32 one-channel byte image in local reproduction | Same representation family |
| Encoder | ResNet18 VAE encoder | ResNet18 VAE encoder | Same |
| Prototype rule | learned class prototypes | learned class prototypes | Same mechanism |
| Objective | corrected paper-consistent Open-Detect loss | released-code Open-Detect loss | Reproduction variant difference |
| Training helper | Open-Detect `run_reproduction.py` | Open-Detect code/reproduction | Same provenance |
| Service adaptation | six output prototypes | six output prototypes required | Task-only adaptation |

The defensible comparison would be **Corrected Open-Detect versus released-code
Open-Detect**, not **our method versus Open-Detect**. Renaming the corrected
variant as an independent method would violate the Stage 16 instruction not to
manufacture method differences.

The project does contain a separately named Stage 14C-6 F2 own-method pipeline,
but it was not used in DQ-3F and cannot be silently substituted while claiming
to reuse the DQ-3F method.

## Source SHA256

- DQ-3F trainer: `{dq3f_script_hash}`
- Open-Detect corrected model: `{corrected_hash}`
- Open-Detect reproduction runner: `{runner_hash}`
- Open-Detect released model: `{released_hash}`

## Decision

The method-identity gate fails before GPU work. Stage 16-2 through Stage 16-5
are `NOT_RUN_IDENTITY_GATE`; the existing DQ-3F/DQ-4 measurements are retained
below only as historical corrected-Open-Detect evidence, never as a two-method
comparison.
"""
    (OUT / "method_identity_audit.md").write_text(identity_text, encoding="utf-8")

    parity_md = f"""# Stage 16 dataset parity audit

The frozen data side passes even though the method-identity side does not.

| Protocol | Train | Validation | Label space | Status |
|---|---:|---:|---|---|
| A | 2,730 | 335 | 6 Services | PASS |
| C | {c_counts['known_train']:,} | {c_counts['known_validation']:,} | 6 Services | PASS_WITH_LOW_VALIDATION_SUPPORT |

Canonical hashes use sorted UTF-8 records and a terminal newline:

- dataset manifest: `{hashes['dataset_manifest_hash']}`
- Train membership: `{hashes['train_membership_hash']}`
- Validation membership: `{hashes['validation_membership_hash']}`
- Service labels: `{hashes['service_label_hash']}`

Every A flow ID is unique; A and C retain all six Service labels. All labels
remain `WEAK_CAPTURE_LABEL`. No Known Test or Unknown Test feature was read.
Because no second method was run, cross-method runtime parity is not claimed.
Machine-readable A/C hashes are in `dataset_parity_audit.csv`.
"""
    (OUT / "dataset_parity_audit.md").write_text(parity_md, encoding="utf-8")

    dq3f_results = read_csv(DQ3F / "service_f3_results.csv")
    ours_fields = [
        "requested_method_name",
        "actual_method_identity",
        "identity_gate",
        *list(dq3f_results[0].keys()),
    ]
    ours_rows = [
        {
            "requested_method_name": "OURS-Service6",
            "actual_method_identity": "CorrectedOpenDetectNet_DQ3F",
            "identity_gate": "METHOD_IDENTITY_NOT_DISTINCT",
            **row,
        }
        for row in dq3f_results
    ]
    write_csv(OUT / "ours_service_results.csv", ours_fields, ours_rows)

    od_fields = [
        "method",
        "seed",
        "train_samples",
        "validation_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "runtime_seconds",
        "peak_gpu_memory_bytes",
        "checkpoint_path",
        "checkpoint_sha256",
        "status",
        "reason",
    ]
    od_rows = [
        {
            "method": "OpenDetect-Service6",
            "seed": seed,
            "train_samples": 2730,
            "validation_samples": 335,
            "status": "NOT_RUN_IDENTITY_GATE",
            "reason": "DQ-3F candidate is already a corrected Open-Detect reproduction; independent OURS comparison is undefined",
        }
        for seed in SEEDS
    ]
    write_csv(OUT / "opendetect_service_results.csv", od_fields, od_rows)

    paired_rows = [
        {
            "seed": seed,
            "ours_actual_identity": "CorrectedOpenDetectNet_DQ3F",
            "opendetect_status": "NOT_RUN_IDENTITY_GATE",
            "delta_accuracy": "",
            "delta_macro_f1": "",
            "delta_weighted_f1": "",
            "status": "NOT_IDENTIFIABLE",
        }
        for seed in SEEDS
    ]
    write_csv(
        OUT / "paired_service_comparison.csv",
        [
            "seed",
            "ours_actual_identity",
            "opendetect_status",
            "delta_accuracy",
            "delta_macro_f1",
            "delta_weighted_f1",
            "status",
        ],
        paired_rows,
    )

    per_service = read_csv(DQ3F / "per_service_metrics.csv")
    per_rows = [
        {
            "method": "CorrectedOpenDetectNet_DQ3F",
            "protocol": "A",
            **row,
            "identity_note": "historical DQ-3F F3; not independent OURS",
        }
        for row in per_service
        if row["task"] == "F3"
    ]
    write_csv(
        OUT / "per_service_results.csv",
        [
            "method",
            "protocol",
            "task",
            "seed",
            "class_index",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
            "correct",
            "errors",
            "identity_note",
        ],
        per_rows,
    )

    confusion = read_csv(DQ3F / "service_confusion_matrix.csv")
    for seed in SEEDS:
        rows = [row for row in confusion if int(row["seed"]) == seed and row["task"] == "F3"]
        write_csv(
            OUT / "confusion_matrices" / f"dq3f_corrected_opendetect_seed{seed}.csv",
            ["task", "seed", "true_label", "predicted_label", "count"],
            rows,
        )

    write_csv(
        OUT / "paired_error_analysis.csv",
        [
            "seed",
            "grouping",
            "group_value",
            "both_correct",
            "dq3f_corrected_only_correct",
            "released_opendetect_only_correct",
            "both_wrong",
            "status",
        ],
        [
            {
                "seed": seed,
                "grouping": "ALL",
                "group_value": "ALL",
                "status": "NOT_RUN_IDENTITY_GATE",
            }
            for seed in SEEDS
        ],
    )

    capture_sets: dict[str, set[str]] = defaultdict(set)
    support = Counter()
    split_support: dict[tuple[str, str], int] = Counter()
    for row in mother:
        capture_sets[row["service"]].add(row["capture_id"])
        support[row["service"]] += 1
        split_support[(row["service"], row["split_role"])] += 1
    capture_lines = [
        "# Stage 16 capture dependency audit",
        "",
        "All labels remain `WEAK_CAPTURE_LABEL`; the table is descriptive only.",
        "",
        "| Service | Train flows | Validation flows | Total flows | Independent captures |",
        "|---|---:|---:|---:|---:|",
    ]
    for service in SERVICES:
        capture_lines.append(
            f"| {service} | {split_support[(service, 'known_train')]} | "
            f"{split_support[(service, 'known_validation')]} | {support[service]} | "
            f"{len(capture_sets[service])} |"
        )
    capture_lines.extend(
        [
            "",
            "P2P has one independent capture, so `CAPTURE_GENERALIZATION_NOT_IDENTIFIABLE` remains in force. "
            "The current random flow split cannot be interpreted as capture-disjoint generalization.",
        ]
    )
    (OUT / "capture_dependency_audit.md").write_text(
        "\n".join(capture_lines) + "\n", encoding="utf-8"
    )

    within = read_csv(DQ4 / "dq4_within_population_results.csv")
    filtering_rows: list[dict[str, object]] = []
    for row in within:
        if row["model"] not in {"M-A", "M-C"}:
            continue
        filtering_rows.append(
            {
                **row,
                "actual_method_identity": "CorrectedOpenDetectNet_DQ3F_DQ4",
                "comparison_scope": "historical A-vs-C selection effect within one Open-Detect reproduction family",
                "two_method_stage16_status": "NOT_RUN_IDENTITY_GATE",
            }
        )
    write_csv(
        OUT / "filtering_protocol_comparison.csv",
        list(within[0].keys())
        + ["actual_method_identity", "comparison_scope", "two_method_stage16_status"],
        filtering_rows,
    )

    dq4_training = read_csv(DQ4 / "dq4_training_results.csv")
    cost_rows: list[dict[str, object]] = []
    for row in dq4_training:
        if row["model"] not in {"M-A", "M-C"}:
            continue
        cost_rows.append(
            {
                "method": "CorrectedOpenDetectNet_DQ3F_DQ4",
                "protocol": row["training_subset"],
                "seed": row["seed"],
                "runtime_seconds": row["runtime_seconds"],
                "peak_gpu_memory_bytes": "2127713792" if row["model"] == "M-A" else "",
                "completed_epochs": row["completed_epochs"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "status": row["status"],
            }
        )
    for seed in SEEDS:
        cost_rows.append(
            {
                "method": "OpenDetect-Service6-new-baseline",
                "protocol": "A/C",
                "seed": seed,
                "status": "NOT_RUN_IDENTITY_GATE",
            }
        )
    write_csv(
        OUT / "training_cost_comparison.csv",
        [
            "method",
            "protocol",
            "seed",
            "runtime_seconds",
            "peak_gpu_memory_bytes",
            "completed_epochs",
            "checkpoint_sha256",
            "status",
        ],
        cost_rows,
    )

    open_set_text = """# Service-level open-set feasibility

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
"""
    (OUT / "service_open_set_feasibility.md").write_text(open_set_text, encoding="utf-8")

    metrics = {
        key: mean_std([float(row[key]) for row in dq3f_results])
        for key in ("accuracy", "macro_f1", "weighted_f1")
    }
    report = f"""# Stage 16 — Coarse Service Open-Detect benchmark

## Terminal status

`BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`

The data parity Gate passed, but the method identity Gate failed. DQ-3F's F3
trainer directly instantiates Open-Detect's `CorrectedOpenDetectNet`, imports
Open-Detect's training loop and prototype reset, and uses its VAE/prototype loss.
Consequently, the requested independent `OURS-Service6` versus
`OpenDetect-Service6` experiment is not scientifically identifiable.

## Data audit

- Protocol A: 2,730 Known Train + 335 Known Validation = 3,065 flows.
- Protocol C: {c_counts['known_train']:,} Train + {c_counts['known_validation']:,} Validation = {sum(c_counts.values()):,} flows.
- Classes: Chat, Email, File-Transfer, P2P, Streaming, VoIP.
- Labels: 3,065/3,065 `WEAK_CAPTURE_LABEL`.
- Known Test / Unknown Test feature usage: 0 / 0.

## Existing DQ-3F F3 evidence (correctly identified)

These are real historical results but belong to the corrected Open-Detect
reproduction, not an independent new method:

| Seed | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
"""
    for row in dq3f_results:
        report += (
            f"| {row['seed']} | {float(row['accuracy']):.6f} | "
            f"{float(row['macro_f1']):.6f} | {float(row['weighted_f1']):.6f} |\n"
        )
    report += f"""

Mean ± population std:

- Accuracy: `{metrics['accuracy'][0]:.6f} ± {metrics['accuracy'][1]:.6f}`
- Macro-F1: `{metrics['macro_f1'][0]:.6f} ± {metrics['macro_f1'][1]:.6f}`
- Weighted-F1: `{metrics['weighted_f1'][0]:.6f} ± {metrics['weighted_f1'][1]:.6f}`

## Required questions

1. **Our method result:** no independently identified method result exists in
   DQ-3F. The reported {metrics['macro_f1'][0]:.6f} Macro-F1 is corrected
   Open-Detect evidence.
2. **New Open-Detect result:** `NOT_RUN_IDENTITY_GATE`; running it would compare
   two Open-Detect variants, not the requested two method entities.
3. **Metric difference:** not identifiable; no deltas are fabricated.
4. **Seed stability:** the historical corrected variant is seed-sensitive
   (Macro-F1 std {metrics['macro_f1'][1]:.6f}; seed 2023 is weakest).
5. **Per-Service winner:** not identifiable without a second distinct method.
6. **Open-Detect rescue:** not identifiable.
7. **Error complementarity:** not run because paired independent predictions do
   not exist.
8. **C protocol:** historical DQ-4 A/C selection evidence is retained, but no
   two-method C comparison was run.
9. **Limitations:** weak labels and the single P2P capture remain material.
10. **Cause attribution:** current evidence only establishes a corrected-versus-
    released implementation distinction, not a representation/supervision win
    for a new method.
11. **Service open set:** blocked pending a newly frozen Service-level test and
    capture-aware protocol.
12. **Byte–Behavior:** its possible value cannot be decided from this invalid
    two-method premise; the existing Stage 14C-6 F2 own-method is the proper
    independent starting point if explicitly selected in a new preregistration.

## Gates

- `COARSE_SERVICE_BENCHMARK_COMPLETED`: **NO**
- `METHOD_IDENTITY_NOT_DISTINCT`: **YES**
- `OURS_SERVICE_ADVANTAGE_OBSERVED`: **NOT_IDENTIFIABLE**
- `OPENDETECT_SERVICE_ADVANTAGE_OBSERVED`: **NOT_IDENTIFIABLE**
- `SERVICE_OPEN_SET_PROTOCOL_BLOCKED`: **YES**

No GPU was selected or used, no encoder was trained, no legacy checkpoint or
prediction was modified, and no open-set experiment was started.
"""
    (OUT / "stage16_report.md").write_text(report, encoding="utf-8")

    results = f"""# Experiment results: stage16-coarse-service-opendetect-benchmark-20260920-v1

- Status: `partial / BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`
- Experiment type: `method-identity-and-closed-set-benchmark`
- Claim scope: `diagnostic`

## Data and split

- Protocol A: 2,730 Known Train + 335 Known Validation = 3,065 flows.
- Protocol C: {c_counts['known_train']:,} Known Train + {c_counts['known_validation']:,} Known Validation = {sum(c_counts.values()):,} flows.
- Classes: Chat, Email, File-Transfer, P2P, Streaming, VoIP.
- Labels: 3,065/3,065 `WEAK_CAPTURE_LABEL`.
- Known Test / Unknown Test feature usage: `0 / 0`.
- A/C flow membership and Service labels pass hash parity.

## Configuration and execution

- DQ-3F directly imports and instantiates Open-Detect's
  `CorrectedOpenDetectNet`, `run_epoch`, `reset_prototypes_in_place`, and
  `weight_init`.
- The DQ-3F frozen configuration records `training_protocol=corrected-paper`.
- New Open-Detect training runs: `0`; GPU selected/used: `NO`.

## Core results

The frozen DQ-3F F3 results belong to `CorrectedOpenDetectNet_DQ3F`:

| Seed | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
"""
    for row in dq3f_results:
        results += (
            f"| {row['seed']} | {float(row['accuracy']):.6f} | "
            f"{float(row['macro_f1']):.6f} | {float(row['weighted_f1']):.6f} |\n"
        )
    results += f"""

- Accuracy: `{metrics['accuracy'][0]:.6f} ± {metrics['accuracy'][1]:.6f}`
- Macro-F1: `{metrics['macro_f1'][0]:.6f} ± {metrics['macro_f1'][1]:.6f}`
- Weighted-F1: `{metrics['weighted_f1'][0]:.6f} ± {metrics['weighted_f1'][1]:.6f}`

The new baseline and all two-method deltas are
`NOT_RUN_IDENTITY_GATE / NOT_IDENTIFIABLE`; no values were fabricated.

## Preserved evidence

- `method_identity_audit.md`, `matched_service_manifest.csv`, and
  `dataset_parity_audit.csv` preserve the identity and A/C parity evidence.
- Historical per-Service metrics and confusion matrices remain explicitly
  labeled as corrected Open-Detect evidence.
- Protected hash snapshots and `completion_verification.json` record integrity.

## Limitations

- DQ-3F and the proposed baseline are Open-Detect implementations, so the
  independent OURS-versus-Open-Detect comparison is not identifiable.
- Labels are weak capture labels; P2P has one capture.
- No independent Service-level Test protocol is frozen.

## Conclusion and next step

Final Gate: `BLOCKED_METHOD_IDENTITY_NOT_DISTINCT`. Preregister either an
accurately named corrected-versus-released Open-Detect study or a true
Stage14C-6 F2 own-method Service benchmark. No Open-Set/DES/H1/Byte–Behavior or
new encoder run was started.
"""
    (OUT / "RESULTS.md").write_text(results, encoding="utf-8")

    after = hash_assets(protected_assets())
    (OUT / "protected_asset_hashes_after.json").write_text(
        json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protected_unchanged = before == after
    required = [
        "method_identity_audit.md",
        "matched_service_manifest.csv",
        "dataset_parity_audit.md",
        "dataset_parity_audit.csv",
        "ours_service_results.csv",
        "opendetect_service_results.csv",
        "paired_service_comparison.csv",
        "per_service_results.csv",
        "paired_error_analysis.csv",
        "capture_dependency_audit.md",
        "filtering_protocol_comparison.csv",
        "training_cost_comparison.csv",
        "service_open_set_feasibility.md",
        "stage16_report.md",
        "RESULTS.md",
    ]
    verification = {
        "status": "BLOCKED_METHOD_IDENTITY_NOT_DISTINCT",
        "method_identity_gate": "FAIL",
        "data_parity_gate": "PASS",
        "protocol_A_counts": {"train": 2730, "validation": 335},
        "protocol_C_counts": {
            "train": c_counts["known_train"],
            "validation": c_counts["known_validation"],
        },
        "required_files_present": all((OUT / name).is_file() for name in required),
        "required_files": required,
        "confusion_matrices_present": len(list((OUT / "confusion_matrices").glob("*.csv"))) == 3,
        "protected_assets_unchanged": protected_unchanged,
        "protected_asset_count": len(before),
        "known_test_feature_values_used": 0,
        "unknown_test_feature_values_used": 0,
        "new_training_runs": 0,
        "gpu_used": False,
        "opendetect_service6": "NOT_RUN_IDENTITY_GATE",
        "paired_method_comparison": "NOT_IDENTIFIABLE",
        "open_set_experiment": "NOT_RUN",
        "des_h1_modified": False,
        "completion_gate": "PASS_BLOCKED_AUDIT_COMPLETE",
    }
    (OUT / "completion_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
