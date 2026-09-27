"""Freeze Stage39 service roles from existing immutable metadata, before model work."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DESIGN = PROJECT / "stage39_coarse_open_set_task_design"
STAGE38 = PROJECT / "stage38_three_view_des_open_set_transfer"
sys.path.insert(0, str(DESIGN))
from audit_semantics import load_metadata, sha  # noqa: E402


def write_json(path: Path, value: object):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit = json.loads((DESIGN / "completion_verification.json").read_text())
    if audit["status"] != "PASS" or not audit["source_hashes_unchanged"]:
        raise RuntimeError("Stage39 metadata audit missing or failed")
    paths = {"iscx_vpn": PROJECT / "stage12_dual_external_validation/protocol/iscx_vpn/split_manifest.csv",
             "iscx_tor": PROJECT / "stage12_dual_external_validation/protocol/iscx_tor/split_manifest.csv",
             "VNAT": PROJECT / "stage14b_vnat_protocol_freeze/vnat_split_manifest.csv"}
    for path in paths.values():
        if sha(path) != audit["source_hashes_after"][str(path)]:
            raise RuntimeError(f"source metadata hash changed: {path}")
    lock_path = STAGE38 / "stage38b_training_config_lock.json"
    locked_code = json.loads(lock_path.read_text())["source_code_sha256"]
    for label, digest in locked_code.items():
        if sha(PROJECT / label) != digest:
            raise RuntimeError(f"Stage38 source training code changed: {label}")
    plan_sha = sha(DESIGN / "PLAN.md")
    expected = {(r["dataset"], r["heldout_known_service"]): r for r in
                csv.DictReader((DESIGN / "service_holdout_feasibility.csv").open())}
    indexes = []
    for dataset, path in paths.items():
        rows = load_metadata(dataset, path)
        known = {r["service"] for r in rows if r["role"] == "known_train"}
        all_services = {r["service"] for r in rows}
        heldouts = ([None] if all_services - known else []) + sorted(known)
        for heldout in heldouts:
            key = heldout if heldout else "NONE_ALREADY_UNSEEN_SERVICE"
            new_known = known - ({heldout} if heldout else set())
            new_unknown = all_services - new_known
            slug = "p2p_only" if heldout is None else heldout.lower().replace("-", "_")
            unit = ROOT / "settings" / dataset / slug
            unit.mkdir(parents=True, exist_ok=False)
            assigned = []
            for row in sorted(rows, key=lambda item: item["uid"]):
                new_role = ("unknown_eval" if row["service"] in new_unknown else
                            "auxiliary_known_service_unseen_app" if row["role"] == "unknown_test" else row["role"])
                assigned.append({"sample_id": row["uid"], "role": new_role,
                    "original_role": row["role"], "service": row["service"],
                    "application": row["application"], "classifier_label": row["classifier_label"],
                    "domain": row["domain"], "group_id": row["group"]})
            role_path = unit / "role_manifest.csv"
            write_csv(role_path, assigned)
            counts = Counter(row["role"] for row in assigned)
            original = expected[dataset, key]
            for name in ("known_train", "known_validation", "known_test", "unknown_eval",
                         "auxiliary_known_service_unseen_app"):
                if counts[name] != int(original[name]):
                    raise RuntimeError(f"{dataset}/{key} role count changed: {name}")
            if len(assigned) != len({r["sample_id"] for r in assigned}):
                raise RuntimeError("duplicate sample ID")
            if any(r["original_role"] in ("known_test", "unknown_test") and
                   r["role"] in ("known_train", "known_validation") for r in assigned):
                raise RuntimeError("historical Test promoted into fitting")
            classes = sorted({r["classifier_label"] for r in assigned if r["role"] == "known_train"})
            if classes != original["classifier_classes"].split(";"):
                raise RuntimeError("known classifier labels changed")
            if any(not any(r["classifier_label"] == label and r["role"] == role
                           for r in assigned) for label in classes for role in
                   ("known_train", "known_validation", "known_test")):
                raise RuntimeError("Known class missing a role")
            frozen = {"status": "FROZEN_PRETRAIN", "dataset": dataset, "heldout_service": key,
                "slug": slug, "protocol_id": "medium_seed2025" if dataset == "VNAT" else "medium_seed2022",
                "training_seed": 2022, "known_services": sorted(new_known),
                "unknown_services": sorted(new_unknown), "known_classifier_labels": classes,
                "counts": dict(counts), "role_manifest_sha256": sha(role_path),
                "source_manifest": str(path), "source_manifest_sha256": sha(path),
                "stage39_plan_sha256": plan_sha, "stage38_training_lock_sha256": sha(lock_path),
                "source_training_code_sha256": locked_code,
                "historical_test_promoted_to_training": 0,
                "prior_unknown_promoted_to_training": 0,
                "all_input_samples_accounted_for": True,
                "checkpoint_reuse": heldout is None and dataset == "iscx_tor",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(unit / "protocol.json", frozen)
            indexes.append({"dataset": dataset, "heldout_service": key, "slug": slug,
                            "new_training_required": not frozen["checkpoint_reuse"],
                            "known_train": counts["known_train"],
                            "known_validation": counts["known_validation"],
                            "known_test": counts["known_test"],
                            "unknown_eval": counts["unknown_eval"],
                            "auxiliary_known_service_unseen_app": counts["auxiliary_known_service_unseen_app"],
                            "role_manifest_sha256": frozen["role_manifest_sha256"]})
    if len(indexes) != 13:
        raise RuntimeError(f"expected 13 predefined settings, got {len(indexes)}")
    write_csv(ROOT / "protocol_index.csv", indexes)
    write_json(ROOT / "freeze_verification.json", {"status": "PASS", "settings": len(indexes),
        "new_training_settings": sum(x["new_training_required"] for x in indexes),
        "source_hashes": {str(p): sha(p) for p in paths.values()},
        "unknown_train_val_samples": 0, "historical_test_promotions": 0,
        "test_feature_values_loaded": 0, "unknown_feature_values_loaded": 0})
    print(json.dumps({"status": "PASS", "settings": len(indexes),
                      "new_training_settings": 12}), flush=True)


if __name__ == "__main__":
    main()
