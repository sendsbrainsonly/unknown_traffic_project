"""Metadata-only audit; creates feasibility counts, never model inputs or splits."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ISCX_SERVICE = {
    "Chat": "Communication", "Email": "Communication", "VoIP": "Communication",
    "Audio": "Streaming", "Video": "Streaming", "Streaming": "Streaming",
    "File-Transfer": "File-Transfer", "P2P": "P2P", "Browsing": "Browsing",
}
# Proposed task taxonomy, not an assertion of authoritative per-flow activity.
VNAT_SERVICE = {
    "netflix": "Streaming", "youtube": "Streaming", "vimeo": "Streaming",
    "rsync": "File-Transfer", "scp": "File-Transfer", "sftp": "File-Transfer",
    "skype": "Communication", "zoiper": "Communication",
    "rdp": "Remote-Access", "ssh": "Remote-Access",
}
ROLES = ("known_train", "known_validation", "known_test", "unknown_test")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(name, rows):
    with (ROOT / name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(dataset, path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if dataset == "VNAT":
                if row["protocol_id"] != "medium_seed2025":
                    continue
                app = row["application"]
                role = "unknown_test" if row["class_role"] == "unknown" else "known_" + row["split"]
                role = {"known_val": "known_validation"}.get(role, role)
                item = dict(uid=row["flow_uid"], application=app, role=role,
                            service=VNAT_SERVICE[app], group=row["group_id"],
                            domain=row["vpn_status"], classifier_label="streaming" if app in ("netflix", "youtube") else app)
            else:
                if row["setting"] != "medium":
                    continue
                service = ISCX_SERVICE[row["official_category"]]
                item = dict(uid=row["flow_id_sha256"], application=row["canonical_class"],
                            role=row["role"], service=service, group=row["source_file"],
                            domain=row["domain_state"], classifier_label=service)
            assert item["role"] in ROLES, item["role"]
            rows.append(item)
    assert len(rows) == len({row["uid"] for row in rows})
    return rows


def main():
    inputs = {name: PROJECT / "stage12_dual_external_validation/protocol" / name / "split_manifest.csv"
              for name in ("iscx_vpn", "iscx_tor")}
    inputs["VNAT"] = PROJECT / "stage14b_vnat_protocol_freeze/vnat_split_manifest.csv"
    before = {str(path): sha(path) for path in inputs.values()}
    overlaps, supports, folds, summaries = [], [], [], []
    for dataset, path in inputs.items():
        rows = load_metadata(dataset, path)
        known = {r["service"] for r in rows if r["role"] == "known_train"}
        all_services = {r["service"] for r in rows}
        unknown_counts = Counter((r["application"], r["service"]) for r in rows if r["role"] == "unknown_test")
        for (app, service), count in sorted(unknown_counts.items()):
            overlaps.append(dict(dataset=dataset, legacy_unknown_application=app, service=service,
                                 flows=count, service_seen_in_known_train=service in known,
                                 interpretation="unseen_application_of_known_service" if service in known else "unseen_service"))
        for service in sorted(all_services):
            subset = [r for r in rows if r["service"] == service]
            counts = Counter(r["role"] for r in subset)
            supports.append(dict(dataset=dataset, service=service, flows=len(subset),
                                 **{role: counts[role] for role in ROLES}, groups=len({r["group"] for r in subset}),
                                 domains=json.dumps(dict(Counter(r["domain"] for r in subset)), sort_keys=True)))
        # No historical Unknown sample is reassigned to Train/Validation.
        # Services absent from historical Known Train remain Unknown in every fold.
        heldouts = ([None] if all_services - known else []) + sorted(known)
        for heldout in heldouts:
            new_known = known - ({heldout} if heldout else set())
            new_unknown = all_services - new_known
            core = [r for r in rows if r["service"] in new_known and r["role"] != "unknown_test"]
            positive = [r for r in rows if r["service"] in new_unknown]
            auxiliary = [r for r in rows if r["service"] in new_known and r["role"] == "unknown_test"]
            counts = Counter(r["role"] for r in core)
            classes = sorted({r["classifier_label"] for r in core if r["role"] == "known_train"})
            minima = {role: min(sum(r["role"] == role and r["classifier_label"] == cls for r in core)
                                for cls in classes) for role in ROLES[:3]}
            assert len(core) + len(positive) + len(auxiliary) == len(rows)
            assert all(minima[r] > 0 for r in ROLES[:3])
            assert minima["known_train"] >= 10
            folds.append(dict(dataset=dataset, heldout_known_service=heldout or "NONE_ALREADY_UNSEEN_SERVICE",
                              known_services=";".join(sorted(new_known)), unknown_services=";".join(sorted(new_unknown)),
                              classifier_classes=";".join(classes), num_known_classes=len(classes),
                              **{role: counts[role] for role in ROLES[:3]}, unknown_eval=len(positive),
                              auxiliary_known_service_unseen_app=len(auxiliary),
                              **{"min_" + role + "_per_class": minima[role] for role in ROLES[:3]},
                              all_samples_accounted_for=True, historical_test_promoted_to_training=0,
                              reuse_stage38_checkpoint=heldout is None, status="METADATA_FEASIBLE_NOT_FROZEN"))
        old_unknown = sum(unknown_counts.values())
        overlap = sum(count for (_, service), count in unknown_counts.items() if service in known)
        summaries.append(dict(dataset=dataset, old_unknown_flows=old_unknown,
                              old_unknown_in_known_services=overlap, overlap_ratio=overlap / old_unknown))
    after = {str(path): sha(path) for path in inputs.values()}
    assert before == after
    write_csv("semantic_overlap_audit.csv", overlaps)
    write_csv("service_support.csv", supports)
    write_csv("service_holdout_feasibility.csv", folds)
    verification = dict(status="PASS", scope="metadata_audit_only", source_hashes_before=before,
                        source_hashes_after=after, source_hashes_unchanged=True, feature_values_read=0,
                        models_trained=0, formal_splits_generated=0, feasibility_rows=len(folds))
    (ROOT / "completion_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest.update(status="success", updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    inputs=[dict(path=p, sha256=h, access="CSV metadata only") for p, h in before.items()],
                    core_results=summaries,
                    configuration=dict(files=["audit_semantics.py", "PLAN.md"], seeds=[2022],
                                       parameters=dict(training_launched=False, unknown_unit="service")),
                    execution=dict(command="python stage39_coarse_open_set_task_design/audit_semantics.py",
                                   tmux_session="stage39_design_audit_0926", environment="2025-10-8-WXY-dgl_py310",
                                   physical_gpu_ids=[], exit_code=0),
                    limitations=["Proposed VNAT taxonomy; ISCX capture-level weak labels",
                                 "Metadata feasibility is not evidence of improved detection",
                                 "Flow splits may share capture groups; exposed development datasets"],
                    next_step="Freeze service-role derivations and source training config before training; see PLAN.md")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", summary=summaries, feasibility_rows=len(folds)), indent=2))


if __name__ == "__main__":
    main()
