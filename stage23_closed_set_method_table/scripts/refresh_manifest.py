#!/usr/bin/env python3
"""Finalize Stage23 preservation manifest after all mutable runs stop."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MANIFEST = ROOT / "manifest.json"
VERIFICATION = ROOT / "completion_verification.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    verified = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    if verified["status"] != "PASS_FIVE_METHOD_MATCHED_TABLE" or verified["verified_runs"] != 20:
        raise RuntimeError("independent Stage23 final verification not PASS")
    frozen = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
    if sha(frozen) != verified["stage20_manifest_sha256"]:
        raise RuntimeError("Stage20 frozen hash changed")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    data["status"] = "complete"
    data["code"]["changes"] = sorted(str(p.relative_to(PROJECT)) for p in (ROOT / "scripts").glob("*.py"))
    data["execution"] = {
        "status": "all_training_and_evaluation_jobs_finished",
        "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
        "max_concurrent_physical_gpus": 6,
        "four_card_release": "RoNeTC GPU0-3 jobs completed and released before 2026-09-24 08:00 Asia/Shanghai",
        "tmux_evidence_root": "../.tmux-task",
    }
    data["configuration"] = {
        "files": ["EXPERIMENT_PLAN.md", "PROTOCOL.md", "METHOD_ADMISSION_AUDIT.md"],
        "datasets": ["iscx_vpn", "iscx_tor"], "seeds": [2022, 2023],
        "frozen_flow_manifest_sha256": verified["stage20_manifest_sha256"],
        "methods": ["TrafficFormer-pretrained", "OURS-E3-T8-pretrained",
                    "Open-Detect corrected-paper", "RoNeTC-runnable-reconstruction",
                    "YaTC-official-pretrained-Stage20"],
        "checkpoint_selection": "Known Validation only",
        "test_selection_samples": 0, "unknown_samples": 0,
    }
    data["core_results"] = [
        {"name": "independently_verified_method_rows", "value": 20},
        {"name": "independent_replay_checks", "value": verified["checks"]},
        {"name": "matched_flow_count", "value": 22136},
        {"name": "vpn_best_macro_f1", "value": 0.8870850180189982, "method": "YaTC"},
        {"name": "tor_best_macro_f1", "value": 0.8273933736836749, "method": "YaTC"},
        {"name": "strict_native_exclusions", "value": "ET-BERT,TFE-GNN,Trident,UnDiff"},
    ]
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        role = ("checkpoint" if path.suffix in (".pt", ".pth") else
                "metrics-or-data" if path.suffix in (".csv", ".json", ".jsonl", ".npy") else
                "artifact")
        artifacts.append({
            "path": relative, "scope": "bundle", "role": role,
            "size_bytes": path.stat().st_size, "sha256": sha(path), "hash_status": "verified",
        })
    data["artifacts"] = artifacts
    data["artifact_count"] = len(artifacts)
    data["artifact_bytes"] = sum(item["size_bytes"] for item in artifacts)
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "artifacts": len(artifacts),
                      "bytes": data["artifact_bytes"], "verification_checks": verified["checks"]}))


if __name__ == "__main__":
    main()
