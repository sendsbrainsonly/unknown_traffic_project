#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone

from stage21_common import CLOSED_MANIFEST, CONFIG, OUT, cache_path, sha256_file, write_json


def main() -> None:
    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == OUT / "manifest.json" or "__pycache__" in path.parts:
            continue
        artifacts.append({
            "path": str(path.relative_to(OUT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schema_version": 1,
        "experiment_id": "stage21-coarse-service-ours-e3-t8-20260922-v1",
        "created_at_utc": "2026-09-22T02:07:27Z",
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "success",
        "terminal_label": "CLOSED_SET_DIAGNOSTIC_COMPLETE",
        "experiment_type": "benchmark",
        "claim_scope": "diagnostic",
        "objective": "Train and test the independent OURS-E3-T8 fusion method on frozen Stage20 VPN-6 and TOR-7 coarse-Service closed-set tasks.",
        "method_identity": {
            "method": "OURS-E3-T8",
            "branches": ["recovered TrafficFormer first-5 packets", "project FIG/TAGCN first-8 packets"],
            "fusion": "Known-Train z-score per branch, 768+128 concatenation, linear classifier",
            "excluded": ["OpenDetectNet", "Open-Detect prototype loss", "Open-Detect classifier"],
        },
        "inputs": [
            {"path": str(CLOSED_MANIFEST), "sha256": sha256_file(CLOSED_MANIFEST), "role": "frozen Stage20 membership"},
            {"path": str(cache_path("iscx_vpn")), "sha256": sha256_file(cache_path("iscx_vpn")), "role": "VPN E3-T8 inputs"},
            {"path": str(cache_path("iscx_tor")), "sha256": sha256_file(cache_path("iscx_tor")), "role": "TOR E3-T8 inputs"},
        ],
        "code": {
            "configuration_sha256": sha256_file(CONFIG),
            "scripts": [item["path"] for item in artifacts if item["path"].startswith("scripts/") and item["path"].endswith(".py")],
        },
        "execution": {
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "formal_tmux_session": "stage21-ours-e3-t8-formal-20260922",
            "actual_training_physical_gpu_ids": [1, 2],
            "parallelism": "two independent single-GPU processes; no DataParallel",
            "formal_runs": 4,
            "successful_runs": 4,
            "seeds": [2022, 2023],
            "test_selection_samples": 0,
            "interrupted_attempt": {
                "dataset": "iscx_vpn", "seed": 2024, "physical_gpu_id": 4,
                "exit_code": -15, "reason": "user-requested reduction to two GPUs and two seeds",
                "excluded_from_aggregates": True,
            },
        },
        "configuration": {
            "file": "config.json", "sha256": sha256_file(CONFIG),
            "datasets": {"iscx_vpn": {"flows": 10955, "classes": 6}, "iscx_tor": {"flows": 11181, "classes": 7}},
        },
        "core_results": [
            {"dataset": "iscx_vpn", "method": "OURS-E3-T8", "seeds": 2, "accuracy_mean": 0.520128087831656, "accuracy_std": 0.03888380603842634, "macro_f1_mean": 0.5249345042909267, "macro_f1_std": 0.04791679462009599, "weighted_f1_mean": 0.512889026336405, "weighted_f1_std": 0.04507485308921194},
            {"dataset": "iscx_tor", "method": "OURS-E3-T8", "seeds": 2, "accuracy_mean": 0.6222023276633841, "accuracy_std": 0.0134288272157565, "macro_f1_mean": 0.5911243794949889, "macro_f1_std": 0.014552519444146261, "weighted_f1_mean": 0.6176433442573146, "weighted_f1_std": 0.012695234580820947},
        ],
        "verification": {"path": "completion_verification.json", "status": "PASS", "checks": 98, "failures": 0},
        "artifacts": artifacts,
        "limitations": [
            "Only two seeds; standard deviations are descriptive, not confidence intervals.",
            "Closed-set only; no open-set detection claim.",
            "Stage20 weak capture labels and non-uniform capture-disjointness remain.",
            "FIG is restricted to eight packets, unlike Stage17's 30-packet pilot.",
            "Strict random-init TrafficFormer with three epochs is severely underfit.",
            "Historical baselines use different populations or protocols and are not paired comparisons.",
        ],
        "next_step": "Run a separate Known-Validation-only optimization and T8/T16/T30 ablation without modifying this baseline bundle.",
    }
    write_json(OUT / "manifest.json", manifest)
    print({"status": "SUCCESS", "artifacts": len(artifacts)})


if __name__ == "__main__":
    main()
