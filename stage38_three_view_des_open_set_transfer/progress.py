#!/usr/bin/env python3
"""Print one read-only Stage38 training and detection progress snapshot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
JOBS = (
    ("VPN graph", "stage38b_vpn_graph_gpu2_v2_0926", 50),
    ("VPN TrafficFormer", "stage38b_vpn_tf_gpu3_v2_0926", 20),
    ("VPN YaTC", "stage38b_vpn_yatc_gpu4_v2_0926", 200),
    ("VPN fusion", "stage38b_vpn_fusion_auto_0926", 60),
    ("Tor graph", "stage38b_tor_graph_gpu2_0926", 50),
    ("Tor TrafficFormer", "stage38b_tor_tf_auto_0926", 20),
    ("Tor YaTC", "stage38b_tor_yatc_gpu2_0926", 200),
    ("Tor fusion", "stage38b_tor_fusion_auto_0926", 60),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def epoch(log: Path) -> tuple[str, str]:
    if not log.is_file():
        return "-", "-"
    latest = None
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in row and any(key in row for key in ("val_macro_f1", "val_mean_macro_f1")):
            latest = row
    if latest is None:
        return "-", "-"
    phase = latest.get("phase")
    value = latest.get("val_macro_f1", latest.get("val_mean_macro_f1"))
    return (f"{phase}:{latest['epoch']}" if phase else str(latest["epoch"]),
            f"{value:.6f}" if isinstance(value, (int, float)) else "-")


def rows(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["method"]: r for r in csv.DictReader(handle)}


def main() -> None:
    vnat = read_json(ROOT / "vnat/stage38a_verification.json")
    print(f"VNAT frozen evaluation: {vnat.get('status', 'PENDING')}")
    for name in ("training_queue", "iscx_vpn_detection_queue", "iscx_tor_detection_queue",
                 "finalizer_queue"):
        item = read_json(ROOT / f"{name}_progress.json")
        print(f"{name}: {item.get('status', 'PENDING')} / {item.get('phase', '-')}"
              f" / updated {item.get('updated_at_utc', '-')}")
    print("branch                 state      epoch/budget   val Macro-F1")
    for label, session, budget in JOBS:
        folder = PROJECT / ".tmux-task" / session
        exit_file = folder / "exit.status"
        state = (f"exit={exit_file.read_text().strip()}" if exit_file.is_file()
                 else "running" if folder.is_dir() else "pending")
        current, score = epoch(folder / "output.log")
        print(f"{label:<22} {state:<10} {current + '/' + str(budget):<14} {score}")
    for dataset, label in (("iscx_vpn", "VPN"), ("iscx_tor", "Tor")):
        result = ROOT / "runs" / dataset / "medium_seed2022/detection"
        closed = read_json(result / "closed_set_results.json")
        if closed:
            print(f"{label} Known Test Macro-F1: {closed['metrics']['macro_f1']:.6f}")
        observed = rows(result / "open_set_results.csv")
        if observed:
            print(f"{label} AUROC / UFAR: " + ", ".join(
                f"{method} {float(observed[method]['auroc']):.4f}/{float(observed[method]['ufar']):.4f}"
                for method in ("msp", "energy", "centroid", "des_v1")))


if __name__ == "__main__":
    main()
