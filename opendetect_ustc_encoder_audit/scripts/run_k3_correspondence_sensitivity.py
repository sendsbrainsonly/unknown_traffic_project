#!/usr/bin/env python3
"""Appendix-only K=3 cross-encoder correspondence sensitivity."""

from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_component_correspondence_audit import (
    AUDIT_ROOT,
    CLASSES,
    assert_flow_identity,
    build_od_assignments,
    build_tf_assignments,
    correspondence,
    prepare_output,
    train_val_stability,
)


K = 3
FINAL_OUTPUT = AUDIT_ROOT / "outputs/component_correspondence_audit/k3_sensitivity"


def upsert_section(path: Path, heading: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = rf"{re.escape(heading)}\n.*?(?=\n## |\Z)"
    replacement = f"{heading}\n\n{body.strip()}\n"
    if re.search(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + replacement
    path.write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    stage = prepare_output(FINAL_OUTPUT)
    try:
        print("[1/5] Replay TrafficFormer formal K=3 and verify Stage 2.5 NLL")
        tf, tf_meta, tf_weights = build_tf_assignments(K)
        print("[2/5] Reuse Open-Detect frozen K=3 artifacts and verify provenance")
        od, od_meta, od_weights = build_od_assignments(K)
        assertions = assert_flow_identity(tf, od)
        cross, confusion = correspondence(tf, od, tf_weights, od_weights)
        stability = train_val_stability(
            {"TrafficFormer": tf, "Open-Detect": od},
            {"TrafficFormer": tf_weights, "Open-Detect": od_weights},
        )
        print("[3/5] Write appendix assignments, correspondence and stability")
        tf.to_parquet(stage / "tf_component_assignments_k3.parquet", index=False)
        od.to_parquet(stage / "od_component_assignments_k3.parquet", index=False)
        cross.to_csv(stage / "cross_encoder_correspondence_k3.csv", index=False)
        confusion.to_csv(stage / "cross_encoder_confusion_k3.csv", index=False)
        stability.to_csv(stage / "train_val_stability_k3.csv", index=False)
        pd.DataFrame(assertions).to_csv(stage / "flow_id_assertions_k3.csv", index=False)
        figures = stage / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        for class_name in CLASSES:
            selected = confusion[
                (confusion.class_name == class_name) & (confusion.split == "val")
            ]
            matrix = np.zeros((K, K), dtype=int)
            for row in selected.itertuples(index=False):
                matrix[row.tf_component_id, row.od_component_id] = row.count
            fig, ax = plt.subplots(figsize=(5.3, 4.5))
            image = ax.imshow(matrix, cmap="Blues")
            for left in range(K):
                for right in range(K):
                    ax.text(right, left, str(matrix[left, right]), ha="center", va="center")
            ax.set_xticks(range(K), [f"OD C{x}" for x in range(K)])
            ax.set_yticks(range(K), [f"TF C{x}" for x in range(K)])
            ax.set_title(f"{class_name}: validation K=3 sensitivity")
            fig.colorbar(image, ax=ax, label="flow count")
            fig.tight_layout()
            fig.savefig(figures / f"{class_name}_validation_k3_contingency.png", dpi=180)
            plt.close(fig)
        validation = cross[cross.split == "val"].copy()
        lines = [
            "# K=3 Cross-Encoder Correspondence Sensitivity",
            "",
            "This is an appendix-only robustness check. K=2 remains the fixed main analysis and the main Gate is not reselected.",
            "",
            "| class | NMI | AMI | ARI | Hungarian accuracy | label |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in validation.itertuples(index=False):
            lines.append(
                f"| {row.class_name} | {row.nmi:.6f} | {row.ami:.6f} | {row.ari:.6f} | {row.hungarian_accuracy:.6f} | {row.correspondence_label} |"
            )
        lines.extend(
            [
                "",
                "K=3 does not establish semantic modes and is not used to alter the K=2 SIMPLE-STATISTICS PERSISTENCE Gate.",
            ]
        )
        summary = "\n".join(lines) + "\n"
        (stage / "k3_sensitivity_summary.md").write_text(summary, encoding="utf-8")
        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "role": "appendix-only K=3 correspondence sensitivity",
            "main_k2_gate_unchanged": "C. SIMPLE-STATISTICS PERSISTENCE",
            "K": K,
            "classes": list(CLASSES),
            "primary_split": "validation",
            "trafficformer": tf_meta,
            "opendetect": od_meta,
            "flow_id_assertions": assertions,
            "test_loaded_or_used": False,
            "encoder_retrained": False,
            "k_selected_or_searched": False,
            "unknown_detection_run": False,
            "elapsed_seconds": time.time() - started,
            "actual_command": f"python {Path(__file__).resolve()}",
        }
        metadata["output_files"] = sorted(
            {
                *(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file()),
                "run_metadata.json",
            }
        )
        (stage / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if FINAL_OUTPUT.exists():
            FINAL_OUTPUT.rmdir()
        stage.rename(FINAL_OUTPUT)
        print("[4/5] Publish appendix and update summaries without changing K=2 Gate")
        appendix_body = "\n".join(lines[2:])
        upsert_section(
            AUDIT_ROOT / "outputs/component_correspondence_audit/audit_summary.md",
            "## K=3 Correspondence Sensitivity",
            appendix_body,
        )
        upsert_section(
            AUDIT_ROOT / "README.md",
            "## K=3 Correspondence Sensitivity",
            appendix_body,
        )
        print("[5/5] Complete")
        print(json.dumps({"status": "PASS", "rows": len(cross), "output": str(FINAL_OUTPUT)}))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
