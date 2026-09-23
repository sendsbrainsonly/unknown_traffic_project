#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from stage22_common import CONFIG, OUT, PRETRAINED_MODEL, cache_path, dataset_rows, protected_assets, read_json, sha256_file, write_json


def main() -> None:
    config = read_json(CONFIG)
    checks: list[dict] = []
    hashes = {name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in protected_assets().items()}
    expected_hash = config["pretrained_model_sha256"]
    checks.append({"name": "official_pretrained_sha256", "passed": hashes["official_pretrained_model"]["sha256"] == expected_hash, "detail": hashes["official_pretrained_model"]["sha256"]})
    checks.append({"name": "pretrained_nonempty", "passed": PRETRAINED_MODEL.stat().st_size > 0, "detail": PRETRAINED_MODEL.stat().st_size})

    for dataset, spec in config["datasets"].items():
        rows = dataset_rows(dataset)
        counts = Counter(row["closed_role"] for row in rows)
        cache = np.load(cache_path(dataset), allow_pickle=False)
        cache_ids = [str(value) for value in cache["flow_ids"]]
        manifest_ids = [row["flow_id"] for row in rows]
        checks.extend([
            {"name": f"{dataset}_total", "passed": len(rows) == spec["expected_flows"], "detail": len(rows)},
            {"name": f"{dataset}_train", "passed": counts["known_train"] == spec["expected_train"], "detail": counts["known_train"]},
            {"name": f"{dataset}_validation", "passed": counts["known_validation"] == spec["expected_validation"], "detail": counts["known_validation"]},
            {"name": f"{dataset}_test", "passed": counts["known_test"] == spec["expected_test"], "detail": counts["known_test"]},
            {"name": f"{dataset}_cache_membership", "passed": set(cache_ids) == set(manifest_ids) and len(cache_ids) == len(manifest_ids), "detail": {"cache": len(cache_ids), "manifest": len(manifest_ids), "missing": len(set(manifest_ids) - set(cache_ids)), "extra": len(set(cache_ids) - set(manifest_ids))}},
            {"name": f"{dataset}_finite_graph", "passed": bool(np.isfinite(cache["fig_x"]).all()), "detail": list(cache["fig_x"].shape)},
            {"name": f"{dataset}_token_shape", "passed": cache["token_ids"].shape == (len(rows), 320), "detail": list(cache["token_ids"].shape)},
        ])

    passed = all(item["passed"] for item in checks)
    write_json(OUT / "protected_asset_hashes_before.json", hashes)
    write_json(OUT / "preflight.json", {"status": "PASS" if passed else "FAIL", "checks": checks})
    print(json.dumps({"status": "PASS" if passed else "FAIL", "checks": len(checks)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
