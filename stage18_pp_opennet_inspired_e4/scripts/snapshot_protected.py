#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stage18_common import OUT, PROJECT, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [
        PROJECT / "stage17_encoder_recovery_and_open_set_pilot/feature_cache/service3065_historical_inputs.npz",
        PROJECT / "stage16s_service_open_set_benchmark/service_unknown_protocol_manifest.csv",
        PROJECT / "stage17_encoder_recovery_and_open_set_pilot/encoder_pilot_results.csv",
        PROJECT / "stage17_encoder_recovery_and_open_set_pilot/encoder_pilot_predictions.csv",
        PROJECT / "stage17_encoder_recovery_and_open_set_pilot/service_loso_pilot_configs.json",
        PROJECT / "stage17_encoder_recovery_and_open_set_pilot/completion_verification.json",
        PROJECT / "stage16s_service_open_set_benchmark/service_open_set_six_metrics.csv",
        PROJECT / "stage16s_service_open_set_benchmark/manifest.json",
    ]
    paths += sorted((PROJECT / "stage17_encoder_recovery_and_open_set_pilot/runs").glob("loso_*/seed*/E3_model_best.pt"))
    paths += sorted((PROJECT / "stage16s_service_open_set_benchmark/runs").glob("loso_email/seed*/checkpoint.sha256"))
    paths += sorted((PROJECT / "stage16s_service_open_set_benchmark/runs").glob("loso_streaming/seed*/checkpoint.sha256"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "status": "PASS",
        "files": len(paths),
        "hashes": {str(path.relative_to(PROJECT)): sha256_file(path) for path in paths},
    }
    target = OUT / args.output
    write_json(target, payload)
    print(target, payload["files"])


if __name__ == "__main__":
    main()
