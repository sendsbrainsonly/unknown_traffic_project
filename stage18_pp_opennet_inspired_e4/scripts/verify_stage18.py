#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stage18_common import OUT, SEEDS, SERVICES, VARIANT_ORDER, protocol_id, read_json, sha256_file, write_json


def main() -> None:
    required = [
        "RESULTS.md", "stage18_report.md", "e4_closed_set_results.csv", "e4_open_set_results.csv",
        "e4_per_class_results.csv", "e4_predictions.csv", "e4_training_dynamics.csv",
        "closed_set_summary.csv", "open_set_summary.csv", "paired_comparison.csv",
        "component_ablation.csv", "component_ablation_summary.csv", "confusion_matrices.csv",
        "e4_e3_error_complementarity.csv", "gate_decision.json", "replay_verification.json",
        "protected_asset_hashes_before.json", "protected_asset_hashes_after.json", "smoke_verification.json",
    ]
    missing=[name for name in required if not (OUT/name).is_file()]
    if missing: raise FileNotFoundError(missing)
    closed=pd.read_csv(OUT/"e4_closed_set_results.csv");opened=pd.read_csv(OUT/"e4_open_set_results.csv");per_class=pd.read_csv(OUT/"e4_per_class_results.csv");pred=pd.read_csv(OUT/"e4_predictions.csv");dynamics=pd.read_csv(OUT/"e4_training_dynamics.csv");ablation=pd.read_csv(OUT/"component_ablation.csv")
    if len(closed)!=42 or len(opened)!=378 or len(per_class)!=210 or len(pred)!=35049 or len(dynamics)!=4200:
        raise RuntimeError({"closed":len(closed),"opened":len(opened),"per_class":len(per_class),"predictions":len(pred),"dynamics":len(dynamics)})
    if len(ablation) != 36:
        raise RuntimeError({"ablation_rows": len(ablation)})
    expected_ablation_pairs = {
        "time_information_vs_length_only": ("E4_TIME_LENGTH", "E4_LENGTH"),
        "length_information_vs_time_only": ("E4_TIME_LENGTH", "E4_TIME"),
        "direction_information_vs_time_length": ("E4_FULL", "E4_TIME_LENGTH"),
    }
    for comparison, expected_pair in expected_ablation_pairs.items():
        observed = set(
            map(
                tuple,
                ablation.loc[
                    ablation.comparison == comparison,
                    ["full_variant", "reference_variant"],
                ].drop_duplicates().to_numpy(),
            )
        )
        if observed != {expected_pair}:
            raise RuntimeError({"comparison": comparison, "observed_pairs": sorted(observed)})
    manifests=[];checkpoint_count=0
    for service in SERVICES:
        for seed in SEEDS:
            run=OUT/"runs"/protocol_id(service)/f"seed{seed}";manifest=read_json(run/"run_manifest.json");manifests.append(manifest)
            if manifest["status"]!="SUCCESS" or not manifest["strict_unknown_free"]: raise RuntimeError(run)
            for key in ("unknown_training_samples","unknown_validation_samples","unknown_support_samples","unknown_normalization_samples","unknown_threshold_samples","known_test_selection_samples"):
                if manifest[key]!=0: raise RuntimeError(f"{run}: {key}")
            for variant in VARIANT_ORDER:
                checkpoint=run/f"{variant}_best.pt";checkpoint_count+=1
                row=closed[(closed.protocol_id==protocol_id(service))&(closed.seed==seed)&(closed.variant==variant)].iloc[0]
                if sha256_file(checkpoint)!=row.checkpoint_sha256: raise RuntimeError(f"checkpoint hash mismatch {checkpoint}")
    before=read_json(OUT/"protected_asset_hashes_before.json");after=read_json(OUT/"protected_asset_hashes_after.json")
    if before["hashes"]!=after["hashes"]: raise RuntimeError("protected Stage16/17 assets changed")
    replay=read_json(OUT/"replay_verification.json")
    if replay["status"]!="PASS" or replay["models"]!=42: raise RuntimeError("replay failed")
    result={"status":"PASS","required_outputs":len(required),"formal_runs":len(manifests),"variants_per_run":7,"trained_checkpoints":checkpoint_count,"closed_rows":len(closed),"open_rows":len(opened),"per_class_rows":len(per_class),"prediction_rows":len(pred),"training_dynamic_rows":len(dynamics),"ablation_rows":len(ablation),"strict_single_feature_ablation":True,"strict_unknown_free":True,"unknown_usage":{"training":0,"validation":0,"support":0,"normalization":0,"threshold":0},"known_test_selection_samples":0,"protected_assets_unchanged":True,"replay_models":replay["models"],"replay_max_abs_difference":replay["max_abs_tensor_difference"],"final_gate":read_json(OUT/"gate_decision.json")["final_gate"],"fusion_started":False}
    write_json(OUT/"completion_verification.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
