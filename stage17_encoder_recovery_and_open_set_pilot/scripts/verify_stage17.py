#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage17_common import CACHE, OUT, read_csv, read_json, sha256_file, write_json


REQUIRED=(
"historical_model_registry.csv","historical_feature_inventory.csv","historical_checkpoint_inventory.csv","historical_model_lineage.md","historical_feature_lineage.md","historical_checkpoint_exposure_audit.md","encoder_architecture_comparison.md","model_recovery_verification.md","service_loso_pilot_manifest.csv","service_loso_pilot_configs.json","encoder_pilot_results.csv","encoder_pilot_predictions.csv","per_service_encoder_comparison.csv","known_classification_comparison.csv","open_set_comparison.csv","paired_error_analysis.csv","embedding_geometry_diagnostics.md","graph_leakage_audit.md","training_failure_log.md","stage17_report.md","RESULTS.md","manifest.json")


def require(value,message):
    if not value: raise AssertionError(message)


def main():
    missing=[name for name in REQUIRED if not (OUT/name).is_file()];require(not missing,f"missing outputs: {missing}")
    cache=np.load(CACHE,allow_pickle=False);require(cache["flow_ids"].shape==(3065,),"cache flow count");require(cache["token_ids"].shape==(3065,320),"token shape");require(cache["fig_x"].shape==(3065,30,7),"FIG shape");require(np.isfinite(cache["fig_x"]).all(),"nonfinite FIG")
    manifest=read_csv(OUT/"service_loso_pilot_manifest.csv");require(len(manifest)==24,"expected 24 encoder run manifest rows")
    require(all(row["strict_unknown_free"]=="True" for row in manifest),"strict flag")
    for row in manifest:
        require(int(row["unknown_training_samples"])==0 and int(row["unknown_validation_samples"])==0 and int(row["unknown_support_samples"])==0 and int(row["unknown_normalization_samples"])==0 and int(row["unknown_threshold_samples"])==0,"unknown leakage")
        path=Path(row["checkpoint_path"]);require(path.is_file(),f"checkpoint missing {path}");require(sha256_file(path)==row["checkpoint_sha256"],f"checkpoint hash {path}")
    results=read_csv(OUT/"encoder_pilot_results.csv");require(len(results)==30,"expected 30 track A/B results")
    require(len([r for r in results if r["detector"]=="DES-v1"])==24,"expected 24 common DES-v1 results")
    require(all(np.isfinite(float(r[k])) for r in results for k in ("auroc","auprc","ufar","known_frr","known_test_macro_f1")),"nonfinite metric")
    before=read_json(OUT/"protected_asset_hashes_before.json");after=read_json(OUT/"protected_asset_hashes_after.json");require(before==after,"protected assets changed")
    report=(OUT/"stage17_report.md").read_text(encoding="utf-8");require("Email" in report and "Streaming" in report,"report protocols")
    verification={"status":"PASS","required_outputs":len(REQUIRED),"cache_flows":3065,"formal_protocol_seed_runs":6,"encoder_run_manifest_rows":24,"trained_checkpoints":18,"reused_e0_checkpoints":6,"common_des_v1_result_rows":24,"track_a_b_result_rows":30,"strict_unknown_free":True,"unknown_usage":{"training":0,"validation":0,"support":0,"normalization":0,"threshold":0},"protected_assets_unchanged":True,"cache_sha256":sha256_file(CACHE)}
    write_json(OUT/"completion_verification.json",verification);print(json.dumps(verification))


if __name__=="__main__":main()
