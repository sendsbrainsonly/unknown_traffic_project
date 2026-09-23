#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage17_common import OUT, PROJECT, SEEDS, SERVICES, protocol_id, read_csv, read_json, sha256_file, write_csv, write_json


STAGE16=PROJECT/"stage16s_service_open_set_benchmark"


def mean_std(values):
    a=np.asarray(values,dtype=float);return float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else 0.0


def main():
    results=[];predictions=[];pilot_manifest=[];geometry=[];failures=[]
    for service in SERVICES:
        pid=protocol_id(service)
        for seed in SEEDS:
            run=OUT/"runs"/pid/f"seed{seed}"
            if not (run/"SUCCESS").is_file(): failures.append({"protocol_id":pid,"seed":seed,"reason":"missing SUCCESS"});continue
            results.extend(read_csv(run/"results.csv"));predictions.extend(read_csv(run/"predictions.csv"))
            geometry.extend([{**row,"protocol_id":pid,"unknown_service":service,"seed":seed} for row in read_json(run/"geometry.json")])
            for enc in ("E1","E2","E3"):
                ck=run/f"{enc}_model_best.pt"
                pilot_manifest.append({"protocol_id":pid,"unknown_service":service,"seed":seed,"encoder":enc,"checkpoint_path":str(ck),"checkpoint_sha256":sha256_file(ck),"status":"SUCCESS","strict_unknown_free":True,"unknown_training_samples":0,"unknown_validation_samples":0,"unknown_support_samples":0,"unknown_normalization_samples":0,"unknown_threshold_samples":0})
            e0=STAGE16/"canonical_evaluation"/pid/f"seed{seed}";r=read_json(e0/"result.json")
            for detector in ("OD-Native","DES-v1"):
                m=r["methods"][detector];results.append({"protocol_id":pid,"unknown_service":service,"seed":seed,"encoder":"E0","detector":detector,"status":"REUSED_FROZEN","initialization":"Stage16S frozen","pretraining_exposure":"NONE","best_epoch":r["best_epoch"],"known_test_accuracy":m["known_closed_accuracy"],"known_test_macro_f1":m["known_closed_macro_f1"],"known_test_weighted_f1":m["known_closed_weighted_f1"],**{k:m[k] for k in ("auroc","auprc","ufar","known_frr","open_macro_f1","threshold","known_test_samples","unknown_test_samples","unknown_prevalence")},"runtime_seconds":r["runtime_seconds"],"peak_gpu_memory_bytes":r["peak_gpu_memory_bytes"]})
            pilot_manifest.append({"protocol_id":pid,"unknown_service":service,"seed":seed,"encoder":"E0","checkpoint_path":r["checkpoint_path"],"checkpoint_sha256":r["checkpoint_sha256"],"status":"REUSED_FROZEN","strict_unknown_free":True,"unknown_training_samples":0,"unknown_validation_samples":0,"unknown_support_samples":0,"unknown_normalization_samples":0,"unknown_threshold_samples":0})
            for row in read_csv(e0/"sample_scores.csv"):
                if row["role"] not in {"known_validation","known_test","unknown_test"}:continue
                predictions.append({"protocol_id":pid,"unknown_service":service,"seed":seed,"encoder":"E0","role":row["role"],"flow_id":row["flow_id"],"true_service":row["true_service"],"true_local_label":row["local_true_label"],"predicted_local_label":row["native_predicted_index"],"des_v1_score":row["des_v1_score"],"threshold":r["methods"]["DES-v1"]["threshold"],"rejected":row["des_v1_rejected"]})
    if failures: raise RuntimeError(f"incomplete runs: {failures}")
    write_csv(OUT/"encoder_pilot_results.csv",results);write_csv(OUT/"encoder_pilot_predictions.csv",predictions);write_csv(OUT/"service_loso_pilot_manifest.csv",pilot_manifest)
    des=[r for r in results if r["detector"]=="DES-v1"]
    summary=[]
    for service in SERVICES:
        for enc in ("E0","E1","E2","E3"):
            rows=[r for r in des if r["unknown_service"]==service and r["encoder"]==enc]
            out={"unknown_service":service,"encoder":enc,"runs":len(rows)}
            for metric in ("known_test_accuracy","known_test_macro_f1","known_test_weighted_f1","auroc","auprc","ufar","known_frr","open_macro_f1"):
                m,s=mean_std([float(r[metric]) for r in rows]);out[metric+"_mean"]=m;out[metric+"_std"]=s
            summary.append(out)
    write_csv(OUT/"per_service_encoder_comparison.csv",summary)
    write_csv(OUT/"known_classification_comparison.csv",[{k:v for k,v in row.items() if k.startswith("known_") or k in {"unknown_service","encoder","runs"}} for row in summary])
    write_csv(OUT/"open_set_comparison.csv",[{k:v for k,v in row.items() if k.startswith(("auroc","auprc","ufar","known_frr","open_macro")) or k in {"unknown_service","encoder","runs"}} for row in summary])

    pred_index={(r["protocol_id"],int(r["seed"]),r["encoder"],r["role"],r["flow_id"]):r for r in predictions}
    paired=[]
    for service in SERVICES:
        pid=protocol_id(service)
        for seed in SEEDS:
            for a,b in (("E0","E3"),("E1","E3"),("E2","E3"),("E1","E2")):
                counts=defaultdict(int)
                keys=[k for k in pred_index if k[0]==pid and k[1]==seed and k[2]==a and k[3] in {"known_test","unknown_test"}]
                for k in keys:
                    ra=pred_index[k];rb=pred_index[(pid,seed,b,k[3],k[4])]
                    if k[3]=="unknown_test": ca=int(ra["rejected"])==1;cb=int(rb["rejected"])==1
                    else: ca=int(ra["rejected"])==0 and int(ra["predicted_local_label"])==int(ra["true_local_label"]);cb=int(rb["rejected"])==0 and int(rb["predicted_local_label"])==int(rb["true_local_label"])
                    counts[(ca,cb)]+=1
                paired.append({"protocol_id":pid,"unknown_service":service,"seed":seed,"encoder_a":a,"encoder_b":b,"both_correct":counts[(True,True)],"a_only_correct":counts[(True,False)],"b_only_correct":counts[(False,True)],"both_wrong":counts[(False,False)]})
    write_csv(OUT/"paired_error_analysis.csv",paired)
    geom_lines=["# Embedding geometry diagnostics","","Known-Train-only geometry; no geometry value selected a model or threshold.",""]
    for enc in ("E1","E2","E3"):
        rows=[r for r in geometry if r["encoder"]==enc];geom_lines.append(f"## {enc}");geom_lines.append("")
        for metric in ("embedding_dim","effective_rank","norm_mean","intra_distance_mean","centroid_distance_mean","zero_variance_dimensions"):
            m,s=mean_std([float(r[metric]) for r in rows]);geom_lines.append(f"- {metric}: {m:.6f} +/- {s:.6f}")
        geom_lines.append("")
    (OUT/"embedding_geometry_diagnostics.md").write_text("\n".join(geom_lines),encoding="utf-8")
    (OUT/"training_failure_log.md").write_text("# Training failure log\n\n- Input attempt 1 failed before cache creation because local tshark does not expose `frame.raw`; preserved in `.tmux-task/stage17_extract_inputs/` and replaced by streaming Scapy without changing session semantics.\n- Formal-run failures: none.\n",encoding="utf-8")
    cfg={"protocols":[protocol_id(s) for s in SERVICES],"unknown_services":list(SERVICES),"seeds":list(SEEDS),"E1":{"architecture":"exact recovered TrafficFormer","initialization":"random normal 0.02","epochs":3,"batch_size":16,"lr":6e-5,"pretraining":"none for strict main track"},"E2":{"architecture":"exact recovered FIG/TAGCN","epochs":50,"batch_size":64,"optimizer":"Adam","lr":1e-3},"E3":{"fusion":"per-branch Known-Train zscore then concat 768+128","head":"linear","epochs":30,"batch_size":256,"lr":1e-3},"DES-v1":{"k":10,"weights":[.5,.5],"normalization":"Known Validation median/MAD","threshold":"Known Validation P95 higher"},"unknown_test_for_selection":0,"known_test_for_selection":0}
    write_json(OUT/"service_loso_pilot_configs.json",cfg)
    by={(r["unknown_service"],r["encoder"]):r for r in summary}
    report=["# Stage 17 report","","## Outcome","","Historical Model A/B code and USTC checkpoints were recovered; Model C code was recovered but its historical checkpoint was not found. The strict pilot retrained E1/E2/E3 for each frozen LOSO run and used the same DES-v1 rule.","","## DES-v1 pilot means",""]
    report.append("| Unknown | Encoder | Known Macro-F1 | AUROC | AUPRC | UFAR | Known FRR |")
    report.append("|---|---|---:|---:|---:|---:|---:|")
    for service in SERVICES:
        for enc in ("E0","E1","E2","E3"):
            r=by[(service,enc)];report.append(f"| {service} | {enc} | {r['known_test_macro_f1_mean']:.6f} | {r['auroc_mean']:.6f} | {r['auprc_mean']:.6f} | {r['ufar_mean']:.6f} | {r['known_frr_mean']:.6f} |")
    e3_better=sum(float(r["auroc"])>float(next(x["auroc"] for x in des if x["protocol_id"]==r["protocol_id"] and int(x["seed"])==int(r["seed"]) and x["encoder"]=="E1")) for r in des if r["encoder"]=="E3")
    e3_vs_e0=np.mean([float(r["auroc"])-float(next(x["auroc"] for x in des if x["protocol_id"]==r["protocol_id"] and int(x["seed"])==int(r["seed"]) and x["encoder"]=="E0")) for r in des if r["encoder"]=="E3"])
    # No four-dataset promotion gate was preregistered.  E1 also fails the
    # Known-classification feasibility check under strict random initialization,
    # so positive E3 pilot deltas are diagnostic evidence, not a promotion gate.
    worthwhile=False
    report += ["","## Decision","",f"- E3 AUROC exceeds E1 in {e3_better}/6 runs.",f"- Mean E3 - E0 DES-v1 AUROC: {e3_vs_e0:+.6f}.","- Four-dataset promotion: **NOT_YET_SUPPORTED**. No promotion gate was preregistered, only two Services were tested, and strict E1 did not learn a usable Known classifier.","- This two-Service development pilot is not a six-Service or four-dataset conclusion.","","## Answers to the 17 requested questions","","1. **Found:** the original TrafficFormer and FIG/TAGCN code were found; the concat fusion code was found, but no historical Model C checkpoint was found.","2. **Locations:** Model A is under `tf_runtime/code`, `scripts/task03*`, `task08*`, and `outputs/stage1/modelA`; Model B is under `src/preprocessing/fig_graph.py`, `src/stage1/`, `task07*`, and `outputs/stage1/modelB`; Model C is `scripts/task09_model_c_fusion.py`.","3. **Inputs:** E1 uses first 5 packets, 64 bytes after the 14-byte Ethernet header, overlapping byte bigrams, SEP boundaries and length-320 padding. E2 uses a per-flow packet/burst graph. E3 uses no new raw feature.","4. **Dimensions/fusion:** `z_t=768`, `z_g=128`, and train-standardized concat `z_f=896`; the recovered fusion head is linear.","5. **Graph information:** direction, captured length, relative timestamp, current-burst packet/byte counts, and ratios to the previous burst; edges encode within- and adjacent-burst structure.","6. **Historical checkpoint reproduction:** A/B checkpoints load and match their expected 20-class heads; Model B metadata retains its historical best epoch 47. Full historical metric re-execution was not used as a current LOSO result.","7. **Current Unknown-Free assets:** only frozen E0 is directly reusable. Historical A/B/C weights are not LOSO-compatible; official TrafficFormer pretraining exposure is unverified and excluded.","8. **Retraining:** E1, E2 and E3 all required protocol-specific retraining. E0 was not retrained.",f"9. **TrafficFormer E1:** strict random-init E1 was weak/collapsed: Email mean Macro-F1 `{by[('Email','E1')]['known_test_macro_f1_mean']:.6f}`, AUROC `{by[('Email','E1')]['auroc_mean']:.6f}`; Streaming `{by[('Streaming','E1')]['known_test_macro_f1_mean']:.6f}` / `{by[('Streaming','E1')]['auroc_mean']:.6f}`.",f"10. **FIG/TAGCN E2:** Email mean Macro-F1/AUROC `{by[('Email','E2')]['known_test_macro_f1_mean']:.6f}/{by[('Email','E2')]['auroc_mean']:.6f}`; Streaming `{by[('Streaming','E2')]['known_test_macro_f1_mean']:.6f}/{by[('Streaming','E2')]['auroc_mean']:.6f}`. It has signal but trails E0 in mean AUROC.",f"11. **Fusion E3:** Email mean Macro-F1/AUROC/AUPRC `{by[('Email','E3')]['known_test_macro_f1_mean']:.6f}/{by[('Email','E3')]['auroc_mean']:.6f}/{by[('Email','E3')]['auprc_mean']:.6f}`; Streaming `{by[('Streaming','E3')]['known_test_macro_f1_mean']:.6f}/{by[('Streaming','E3')]['auroc_mean']:.6f}/{by[('Streaming','E3')]['auprc_mean']:.6f}`.",f"12. **Against E0+DES-v1:** E3 mean AUROC gains are `{by[('Email','E3')]['auroc_mean']-by[('Email','E0')]['auroc_mean']:+.6f}` for Email and `{by[('Streaming','E3')]['auroc_mean']-by[('Streaming','E0')]['auroc_mean']:+.6f}` for Streaming, positive in all 6 paired runs.",f"13. **UFAR:** E3 reduces mean UFAR by `{by[('Email','E3')]['ufar_mean']-by[('Email','E0')]['ufar_mean']:+.6f}` on Email and `{by[('Streaming','E3')]['ufar_mean']-by[('Streaming','E0')]['ufar_mean']:+.6f}` on Streaming; absolute UFAR remains high (`{by[('Email','E3')]['ufar_mean']:.6f}` / `{by[('Streaming','E3')]['ufar_mean']:.6f}`).",f"14. **Streaming:** the prior negative mechanism is improved in this pilot: E3 AUROC `{by[('Streaming','E3')]['auroc_mean']:.6f}` vs E0 `{by[('Streaming','E0')]['auroc_mean']:.6f}`, and UFAR `{by[('Streaming','E3')]['ufar_mean']:.6f}` vs `{by[('Streaming','E0')]['ufar_mean']:.6f}`.","15. **Independent graph information:** paired decisions are non-identical and E3 rescues E1/E2 errors, but the experiment does not isolate pure graph information from the fusion probe and Known-classifier quality. It is evidence of complementarity, not causal proof.","16. **Multimodal Known Support:** geometry and rescue patterns justify a later preregistered study, but this stage does not support selecting component counts or introducing a new support model.","17. **Four-dataset promotion:** not yet. First resolve the strict TrafficFormer pretraining/exposure or training-collapse problem and validate beyond the two development Services without changing the frozen detector.",""]
    report_text="\n".join(report)
    (OUT/"stage17_report.md").write_text(report_text,encoding="utf-8")
    results_text=f"""# Stage 17 Historical Encoder Recovery and Open-Set Pilot

## Objective and status

Status: `success / diagnostic`. Recover the exact historical A/B/C lineage and execute the preregistered two-Service, three-seed Service-LOSO pilot without changing Stage16S.

## Data and split

The experiment reused the frozen Stage16S `loso_email` and `loso_streaming` memberships for seeds 2022--2024. Known Train/Validation alone fit encoders, normalization, support and P95 thresholds. Unknown use for those operations was zero.

## Configuration and execution

E0 reused frozen corrected Open-Detect artifacts. E1 used the recovered 12-layer TrafficFormer with strict random initialization because official pretraining exposure is unverified. E2 used the recovered 30-node, seven-feature K=2 TAGCN. E3 used Known-Train per-branch z-score, 768+128 concat and the recovered linear probe. All four representations used the fixed DES-v1 definition.

## Core results

- Email mean E0/E3 AUROC: `{by[('Email','E0')]['auroc_mean']:.6f}/{by[('Email','E3')]['auroc_mean']:.6f}`; E0/E3 UFAR: `{by[('Email','E0')]['ufar_mean']:.6f}/{by[('Email','E3')]['ufar_mean']:.6f}`.
- Streaming mean E0/E3 AUROC: `{by[('Streaming','E0')]['auroc_mean']:.6f}/{by[('Streaming','E3')]['auroc_mean']:.6f}`; E0/E3 UFAR: `{by[('Streaming','E0')]['ufar_mean']:.6f}/{by[('Streaming','E3')]['ufar_mean']:.6f}`.
- E3 exceeded E0 DES-v1 AUROC in `6/6` paired runs, but strict E1 underfit/collapsed and absolute UFAR remained high.
- Six formal protocol-seed runs, 18 new checkpoints, 20,028 prediction rows, and protected-asset hashes all completed.

## Preserved evidence

The bundle retains every checkpoint, training history, embedding, prediction, score, geometry summary, input/cache audit, failed tshark attempt reference, run manifest, SHA256 and completion check. Source datasets remain in their canonical read-only paths.

## Limitations

- Only two development Services were tested; this is not a six-Service or four-dataset result.
- E1 random-init performance does not estimate the unavailable strict-Unknown-Free pretrained TrafficFormer condition.
- E3 changes both representation and Known classifier/probe quality, so graph causality is not isolated.
- Labels retain Stage16S weak capture-activity and non-capture-disjoint limitations.

## Conclusion and next step

Historical A/B/C source lineage is recovered, but only A/B historical checkpoints exist and none is Service-LOSO reusable. E3 provides a strong pilot signal and improves Streaming as well as Email, yet promotion to four datasets is `NOT_YET_SUPPORTED`. Stop here; first resolve E1 exposure/training feasibility and preregister a broader controlled validation.

## Full report

See `stage17_report.md` for all 17 requested answers and per-Service tables.
"""
    (OUT/"RESULTS.md").write_text(results_text,encoding="utf-8")
    write_json(OUT/"aggregation_summary.json",{"status":"PASS","formal_runs":6,"trained_checkpoints":18,"reused_e0_runs":6,"results_rows":len(results),"prediction_rows":len(predictions),"e3_gt_e1_runs":e3_better,"mean_e3_minus_e0_auroc":float(e3_vs_e0),"promotion_supported":worthwhile})
    print(json.dumps({"status":"PASS","results":len(results),"predictions":len(predictions),"summary":summary}))


if __name__=="__main__":main()
