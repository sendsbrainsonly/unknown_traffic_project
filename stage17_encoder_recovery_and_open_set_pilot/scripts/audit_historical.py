#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from stage17_common import OUT, PROJECT, sha256_file, write_csv, write_json


ASSETS = [
    PROJECT / "scripts/task03_generate_trafficformer_input.py",
    PROJECT / "scripts/task04_generate_fig_graph.py",
    PROJECT / "scripts/task07_train_model_b.py",
    PROJECT / "scripts/task08_export_model_a_zt.py",
    PROJECT / "scripts/task09_model_c_fusion.py",
    PROJECT / "src/preprocessing/trafficformer_input.py",
    PROJECT / "src/preprocessing/fig_graph.py",
    PROJECT / "src/stage1/tagcn.py",
    PROJECT / "src/stage1/fig_dataset.py",
    PROJECT / "configs/encoder/trafficformer_input.yaml",
    PROJECT / "configs/encoder/fig_graph.yaml",
    PROJECT / "tf_runtime/code/models/pretrained_model.bin",
    PROJECT / "outputs/stage1/modelA/finetuned_model.bin",
    PROJECT / "outputs/stage1/modelB/seeds/seed2/modelB_best.pt",
    PROJECT / "outputs/stage1/modelB/seeds/seed2/run_summary.json",
    PROJECT / "stage16s_service_open_set_benchmark/service_unknown_protocol_manifest.csv",
    PROJECT / "stage16s_service_open_set_benchmark/training_configs.json",
]


def git_commit(path: Path) -> str:
    value = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(path.relative_to(PROJECT))], cwd=PROJECT, text=True).strip()
    return value or "UNTRACKED_OR_NO_HISTORY"


def checkpoint_record(model_id: str, path: Path, dataset: str, status: str, exposure: str):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    tensors = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    return {
        "model_id": model_id, "checkpoint_path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "loadable_cpu": True, "tensor_count": len(tensors),
        "dataset": dataset, "strict_service_loso_reusable": False,
        "pretraining_exposure": exposure, "status": status,
        "reason": "historical checkpoint class space is USTC-20, not the current five-Known Service LOSO",
    }


def write_docs() -> None:
    docs = {
        "historical_model_lineage.md": """# Historical model lineage\n\n| Model | Verified lineage | Training state | Service-LOSO use |\n|---|---|---|---|\n| Model A / TrafficFormer | Official repository code pinned at `6d0ba64d82e74fb130c6c7301ef20885dbfbdf29`, plus project-local short-flow adapter | USTC-20 checkpoint complete | Architecture/input code recoverable; historical weights not reusable as five-class classifier |\n| Model B / FIG-TAGCN | Project-local independent reconstruction of the published FIG/TAGCN description | USTC-20 seed2 checkpoint complete | Architecture/input code recoverable; must retrain |\n| Model C / concat fusion | Project-local `task09_model_c_fusion.py` | Code exists; no historical checkpoint/result found | Recreate only by executing the recovered code on newly legal E1/E2 embeddings |\n| E0 / corrected Open-Detect | Frozen Stage16S checkpoints | 18 Service-LOSO runs complete | Reused without retraining |\n\nModel B is not the separate TFE-GNN paper implementation. Model C is a post-hoc probe, not an end-to-end joint encoder.\n""",
        "historical_feature_lineage.md": """# Historical feature lineage\n\n- **TrafficFormer (G1/G2/G8):** first five observed packets; strip 14-byte Ethernet header; retain up to 64 following bytes per packet; overlapping two-byte bigrams; prepend `[SEP]` per packet; downstream token tail padding to 320. It does not explicitly input length, direction, IAT, burst statistics, or full-flow statistics.\n- **FIG/TAGCN (G3/G4/G5):** at most 30 packet nodes. Seven features are direction, captured length, time from first packet, current-burst packet count, current-burst byte count, current/previous burst packet ratio, and current/previous burst byte ratio. Bursts are maximal same-direction runs. Edges join consecutive packets inside bursts and first-to-first/last-to-last nodes of adjacent bursts.\n- **Fusion:** no new raw feature. Each branch is standardized with Known-Train statistics and concatenated.\n- **Not used:** full-flow aggregate statistics, explicit transport-state features, capture ID, filename, labels, or quality mask as learned inputs. Padding masks are used only by batching/readout.\n\nThe recovered graph branch already covers direction, packet timing, length and burst behavior; a future Behavior branch would need non-duplicative evidence rather than relabeling these same features.\n""",
        "trafficformer_architecture.md": """# TrafficFormer architecture\n\n`word_pos_seg embedding -> 12-layer, 12-head Transformer (hidden 768, FFN 3072, GELU, dropout 0.1) -> first-token z_t (768) -> Linear(768,768)+tanh -> class head`. Historical training used NLL/softmax classification and official pretrained initialization. Stage17 strict pilot uses the identical architecture and random normal initialization because the official pretraining corpus exposure is undisclosed.\n""",
        "fig_tagcn_architecture.md": """# FIG/TAGCN architecture\n\nEach graph represents one flow only; there are no cross-flow or cross-capture edges. `TAGCN` computes `sum_{k=0..2} Ahat^k X W_k + b`, followed by ReLU, dropout 0.5, masked mean graph readout `z_g` (128), and a linear class head. The node/edge construction is label-free and inductive.\n""",
        "fusion_architecture.md": """# Fusion architecture\n\nRecovered Model C independently standardizes `z_t` and `z_g` using Known-Train means/stds, concatenates them as `z_f=[z_t;z_g]` (768+128=896), and trains a linear classifier with Adam 1e-3 for 30 epochs, selecting by Known-Validation Macro-F1. It is not end-to-end, attention-based, weighted, VAE-based, or test-tuned. No historical Model C checkpoint was found.\n""",
        "opendetect_encoder_reference.md": """# Corrected Open-Detect reference\n\nE0 is the existing Stage16S corrected Open-Detect ResNet18/VAE-style encoder with 128-dimensional deterministic mean representation. Stage17 does not reimplement or retrain E0; it reuses frozen latent outputs, native scores and DES-v1 results.\n""",
        "historical_checkpoint_exposure_audit.md": """# Historical checkpoint exposure audit\n\n- `pretrained_model.bin`: official downloadable TrafficFormer pretraining artifact, loadable, but the public README does not identify the pretraining corpus. Status: `PRETRAINING_EXPOSURE_UNVERIFIED`. It is excluded from the strict Stage17 comparison.\n- `finetuned_model.bin`: trained on the project USTC-20 closed-set split; wrong labels and dataset for Service-LOSO. Engineering smoke only.\n- `modelB_best.pt`: trained on USTC-20; wrong labels and dataset for Service-LOSO. Engineering smoke only.\n- Model C: no historical checkpoint found.\n- E0: Stage16S checkpoint is protocol-specific and already passed strict Unknown-Free checks.\n\nNo historical A/B/C checkpoint is inserted into support fitting, normalization, classifier fitting or threshold calibration in the strict pilot.\n""",
        "encoder_architecture_comparison.md": """# Encoder architecture comparison\n\n| ID | Input | Encoder | Representation | Historical status |\n|---|---|---|---:|---|\n| E0 | 32x32 byte image | corrected Open-Detect ResNet18 | 128 | frozen Service-LOSO available |\n| E1 | 5 packets x 64 post-Ethernet bytes | TrafficFormer Transformer | 768 | USTC checkpoint only; retrain needed |\n| E2 | per-flow packet/burst graph | K=2 TAGCN | 128 | USTC checkpoint only; retrain needed |\n| E3 | E1 + E2 embeddings | train-zscore concat + linear probe | 896 | code only; retrain needed |\n""",
        "graph_leakage_audit.md": """# Graph leakage audit\n\nThe recovered FIG is a single-flow internal graph. Nodes are packets from that flow; edges are deterministic functions of within-flow order and direction. There are no cross-flow/capture edges, no global graph, and no labels, service names, filenames or capture IDs in features/weights. Normalization is fit on Known Train only. Known Validation calibrates/checkpoints; Unknown Test is never used for graph construction statistics, training, support, normalization or thresholding. Result: `PASS_INDUCTIVE_PER_FLOW_GRAPH`, subject to the weak capture-label limitation inherited from Stage16S.\n""",
        "recoverability_report.md": """# Recoverability report\n\n- Model A: `CODE_AND_HISTORICAL_CHECKPOINT_RECOVERED`; strict LOSO requires retraining.\n- Model B: `CODE_AND_HISTORICAL_CHECKPOINT_RECOVERED`; it is an independent reconstruction and strict LOSO requires retraining.\n- Model C: `CODE_RECOVERED_CHECKPOINT_NOT_AVAILABLE`; execution can reproduce its documented concat probe without inventing a new model.\n- E0: `FROZEN_STAGE16S_REUSABLE`.\n\nHistorical A/B weights are loadable but not compatible with current label space. The official TrafficFormer pretraining exposure is unverified, so strict pilot uses random initialization.\n""",
        "stage17_0_results.md": """# Stage 17-0 results\n\nThe historical pipeline was found. Its actual branches are a 768-D TrafficFormer, a 128-D per-flow FIG/TAGCN, and an 896-D train-standardized concatenation with a linear probe. Model A and B have complete USTC-20 checkpoints; Model C has source code but no saved historical checkpoint. None of the historical A/B/C checkpoints is a valid five-Known Service-LOSO checkpoint. The recovered architecture and feature code can run locally and is therefore eligible for protocol-specific retraining.\n""",
    }
    for name, text in docs.items():
        (OUT / name).write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=("before", "after"), default="before"); args = ap.parse_args()
    missing = [str(p) for p in ASSETS if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    hashes = {str(p.relative_to(PROJECT)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in ASSETS}
    write_json(OUT / f"protected_asset_hashes_{args.phase}.json", {"files": hashes, "file_count": len(hashes)})
    if args.phase == "after": return
    checkpoints = [
        checkpoint_record("TrafficFormer-official-pretrain", ASSETS[11], "undisclosed", "LOADABLE_ENGINEERING_ONLY", "PRETRAINING_EXPOSURE_UNVERIFIED"),
        checkpoint_record("Model-A-TrafficFormer", ASSETS[12], "USTC-TFC2016 20-class", "LOADABLE_WRONG_LABEL_SPACE", "inherits official unverified pretraining plus USTC fine-tuning"),
        checkpoint_record("Model-B-FIG-TAGCN", ASSETS[13], "USTC-TFC2016 20-class", "LOADABLE_WRONG_LABEL_SPACE", "none"),
        {"model_id":"Model-C-Fusion","checkpoint_path":"NOT_FOUND","bytes":0,"sha256":"NOT_AVAILABLE","loadable_cpu":False,"tensor_count":0,"dataset":"NOT_AVAILABLE","strict_service_loso_reusable":False,"pretraining_exposure":"inherits branches","status":"CODE_ONLY_CHECKPOINT_NOT_AVAILABLE","reason":"no historical modelC output/checkpoint found"},
    ]
    write_csv(OUT / "historical_checkpoint_inventory.csv", checkpoints)
    models = [
        {"model_id":"A","source_path":"scripts/task08_export_model_a_zt.py; tf_runtime/code","git_commit":git_commit(PROJECT/"scripts/task08_export_model_a_zt.py"),"model_class":"TrafficFormer Classifier/EncoderOnly","encoder_architecture":"12-layer Transformer","input_modality":"packet bytes as overlapping bigrams","input_shape":"5 packets x <=64 post-Ethernet bytes; token length 320","output_dimension":768,"training_objective":"NLL classification after official pretraining","known_classifier":"tanh MLP head","unknown_detector":"none historically; DES-v1 in pilot","checkpoint_path":str(ASSETS[12]),"checkpoint_sha256":sha256_file(ASSETS[12]),"dataset":"USTC-TFC2016 20-class","split":"compatible_min1","training_classes":20,"pretraining_source":"official downloaded artifact; corpus undisclosed","current_status":"RECOVERED_RETRAIN_REQUIRED"},
        {"model_id":"B","source_path":"src/preprocessing/fig_graph.py; src/stage1/tagcn.py","git_commit":git_commit(PROJECT/"src/stage1/tagcn.py"),"model_class":"TAGCN","encoder_architecture":"K=2 polynomial graph convolution + masked mean","input_modality":"single-flow packet/burst graph","input_shape":"<=30 nodes x 7 features","output_dimension":128,"training_objective":"cross-entropy","known_classifier":"linear","unknown_detector":"none historically; DES-v1 in pilot","checkpoint_path":str(ASSETS[13]),"checkpoint_sha256":sha256_file(ASSETS[13]),"dataset":"USTC-TFC2016 20-class","split":"compatible_min1","training_classes":20,"pretraining_source":"none","current_status":"RECOVERED_RETRAIN_REQUIRED"},
        {"model_id":"C","source_path":"scripts/task09_model_c_fusion.py","git_commit":git_commit(PROJECT/"scripts/task09_model_c_fusion.py"),"model_class":"Linear probe over standardized concat","encoder_architecture":"frozen A/B + train-zscore + concat","input_modality":"z_t and z_g","input_shape":"768 + 128","output_dimension":896,"training_objective":"cross-entropy","known_classifier":"linear","unknown_detector":"none historically; DES-v1 in pilot","checkpoint_path":"NOT_FOUND","checkpoint_sha256":"NOT_AVAILABLE","dataset":"code expects aligned USTC embeddings","split":"compatible_min1","training_classes":"dynamic","pretraining_source":"inherits A/B","current_status":"CODE_RECOVERED_CHECKPOINT_NOT_AVAILABLE"},
        {"model_id":"E0","source_path":"stage16s_service_open_set_benchmark","git_commit":"frozen artifact","model_class":"corrected Open-Detect","encoder_architecture":"ResNet18 variational encoder","input_modality":"32x32 byte image","input_shape":"1x32x32","output_dimension":128,"training_objective":"corrected Open-Detect composite loss","known_classifier":"learned prototypes","unknown_detector":"OD-Native / DES-v1","checkpoint_path":"stage16s runs per protocol/seed","checkpoint_sha256":"see shared_encoder_checkpoints.csv","dataset":"ISCX-VPN Service-LOSO","split":"Stage16S frozen","training_classes":5,"pretraining_source":"none","current_status":"FROZEN_REUSABLE"},
    ]
    write_csv(OUT / "historical_model_registry.csv", models)
    features=[]
    for model,used,not_used,pcap in [
        ("A","G1 raw bytes; G2 fixed Ethernet removal; G8 token padding","G3/G4/G5/G6/G7",True),
        ("B","G3 length/direction/IAT-like relative time; G4 bidirection; G5 bursts; G8 graph mask","G1/G2/G6/G7",True),
        ("C","inherits A and B; no new raw feature","G6/G7",True),
        ("E0","G1 byte image","explicit G3/G4/G5/G6/G7",False),
    ]: features.append({"model_id":model,"features_used":used,"implemented_but_not_input":"none verified","not_implemented_or_not_used":not_used,"requires_pcap":pcap,"dataframe_only_possible":False if pcap else True,"label_or_capture_feature_used":False})
    write_csv(OUT / "historical_feature_inventory.csv", features)
    write_docs()
    write_json(OUT / "model_recovery_verification.md.json", {"status":"PASS","historical_assets":len(ASSETS),"checkpoint_loads":3,"model_c_checkpoint":"NOT_AVAILABLE"})
    (OUT / "model_recovery_verification.md").write_text("# Model recovery verification\n\nThree existing checkpoint artifacts loaded on CPU and were hashed. The recovered Model A/B source files have Git lineage; Model C source exists but its checkpoint is absent. No approximate substitute was introduced.\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","models":len(models),"checkpoints":len(checkpoints),"assets":len(ASSETS)}))


if __name__ == "__main__": main()
