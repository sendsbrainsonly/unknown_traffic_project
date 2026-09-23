#!/usr/bin/env python3
"""Build the Stage 10A static paper-code-reproduction audit outputs.

This script only reads already-frozen sources and results.  It does not import
models, open PCAPs, run inference, fit a transform, calibrate a threshold, or
change any Stage 6--9 artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "stage10a_opendetect_protocol_audit"
OUT = AUDIT / "outputs"
OPENDETECT_PROJECT = PROJECT.parent / "Open-Detect"
OFFICIAL = OPENDETECT_PROJECT / "code"
PAPER_DIR = OPENDETECT_PROJECT / "paper"
REPRODUCTION = OPENDETECT_PROJECT / "reproduction"
FINAL_MATRIX = (
    OPENDETECT_PROJECT
    / "artifacts/paper-reproduction/final-reproduction-matrix-v3"
)
USTC_V6_SPLITS = OPENDETECT_PROJECT / "artifacts/v6-local-pcap-fivefold-splits-v1"
USTC_V6 = (
    OPENDETECT_PROJECT
    / "artifacts/paper-reproduction/v6-local-pcap-fivefold-corrected-v1"
)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def git_status(path: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--short", "--untracked-files=no"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


for directory in (
    OUT / "paper",
    OUT / "official_code",
    OUT / "ustc_reproduction",
    OUT / "cipherspectrum",
    OUT / "summary",
    AUDIT / "sources",
):
    directory.mkdir(parents=True, exist_ok=True)


paper_protocol = [
    {"item": "Dataset", "paper_value": "USTC-TFC2016 and Malicious TLS", "page": "9", "section": "V-A", "equation/table": "Table II", "evidence": "Evaluation uses USTC-TFC2016 and a 24-class Malicious TLS corpus.", "confidence": "HIGH"},
    {"item": "Class count", "paper_value": "USTC 20 (10 benign + 10 malware); Malicious TLS 24", "page": "9", "section": "V-A", "equation/table": "Table II", "evidence": "Table II enumerates 20 USTC and 24 Malicious TLS classes.", "confidence": "HIGH"},
    {"item": "Train/Val/Test split", "paper_value": "8:1:1", "page": "9", "section": "V-A", "equation/table": "-", "evidence": "The datasets are divided into training, validation, and testing sets at 8:1:1.", "confidence": "HIGH"},
    {"item": "Split isolation", "paper_value": "GROUP_ISOLATION_NOT_SPECIFIED", "page": "9-10", "section": "V-A/V-B", "equation/table": "-", "evidence": "The paper states 8:1:1 and random class selection but gives no capture/session/group/temporal isolation rule.", "confidence": "HIGH"},
    {"item": "Open-world scenario", "paper_value": "A1-A3 USTC; B1-B3 Malicious TLS; C1-C2 cross-dataset", "page": "10", "section": "V-B", "equation/table": "Table III", "evidence": "Eight scenarios vary withheld classes or use one corpus as Known and the other as Unknown.", "confidence": "HIGH"},
    {"item": "Known/Unknown class selection", "paper_value": "A1 19/1, A2 17/3, A3 15/5; B1 23/1, B2 21/3, B3 19/5; C1 24/20, C2 20/24", "page": "10", "section": "V-B", "equation/table": "Table III", "evidence": "Table III lists counts and named withheld classes for each scenario.", "confidence": "HIGH"},
    {"item": "Input format", "paper_value": "1-channel 32x32 grayscale byte image (1024 bytes)", "page": "5-6", "section": "III-A/III-B", "equation/table": "Fig. 2", "evidence": "k(k1+k2)=8*(80+48)=1024 bytes are reshaped into 32x32.", "confidence": "HIGH"},
    {"item": "Packet/flow preprocessing", "paper_value": "first 8 IP packets; 80 header bytes + 48 payload bytes per packet; truncate/zero-pad", "page": "5-6,9", "section": "III-A/V-A", "equation/table": "Fig. 2", "evidence": "Non-IP protocols are excluded, Ethernet is stripped, and fixed-length header/payload segments are padded with zero.", "confidence": "HIGH"},
    {"item": "IP/port handling", "paper_value": "IP addresses masked; port removal not stated", "page": "5", "section": "III-A", "equation/table": "-", "evidence": "Paper explicitly masks IP addresses but does not state that TCP/UDP ports are masked.", "confidence": "MEDIUM"},
    {"item": "Encoder architecture", "paper_value": "ResNet18 encoder and mirrored decoder", "page": "9", "section": "V-A", "equation/table": "-", "evidence": "Implementation details specify ResNet18 and a mirror-structured decoder.", "confidence": "HIGH"},
    {"item": "Latent dimension", "paper_value": "128", "page": "9", "section": "V-A", "equation/table": "-", "evidence": "Latent dimension d is set to 128.", "confidence": "HIGH"},
    {"item": "VAE sampling", "paper_value": "TRAIN_LATENT_MODE=SAMPLED_Z", "page": "6", "section": "III-B", "equation/table": "Eq. 4-6", "evidence": "Training defines q_phi(z|x) and reparameterized sampling; inference substitution is not specified.", "confidence": "HIGH"},
    {"item": "Inference classification latent", "paper_value": "UNCLEAR", "page": "6-7", "section": "III-B/III-C", "equation/table": "Eq. 17-18", "evidence": "Equations use z/q(z|x), but no inference pseudocode says whether z is sampled or replaced by mu_x.", "confidence": "HIGH"},
    {"item": "Inference detection latent", "paper_value": "UNCLEAR", "page": "7", "section": "III-D", "equation/table": "Eq. 21-22", "evidence": "Detection text says latent representation z without an inference-time sampling rule.", "confidence": "HIGH"},
    {"item": "Gaussian prototype definition", "paper_value": "one Gaussian prototype per Known class, centered at mu_y", "page": "6-7", "section": "III-B/III-C", "equation/table": "Eq. 12-18", "evidence": "Each class is represented by a Gaussian prior prototype.", "confidence": "HIGH"},
    {"item": "Prior definition", "paper_value": "p(z|y)=N(mu_y,I)", "page": "6-7", "section": "III-B", "equation/table": "Eq. 12,15", "evidence": "Prototype covariance is identity; only the class mean varies.", "confidence": "HIGH"},
    {"item": "Prototype loss", "paper_value": "KL[q_phi(z|x)||N(mu_y,I)] plus discriminative prototype classification", "page": "7", "section": "III-C", "equation/table": "Eq. 15-20", "evidence": "Paper describes KL-to-target prototype and a prototype-based classification objective.", "confidence": "HIGH"},
    {"item": "Detection score", "paper_value": "minimum distance from latent distribution/representation to a Gaussian prototype", "page": "7", "section": "III-D", "equation/table": "Eq. 21", "evidence": "dist=min_i dis(z,Prototype_i); nearby equations define Gaussian KL, but Eq.21 does not restate the exact dis function.", "confidence": "MEDIUM"},
    {"item": "Unknown decision rule", "paper_value": "Known if minimum distance < threshold; Unknown otherwise", "page": "7", "section": "III-D", "equation/table": "Eq. 22", "evidence": "A single threshold separates Known from Unknown by nearest-prototype distance.", "confidence": "HIGH"},
    {"item": "Threshold calibration", "paper_value": "global threshold chosen for 95% Known validation acceptance", "page": "7", "section": "III-D", "equation/table": "-", "evidence": "Threshold is set so that 95% of validation samples are classified as Known; no per-class rule is stated.", "confidence": "HIGH"},
    {"item": "Known validation acceptance target", "paper_value": "95% overall Known validation; per-class acceptance not stated", "page": "7", "section": "III-D", "equation/table": "-", "evidence": "The text says 95% of samples in the validation set, not 95% within every class.", "confidence": "HIGH"},
    {"item": "Metrics", "paper_value": "Accuracy/F1 for classification and open-world detection; AUROC curves", "page": "10-13", "section": "V-C/V-D", "equation/table": "Tables IV-IX, Figs. 5-6", "evidence": "Closed-world and open-world tables report Accuracy and F1; figures report AUROC.", "confidence": "HIGH"},
    {"item": "F1 definition", "paper_value": "Open-world likely binary Known-vs-Unknown F1 (Unknown positive), exact averaging not explicit; closed-world likely weighted multiclass", "page": "10-13", "section": "V-C/V-D", "equation/table": "Tables IV-IX", "evidence": "Paper gives no F1 formula/averaging. Released code uses default binary F1 for open-set and weighted F1 for closed-set.", "confidence": "MEDIUM"},
    {"item": "Repeated runs/seeds", "paper_value": "five-fold results reported as mean +/- std; exact fold manifests and seeds absent", "page": "10", "section": "V-B", "equation/table": "-", "evidence": "Paper states five-fold evaluation but does not publish exact memberships or random seeds.", "confidence": "HIGH"},
    {"item": "Data augmentation", "paper_value": "random crop and horizontal flip", "page": "9", "section": "V-A", "equation/table": "-", "evidence": "Paper describes random center cropping/random level flipping; code implements RandomCrop(padding=4) and RandomHorizontalFlip.", "confidence": "HIGH"},
    {"item": "Hyperparameters", "paper_value": "lambda=0.005, gamma=1, d=128, k=8, k1=80, k2=48, Adam", "page": "9", "section": "V-A", "equation/table": "-", "evidence": "Implementation details list the fixed input/model/loss hyperparameters.", "confidence": "HIGH"},
]
write_csv(OUT / "paper/paper_protocol.csv", paper_protocol)


official_map = [
    {"function": "training entry and hyperparameters", "file": "train.py", "line_start": 15, "line_end": 86, "behavior": "seed 2022; Adam; 100 epochs; prototype resets at epochs 50 and 80", "paper_consistent": "PARTIAL_MATCH", "notes": "Released validation loader is the test NPZ; exact paper folds absent."},
    {"function": "evaluation entry", "file": "test.py", "line_start": 70, "line_end": 111, "behavior": "loads checkpoint; evaluates closed/open sets; open score selected from KL output", "paper_consistent": "PARTIAL_MATCH", "notes": "Threshold selection differs from paper."},
    {"function": "prototype parameter", "file": "model.py", "line_start": 13, "line_end": 19, "behavior": "one trainable Cx128 mean parameter initialized with Kaiming normal", "paper_consistent": "MATCH", "notes": "No learned prototype covariance."},
    {"function": "VAE sampler", "file": "model.py", "line_start": 21, "line_end": 28, "behavior": "training: mu+std*eps; evaluation: mu", "paper_consistent": "PARTIAL_MATCH", "notes": "Paper inference mode is unclear; code is unambiguous."},
    {"function": "squared Euclidean distance", "file": "model.py", "line_start": 30, "line_end": 35, "behavior": "computes ||z-mu_y||^2", "paper_consistent": "MATCH", "notes": "Used for training discriminative prediction/loss."},
    {"function": "Gaussian KL", "file": "model.py", "line_start": 37, "line_end": 40, "behavior": "0.5*(||mu_x-mu_y||^2 + sum(exp(logvar)-logvar-1))", "paper_consistent": "MATCH", "notes": "KL[N(mu_x,diag(exp(logvar))) || N(mu_y,I)]; includes variance and log determinant term."},
    {"function": "encoder forward", "file": "model.py", "line_start": 42, "line_end": 48, "behavior": "encoder returns mu/logvar; sampler produces latent_z; returns Euclidean and KL matrices", "paper_consistent": "PARTIAL_MATCH", "notes": "Evaluation latent_z is mu_x."},
    {"function": "training objective", "file": "model.py", "line_start": 50, "line_end": 76, "behavior": "MSE reconstruction, target KL, entropy term, and Euclidean discriminative CE", "paper_consistent": "MISMATCH", "notes": "Paper's discriminative probability is written from KL; released code trains it from sampled-z Euclidean distance."},
    {"function": "prototype reset", "file": "utils.py", "line_start": 43, "line_end": 55, "behavior": "model.eval; encoder mu_x; per-class mean becomes a new nn.Parameter", "paper_consistent": "PARTIAL_MATCH", "notes": "Replacing the Parameter after optimizer creation disconnects it from the optimizer."},
    {"function": "native score selection", "file": "test.py", "line_start": 91, "line_end": 105, "behavior": "uses per-sample minimum Gaussian KL and nearest-KL class prediction", "paper_consistent": "MATCH", "notes": "Euclidean outputs are computed but discarded for open-set evaluation."},
    {"function": "threshold calibration", "file": "test.py", "line_start": 46, "line_end": 66, "behavior": "global threshold maximizing TPR-FPR on labeled Known+Unknown test scores", "paper_consistent": "MISMATCH", "notes": "Oracle test Youden-J; paper says 95% Known validation acceptance."},
    {"function": "open-set labels and F1", "file": "test.py", "line_start": 46, "line_end": 58, "behavior": "Known=0, Unknown=1; sklearn default binary f1_score", "paper_consistent": "UNKNOWN", "notes": "Paper does not define F1 averaging/positive class."},
    {"function": "closed-set metrics", "file": "test.py", "line_start": 28, "line_end": 44, "behavior": "Accuracy and weighted Precision/Recall/F1", "paper_consistent": "PARTIAL_MATCH", "notes": "Likely explains paper closed-world F1, but formula is not stated in paper."},
    {"function": "data loader", "file": "data/dataset.py", "line_start": 37, "line_end": 88, "behavior": "loads prebuilt train/test NPZ arrays; no separate validation NPZ", "paper_consistent": "MISMATCH", "notes": "Released train.py treats test NPZ as validation."},
    {"function": "augmentation", "file": "data/dataset.py", "line_start": 90, "line_end": 120, "behavior": "RandomCrop(32,padding=4), RandomHorizontalFlip, ToTensor", "paper_consistent": "MATCH", "notes": "No dataset mean/std normalization."},
    {"function": "PCAP fixed-length encoding", "file": "data/Preprocessing/utils.py", "line_start": 11, "line_end": 44, "behavior": "first 8 packets, fixed 80-byte header and 48-byte payload, zero padding", "paper_consistent": "MATCH", "notes": "rdpcap loads each flow PCAP; exceptions become zero packet segments."},
    {"function": "IP masking/header-payload split", "file": "data/Preprocessing/utils.py", "line_start": 47, "line_end": 67, "behavior": "serializes from IP layer, masks src/dst to 0.0.0.0, retains transport ports, separates Raw payload", "paper_consistent": "MATCH", "notes": "Ethernet is excluded because serialization starts from IP."},
    {"function": "32x32 image write", "file": "data/Preprocessing/pcap2png.py", "line_start": 14, "line_end": 31, "behavior": "hex bytes converted to uint8 and reshaped to 32x32 grayscale", "paper_consistent": "MATCH", "notes": "Paper image construction reproduced."},
    {"function": "class/split definitions", "file": "data/splits.py", "line_start": 1, "line_end": 111, "behavior": "hard-coded Known/Unknown class lists per dataset scenario", "paper_consistent": "PARTIAL_MATCH", "notes": "No released exact five-fold flow/capture membership."},
]
write_csv(OUT / "official_code/official_implementation_map.csv", official_map)


prototype_md = """# Official Open-Detect Prototype Definition

## Exact structure

- There is exactly **one prototype mean per Known class**: `prototypes` has shape `n_classes x 128` (`model.py:13-19`).
- A prototype is **only** `mu_y`; there is no learned class covariance.  Its Gaussian is `N(mu_y, I)`.
- The query posterior is `q_x=N(mu_x, diag(exp(logvar_x)))`.
- The exact code KL is:

  `0.5 * (||mu_x-mu_y||^2 + sum(exp(logvar_x)-logvar_x-1))`.

  It therefore includes posterior variance and the log-determinant contribution; the prototype covariance is fixed identity.

## Lifecycle

1. Means are Kaiming-initialized trainable parameters.
2. The training discriminative term updates them through squared Euclidean distance to sampled `z`.
3. At epochs 50 and 80, `reset_prototype` switches to evaluation mode, computes deterministic encoder means, and replaces each prototype with its training-class mean.
4. The replacement is a new `nn.Parameter` after the optimizer was created.  The released optimizer is not rebuilt, so the reset parameter is no longer in that optimizer's parameter list.  This is an implementation hazard, not a paper-defined algorithmic step.

## Native detector

The released open-set detector does not use a GMM, Mahalanobis fit, or symmetric KL.  It uses the minimum forward KL from the query posterior to the fixed-identity class prototypes.  One component per class is therefore the precise Native structure.
"""
write_text(OUT / "official_code/prototype_definition.md", prototype_md)


metric_md = """# Metric Definition Audit

## Paper evidence

The paper reports Accuracy and F1 in closed-world and open-world tables, but it does not provide an F1 equation, averaging mode, or positive-class convention.  Therefore the exact paper definition cannot be recovered from the PDF alone.

## Released code evidence

- Closed-set classification uses `f1_score(..., average='weighted')` (`test.py:28-44`).
- Open-set evaluation forms a binary target with Known=0 and Unknown=1 and calls `f1_score(y_true, y_pred)` without an `average` argument (`test.py:46-58`).  This is binary F1 with **Unknown as the positive class**.
- The same function selects its threshold from labeled Known+Unknown test samples, so the released open-set F1 is coupled to oracle test calibration.

## Audit conclusion

- **Paper closed-world F1:** likely weighted multiclass F1, based on released code and the table pattern; confidence MEDIUM.
- **Paper open-world F1:** likely binary Known-vs-Unknown F1 with Unknown positive; confidence MEDIUM.  The PDF itself remains under-specified.
- **Stage9 Known Macro-F1:** macro-average classification F1 over accepted/reported Known classes, not the paper open-world F1.

Accordingly, paper open-world F1 and Stage9 Known Macro-F1 are **NOT_COMPARABLE**.  No claim in this audit treats them as the same metric.
"""
write_text(OUT / "paper/metric_definition_audit.md", metric_md)


final_matrix = load_json(FINAL_MATRIX / "final_reproduction_matrix.json")
v6_campaign = load_json(USTC_V6 / "campaign_summary.json")
v6_split_campaign = load_json(USTC_V6_SPLITS / "campaign_manifest.json")
v6_summaries = {
    scenario: load_json(USTC_V6 / f"summaries/{scenario}_five_repeat_summary.json")
    for scenario in ("a1", "a2", "a3")
}
figure4 = load_json(
    OPENDETECT_PROJECT
    / "artifacts/paper-reproduction/figure4-fivefold-mean-v1/figure_4_fivefold_metrics.json"
)
ustc_closed_summary = load_json(
    OPENDETECT_PROJECT
    / "artifacts/paper-reproduction/local-fivefold-corrected-stable-v1/summaries/c2_five_repeat_summary.json"
)

ustc_open_set_rows = []
v6_checkpoints = []
for scenario in ("a1", "a2", "a3"):
    for fold in range(5):
        run = USTC_V6 / scenario / f"fold{fold}_seed{2022 + fold}"
        config = load_json(run / "run_config.json")
        metrics = load_json(run / "test_metrics.json")
        checkpoint_path = run / "model_best.pt"
        checkpoint_hash = sha256(checkpoint_path)
        v6_checkpoints.append(checkpoint_path)
        primary = metrics["open_world"]["balanced_1to1"]
        paper_threshold = primary["paper_validation_95pct"]
        ustc_open_set_rows.append({
            "setting": config["paper_target"]["scenario"],
            "fold": fold,
            "seed": config["seed"],
            "reproduction_status": "LOCAL_APPROXIMATION",
            "training_protocol": config["training_protocol"],
            "known_classes": len(config["known_classes"]),
            "unknown_classes": len(config["unknown_classes"]),
            "unknown_class_ids": ";".join(map(str, config["unknown_classes"])),
            "known_train_n": config["dataset_sizes"]["train_known"],
            "known_validation_n": config["dataset_sizes"]["validation_known"],
            "known_test_n_all": config["dataset_sizes"]["test_known"],
            "unknown_test_n_all": config["dataset_sizes"]["test_unknown"],
            "balanced_known_test_n": primary["sample_counts"]["known"],
            "balanced_unknown_test_n": primary["sample_counts"]["unknown"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "best_epoch": metrics["training"]["best_epoch"],
            "completed_epochs": metrics["training"]["completed_epochs"],
            "best_validation_accuracy": metrics["training"]["best_validation_accuracy"],
            "validation_macro_f1": "NOT_RECORDED",
            "train_latent_mode": "SAMPLED_Z",
            "inference_classification_latent_mode": "MU_X",
            "inference_detection_latent_mode": "MU_X_AND_LOGVAR_X",
            "native_score": "min_y KL[N(mu_x,diag(exp(logvar_x))) || N(mu_y,I)]",
            "native_threshold": paper_threshold["threshold"],
            "threshold_method": "global P95 (method=higher) on Known validation KL only",
            "closed_known_test_accuracy": metrics["closed_world_known_test"]["accuracy"],
            "closed_known_test_f1_weighted": metrics["closed_world_known_test"]["f1_weighted"],
            "open_accuracy_binary": paper_threshold["accuracy"],
            "open_f1_binary_unknown_positive": paper_threshold["f1"],
            "open_auroc_unknown_positive": primary["auroc"],
            "metric_sample_composition": "balanced 1:1 Known/Unknown test subset",
            "paper_exact_fold": False,
        })
write_csv(OUT / "ustc_reproduction/ustc_open_set_reproduction.csv", ustc_open_set_rows)

scenario_metrics = {row["scenario"]: row for row in v6_campaign["metrics"]}
ustc_protocol = [
    {"item": "project_relationship", "value": "Projects/Open-Detect is a sibling of Projects/unknown_traffic_project; it is not nested in this method project", "evidence": str(OPENDETECT_PROJECT)},
    {"item": "authoritative_status", "value": "LOCAL_APPROXIMATION; paper-level exact reproduction count = 0", "evidence": "artifacts/paper-reproduction/final-reproduction-matrix-v3/final_reproduction_matrix.json"},
    {"item": "latest_ustc_campaign", "value": "v6 local PCAP A-1/A-2/A-3; 15 complete runs", "evidence": "artifacts/paper-reproduction/v6-local-pcap-fivefold-corrected-v1/campaign_summary.json"},
    {"item": "checkpoint_count", "value": len(v6_checkpoints), "evidence": "one model_best.pt for each of 3 scenarios x 5 repeats; hashes are in ustc_open_set_reproduction.csv"},
    {"item": "checkpoint_hashes", "value": "15 independently recomputed SHA-256 values", "evidence": "outputs/ustc_reproduction/ustc_open_set_reproduction.csv"},
    {"item": "dataset", "value": "USTC-TFC2016 local PCAP-derived v6 pool; 34,585 selected flow images", "evidence": "artifacts/v6-local-pcap-fivefold-splits-v1/campaign_manifest.json"},
    {"item": "pool_sha256", "value": v6_split_campaign["pool_sha256"], "evidence": "campaign_manifest.json"},
    {"item": "classes", "value": "20 USTC classes; SMB-1/SMB-2 are one SMB class", "evidence": "fold0 preprocessing_report.json and official splits.py"},
    {"item": "split", "value": "five independently seeded class-stratified grouped-image-disjoint 8:1:1 splits; not author exact fivefold", "evidence": v6_split_campaign["reproduction_boundary"]},
    {"item": "full_pool_train_validation_test", "value": "27,667 / 3,457 / 3,461 per repeat", "evidence": "campaign_manifest.json"},
    {"item": "split_isolation", "value": "zero cross-split flow overlap and zero exact-image overlap in all five local splits", "evidence": "campaign_manifest.json"},
    {"item": "input_preprocessing", "value": "directional local 5-tuple sessions; first 8 packets; 80 IP/header + 48 Raw payload bytes; IPv4 addresses zeroed; ports retained; zero pad/truncate; 32x32 uint8", "evidence": "fold0 preprocessing_report.json; reproduction/prepare_ustc.py"},
    {"item": "source_selection_boundary", "value": "local source fragments for SMB/Weibo were diagnostically inferred from released-image overlap; not an author-disclosed reusable selection protocol", "evidence": "fold0 preprocessing_report.json class_replacement.protocol_note"},
    {"item": "architecture", "value": "CorrectedOpenDetectNet subclasses released OpenDetectNet but connects the omitted decoder block and clamps logvar to [-30,20]", "evidence": "reproduction/corrected_model.py"},
    {"item": "training_objective", "value": "corrected-paper: target KL + reconstruction + KL-based discriminative cross-entropy; released constant entropy term removed", "evidence": "reproduction/corrected_model.py:60-86"},
    {"item": "latent_mode", "value": "TRAIN=SAMPLED_Z; INFERENCE_CLASSIFICATION=MU_X; INFERENCE_DETECTION=MU_X_AND_LOGVAR_X", "evidence": "official model sampler plus reproduction/run_reproduction.py collect_outputs"},
    {"item": "prototype", "value": "one learned mean mu_y per Known class with identity covariance; in-place class-mean resets at loop epochs 50/80", "evidence": "reproduction/run_ustc_reproduction.py and corrected_model.py"},
    {"item": "Native_open_set_score", "value": "min_y 0.5*(||mu_x-mu_y||^2 + sum(exp(logvar_x)-logvar_x-1)); smaller is more Known", "evidence": "reproduction/corrected_model.py:60-65; run_reproduction.py:405-419"},
    {"item": "threshold", "value": "one global P95 KL threshold from Known validation only (numpy quantile method=higher); target is 95% Known acceptance", "evidence": "reproduction/run_reproduction.py:478-519"},
    {"item": "open_metric_definition", "value": "balanced 1:1 Known/Unknown test; binary Accuracy and binary F1 with Unknown=1; AUROC Unknown-positive", "evidence": "reproduction/run_reproduction.py:436-535"},
    {"item": "A-1_result", "value": f"Accuracy={scenario_metrics['A-1']['validation_threshold_accuracy_mean']:.4f}+/-{scenario_metrics['A-1']['validation_threshold_accuracy_std_sample']:.4f}; F1={scenario_metrics['A-1']['validation_threshold_f1_mean']:.4f}+/-{scenario_metrics['A-1']['validation_threshold_f1_std_sample']:.4f}; AUROC={scenario_metrics['A-1']['auroc_mean']:.4f}+/-{scenario_metrics['A-1']['auroc_std_sample']:.4f}", "evidence": "v6 campaign_summary.json"},
    {"item": "A-2_result", "value": f"Accuracy={scenario_metrics['A-2']['validation_threshold_accuracy_mean']:.4f}+/-{scenario_metrics['A-2']['validation_threshold_accuracy_std_sample']:.4f}; F1={scenario_metrics['A-2']['validation_threshold_f1_mean']:.4f}+/-{scenario_metrics['A-2']['validation_threshold_f1_std_sample']:.4f}; AUROC={scenario_metrics['A-2']['auroc_mean']:.4f}+/-{scenario_metrics['A-2']['auroc_std_sample']:.4f}", "evidence": "v6 campaign_summary.json"},
    {"item": "A-3_result", "value": f"Accuracy={scenario_metrics['A-3']['validation_threshold_accuracy_mean']:.4f}+/-{scenario_metrics['A-3']['validation_threshold_accuracy_std_sample']:.4f}; F1={scenario_metrics['A-3']['validation_threshold_f1_mean']:.4f}+/-{scenario_metrics['A-3']['validation_threshold_f1_std_sample']:.4f}; AUROC={scenario_metrics['A-3']['auroc_mean']:.4f}+/-{scenario_metrics['A-3']['auroc_std_sample']:.4f}", "evidence": "v6 campaign_summary.json"},
    {"item": "closed_world_result", "value": f"five-repeat all-known USTC Accuracy={figure4['datasets']['USTC-TFC2016']['accuracy_mean']:.6f}; weighted F1={ustc_closed_summary['aggregate']['closed_f1']['mean']:.6f}", "evidence": "figure4-fivefold-mean-v1/figure_4_fivefold_metrics.json and local-fivefold-corrected-stable-v1/summaries/c2_five_repeat_summary.json"},
    {"item": "legacy_98_62_98_74_record", "value": "NOT the sibling Open-Detect reproduction; it belongs to unknown_traffic_project/opendetect_ustc_encoder_audit and must not be used as layer C", "evidence": "corrected project-boundary audit"},
]
write_csv(OUT / "ustc_reproduction/ustc_reproduction_protocol.csv", ustc_protocol)


paper_vs_ustc = [
    {"item": "dataset", "paper": "USTC Table-II corpus (34,585 samples)", "ustc_reproduction": "local PCAP-derived v6 pool with the same 34,585 class-count target", "alignment": "PARTIAL_MATCH", "reason": "counts align, but flow identities and source-fragment selection are locally reconstructed/inferred"},
    {"item": "classes", "paper": "20 USTC classes", "ustc_reproduction": "20 classes; SMB captures merged into one SMB label", "alignment": "MATCH", "reason": "class taxonomy and per-class target counts align"},
    {"item": "split", "paper": "8:1:1; five-fold; group isolation and seeds unpublished", "ustc_reproduction": "five independent repeated grouped-image-disjoint 8:1:1 splits; seeds 2022-2026", "alignment": "PARTIAL_MATCH", "reason": "ratio/repetition count align; local repeated splits are not the authors' folds"},
    {"item": "preprocessing", "paper": "8x(80 header+48 payload), mask IP, 32x32", "ustc_reproduction": "same byte layout from directional local sessions", "alignment": "PARTIAL_MATCH", "reason": "core encoding aligns; author flow segmentation/source selection is unpublished and local source-fragment selection is inferred"},
    {"item": "encoder", "paper": "ResNet18 CVAE and mirrored decoder", "ustc_reproduction": "CorrectedOpenDetectNet based on released structure with decoder-path and numerical repairs", "alignment": "PARTIAL_MATCH", "reason": "base structure aligns, but this is deliberately not unchanged released-code behavior"},
    {"item": "latent dimension", "paper": "128", "ustc_reproduction": "128", "alignment": "MATCH", "reason": "identical"},
    {"item": "training objective", "paper": "KL-derived prototype classification plus KL/reconstruction", "ustc_reproduction": "corrected-paper KL-based discriminative objective plus target KL/reconstruction", "alignment": "PARTIAL_MATCH", "reason": "equation family aligns; implementation includes explicit repairs and numerical clamping"},
    {"item": "latent mode", "paper": "training sampled; inference unclear", "ustc_reproduction": "training sampled; classification uses mu_x; detection uses mu_x and logvar_x", "alignment": "UNKNOWN", "reason": "paper inference substitution is under-specified"},
    {"item": "prototype", "paper": "one N(mu_y,I) per class", "ustc_reproduction": "one learned/reset mean with identity covariance per Known class", "alignment": "PARTIAL_MATCH", "reason": "prototype family aligns; optimizer-preserving in-place reset is a local repair"},
    {"item": "score", "paper": "nearest Gaussian-prototype distance", "ustc_reproduction": "minimum forward Gaussian KL q_x || N(mu_y,I)", "alignment": "PARTIAL_MATCH", "reason": "consistent with surrounding paper KL equations and released implementation, but Eq.21 does not restate dis exactly"},
    {"item": "threshold", "paper": "global 95% Known-validation acceptance", "ustc_reproduction": "global P95 Known-validation KL using quantile method=higher", "alignment": "MATCH", "reason": "validation-only threshold directly implements the stated paper rule"},
    {"item": "metrics", "paper": "five-fold Accuracy/F1; open F1 averaging and sample composition under-specified", "ustc_reproduction": "five-repeat mean/std; balanced 1:1 binary Accuracy/F1 with Unknown positive plus AUROC", "alignment": "PARTIAL_MATCH", "reason": "metric implementation is explicit locally, but author fold/sample composition and F1 convention are unpublished"},
    {"item": "open-world class scenarios", "paper": "A-1/A-2/A-3 with 19/1,17/3,15/5", "ustc_reproduction": "same official A-1/A-2/A-3 class lists", "alignment": "MATCH", "reason": "reproduction imports the released splits.py definitions"},
    {"item": "unknown exclusion", "paper": "Unknown classes excluded from Known training", "ustc_reproduction": "scenario filtering excludes Unknown labels from training and Known validation threshold", "alignment": "MATCH", "reason": "run_ustc_reproduction constructs Known-only train/validation loaders"},
    {"item": "reproduction claim", "paper": "author five-fold result", "ustc_reproduction": "LOCAL_APPROXIMATION; 0 paper-level exact rows in final matrix", "alignment": "MISMATCH", "reason": "author flow identities/folds/seeds and exact raw-to-NPZ provenance are unavailable"},
    {"item": "legacy 98.6158/98.7371 record", "paper": "not applicable", "ustc_reproduction": "belongs to a nested unknown_traffic_project audit, not the sibling Projects/Open-Detect reproduction", "alignment": "MISMATCH", "reason": "the corrected task boundary requires the sibling method project as layer C"},
]
write_csv(OUT / "summary/paper_vs_ustc.csv", paper_vs_ustc)


stage7_rows = []
stage7_counts = {}
for setting in ("low", "medium", "high"):
    fold = load_json(PROJECT / f"stage6_cipherspectrum_protocol/outputs/splits/{setting}_fold.json")
    run = next((PROJECT / "stage7_cipherspectrum_known_training/runs").glob(f"formal-{setting}-*"))
    selection = load_json(run / "training_checkpoint_selection.json")
    record = {
        "setting": setting,
        "known_classes": len(fold["known_classes"]),
        "unknown_classes": len(fold["unknown_classes"]),
        "train_n": fold["known_train_count"],
        "val_n": fold["known_validation_count"],
        "known_test_n": fold["known_test_count"],
        "unknown_test_n": fold["unknown_test_count"],
        "group_aware": True,
        "checkpoint": str((run / "artifacts/training_best_checkpoint.pt").relative_to(PROJECT)),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "best_epoch": selection["best_epoch"],
        "stop_epoch": selection["stop_epoch"],
        "val_accuracy": selection["val_accuracy"],
        "val_macro_f1": selection["val_macro_f1"],
        "architecture_alignment": "MODIFIED",
        "architecture_notes": "same released structural architecture through Stage3OpenDetectNet; max-logvar=20 stability guard and local training wrapper/protocol; recorded train/val guard activations were zero",
    }
    stage7_rows.append(record)
    stage7_counts[setting] = record
write_csv(OUT / "cipherspectrum/stage7_encoder_audit.csv", stage7_rows)


native_audit = [
    {"item": "architecture", "official": "custom one-channel ResNet18-like CVAE", "stage9_native": "Stage3OpenDetectNet subclass of the audited released adapter", "alignment": "MODIFIED", "evidence": "stage3_unknown_utility/scripts/stage3_model.py:18-42"},
    {"item": "training latent", "official": "sampled z", "stage9_native": "Stage7 trained with sampled z", "alignment": "MATCH", "evidence": "official model.py:21-28; local adapter"},
    {"item": "inference latent", "official": "mu_x", "stage9_native": "deterministic mu_x", "alignment": "MATCH", "evidence": "official model.py:21-28; extract_test_representations.py:75-85"},
    {"item": "query posterior covariance", "official": "diag(exp(logvar_x))", "stage9_native": "diag(exp(logvar_x)); logvar upper-clamped at 20 by stability wrapper", "alignment": "MODIFIED", "evidence": "stage3_model.py:18-42"},
    {"item": "prototype", "official": "one learned/reset mu_y per class; covariance I", "stage9_native": "frozen Stage7 prototype means; covariance I", "alignment": "MATCH", "evidence": "checkpoint plus extract_test_representations.py:81"},
    {"item": "distance/KL", "official": "forward KL q_x || N(mu_y,I)", "stage9_native": "same KL expression", "alignment": "MATCH", "evidence": "official model.py:37-40; local adapter kl_div_to_prototypes"},
    {"item": "stored score sign", "official": "min KL, higher means more Unknown", "stage9_native": "-min KL, higher means more Known", "alignment": "EQUIVALENT_SIGN_REVERSAL", "evidence": "extract_test_representations.py:81-85"},
    {"item": "class prediction", "official": "argmin KL", "stage9_native": "argmin KL", "alignment": "MATCH", "evidence": "official test.py:91-105; local extract line 82"},
    {"item": "threshold scope", "official": "one global threshold", "stage9_native": "one global threshold per frozen setting", "alignment": "MATCH", "evidence": "native_threshold.json"},
    {"item": "threshold data", "official": "mixed labeled Known+Unknown test", "stage9_native": "Known validation only; no test/Unknown calibration", "alignment": "MISMATCH", "evidence": "official test.py:46-58; Stage8A native_threshold.json"},
    {"item": "threshold rule", "official": "Youden J = argmax(TPR-FPR)", "stage9_native": "linear P05 giving about 95% overall Known-validation acceptance", "alignment": "MISMATCH", "evidence": "official test.py:50-54; Stage8A native_threshold.json"},
    {"item": "overall detector identity", "official": "released Native detector", "stage9_native": "same latent/prototype/KL family but different calibration and stability wrapper", "alignment": "CONCEPT_ONLY_NOT_EXACT", "evidence": "aggregate static audit"},
]
write_csv(OUT / "cipherspectrum/native_implementation_audit.csv", native_audit)


preprocessing = [
    {"aspect": "packet count", "paper": "first 8", "official_code": "first 8", "ustc_reproduction": "first 8", "cipherspectrum": "first 8", "alignment": "MATCH"},
    {"aspect": "bytes per packet", "paper": "80 header + 48 payload", "official_code": "80 + 48", "ustc_reproduction": "80 + 48", "cipherspectrum": "80 + 48", "alignment": "MATCH"},
    {"aspect": "direction", "paper": "capture order; no canonical direction stated", "official_code": "PCAP order; no canonical direction", "ustc_reproduction": "directional local sessions in capture order", "cipherspectrum": "single-flow PCAP order", "alignment": "PARTIAL_MATCH"},
    {"aspect": "padding", "paper": "zero pad", "official_code": "zero pad packets/segments", "ustc_reproduction": "zero pad", "cipherspectrum": "zero pad", "alignment": "MATCH"},
    {"aspect": "truncation", "paper": "fixed k/k1/k2", "official_code": "first 8 and slice segments", "ustc_reproduction": "same", "cipherspectrum": "same", "alignment": "MATCH"},
    {"aspect": "image construction", "paper": "1024 bytes -> 32x32 grayscale", "official_code": "uint8 reshape 32x32", "ustc_reproduction": "NumPy uint8 32x32", "cipherspectrum": "NumPy uint8 32x32", "alignment": "MATCH"},
    {"aspect": "IP handling", "paper": "mask IP addresses", "official_code": "IPv4 src/dst -> 0.0.0.0", "ustc_reproduction": "zero IPv4 addresses", "cipherspectrum": "zero IPv4 addresses", "alignment": "MATCH"},
    {"aspect": "port handling", "paper": "not stated", "official_code": "transport ports retained", "ustc_reproduction": "ports retained", "cipherspectrum": "ports retained", "alignment": "UNKNOWN_PAPER; CODE_MATCH"},
    {"aspect": "header/payload", "paper": "separate fixed slices", "official_code": "IP bytes minus Raw payload; Raw payload separate", "ustc_reproduction": "same", "cipherspectrum": "same", "alignment": "MATCH"},
    {"aspect": "normalization", "paper": "not stated", "official_code": "ToTensor scale only; no mean/std", "ustc_reproduction": "same model input scale", "cipherspectrum": "same model input scale", "alignment": "PARTIAL_MATCH"},
    {"aspect": "augmentation", "paper": "random crop/flip", "official_code": "RandomCrop padding4 + horizontal flip", "ustc_reproduction": "same", "cipherspectrum": "same Stage7 training transforms", "alignment": "MATCH"},
    {"aspect": "invalid/non-IP handling", "paper": "non-IP excluded", "official_code": "exceptions/non-IP silently become zero segments", "ustc_reproduction": "strict on-packet-error failure; non-TCP/UDP IPv4 and undecoded later fragments excluded", "cipherspectrum": "strict failure gate", "alignment": "PARTIAL_MATCH"},
    {"aspect": "flow construction", "paper": "details beyond fixed packet layout not fully specified", "official_code": "expects pre-split single-flow PCAPs/NPZ", "ustc_reproduction": "directional 5-tuple sessions; no timeout/SYN-FIN-RST boundaries; local source-fragment selection partly inferred", "cipherspectrum": "dataset-provided single-flow PCAPs", "alignment": "PARTIAL_MATCH"},
]
write_csv(OUT / "summary/preprocessing_comparison.csv", preprocessing)


results_path = PROJECT / "stage9_cipherspectrum_final_test/outputs/summary/cross_setting_results.csv"
with results_path.open(newline="", encoding="utf-8-sig") as handle:
    stage9_results = list(csv.DictReader(handle))
selected = {(row["setting"], row["method"]): row for row in stage9_results}

result_alignment = [
    {"dataset": "Paper USTC", "known classes": "A1/A2/A3: 19/17/15", "unknown classes": "1/3/5", "classification metric": "closed-world Accuracy/F1", "classification value": "Open-Detect 99.28% Accuracy; 99.28% F1", "open-set metric": "A1/A2/A3 Accuracy-F1: 98.20-98.24 / 89.22-89.10 / 95.51-91.17 (%)", "UFAR if available": "NOT_AVAILABLE", "AUROC if available": "curves only", "F1 definition": "under-specified; likely binary Unknown-positive for open set", "split type": "8:1:1, five-fold; group isolation unspecified"},
    {"dataset": "Paper Malicious TLS", "known classes": "B1/B2/B3: 23/21/19", "unknown classes": "1/3/5", "classification metric": "closed-world Accuracy/F1", "classification value": "Open-Detect 99.02% Accuracy; 99.01% F1", "open-set metric": "B1/B2/B3 Accuracy-F1: 90.20-90.71 / 90.34-89.62 / 85.94-85.97 (%)", "UFAR if available": "NOT_AVAILABLE", "AUROC if available": "curves only", "F1 definition": "under-specified; likely binary Unknown-positive for open set", "split type": "8:1:1, five-fold; group isolation unspecified"},
]
for scenario in ("A-1", "A-2", "A-3"):
    metric = scenario_metrics[scenario]
    known_count, unknown_count = {"A-1": (19, 1), "A-2": (17, 3), "A-3": (15, 5)}[scenario]
    result_alignment.append({
        "dataset": f"Sibling Open-Detect USTC v6 {scenario}",
        "known classes": known_count,
        "unknown classes": unknown_count,
        "classification metric": "Known-test weighted F1 is recorded per run; campaign summary prioritizes open-world metrics",
        "classification value": "see outputs/ustc_reproduction/ustc_open_set_reproduction.csv",
        "open-set metric": f"validation-threshold binary Accuracy/F1 = {metric['validation_threshold_accuracy_mean']:.4f}+/-{metric['validation_threshold_accuracy_std_sample']:.4f} / {metric['validation_threshold_f1_mean']:.4f}+/-{metric['validation_threshold_f1_std_sample']:.4f}",
        "UFAR if available": "NOT_REPORTED_AS_UFAR",
        "AUROC if available": f"{metric['auroc_mean']:.4f}+/-{metric['auroc_std_sample']:.4f}",
        "F1 definition": "binary Unknown-positive on balanced 1:1 Known/Unknown test subset",
        "split type": "five independent repeated grouped-image-disjoint 8:1:1 splits; not author folds",
    })
for setting in ("low", "medium", "high"):
    native = selected[(setting, "Native")]
    counts = stage7_counts[setting]
    result_alignment.append({
        "dataset": f"CipherSpectrum {setting.title()}",
        "known classes": counts["known_classes"],
        "unknown classes": counts["unknown_classes"],
        "classification metric": "Known test Accuracy / Known Macro-F1",
        "classification value": f"{float(native['Known Accuracy']):.6f} / {float(native['Known Macro-F1']):.6f}",
        "open-set metric": "Native frozen Known-vs-Unknown detection",
        "UFAR if available": f"{float(native['UFAR']):.6f}",
        "AUROC if available": f"{float(native['AUROC']):.6f}",
        "F1 definition": "Known multiclass Macro-F1; NOT_COMPARABLE to paper open-world F1",
        "split type": "frozen group-aware cross-group split",
    })
write_csv(OUT / "summary/result_definition_alignment.csv", result_alignment)


same_encoder = []
for setting in ("low", "medium", "high"):
    for method in ("Native", "Single-Full-K1", "Multi-Global-K2"):
        row = selected[(setting, method)]
        same_encoder.append({
            "setting": setting,
            "method": method,
            "same_stage7_encoder": True,
            "same_mu_x": True,
            "representation_after_mu": "Native uses mu_x+logvar_x KL; K1/K2 use frozen StandardScaler/PCA64(mu_x)",
            "known_macro_f1": row["Known Macro-F1"],
            "UFAR": row["UFAR"],
            "AUROC": row["AUROC"],
        })
write_csv(OUT / "summary/same_encoder_detector_comparison.csv", same_encoder)


failure_decomposition = [
    {"factor": "F1 Representation limitation", "evidence_level": "EVIDENCE_MODERATE", "evidence": "Stage7/Stage9 Known Macro-F1 is only 0.70-0.81, but the same mu_x yields much stronger K1/K2 AUROC, so representation alone is insufficient."},
    {"factor": "F2 Native detector-score mismatch", "evidence_level": "EVIDENCE_STRONG", "evidence": "With the same encoder and mu_x, Native AUROC is 0.472-0.501 while K1/K2 is 0.780-0.892. The fixed-identity prototype KL is poorly aligned with CipherSpectrum density geometry/calibration."},
    {"factor": "F3 Dataset/domain shift", "evidence_level": "EVIDENCE_MODERATE", "evidence": "CipherSpectrum differs from USTC/Malicious TLS in collection/domain/label construction; no controlled causal isolation was run."},
    {"factor": "F4 Protocol difficulty shift", "evidence_level": "EVIDENCE_MODERATE", "evidence": "CipherSpectrum enforces group-aware cross-group generalization; the paper states 8:1:1 but no group isolation. This is a potential explanation, not a proven cause."},
    {"factor": "F5 Implementation mismatch", "evidence_level": "EVIDENCE_STRONG", "evidence": "Stage9 and official code agree on mu_x/prototype/KL, but differ critically in threshold data/rule; Stage7 also has an upper-logvar stability wrapper and a local training/split pipeline."},
    {"factor": "F6 Metric-definition mismatch", "evidence_level": "EVIDENCE_STRONG", "evidence": "Paper open-world F1 is under-specified/likely binary Unknown-positive, whereas Stage9 reports Known multiclass Macro-F1; direct F1 comparison is invalid."},
]
write_csv(OUT / "summary/failure_decomposition.csv", failure_decomposition)


latent_modes = [
    {"layer": "Paper", "train_latent_mode": "SAMPLED_Z", "inference_classification_latent_mode": "UNCLEAR", "inference_unknown_detection_latent_mode": "UNCLEAR", "evidence": "paper Eq.4-6,17-18,21-22; no inference pseudocode"},
    {"layer": "Official code", "train_latent_mode": "SAMPLED_Z", "inference_classification_latent_mode": "MU_X", "inference_unknown_detection_latent_mode": "MU_X", "evidence": "model.py:21-28,42-48; test.py:91-105"},
    {"layer": "Sibling Open-Detect USTC v6 reproduction", "train_latent_mode": "SAMPLED_Z", "inference_classification_latent_mode": "MU_X", "inference_unknown_detection_latent_mode": "MU_X", "evidence": "Projects/Open-Detect/reproduction/corrected_model.py and run_reproduction.py:405-419"},
    {"layer": "CipherSpectrum Stage7-9", "train_latent_mode": "SAMPLED_Z", "inference_classification_latent_mode": "MU_X", "inference_unknown_detection_latent_mode": "MU_X", "evidence": "Stage3OpenDetectNet; extract_test_representations.py:75-85"},
]
write_csv(OUT / "summary/latent_mode_audit.csv", latent_modes)


threshold_rows = [
    {"layer": "Paper", "scope": "global", "fit_data": "Known validation", "method": "accept 95% of validation samples", "per_class": False, "leakage_risk": "NOT_SUPPORTED"},
    {"layer": "Official code", "scope": "global", "fit_data": "labeled Known+Unknown test", "method": "Youden J argmax(TPR-FPR)", "per_class": False, "leakage_risk": "SUPPORTED_TEST_ORACLE"},
    {"layer": "Sibling Open-Detect USTC v6 reproduction", "scope": "global per scenario/repeat", "fit_data": "Known validation only", "method": "P95 of KL distance with numpy quantile method=higher", "per_class": False, "leakage_risk": "NOT_SUPPORTED"},
]
for setting in ("low", "medium", "high"):
    payload = load_json(PROJECT / f"stage8a_cipherspectrum_known_density/artifacts/{setting}/native_threshold.json")
    threshold_rows.append({"layer": f"CipherSpectrum {setting}", "scope": "global", "fit_data": payload["fit_split"], "method": payload["selection_method"], "per_class": False, "leakage_risk": "NOT_SUPPORTED", "threshold": payload["threshold"], "actual_known_acceptance": payload["actual_known_validation_acceptance"]})
write_csv(OUT / "summary/threshold_comparison.csv", threshold_rows)


leakage_rows = [
    {"feature_or_shortcut": "filename", "paper": "POSSIBLE: split mechanism not published", "official_code": "NOT_SUPPORTED as model input; filenames organize samples", "ustc_reproduction": "NOT_SUPPORTED as model input; filenames retained only in provenance", "cipherspectrum": "NOT_SUPPORTED as input; used only for manifests"},
    {"feature_or_shortcut": "capture identity", "paper": "POSSIBLE: group isolation unspecified", "official_code": "POSSIBLE: NPZ provenance/membership unavailable", "ustc_reproduction": "POSSIBLE: flow/image groups are isolated, but source-capture isolation is not claimed", "cipherspectrum": "NOT_SUPPORTED: group-aware isolation enforced"},
    {"feature_or_shortcut": "flow ordering", "paper": "POSSIBLE", "official_code": "POSSIBLE: prebuilt NPZ", "ustc_reproduction": "POSSIBLE: deterministic local selection/splitting is documented but author selection is unknown", "cipherspectrum": "NOT_SUPPORTED by frozen group split"},
    {"feature_or_shortcut": "IP address", "paper": "NOT_SUPPORTED: explicitly masked", "official_code": "NOT_SUPPORTED: masked", "ustc_reproduction": "NOT_SUPPORTED: masked", "cipherspectrum": "NOT_SUPPORTED: masked"},
    {"feature_or_shortcut": "port", "paper": "POSSIBLE: removal not stated", "official_code": "SUPPORTED as retained input feature", "ustc_reproduction": "SUPPORTED as retained input feature", "cipherspectrum": "SUPPORTED as retained input feature"},
    {"feature_or_shortcut": "domain/SNI", "paper": "NOT_SUPPORTED as explicit field", "official_code": "NOT_SUPPORTED as explicit field; may remain in payload bytes", "ustc_reproduction": "NOT_SUPPORTED as explicit field; may remain in payload bytes", "cipherspectrum": "NOT_SUPPORTED as explicit field; may remain in payload bytes"},
    {"feature_or_shortcut": "session/PCAP source", "paper": "POSSIBLE: isolation unspecified", "official_code": "POSSIBLE: prebuilt split provenance absent", "ustc_reproduction": "POSSIBLE: flow and exact-image isolation pass, but source-PCAP isolation is not enforced; SMB/Weibo source fragments were locally inferred", "cipherspectrum": "NOT_SUPPORTED: frozen group-aware split"},
]
write_csv(OUT / "summary/leakage_risk_audit.csv", leakage_rows)


paper_pdf = next(PAPER_DIR.glob("*.pdf"))
source_files = [
    paper_pdf,
    OFFICIAL / "train.py",
    OFFICIAL / "test.py",
    OFFICIAL / "model.py",
    OFFICIAL / "utils.py",
    OFFICIAL / "data/dataset.py",
    OFFICIAL / "data/splits.py",
    OFFICIAL / "data/Preprocessing/utils.py",
    REPRODUCTION / "run_ustc_reproduction.py",
    REPRODUCTION / "run_reproduction.py",
    REPRODUCTION / "corrected_model.py",
    FINAL_MATRIX / "final_reproduction_matrix.json",
    USTC_V6 / "campaign_summary.json",
    USTC_V6_SPLITS / "campaign_manifest.json",
    OPENDETECT_PROJECT / "artifacts/paper-reproduction/local-fivefold-corrected-stable-v1/summaries/c2_five_repeat_summary.json",
    PROJECT / "stage9_cipherspectrum_final_test/outputs/summary/cross_setting_results.csv",
]
source_manifest = {
    "audit_type": "STATIC_ONLY",
    "paper_pdf_found": True,
    "paper_pdf": str(paper_pdf),
    "paper_pdf_sha256": sha256(paper_pdf),
    "opendetect_project_relation": "sibling project under Projects/; read-only source",
    "official_parent_commit": git_head(OPENDETECT_PROJECT),
    "official_code_commit": git_head(OFFICIAL),
    "official_parent_tracked_status": git_status(OPENDETECT_PROJECT),
    "official_code_tracked_status": git_status(OFFICIAL),
    "unknown_project_commit": git_head(PROJECT),
    "files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in source_files],
    "v6_checkpoints": [
        {
            "path": row["checkpoint"],
            "size_bytes": Path(row["checkpoint"]).stat().st_size,
            "sha256": row["checkpoint_sha256"],
        }
        for row in ustc_open_set_rows
    ],
    "source_boundaries": {
        "paper_supplementary_found": False,
        "official_code_auditable": True,
        "author_exact_reproduction_complete": False,
        "sibling_reproduction_status": final_matrix["counts"],
        "paper_level_exact_reproduction_count": final_matrix["exact_reproduction_count"],
        "missing_for_author_exact": ["published fold memberships", "author random seeds", "author USTC flow identities and raw-to-NPZ provenance", "raw Malicious TLS PCAPs", "unambiguous paper F1 definition", "paper inference pseudocode"],
    },
}
write_json(AUDIT / "sources/source_manifest.json", source_manifest)
source_md = f"""# Source Manifest

- Paper PDF: `{paper_pdf}`
- Paper SHA-256: `{source_manifest['paper_pdf_sha256']}`
- Extracted text: `sources/open_detect_paper.txt`
- Open-Detect parent commit: `{source_manifest['official_parent_commit']}`
- Official nested code commit: `{source_manifest['official_code_commit']}`
- Sibling USTC v6 checkpoints: `{len(source_manifest['v6_checkpoints'])}` complete `model_best.pt` files; each SHA-256 is recorded in `outputs/ustc_reproduction/ustc_open_set_reproduction.csv`
- Sibling final matrix: `0` paper-exact, `8` local approximations, `1` blocked by missing data, `1` excluded by scope
- Stage9 result source: `stage9_cipherspectrum_final_test/outputs/summary/cross_setting_results.csv`

`Projects/Open-Detect` and `Projects/unknown_traffic_project` are sibling method projects.  This audit reads the former and writes only the latter.  No supplementary document was found locally.  The released code is sufficient for implementation auditing, but published materials do not contain the exact author folds/seeds and do not make an author-exact reproduction possible.  The nested official checkout has pre-existing tracked `__pycache__/*.pyc` changes; no tracked Python source change was observed or made by this audit.
"""
write_text(AUDIT / "sources/source_manifest.md", source_md)


report = """# Stage 10A Open-Detect Paper-Code-Reproduction Protocol Audit

## Executive Summary

The CipherSpectrum Native result cannot yet be attributed directly to an Open-Detect generalization failure.  The official code and Stage9 agree on inference `mu_x`, one fixed-identity Gaussian prototype per class, forward Gaussian KL, and nearest-KL prediction.  They do **not** agree on threshold calibration: released code tunes a global Youden-J threshold on labeled Known+Unknown test data, whereas Stage9 freezes a global P05 threshold using Known Validation only.  Stage7 also uses a max-logvar stability wrapper and a much stricter group-aware external protocol.  Final decision: **Gate B — HIGH-PRIORITY PROTOCOL MISMATCH**; claim status: `EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE`.

## Paper Protocol

The paper uses 20-class USTC-TFC2016 and 24-class Malicious TLS, an 8:1:1 split, and eight open-world scenarios.  It does not specify capture/session/group isolation, exact fold memberships, or seeds.  Input is the first eight IP packets, each contributing 80 header and 48 payload bytes, zero-padded/truncated to a 1024-byte 32x32 image.  Full evidence is in `outputs/paper/paper_protocol.csv`.

## Official Code

The released source is complete enough to recover model, score, threshold, metrics, loader, and preprocessing behavior.  It is not complete enough for author-exact reproduction because exact folds and raw-to-NPZ provenance are absent.  The source tree also exposes paper-code differences: training discriminative classification uses sampled-z squared Euclidean distance, while the paper writes a KL-derived class probability; train.py uses the released test NPZ as validation.

## USTC Reproduction

The reproduction layer is the **sibling method project** `Projects/Open-Detect`, not `unknown_traffic_project/opendetect_ustc_encoder_audit` and not `stage3_unknown_utility`.  Its authoritative final matrix records zero paper-exact reproductions, eight local approximations, one missing-data block, and one scope exclusion.

The latest USTC evidence is the v6 local-PCAP campaign: one 34,585-image pool, five independently seeded class-stratified grouped-image-disjoint 8:1:1 splits, and A-1/A-2/A-3 training on every split (15 complete checkpoints).  All five full-pool splits contain 27,667/3,457/3,461 train/validation/test images and record zero cross-split flow and exact-image overlap.  These are repeated local splits, **not the authors' exact five folds**.  The local `corrected-paper` model subclasses released Open-Detect but repairs the decoder path, uses the paper KL discriminative family, preserves optimizer linkage at prototype reset, and clamps log variance.

With a global Known-validation P95 KL threshold and balanced 1:1 binary test composition, five-repeat Accuracy/F1/AUROC are A-1 0.9785/0.9789/0.9952, A-2 0.8253/0.7980/0.8534, and A-3 0.9204/0.9167/0.9699.  The separate all-known USTC approximation records Accuracy 0.993909 and weighted F1 0.993832.  Per-run checkpoint hashes and metrics are in `outputs/ustc_reproduction/ustc_open_set_reproduction.csv`.

The earlier 0.986158 validation Accuracy / 0.987371 validation Macro-F1 record belongs to a nested audit inside `unknown_traffic_project`; it is not the sibling Open-Detect reproduction requested here and is excluded from layer C.  The sibling v6 trainer selects checkpoints by validation Accuracy and does not record validation Macro-F1, so no replacement validation Macro-F1 is invented.

## CipherSpectrum Protocol

Stage6 freezes group-aware Low/Medium/High splits.  Stage7 uses the same custom Open-Detect structural architecture through a local subclass, but adds a max-logvar=20 stability path and local early-stopping/data wrappers.  The recorded stability guard did not activate on Stage7 train/validation.  Stage8A freezes deterministic `mu_x`, thresholds, scaler/PCA, and density models using Known train/validation only.  Stage9 performs the one-shot frozen test without refitting.

## Latent Mode

- Paper: training is `SAMPLED_Z`; both inference modes are `UNCLEAR` because no inference pseudocode chooses sampled z versus mu_x.
- Official code: training is `SAMPLED_Z`; classification and unknown detection inference are `MU_X`.
- Sibling USTC v6 reproduction: training is `SAMPLED_Z`; evaluation classification uses `MU_X`, and Native detection uses `mu_x` plus `logvar_x` in the forward KL.
- CipherSpectrum: Stage7 training sampled; Stage8A/9 classification/detection uses deterministic `MU_X`.

Therefore there is **no sampled-z versus mu_x mismatch** between official inference and Stage9.

## Prototype Definition

Each Known class has one 128-dimensional mean `mu_y`.  Prototype covariance is fixed identity.  Means are trainable, then reset to deterministic training-class means at epochs 50/80; the released parameter replacement disconnects the new parameter from the original optimizer.  See `outputs/official_code/prototype_definition.md`.

## Native Score

Official Native anomaly score is:

`s_unknown(x)=min_y 0.5*(||mu_x-mu_y||^2 + sum(exp(logvar_x)-logvar_x-1))`.

It is forward `KL[q_x || N(mu_y,I)]`; it is not Euclidean-only, Mahalanobis refitting, symmetric KL, or a GMM.  Stage9 stores the negative of the same minimum KL so higher means more Known.  This sign/label reversal preserves ranking when applied consistently.  Stage9 uses the stability-clamped logvar path.

## Threshold Calibration

The paper states a global threshold chosen so 95% of overall Known validation is accepted.  It does not state per-class 95%.  Released code instead maximizes Youden J on labeled Known+Unknown test scores.  Stage9 uses a global linear P05 of Known Validation Native scores, achieving 0.94994/0.94995/0.95000 validation acceptance for Low/Medium/High.  This is paper-aligned and leakage-resistant, but not an exact reproduction of the released evaluator.

## Metric Definitions

Paper F1 averaging is not explicitly defined.  Released closed-set code uses weighted multiclass F1; released open-set code uses binary F1 with Unknown positive.  Stage9 reports Known multiclass Macro-F1 separately from UFAR and Known-positive AUROC.  Paper open-world F1 and Stage9 Known Macro-F1 are `NOT_COMPARABLE`.

## Data Split

Paper: 8:1:1, five-fold, group isolation unspecified.  Released code: prebuilt train/test NPZ, with test reused for validation and threshold selection.  The sibling USTC v6 reproduction uses five independently seeded grouped-image-disjoint 8:1:1 splits over one locally reconstructed flow-image pool; it is not capture-isolated and is not the author fold assignment.  CipherSpectrum uses a frozen group-aware split.  Cross-group generalization is a **potential explanation** for lower performance, not a causally proven explanation.

## Preprocessing

The four layers agree on the central 8x(80+48) 32x32 byte representation, IP masking, port retention in released/local code, padding, truncation, and training crop/flip.  Important uncertainty/differences remain in original flow construction, invalid/non-IP handling, direction/capture structure, and split provenance.  See `outputs/summary/preprocessing_comparison.csv`.

## Same-Encoder Detector Comparison

Native, Single-Full-K1, and Multi-Global-K2 all reuse the same Stage7 checkpoint and the same deterministic `mu_x`.  Native additionally uses `logvar_x`; K1/K2 apply the frozen train-only StandardScaler/PCA64 to `mu_x` and different frozen density scores.

| Setting | Native AUROC | K1 AUROC | K2 AUROC |
|---|---:|---:|---:|
| Low | 0.500830 | 0.880005 | 0.892480 |
| Medium | 0.497970 | 0.779753 | 0.795385 |
| High | 0.472133 | 0.804478 | 0.821250 |

This is strong evidence that score/distribution modeling, not only the encoder, explains a substantial part of the external gap.

## Failure Decomposition

| Factor | Evidence |
|---|---|
| F1 Representation limitation | EVIDENCE_MODERATE |
| F2 Native detector-score mismatch | EVIDENCE_STRONG |
| F3 Dataset/domain shift | EVIDENCE_MODERATE |
| F4 Protocol difficulty shift | EVIDENCE_MODERATE |
| F5 Implementation mismatch | EVIDENCE_STRONG |
| F6 Metric-definition mismatch | EVIDENCE_STRONG |

The strongest direct explanation is that the Native fixed-identity prototype-posterior KL score is poorly aligned/calibrated for CipherSpectrum geometry: changing only the frozen post-encoder density model raises AUROC by roughly 0.28-0.39.  Domain and group-aware protocol shifts plausibly compound this, but were not causally isolated.

## What Can Be Claimed

- The sibling USTC project has a complete 15-run v6 local-PCAP A-scenario campaign and a separate high all-known approximation; its final matrix correctly labels both as local approximations rather than paper-exact results.
- A-1 is close to the paper locally, while A-2 and A-3 show material protocol-level gaps; the local campaign does not support a blanket statement that every USTC open-world scenario was reproduced closely.
- Official inference and Stage9 both use deterministic `mu_x`; sampled-z inference mismatch is not the issue.
- Stage9 Native shares the official prototype/KL family but is not an exact released-evaluator reproduction.
- Under the frozen CipherSpectrum protocol, Native ranking is near random and K1/K2 density scores over the same `mu_x` are substantially stronger.
- The paper/code/local protocols and F1 definitions have material comparability limits.

## What Cannot Be Claimed

- It cannot be claimed that Open-Detect universally fails to generalize.
- It cannot be claimed that paper open-world F1 equals Stage9 Known Macro-F1.
- It cannot be claimed that group-aware splitting, domain shift, representation quality, or score mismatch alone is the proven cause.
- It cannot be claimed that the old nested 98.6158%/98.7371% result is the sibling Open-Detect reproduction.
- It cannot be claimed that the sibling v6 result is author-exact: author flow identities, folds and seeds are unavailable, and local source-fragment selection was partly inferred.
- It cannot be claimed that Stage9 is bit-for-bit identical to the released evaluator.

## Final Gate

**Gate B — HIGH-PRIORITY PROTOCOL MISMATCH**

`EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE`

The decisive mismatch is threshold calibration (official mixed labeled test/Youden-J versus Stage9 Known-validation/P05), compounded by the stability wrapper, split difficulty, and metric-definition differences.  No retraining or test rerun was performed.
"""
write_text(AUDIT / "AUDIT_REPORT.md", report)


readme = """# Stage 10A — Open-Detect Protocol Audit

## Goal

Audit the Open-Detect paper, released implementation, sibling `Projects/Open-Detect` USTC reproduction, and frozen CipherSpectrum Stage6-9 pipeline without training, inference, refitting, threshold tuning, or upstream changes.

## Why This Audit Was Needed

Open-Detect is strong in the paper and has substantial local USTC reproduction evidence, while its Native detector is nearly random on frozen CipherSpectrum.  Before attributing that gap, the method-project boundary, latent mode, score, prototype, threshold, split, preprocessing, and metrics must be aligned.

## Sources

The local 2025 TIFS PDF, nested official checkout, sibling Open-Detect reproduction artifacts, 15 v6 checkpoints, and frozen Stage6-9 code/results are enumerated and hashed in `sources/source_manifest.json`.  Extracted paper text is preserved in `sources/open_detect_paper.txt`.  `Projects/Open-Detect` is read-only and is a peer of `Projects/unknown_traffic_project`.

## Paper vs Code

The core Gaussian prototype and KL equations are represented in code.  Released inference deterministically uses `mu_x`.  Important differences include the sampled-z Euclidean discriminative training term, test-NPZ validation, oracle test Youden-J threshold, decoder/entropy implementation issues, and prototype optimizer disconnection after reset.

## Paper vs USTC Reproduction

Overall `PARTIAL_MATCH`: the 20-class taxonomy, target count, central preprocessing, latent dimension, scenario classes, prototype family, and paper-style Known-validation threshold largely align.  The local v6 campaign uses corrected-paper code, locally inferred flow/source reconstruction, and five repeated grouped-image-disjoint splits rather than author folds.  Its final matrix records zero paper-exact reproductions.  The old nested 98.6158%/98.7371% result is explicitly excluded from this layer.

## Paper vs CipherSpectrum

CipherSpectrum preserves the central byte image, structural encoder, deterministic inference mean, and prototype KL family, but uses a different domain, group-aware split, local numerical wrapper, validation-only threshold, and different reporting metrics.

## Key Protocol Differences

The highest-priority difference is official test-oracle Youden-J calibration versus Stage9 Known-validation P05.  Group-aware versus unspecified/random-flow splitting and under-specified F1 definitions further limit direct comparison.

## Native Score Audit

Native is minimum forward KL from `N(mu_x,diag(exp(logvar_x)))` to one `N(mu_y,I)` prototype per Known class.  Stage9 stores its negative, an equivalent orientation reversal.  It is not a GMM or symmetric KL.

## Latent Mode Audit

Paper inference is `UNCLEAR`; official and Stage9 inference are both `MU_X`.  No sampled-z/mu_x mismatch was found.

## Metric Definition Audit

Paper F1 is under-specified.  Released open-set F1 is binary Unknown-positive, while Stage9 Known Macro-F1 is multiclass.  These are `NOT_COMPARABLE`.

## Final Gate

**Gate B — HIGH-PRIORITY PROTOCOL MISMATCH**; `EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE`.

## Next Step

No next stage is launched by this audit.  Any future action requires a separate pre-registered task and must not tune against the already-opened CipherSpectrum test set.
"""
write_text(AUDIT / "README.md", readme)


print(json.dumps({
    "status": "BUILT",
    "audit_root": str(AUDIT),
    "paper_rows": len(paper_protocol),
    "official_map_rows": len(official_map),
    "ustc_rows": len(ustc_protocol),
    "stage7_rows": len(stage7_rows),
    "stage9_comparison_rows": len(same_encoder),
    "final_gate": "B",
}, ensure_ascii=False, indent=2))
