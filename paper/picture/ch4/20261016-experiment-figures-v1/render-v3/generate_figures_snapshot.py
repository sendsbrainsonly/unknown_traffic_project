#!/usr/bin/env python3
"""Create chapter-4 figures from retained results; never overwrite a render."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ch4-mpl-cache-01a144b7"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image, ImageOps, ImageDraw

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "stage44_vnat_coarse_service_loso_open_set").is_dir())
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#829099"
FOLDS = ["communication", "file_transfer", "remote_access", "streaming"]
FOLD_LABELS = ["Communication", "File transfer", "Remote access", "Streaming"]
SOURCES: dict[str, dict] = {}
RECORDS: list[dict] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    SOURCES[relative] = {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}
    return path


def md_tables(relative: str) -> list[list[list[str]]]:
    text = source(relative).read_text(encoding="utf-8")
    tables, current = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            current.append([v.strip().replace("**", "") for v in line.strip("|").split("|")])
        elif current:
            tables.append(current)
            current = []
    return tables


def csv_rows(relative: str) -> list[dict]:
    with source(relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def add(figure, setting, method, metric, value, path, locator, sd=None, **context):
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError((figure, setting, method, metric, value))
    RECORDS.append({"figure": figure, "setting": setting, "method": method,
                    "metric": metric, "value": value, "sd": sd,
                    "source_path": path, "source_locator": locator, **context})


def extract_data():
    p = "stage23b_tfe_trident_same_flow_adaptation/stage23b_all_methods_comparison.md"
    tables = md_tables(p)
    for dataset, table in zip(["VPN6", "Tor7"], tables[:2], strict=True):
        for row in table[2:]:
            mean, sd = map(float, row[6].split("±"))
            add("fig4_1", dataset, row[0], "macro_f1", mean, p,
                f"{dataset}:Macro-F1 mean±populationSD:{row[0]}", sd=sd,
                seeds=[2022, 2023], n_test=1093 if dataset == "VPN6" else 1117)

    p = "stage32_three_dataset_coarse_single_seed/stage32_report.md"
    for row in md_tables(p)[1][2:]:
        if row[1] != "known_test":
            continue
        add("fig4_2", row[0], row[2], "macro_f1", row[3], p,
            f"same-flow paired:{row[0]}:{row[2]}", seeds=[2022])
        if row[2] == "E3_hard_remap":
            add("fig4_2", row[0], "three_view", "macro_f1", row[4], p,
                f"same-flow paired:{row[0]}:T0", seeds=[2022])

    p = "stage44b_vnat_matched_opendetect/matched_comparison.csv"
    for row in csv_rows(p):
        if row["view"] != "natural" or row["threshold_percentile"] != "95":
            continue
        for method in ["od", "three_view_des_v1"]:
            for metric in ["auroc", "binary_f1", "ufar", "known_frr"]:
                add("fig4_3", row["fold"], method, metric, row[f"{method}_{metric}"],
                    p, f"{row['fold']}:natural:P95:{method}_{metric}",
                    n_known=int(row["known_samples"]), n_unknown=int(row["unknown_samples"]),
                    seeds=[2022], threshold="Known-Val P95")

    p = "stage44_vnat_coarse_service_loso_open_set/stage44_run_results.csv"
    for row in csv_rows(p):
        if row["view"] != "natural" or row["threshold_percentile"] != "95":
            continue
        for metric in ["auroc", "ufar", "known_frr"]:
            add("fig4_4", row["fold"], row["method"], metric, row[metric], p,
                f"{row['fold']}:{row['method']}:natural:P95:{metric}", seeds=[2022])

    cases = [
        ("DoS slowloris", "stage42s_cic_favorable_open_set/RESULTS.md", 3, 5709),
        ("DoS GoldenEye", "stage42t_cic_goldeneye_open_set/RESULTS.md", 3, 7441),
        ("DoS Hulk", "stage42u_cic_hulk_open_set/RESULTS.md", 4, 155168),
        ("FTP-Patator", "stage42x_cic_ftp_patator_open_set/RESULTS.md", 4, 3985),
        ("DDoS", "stage42v_cic_ddos_open_set/RESULTS.md", 4, 76613),
        ("Bot", "stage42w_cic_bot_open_set/RESULTS.md", 5, 1228),
    ]
    for attack, p, ufar_column, count in cases:
        rows = md_tables(p)[0]
        for method in ["MSP", "DES-v1"]:
            row = next(r for r in rows if r[0] == method)
            add("fig4_5", attack, method, "auroc", row[1], p,
                f"natural:{method}:AUROC", n_unknown=count, n_known=2272)
            if method == "DES-v1":
                add("fig4_5", attack, method, "ufar", row[ufar_column], p,
                    "natural:DES-v1:UFAR", n_unknown=count, n_known=2272)
    p = "stage42y_cic_remaining_four_open_set/RESULTS.md"
    for row in md_tables(p)[0][2:]:
        for method, metric, column in [("MSP", "auroc", 2), ("DES-v1", "auroc", 3),
                                       ("DES-v1", "ufar", 5)]:
            add("fig4_5", row[0], method, metric, row[column], p,
                f"natural:{row[0]}:{method}:{metric}",
                n_unknown=int(row[1].replace(",", "")), n_known=2272)

    p = "stage43_cic_mixed_prevalence/stage43_mixed_open_set_report.md"
    tables = md_tables(p)
    for row in tables[0][2:]:
        for metric, column in [("auroc", 1), ("auprc", 2), ("ufar", 3), ("known_frr", 4)]:
            add("fig4_6", row[0], "DES-v1", metric, row[column], p,
                f"prevalence:{row[0]}:{metric}", sampling_repeats=20)
    for row in tables[1][2:]:
        if row[0] in ["all_balanced", "authentication_web", "previously_hard"]:
            add("fig4_6", row[0], "DES-v1", "ufar", row[3], p,
                f"composition:{row[0]}:UFAR", sampling_repeats=20)

    p = "stage41_a123_matched_open_set_comparison/comparison_report.md"
    for row in md_tables(p)[0][2:]:
        for method, columns in [("Open-Detect", [3, 6, 9]), ("Three-view DES-v1", [4, 7, 10])]:
            for metric, column in zip(["auroc", "binary_f1", "ufar"], columns, strict=True):
                add("fig4_7", row[0], method, metric, row[column], p,
                    f"{row[0]}:balanced1to1:P95:{method}:{metric}", seeds=[2022])


def record(figure, setting, method, metric):
    rows = [r for r in RECORDS if (r["figure"], r["setting"], r["method"], r["metric"])
            == (figure, setting, method, metric)]
    if len(rows) != 1:
        raise ValueError((figure, setting, method, metric, len(rows)))
    return rows[0]


def vals(figure, settings, method, metric):
    return np.array([record(figure, s, method, metric)["value"] * 100 for s in settings])


def pct_label(value):
    """Match manuscript half-up rounding without binary midpoint artifacts."""
    return str(Decimal(str(round(float(value), 10))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
                         "legend.fontsize": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.7,
                         "grid.color": "#DDE2E6", "grid.linewidth": 0.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "savefig.facecolor": "white", "figure.facecolor": "white"})


def base(ax, xlabel, xmax=105):
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, xmax)
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def hbars(ax, values, labels, colors, xlabel, errors=None):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=.57, xerr=errors,
            error_kw={"elinewidth": .8, "capsize": 2.5, "ecolor": "#17212A"})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    base(ax, xlabel)
    for i, (yi, value) in enumerate(zip(y, values, strict=True)):
        offset = errors[i] if errors is not None else 0
        ax.text(value + offset + 1.2, yi, pct_label(value), va="center", fontsize=8)


def pair_hbars(ax, settings, left, right, metric, xlabel, xmax=109, legend=False):
    y = np.arange(len(settings)); h = .31
    for offset, data, color, hatch, label in [(-.18, left, BLUE, "", "Open-Detect"),
                                             (.18, right, ORANGE, "///", "Three-view DES-v1")]:
        bars = ax.barh(y + offset, data, height=h, color=color, label=label,
                       hatch=hatch, edgecolor="white", linewidth=.35)
        for b, value in zip(bars, data, strict=True):
            ax.text(value + xmax * .012, b.get_y() + b.get_height()/2,
                    pct_label(value), va="center", fontsize=7.4)
    ax.set_yticks(y, settings); ax.invert_yaxis(); base(ax, xlabel, xmax)
    if legend:
        ax.legend(loc="lower right", frameon=False)


def figures():
    result = []
    order = ["TrafficFormer-pretrained", "YaTC-official-pretrained-Stage20",
             "Open-Detect corrected-paper", "RoNeTC-runnable-reconstruction",
             "TFE-GNN-8-UDP-short", "Trident-early8-86D", "OURS-E3-T8-pretrained"]
    labels = ["TrafficFormer", "YaTC", "Open-Detect (CP)", "RoNeTC (recon.)",
              "TFE-GNN (adapt.)", "Trident (adapt.)", "E3 (two-view)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    for ax, setting, panel in zip(axes, ["VPN6", "Tor7"], ["(a) ISCX-VPN / 6 classes", "(b) ISCX-Tor / 7 classes"], strict=True):
        data = [record("fig4_1", setting, m, "macro_f1") for m in order]
        hbars(ax, [r["value"]*100 for r in data], labels, BLUE,
              panel + "\nMacro-F1 (%) ↑", [r["sd"]*100 for r in data])
    result.append(("fig4_1_closed_baselines", fig))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), layout="constrained")
    order = ["E3_hard_remap", "E3_coarse_head", "YaTC_hard_remap_context", "three_view"]
    labels = ["E3 / label remap", "E3 / coarse head", "YaTC / label remap", "Three-view / coarse head"]
    for ax, setting, panel in zip(axes, ["iscx_vpn", "iscx_tor"], ["(a) ISCX-VPN / 4 classes", "(b) ISCX-Tor / 5 classes"], strict=True):
        data = [record("fig4_2", setting, m, "macro_f1")["value"]*100 for m in order]
        hbars(ax, data, labels, [GRAY, GRAY, BLUE, ORANGE], panel + "\nMacro-F1 (%) ↑")
    result.append(("fig4_2_current_closed", fig))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.1), layout="constrained")
    for ax, metric, label, xmax in zip(axes.flat,
            ["auroc", "binary_f1", "ufar", "known_frr"],
            ["(a) AUROC (%) ↑", "(b) Unknown F1 (%) ↑", "(c) UFAR (%) ↓", "(d) Known FRR (%) ↓"],
            [110, 110, 112, 12.5], strict=True):
        pair_hbars(ax, FOLD_LABELS, vals("fig4_3", FOLDS, "od", metric),
                   vals("fig4_3", FOLDS, "three_view_des_v1", metric), metric, label, xmax)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    result.append(("fig4_3_vnat_matched", fig))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), layout="constrained")
    methods = ["msp", "energy", "centroid", "des_v1"]
    for ax, metric, label, low, high, cmap in zip(axes,
            ["auroc", "ufar", "known_frr"],
            ["(a) AUROC (%) ↑", "(b) UFAR (%) ↓", "(c) Known FRR (%) ↓"],
            [70, 0, 0], [100, 40, 10], ["Blues", "Blues_r", "Blues_r"], strict=True):
        data = np.array([vals("fig4_4", FOLDS, m, metric) for m in methods])
        im = ax.imshow(data, cmap=cmap, vmin=low, vmax=high, aspect="auto")
        ax.set_xticks(range(4), ["Comm.", "File\ntransfer", "Remote\naccess", "Streaming"], fontsize=8)
        ax.set_yticks(range(4), ["MSP", "Energy", "DES-v0", "DES-v1"])
        ax.set_xlabel(label)
        ax.tick_params(length=0)
        for (i, j), value in np.ndenumerate(data):
            rgb = im.cmap(im.norm(value))[:3]
            luminance = sum(a*b for a,b in zip(rgb,[.2126,.7152,.0722]))
            ax.text(j, i, pct_label(value), ha="center", va="center", fontsize=8.4,
                    color="white" if luminance < .55 else "#17212A")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=.07, fraction=.07)
    result.append(("fig4_4_vnat_ablation", fig))

    attacks = ["DoS slowloris", "DoS GoldenEye", "DoS Hulk", "FTP-Patator", "SSH-Patator",
               "Web Attack - Brute Force", "Web Attack - XSS", "DoS Slowhttptest", "DDoS", "Bot"]
    labels = ["DoS slowloris", "DoS GoldenEye", "DoS Hulk", "FTP-Patator", "SSH-Patator",
              "Web Brute Force", "Web XSS", "DoS Slowhttptest", "DDoS", "Bot"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), layout="constrained")
    y = np.arange(10)
    left = vals("fig4_5", attacks, "MSP", "auroc")
    right = vals("fig4_5", attacks, "DES-v1", "auroc")
    axes[0].hlines(y, left, right, color="#BFC8CE", linewidth=1.5)
    axes[0].scatter(left, y, color=BLUE, s=28, marker="o", label="MSP", zorder=3)
    axes[0].scatter(right, y, color=ORANGE, s=32, marker="s", label="DES-v1", zorder=3)
    axes[0].set_yticks(y, labels); axes[0].set_ylim(9.6, -.6)
    axes[0].axvline(50, color="#9CA6AD", linestyle=":", linewidth=.8)
    base(axes[0], "(a) AUROC (%) ↑", 104)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    errors = vals("fig4_5", attacks, "DES-v1", "ufar")
    axes[1].barh(y, errors, color=ORANGE, height=.56)
    for yi, value in zip(y, errors, strict=True):
        axes[1].text(value+1.0, yi, pct_label(value), va="center", fontsize=8.2)
    axes[1].set_yticks(y, [f"n = {record('fig4_5', a, 'DES-v1', 'ufar')['n_unknown']:,}" for a in attacks])
    axes[1].set_ylim(9.6, -.6); base(axes[1], "(b) DES-v1 UFAR (%) ↓", 60)
    result.append(("fig4_5_cic_attack_profiles", fig))

    settings = ["known9_unknown1", "known3_unknown1", "known1_unknown1", "known1_unknown3", "known1_unknown9"]
    prevalence = [10, 25, 50, 75, 90]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), layout="constrained")
    for metric, color, marker, label in [("auroc", BLUE, "o", "AUROC"), ("auprc", ORANGE, "s", "AUPRC")]:
        axes[0].plot(prevalence, vals("fig4_6", settings, "DES-v1", metric), color=color, marker=marker, markersize=4, label=label)
    axes[0].set_ylim(80, 101); axes[0].set_ylabel("Score (%) ↑")
    axes[0].set_xlabel("(a) Unknown prevalence (%)")
    for metric, color, marker, label in [("ufar", ORANGE, "s", "UFAR"), ("known_frr", BLUE, "o", "Known FRR")]:
        axes[1].plot(prevalence, vals("fig4_6", settings, "DES-v1", metric), color=color, marker=marker, markersize=4, label=label)
    axes[1].set_ylim(0, 10); axes[1].set_ylabel("Error rate (%) ↓")
    axes[1].set_xlabel("(b) Unknown prevalence (%)")
    for ax in axes[:2]:
        ax.set_xticks(prevalence); ax.grid(); ax.legend(frameon=False, loc="lower left")
    composition = ["all_balanced", "authentication_web", "previously_hard"]
    errors = vals("fig4_6", composition, "DES-v1", "ufar")
    bars = axes[2].bar(range(3), errors, color=ORANGE, width=.55)
    axes[2].set_xticks(range(3), ["All 10\nbalanced", "Auth. +\nWeb", "Bot +\nDDoS"])
    axes[2].set_ylim(0, 40); axes[2].set_ylabel("UFAR (%) ↓")
    axes[2].set_xlabel("(c) Unknown composition")
    axes[2].set_axisbelow(True); axes[2].grid(axis="y")
    for bar, value in zip(bars, errors, strict=True):
        axes[2].text(bar.get_x()+bar.get_width()/2, value+.7, pct_label(value), ha="center", fontsize=8)
    result.append(("fig4_6_cic_mixture_sensitivity", fig))

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3), layout="constrained")
    scenarios = ["A-1", "A-2", "A-3"]; x = np.arange(3)
    for ax, metric, label in zip(axes, ["auroc", "binary_f1", "ufar"],
                                ["(a) AUROC (%) ↑", "(b) Unknown F1 (%) ↑", "(c) UFAR (%) ↓"], strict=True):
        for shift, method, color, hatch in [(-.18, "Open-Detect", BLUE, ""), (.18, "Three-view DES-v1", ORANGE, "///")]:
            values = vals("fig4_7", scenarios, method, metric)
            bars = ax.bar(x+shift, values, width=.34, color=color, label=method,
                          hatch=hatch, edgecolor="white", linewidth=.35)
            for bar, value in zip(bars, values, strict=True):
                ax.text(bar.get_x()+bar.get_width()/2, value+1.2, pct_label(value),
                        ha="center", va="bottom", fontsize=7.3, rotation=90)
        ax.set_xticks(x, ["A-1\n85 + 85", "A-2\n558 + 558", "A-3\n1,031 + 1,031"])
        ax.set_ylim(0, 116); ax.set_yticks([0,25,50,75,100]); ax.set_xlabel(label)
        ax.set_axisbelow(True); ax.grid(axis="y")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    result.append(("fig4_7_ustc_matched", fig))
    return result


def save_figures(out):
    export_rows = []
    for name, fig in figures():
        fig.canvas.draw()
        for ext in ["pdf", "svg", "png"]:
            target = out / f"{name}.{ext}"
            fig.savefig(target, dpi=300, bbox_inches="tight", pad_inches=.08)
            export_rows.append({"path": target.name, "sha256": sha(target), "bytes": target.stat().st_size})
        plt.close(fig)
    return export_rows


def overview(out):
    files = sorted(out.glob("fig4_*.png"))
    canvas = Image.new("RGB", (1800, 4*650), "#F1F3F5")
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(files):
        image = Image.open(p).convert("RGB")
        image.thumbnail((880, 600), Image.Resampling.LANCZOS)
        x, y = (i%2)*900, (i//2)*650
        canvas.paste(image, (x+(900-image.width)//2, y+35+(600-image.height)//2))
        draw.text((x+20, y+12), p.stem, fill="#17212A")
    canvas.save(out / "overview.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="New render directory; must not exist")
    parser.add_argument("--data", type=Path, help="Optional frozen plot_data.json for rerendering")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.out / "generate_figures_snapshot.py")
    try:
        if args.data:
            payload = json.loads(args.data.read_text())
            RECORDS.extend(payload["records"]); SOURCES.update(payload["sources"])
        else:
            extract_data()
        keys = [(r["figure"], r["setting"], r["method"], r["metric"]) for r in RECORDS]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate plot records")
        expected = {"fig4_1":14, "fig4_2":8, "fig4_3":32, "fig4_4":48,
                    "fig4_5":30, "fig4_6":23, "fig4_7":18}
        counts = {f:sum(r["figure"]==f for r in RECORDS) for f in expected}
        if counts != expected:
            raise ValueError((counts, expected))
        payload = {"schema_version":1, "unit":"fraction; plots display percent",
                   "records":RECORDS, "sources":SOURCES}
        (args.out / "plot_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
        fields = ["figure","setting","method","metric","value","sd","source_path","source_locator"]
        with (args.out / "plot_data.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(RECORDS)
        style(); exports = save_figures(args.out); overview(args.out)
        source_checks = {p:sha(ROOT/p)==s["sha256"] for p,s in SOURCES.items()}
        if not all(source_checks.values()):
            raise ValueError("Source hash changed during render")
        manifest = {"status":"RENDERED_AWAITING_VISUAL_REVIEW", "created_utc":datetime.now(timezone.utc).isoformat(),
                    "python":platform.python_version(), "matplotlib":matplotlib.__version__,
                    "numpy":np.__version__, "figure_count":7, "numeric_records":len(RECORDS),
                    "records_per_figure":counts, "source_hash_checks":source_checks,
                    "script_sha256":sha(Path(__file__)), "outputs":exports,
                    "limits":["No new model evaluations", "Report-precision values where CSVs absent",
                              "No uncertainty bars for single-seed results", "No synthetic ROC/PR curves"]}
        (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({"status":manifest["status"],"figures":7,"exports":len(exports),
                          "records":len(RECORDS),"out":str(args.out)}, ensure_ascii=False))
    except Exception:
        (args.out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
