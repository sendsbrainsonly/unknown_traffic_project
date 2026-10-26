#!/usr/bin/env python3
"""Render Figure 4-1 from the values in Table 4-2 of 4(1).md."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2474A6"
GRAY = "#969696"
ORANGE = "#E87522"


def read_data(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise ValueError(f"Expected eight configurations, found {len(rows)}")
    return rows


def render(data_path: Path, output_dir: Path) -> None:
    rows = read_data(data_path)
    labels = [row["method"] for row in rows]
    y = np.arange(len(rows))
    colors = [GRAY if "E3" in label else ORANGE if "ours" in label else BLUE for label in labels]
    hatches = ["" if "ours" not in label else "///" for label in labels]

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = [
        (axes[0], "vpn_macro_f1_mean", "vpn_macro_f1_sd", "(a) ISCX-VPN / 6 classes"),
        (axes[1], "tor_macro_f1_mean", "tor_macro_f1_sd", "(b) ISCX-Tor / 7 classes"),
    ]
    for ax, mean_key, sd_key, title in panels:
        means = np.array([float(row[mean_key]) for row in rows])
        errors = np.array([float(row[sd_key]) for row in rows])
        bars = ax.barh(y, means, xerr=errors, height=0.64, color=colors,
                       edgecolor="white", linewidth=0.5,
                       error_kw={"ecolor": "#252525", "elinewidth": 1.2, "capsize": 3})
        for bar, hatch in zip(bars, hatches, strict=True):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("#783B16")
                bar.set_linewidth(0.8)
        for yi, mean, error in zip(y, means, errors, strict=True):
            ax.text(mean + error + 0.7, yi, f"{mean:.2f}", va="center", ha="left", fontsize=8.5)
        ax.set_title(title, pad=12)
        ax.set_xlim(0, 101)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel("Macro-F1 (%) ↑")
        ax.grid(axis="x", color="#D8DDE2", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)

    axes[0].set_yticks(y, labels)
    axes[1].tick_params(labelleft=False)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.25, right=0.985, bottom=0.12, top=0.92)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig4_1_closed_baselines"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white",
                metadata={"Title": "Figure 4-1. Closed-set comparison under original service labels",
                          "Description": "Eight configurations; VPN six classes and Tor seven classes; bars show two-seed Macro-F1 means with population standard deviation."})
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("plot_data.csv"))
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    render(args.data, args.outdir)


if __name__ == "__main__":
    main()
