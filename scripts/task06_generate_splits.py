# -*- coding: utf-8 -*-
"""Stage 1 §7.2：划分 train/val/test 并切分 TrafficFormer TSV 三件套。

计划 §7.2 要求 60/20/20，或"已有官方 split 则优先官方划分"。官方代码
（finetuning_data_gen.py）的划分方式是分层 80/10/10（train_test_split，
random_state 41/42），故默认复刻官方方式；--seed 可改为单一种子。

输入：data/trafficformer_input/<policy>/flow_map.csv + trafficformer_all.tsv
输出（data/splits/<name>/）：
  train.txt / val.txt / test.txt              每行一个 flow_id
  train_dataset.tsv / valid_dataset.tsv / test_dataset.tsv   服务器训练三件套
  split_summary.json                          每类计数、行数、sha256

冒烟模式：--name smoke_gmail_zeus --classes Gmail,Zeus --max-per-class 200
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

TSV_HEADER = "label\ttext_a\n"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_flow_map(flow_map_path, classes=None, max_per_class=None, cap_seed=7):
    """返回 [(flow_id, label_id, tsv_line_number)]，按 tsv_line_number 升序。"""
    rows = []
    with flow_map_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if classes and r["class_name"] not in classes:
                continue
            rows.append((r["flow_id"], int(r["label_id"]), int(r["tsv_line_number"])))
    if max_per_class:
        rng = random.Random(cap_seed)
        by_label = {}
        for r in rows:
            by_label.setdefault(r[1], []).append(r)
        capped = []
        for label, group in by_label.items():
            rng.shuffle(group)
            capped.extend(group[:max_per_class])
        rows = sorted(capped, key=lambda r: r[2])
    else:
        rows.sort(key=lambda r: r[2])
    return rows


def split_ids(rows, seed1, seed2):
    """分层 80/10/10。返回 (train, val, test)，元素为 (flow_id, label_id)。"""
    train_r, rest_r = train_test_split(
        rows, test_size=0.2, random_state=seed1, stratify=[r[1] for r in rows])
    val_r, test_r = train_test_split(
        rest_r, test_size=0.5, random_state=seed2, stratify=[r[1] for r in rest_r])
    return (
        [(r[0], r[1]) for r in train_r],
        [(r[0], r[1]) for r in val_r],
        [(r[0], r[1]) for r in test_r],
    )


def main():
    parser = argparse.ArgumentParser(description="Stage 1 §7.2 划分 + TSV 切分")
    parser.add_argument("--project-root", default="D:/unknown_traffic_project")
    parser.add_argument("--policy", default="compatible_min1",
                        choices=["compatible_min1", "strict_min3"])
    parser.add_argument("--name", default=None, help="输出目录名，默认=data/splits/<policy>")
    parser.add_argument("--seed", type=int, default=None,
                        help="单一随机种子；默认 None = 官方 random_state 41/42")
    parser.add_argument("--classes", default=None,
                        help="逗号分隔类名（冒烟模式用），默认全部")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="每类最多流数（冒烟模式用）")
    args = parser.parse_args()

    root = Path(args.project_root)
    tf_dir = root / "data" / "trafficformer_input" / args.policy
    out_dir = root / "data" / "splits" / (args.name or args.policy)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed1 = seed2 = args.seed if args.seed else None
    if not seed1:
        seed1, seed2 = 41, 42  # 官方代码的 random_state

    classes = set(args.classes.split(",")) if args.classes else None
    rows = load_flow_map(tf_dir / "flow_map.csv", classes, args.max_per_class)
    assert len(rows) == len({r[0] for r in rows}), "flow_id 有重复"

    train, val, test = split_ids(rows, seed1, seed2)
    parts = {"train": train, "val": val, "test": test}

    # 1) flow_id 列表
    for name, part in parts.items():
        (out_dir / f"{name}.txt").write_text(
            "\n".join(fid for fid, _ in part) + "\n", encoding="utf-8")

    # 2) 切 TSV 三件套（按行号一次扫描）
    line_to_split = {}
    for name, part in parts.items():
        for fid, _ in part:
            line_to_split[fid] = name
    fid_by_line = {r[2]: r[0] for r in rows}
    out_files = {
        name: (out_dir / {"train": "train_dataset.tsv", "val": "valid_dataset.tsv",
                          "test": "test_dataset.tsv"}[name]).open("w", encoding="utf-8")
        for name in parts
    }
    for fh in out_files.values():
        fh.write(TSV_HEADER)
    written = Counter()
    with (tf_dir / "trafficformer_all.tsv").open(encoding="utf-8") as fh:
        next(fh)  # header
        for line_no, line in enumerate(fh, start=2):
            fid = fid_by_line.get(line_no)
            if fid is not None:
                out_files[line_to_split[fid]].write(line)
                written[line_to_split[fid]] += 1
    for fh in out_files.values():
        fh.close()

    # 3) summary + 校验
    summary = {
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": args.policy,
        "seed1": seed1, "seed2": seed2,
        "classes_filter": sorted(classes) if classes else "all",
        "max_per_class": args.max_per_class,
        "total_flows": len(rows),
        "per_split": {
            name: {
                "flows": len(part),
                "per_class": {str(k): v for k, v in sorted(Counter(l for _, l in part).items())},
            }
            for name, part in parts.items()
        },
        "tsv_lines_written": dict(written),
    }
    for name, part in parts.items():
        fname = {"train": "train_dataset.tsv", "val": "valid_dataset.tsv",
                 "test": "test_dataset.tsv"}[name]
        p = out_dir / fname
        summary["per_split"][name]["tsv_lines"] = sum(1 for _ in p.open(encoding="utf-8")) - 1
        summary["per_split"][name]["tsv_sha256"] = sha256_file(p)
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # 一致性校验
    all_ids = {fid for name in parts for fid, _ in parts[name]}
    assert all_ids == {r[0] for r in rows}, "split 并集 != 输入集合"
    for name in parts:
        assert written[name] == len(parts[name]) == summary["per_split"][name]["tsv_lines"], name
    for name, part in parts.items():
        ratio = len(part) / len(rows)
        print(f"{name}: {len(part)} flows ({ratio:.1%}), "
              f"classes={len({l for _, l in part})}")
    print(f"out: {out_dir}")


if __name__ == "__main__":
    main()
