# -*- coding: utf-8 -*-
"""Stage 0 / Task 0.1+0.2: PCAP -> 双向流 -> flow_index.csv + 每流包级数据 (v2 计划 §6)

用法:
    python scripts/00_prepare_data.py                          # 全部 20 类
    python scripts/00_prepare_data.py --classes Facetime.pcap  # 只跑指定类 (烟测)

产出:
    data/flows/{split}__{class}__{stem}.pkl     每 (类,源文件) 一个: flow_id -> 包级元组列表
    outputs/stage0_data/flow_index.csv          §6.3 交付物
    outputs/stage0_data/dataset_statistics.csv  §6.3 交付物
"""
import argparse
import csv
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.flow_split import (  # noqa: E402
    discover_classes, flow_records, split_pcap, write_flows_pkl,
)


def main():
    ap = argparse.ArgumentParser(description="Stage 0: PCAP -> bidirectional flows")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "dataset", "ustc_tfc2016.yaml"))
    ap.add_argument("--classes", default=None, help="逗号分隔的类条目名 (如 Facetime.pcap,SMB), 默认全部")
    ap.add_argument("--root", default=None, help="覆盖配置里的 raw_pcap_root")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    raw_root = args.root or cfg["raw_pcap_root"]
    flows_dir = os.path.join(PROJECT_ROOT, cfg["outputs"]["flows_dir"])
    stage0_dir = os.path.join(PROJECT_ROOT, cfg["outputs"]["stage0_dir"])
    os.makedirs(flows_dir, exist_ok=True)
    os.makedirs(stage0_dir, exist_ok=True)

    classes = discover_classes(raw_root)
    partial = bool(args.classes)
    if partial:
        wanted = {c.strip().lower().rstrip(".pcap") for c in args.classes.split(",")}
        classes = [(s, c, p) for s, c, p in classes
                   if c.lower().rstrip(".pcap") in wanted]
        if not classes:
            print("no class matched:", args.classes)
            sys.exit(1)
    print(f"raw root: {raw_root}")
    print(f"classes to process: {len(classes)}{' (partial run, 只写 pkl, 不更新 stage0 交付物)' if partial else ''}")

    all_rows = []
    t0 = time.time()
    for split, class_name, pcaps in classes:
        t_class = time.time()
        for stem, path in pcaps:
            flows, skipped = split_pcap(path)
            rows, packets = flow_records(flows, split, class_name, stem)
            write_flows_pkl(packets, split, class_name, stem, skipped, flows_dir)
            all_rows.extend(rows)
            print(f"  [{split}/{class_name}/{stem}] flows={len(rows)} "
                  f"packets={sum(r['packet_count'] for r in rows)} "
                  f"skipped={skipped} ({time.time()-t_class:.1f}s)", flush=True)
        print(f"  [{split}/{class_name}] total {sum(1 for r in all_rows if r['class_name']==class_name)} flows, "
              f"{time.time()-t_class:.1f}s", flush=True)

    if partial:
        print(f"\nDONE (partial): {len(all_rows)} flows, {time.time()-t0:.1f}s; "
              f"flow_index.csv / dataset_statistics.csv 未更新")
        return

    # flow_index.csv (§6.3)
    cols = ["flow_id", "split", "class_name", "source_file",
            "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
            "packet_count", "forward_packets", "backward_packets",
            "start_time", "end_time", "duration_s", "total_bytes"]
    index_path = os.path.join(stage0_dir, "flow_index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    # dataset_statistics.csv (§6.3): 由 flow_index 汇总, 保证两个交付物自洽
    import statistics
    by_class = {}
    for r in all_rows:
        by_class.setdefault(r["class_name"], []).append(r["packet_count"])
    stat_path = os.path.join(stage0_dir, "dataset_statistics.csv")
    with open(stat_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "class_name", "flow_num", "packet_num", "avg_packets_per_flow",
            "median_packets_per_flow", "min_packets_per_flow", "max_packets_per_flow"])
        w.writeheader()
        for cname in sorted(by_class):
            lens = by_class[cname]
            w.writerow({
                "class_name": cname,
                "flow_num": len(lens),
                "packet_num": sum(lens),
                "avg_packets_per_flow": round(statistics.mean(lens), 2),
                "median_packets_per_flow": statistics.median(lens),
                "min_packets_per_flow": min(lens),
                "max_packets_per_flow": max(lens),
            })

    print(f"\nDONE: {len(all_rows)} flows total, {time.time()-t0:.1f}s")
    print("flow_index.csv ->", index_path)
    print("dataset_statistics.csv ->", stat_path)


if __name__ == "__main__":
    main()
