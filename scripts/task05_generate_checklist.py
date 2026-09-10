# -*- coding: utf-8 -*-
"""Task 0.5: 生成人工一致性抽查清单（≥50 流，分层随机）。

计划要求（v2 §6.2 Task 0.5）：随机抽取至少 50 条 flow，人工检查
TrafficFormer 与 FIG 是否来自同一 flow、标签、packet 数量、packet 顺序、
timestamp、direction、空流/单包流处理。

抽样方案：固定随机种子，每类 3 条（20 类 = 60 条），再补齐节点数分桶
{1,2,3-4,5-10,11-20,21-30} 的强制覆盖（每桶至少 2 条）。Stage 0 无空流
（packet_count>=1 已全量验证），故单包流即最小图，列入清单。

输出（写到 --output-dir，默认 D:\\unknown_traffic_project\\outputs\\stage0_data）：
  manual_checklist_50.csv    抽查表（UTF-8 BOM，Excel 可直接打开）
  manual_checklist_50_说明.md  核对方法与判定标准
"""
import argparse
import csv
import json
import pickle
import random
from pathlib import Path

SEED = 20260901
NODE_BUCKETS = ((1, "1"), (2, "2"), (4, "3-4"), (10, "5-10"), (20, "11-20"), (30, "21-30"))


def _bucket(node_count: int) -> str:
    for upper, label in NODE_BUCKETS:
        if node_count <= upper:
            return label
    return "21-30"


def main():
    parser = argparse.ArgumentParser(description="Task 0.5 人工抽查清单生成")
    parser.add_argument(
        "--project-root",
        default="D:/unknown_traffic_project",
        help="项目根目录（含 data/ 与 outputs/）",
    )
    parser.add_argument("--output-dir", default=None, help="清单输出目录")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--min-per-bucket", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.project_root)
    fig_dir = root / "data" / "fig_graph" / "all_flows"
    tf_dir = root / "data" / "trafficformer_input" / "compatible_min1"
    flows_dir = root / "data" / "flows"
    output_dir = Path(args.output_dir) if args.output_dir else root / "outputs" / "stage0_data"

    # 索引：fig_index（含 byte_offset，可随机访问 jsonl）
    with (fig_dir / "fig_index.csv").open(newline="", encoding="utf-8") as fh:
        fig_rows = list(csv.DictReader(fh))
    fig_by_id = {r["flow_id"]: r for r in fig_rows}
    # TF flow_map（含 tsv_line_number）
    with (tf_dir / "flow_map.csv").open(newline="", encoding="utf-8") as fh:
        tf_by_id = {r["flow_id"]: r for r in csv.DictReader(fh)}

    # 分层抽样
    random.seed(args.seed)
    by_class = {}
    for r in fig_rows:
        by_class.setdefault(r["class_name"], []).append(r)
    sampled_ids = set()
    for class_name in sorted(by_class):
        for r in random.sample(by_class[class_name], min(args.per_class, len(by_class[class_name]))):
            sampled_ids.add(r["flow_id"])
    # 节点数分桶强制覆盖
    by_bucket = {}
    for r in fig_rows:
        by_bucket.setdefault(_bucket(int(r["node_count"])), []).append(r["flow_id"])
    for label in ("1", "2", "3-4", "5-10", "11-20", "21-30"):
        if len(sampled_ids & set(by_bucket[label])) < args.min_per_bucket:
            for fid in random.sample(by_bucket[label], args.min_per_bucket):
                sampled_ids.add(fid)

    # 需要读取的 TSV 行（一次顺序扫描，取需要的行号）
    needed_lines = {int(tf_by_id[fid]["tsv_line_number"]): fid for fid in sampled_ids}
    max_line = max(needed_lines)
    tsv_text = {}
    with (tf_dir / "trafficformer_all.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line_no, line in enumerate(fh, start=2):
            if line_no in needed_lines:
                tsv_text[needed_lines[line_no]] = line.rstrip("\n").split("\t", 1)[1]
            if line_no >= max_line:
                break

    pkl_cache = {}
    jsonl_path = fig_dir / "fig_all.jsonl"

    def pkl_packets(source_pkl: str):
        if source_pkl not in pkl_cache:
            with (flows_dir / source_pkl).open("rb") as fh:
                pkl_cache[source_pkl] = pickle.load(fh)["packets"]
        return pkl_cache[source_pkl]

    def fmt_features(feats):
        parts = []
        for f in feats:
            parts.append(
                "(%s,%s,%s|%s,%s,%s,%s)" % tuple(
                    (int(v) if float(v).is_integer() else round(float(v), 4)) for v in f
                )
            )
        return ";".join(parts)

    rows = []
    for fid in sorted(sampled_ids, key=lambda x: (x.split("__")[1], int(fig_by_id[x]["node_count"]), x)):
        fig = fig_by_id[fid]
        tf = tf_by_id[fid]
        packets = pkl_packets(fig["source_pkl"])[fid]
        pkt_dirs = "".join("+" if p[3] == 1 else "-" for p in packets)
        pkt_dirs_short = pkt_dirs if len(pkt_dirs) <= 40 else pkt_dirs[:40] + f"…(共{len(pkt_dirs)}包)"
        rel_ts = [round(p[0] - packets[0][0], 6) for p in packets[:6]]
        text = tsv_text[fid]
        sep_count = text.count("[SEP]")
        preview = text[:48] + ("…" if len(text) > 48 else "")

        with jsonl_path.open(encoding="utf-8") as fh:
            fh.seek(int(fig["byte_offset"]))
            rec = json.loads(fh.readline())
        dirs = "".join("+" if f[0] > 0 else "-" for f in rec["features"])
        dirs_short = dirs if len(dirs) <= 30 else dirs[:30] + "…"
        if len(rec["edges"]) <= 6:
            edges_preview = ";".join(f"{a}-{b}" for a, b in rec["edges"]) or "(无边)"
        else:
            edges_preview = ";".join(f"{a}-{b}" for a, b in rec["edges"][:6]) + f"…(共{len(rec['edges'])}条)"
        first_feats = rec["features"][:2]
        rows.append({
            "seq": len(rows) + 1,
            "flow_id": fid,
            "class_name": fig["class_name"],
            "label_id": fig["label_id"],
            "source_pkl": fig["source_pkl"],
            "pkl_packet_count": len(packets),
            "pkl_direction_seq": pkt_dirs_short,
            "pkl_first6_rel_ts": ",".join(str(t) for t in rel_ts),
            "tf_tsv_line": tf["tsv_line_number"],
            "tf_used_packets": tf["used_packet_count"],
            "tf_sep_count": sep_count,
            "tf_text_preview": preview,
            "fig_node_count": rec["node_count"],
            "fig_burst_count": rec["burst_count"],
            "fig_edge_count": len(rec["edges"]),
            "fig_direction_seq": dirs_short,
            "fig_first2_node_features": fmt_features(first_feats),
            "fig_edges_preview": edges_preview,
            "check_same_flow": "",
            "check_label": "",
            "check_packet_count": "",
            "check_order": "",
            "check_timestamp": "",
            "check_direction": "",
            "note": "",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "manual_checklist_50.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    bucket_counts = {}
    for r in rows:
        bucket_counts[_bucket(int(r["fig_node_count"]))] = bucket_counts.get(
            _bucket(int(r["fig_node_count"])), 0
        ) + 1
    note = (
        f"# Task 0.5 人工抽查清单使用说明\n\n"
        f"- 抽样：随机种子 {args.seed}，每类 {args.per_class} 条 + 节点数分桶强制覆盖，"
        f"共 **{len(rows)} 条**（计划要求 ≥50）。\n"
        f"- 节点数分桶覆盖：{bucket_counts}\n"
        f"- Stage 0 无空流（packet_count≥1 全量已验证），单包流=单节点 0 边图，已包含在清单中。\n\n"
        f"## 核对方法（每行逐项勾选，结论填 ✓ 或 ✗，异常写进 note）\n\n"
        f"1. **同一 flow**：flow_id 在三个文件里指向同一条流（清单行已按 flow_id 汇齐，"
        f"抽查确认即可）。\n"
        f"2. **标签**：class_name / label_id 与 flow_id 中嵌的类名一致。\n"
        f"3. **包数**：pkl_packet_count == tf 使用的包数上限（used_packets=min(5,包数)）"
        f"== fig_node_count=min(30,包数)；tf_sep_count 应等于 tf_used_packets。\n"
        f"4. **包序/时间戳**：pkl_first6_rel_ts 非降；fig 特征第 3 位 ts_delta 与之相同。\n"
        f"5. **方向**：pkl_direction_seq 前 30 位 == fig_direction_seq；burst 切分规则："
        f"同方向连续段为一个 burst，burst 内链式连边、burst 间首连首尾连尾。\n"
        f"6. **单包流**：1 节点、0 边、首 burst 比率特征为 0（fill）。\n\n"
        f"完成后把勾选结果（CSV 或截图）交给报告撰写方，汇总生成 data_validation_report.md。\n"
    )
    (output_dir / "manual_checklist_50_说明.md").write_text(note, encoding="utf-8")
    print(f"checklist rows: {len(rows)}")
    print(f"bucket coverage: {bucket_counts}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
