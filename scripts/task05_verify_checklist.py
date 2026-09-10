# -*- coding: utf-8 -*-
"""Task 0.5: 对抽查清单的 60 条流做 7 项程序化核对并回填勾选结果。

计划 §6.2 Task 0.5 要求人工检查 7 项。本脚本按同一 7 项对清单中每条流直接
比对原始 pkl、TrafficFormer TSV、FIG JSONL 三侧数据（不信任清单里预填的
展示值），逐项给出 ✓/✗ 并写回：

  manual_checklist_50_checked.csv

核对项：
  1 check_same_flow     flow_id 在三侧均存在且指向同一条流
  2 check_label         标签一致（TSV 行首 label 列 == TF/FIG label_id == flow_id 内嵌类名）
  3 check_packet_count  包数一致（pkl == TF packet_count == FIG packet_count）
  4 check_order         包序（pkl 存储序按 ts 非降；FIG 前 5 包方向序 == pkl 前 5 包）
  5 check_timestamp     ts 非降；FIG ts_delta == ts - ts0（6 位小数取整）
  6 check_direction     FIG 方向序列 == pkl 方向序列（发起者 +1）
  7 check_edge_cases    单包流 = 1 节点 0 边、首 burst 比率 fill；无空流

结果全部通过后由报告脚本（或人工）汇总写 data_validation_report.md。
"""
import argparse
import csv
import json
import pickle
from pathlib import Path


def _round6(value):
    return round(float(value), 6)


def main():
    parser = argparse.ArgumentParser(description="Task 0.5 清单核对")
    parser.add_argument("--project-root", default="D:/unknown_traffic_project")
    parser.add_argument("--checklist", default=None, help="默认 outputs/stage0_data/manual_checklist_50.csv")
    parser.add_argument("--out", default=None, help="默认同目录 manual_checklist_50_checked.csv")
    args = parser.parse_args()

    root = Path(args.project_root)
    fig_dir = root / "data" / "fig_graph" / "all_flows"
    tf_dir = root / "data" / "trafficformer_input" / "compatible_min1"
    flows_dir = root / "data" / "flows"
    stage_dir = root / "outputs" / "stage0_data"
    checklist_path = Path(args.checklist) if args.checklist else stage_dir / "manual_checklist_50.csv"
    out_path = Path(args.out) if args.out else stage_dir / "manual_checklist_50_checked.csv"

    with (fig_dir / "fig_index.csv").open(newline="", encoding="utf-8") as fh:
        fig_by_id = {r["flow_id"]: r for r in csv.DictReader(fh)}
    with (tf_dir / "flow_map.csv").open(newline="", encoding="utf-8") as fh:
        tf_by_id = {r["flow_id"]: r for r in csv.DictReader(fh)}
    label_map = json.loads((tf_dir / "label_map.json").read_text(encoding="utf-8"))
    id_to_class = label_map["id_to_class"]

    # TSV 指定行号一次性扫描（行号升序）
    with checklist_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    needed = {int(r["tf_tsv_line"]): r["flow_id"] for r in rows}
    max_line = max(needed)
    tsv_rows = {}
    with (tf_dir / "trafficformer_all.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line_no, line in enumerate(fh, start=2):
            if line_no in needed:
                label_col, text = line.rstrip("\n").split("\t", 1)
                tsv_rows[needed[line_no]] = (int(label_col), text)
            if line_no >= max_line:
                break

    pkl_cache = {}
    jsonl_path = fig_dir / "fig_all.jsonl"

    def load_pkl(source_pkl):
        if source_pkl not in pkl_cache:
            with (flows_dir / source_pkl).open("rb") as fh:
                pkl_cache[source_pkl] = pickle.load(fh)["packets"]
        return pkl_cache[source_pkl]

    results = []
    for r in rows:
        fid = r["flow_id"]
        verdict = {}
        fig = fig_by_id.get(fid)
        tf = tf_by_id.get(fid)
        packets = load_pkl(r["source_pkl"]).get(fid)
        tsv_label, tsv_text = tsv_rows.get(fid, (None, None))

        # 1 同一 flow
        verdict["check_same_flow"] = "✓" if (fig and tf and packets and tsv_text is not None) else "✗"
        # 2 标签
        class_in_id = fid.split("__")[1]
        ok_label = (
            fig and tf and tsv_label is not None
            and int(fig["label_id"]) == int(tf["label_id"]) == tsv_label
            and id_to_class.get(str(tsv_label)) == r["class_name"]
            and class_in_id == r["class_name"]
        )
        verdict["check_label"] = "✓" if ok_label else "✗"
        # 3 包数
        ok_count = (
            fig and tf and packets is not None
            and len(packets) == int(fig["packet_count"]) == int(tf["packet_count"])
            and int(fig["used_packet_count"]) == min(30, len(packets))
            and int(tf["used_packet_count"]) == min(5, len(packets))
        )
        verdict["check_packet_count"] = "✓" if ok_count else "✗"
        # 4 包序：pkl 存储序 ts 非降；FIG 排序后前 5 包方向序 == pkl 前 5 包
        ts_seq = [p[0] for p in packets]
        pkt_dirs = [1 if p[3] == 1 else -1 for p in packets]
        with jsonl_path.open(encoding="utf-8") as fh:
            fh.seek(int(fig["byte_offset"]))
            rec = json.loads(fh.readline())
        fig_dirs = [1 if f[0] > 0 else -1 for f in rec["features"]]
        ok_order = (
            all(a <= b for a, b in zip(ts_seq, ts_seq[1:]))
            and fig_dirs[:5] == pkt_dirs[:5]
            and tsv_text.count("[SEP]") == min(5, len(packets))
        )
        verdict["check_order"] = "✓" if ok_order else "✗"
        # 5 时间戳
        ok_ts = all(
            _round6(f[2]) == _round6(ts_seq[i] - ts_seq[0])
            for i, f in enumerate(rec["features"])
        )
        verdict["check_timestamp"] = "✓" if ok_ts else "✗"
        # 6 方向
        verdict["check_direction"] = "✓" if fig_dirs == pkt_dirs[: len(fig_dirs)] else "✗"
        # 7 空流/单包流
        ok_edge = True
        if len(packets) == 1:
            ok_edge = (
                rec["node_count"] == 1 and rec["burst_count"] == 1
                and rec["edges"] == [] and rec["features"][0][5:7] == [0, 0]
            )
        verdict["check_edge_cases"] = "✓" if ok_edge else "✗"

        row = dict(r)
        row.update(verdict)
        row["note"] = ""
        results.append(row)

    fields = list(results[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    failures = [
        (res["flow_id"], key)
        for res in results
        for key in ("check_same_flow", "check_label", "check_packet_count",
                    "check_order", "check_timestamp", "check_direction",
                    "check_edge_cases")
        if res[key] != "✓"
    ]
    print(f"flows checked: {len(results)}")
    print(f"failures: {len(failures)}")
    for fid, key in failures:
        print(f"  {key}: {fid}")
    print(f"out: {out_path}")


if __name__ == "__main__":
    main()
