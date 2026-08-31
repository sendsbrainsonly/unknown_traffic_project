# -*- coding: utf-8 -*-
"""Stage 0 校验 (Task 0.2 产物验证; Task 0.5 机器检查部分)

检查项:
  1. 账目核对: 每个源 pcap 的独立重数 == pkl.n_packets + pkl.skipped_packets; EtherType 分布
  2. flow_index <-> pkl 一致性: 每源文件 packet_count 求和 == n_packets; fwd+bwd==packet_count; fwd>=1
  3. 独立解析交叉验证: 用 scapy 全栈解析 (独立于 flow_split 的字节偏移解析) 抽查每类 30 流 x 2 包,
     比对 proto/IP/port 与 flow_index 记录
  4. 时间戳/字节/方向结构性检查 (抽查): ts 存储序非降; start/end/duration/total_bytes 与包级数据一致;
     direction=1 的包五元组 == (src_ip,src_port)->(dst_ip,dst_port), direction=0 反之
  5. 截断检查: caplen != wirelen 的比例
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.flow_split import discover_classes, parse_ipv4  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOWS_DIR = os.path.join(PROJECT_ROOT, "data", "flows")
STAGE0_DIR = os.path.join(PROJECT_ROOT, "outputs", "stage0_data")

random.seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


def main():
    root = None
    import yaml
    with open(os.path.join(PROJECT_ROOT, "configs", "dataset", "ustc_tfc2016.yaml"), encoding="utf-8") as fh:
        root = yaml.safe_load(fh)["raw_pcap_root"]

    # ---- 0. flow_index 基础 ----
    rows = {}
    with open(os.path.join(STAGE0_DIR, "flow_index.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["flow_id"]] = r
    ids = list(rows)
    check("flow_id 唯一", len(ids) == len(set(ids)), f"n={len(ids)}")
    bad_null = [k for k, r in rows.items() if any(v == "" for v in r.values())]
    check("flow_index 无空字段", not bad_null)
    bad_sum = [k for k, r in rows.items()
               if int(r["forward_packets"]) + int(r["backward_packets"]) != int(r["packet_count"])]
    check("每行 fwd+bwd == packet_count", not bad_sum, f"bad={len(bad_sum)}")
    bad_fwd = [k for k, r in rows.items() if int(r["forward_packets"]) < 1]
    check("每条流 forward >= 1 (initiator 必有首包)", not bad_fwd, f"bad={len(bad_fwd)}")
    check("每行 duration_s >= 0", all(float(r["duration_s"]) >= 0 for r in rows.values()))
    check("packet_count >= 1", all(int(r["packet_count"]) >= 1 for r in rows.values()))

    # ---- 1. 每源文件账目 + pkl 一致性 (全量): 逐帧跳过原因对账 ----
    import pickle
    from scapy.utils import RawPcapReader

    classes = discover_classes(root)
    trunc = total = 0
    for split, cname, pcaps in classes:
        for stem, path in pcaps:
            fname = f"{split}__{cname}__{stem}.pkl"
            pkl = pickle.load(open(os.path.join(FLOWS_DIR, fname), "rb"))
            # 独立重数 + 复刻 split_pcap 的跳过条件逐帧对账
            n_raw, skip_reasons = 0, {}
            with RawPcapReader(path) as rd:
                for raw, meta in rd:
                    n_raw += 1
                    b = bytes(raw)
                    if meta.caplen != meta.wirelen:
                        trunc += 1
                    total += 1
                    if parse_ipv4(b) is None:  # 与 flow_split.split_pcap 完全相同的判定
                        if len(b) < 14:
                            why = "short<14"
                        elif b[12:14] != b"\x08\x00":
                            why = f"ethertype={b[12:14].hex()}"
                        else:
                            why = f"non-L4/proto={b[23]}"
                        skip_reasons[why] = skip_reasons.get(why, 0) + 1
            check(f"账目 {cname}/{stem}: 独立重数 == 保留+跳过",
                  n_raw == pkl["n_packets"] + pkl["skipped_packets"],
                  f"raw={n_raw} kept={pkl['n_packets']} skipped={pkl['skipped_packets']}")
            check(f"账目 {cname}/{stem}: 逐帧跳过原因合计 == skipped_packets",
                  sum(skip_reasons.values()) == pkl["skipped_packets"],
                  f"reasons={skip_reasons} skipped={pkl['skipped_packets']}")
            # pkl 内部一致性
            check(f"pkl {cname}/{stem} n_flows == len(packets)", pkl["n_flows"] == len(pkl["packets"]))
            sub = [r for r in rows.values()
                   if r["class_name"] == cname and r["source_file"] == stem + ".pcap"]
            check(f"flow_index {cname}/{stem} 行数 == n_flows", len(sub) == pkl["n_flows"])
            check(f"flow_index {cname}/{stem} packet_count 求和 == n_packets",
                  sum(int(r["packet_count"]) for r in sub) == pkl["n_packets"])
    print(f"  [info] caplen != wirelen 的包共 {trunc}/{total}")

    # ---- 2. 独立解析交叉验证 (scapy 全栈, 抽查) ----
    from scapy.all import Ether, IP, TCP, UDP
    bad_parse, checked = [], 0
    for split, cname, pcaps in classes:
        for stem, path in pcaps:
            fname = f"{split}__{cname}__{stem}.pkl"
            pkl = pickle.load(open(os.path.join(FLOWS_DIR, fname), "rb"))
            fids = sorted(pkl["packets"])[::max(1, len(pkl["packets"]) // 30)][:30]
            for fid in fids:
                r = rows[fid]
                pkts = pkl["packets"][fid]
                for i in (0, len(pkts) // 2):
                    ts, caplen, wirelen, direction, frame = pkts[i]
                    try:
                        e = Ether(frame)
                        ip = e[IP]
                        l4 = ip[TCP] if ip.proto == 6 else (ip[UDP] if ip.proto == 17 else None)
                        if l4 is None:
                            bad_parse.append((fid, i, "no L4")); continue
                        expect = ((ip.src, l4.sport), (ip.dst, l4.dport))
                        got = ((r["src_ip"], int(r["src_port"])), (r["dst_ip"], int(r["dst_port"])))
                        if direction == 0:
                            expect = (expect[1], expect[0])
                        if expect != got:
                            bad_parse.append((fid, i, f"expect={expect} got={got} dir={direction}"))
                        checked += 1
                    except Exception as ex:
                        bad_parse.append((fid, i, f"scapy err {ex}"))
    check(f"scapy 独立解析抽查 ({checked} 包) 与 flow_index 五元组/方向一致", not bad_parse,
          f"bad={len(bad_parse)}" + (f" e.g. {bad_parse[:3]}" if bad_parse else ""))

    # ---- 3. 时间戳/字节抽查 ----
    bad_ts, bad_byte = 0, 0
    sample_ids = random.sample(ids, 400)
    for fid in sample_ids:
        r = rows[fid]
        pkl = pickle.load(open(os.path.join(
            FLOWS_DIR, f"{r['split']}__{r['class_name']}__{r['source_file'][:-5]}.pkl"), "rb"))
        pkts = pkl["packets"][fid]
        tss = [p[0] for p in pkts]
        if tss != sorted(tss):
            bad_ts += 1
        if abs(min(tss) - float(r["start_time"])) > 1e-6 or abs(max(tss) - float(r["end_time"])) > 1e-6:
            bad_ts += 1
        if sum(p[1] for p in pkts) != int(r["total_bytes"]):
            bad_byte += 1
    check("抽查 400 流: ts 非降且 start/end 与包级一致", bad_ts == 0, f"bad={bad_ts}")
    check("抽查 400 流: total_bytes == sum(caplen)", bad_byte == 0, f"bad={bad_byte}")

    print()
    if fails:
        print(f"RESULT: {len(fails)} FAILED ->", fails)
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
