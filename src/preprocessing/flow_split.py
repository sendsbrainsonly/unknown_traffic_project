# -*- coding: utf-8 -*-
"""Task 0.1 + Task 0.2: PCAP 流式读取 -> 双向流构建（v2 计划 §6.2）

流定义: 五元组 (srcIP, srcPort, dstIP, dstPort, proto), 反向五元组视为同一双向流。
方向: 以每个流第一个包的方向为 forward (initiator)。
包级字段 (Task 0.1): timestamp / caplen / wirelen / IP / port / L4 proto / raw bytes / direction。
"""
import os
import pickle

from scapy.utils import RawPcapReader

TCP, UDP = 6, 17
MIN_FRAME = 40  # Ethernet(14) + IPv4 基本头(20) + 端口(4) 之后仍有余量


def parse_ipv4(b):
    """Ethernet+IPv4 解析; 非 IPv4 / 非 TCP/UDP 返回 None。"""
    if len(b) < MIN_FRAME or b[12:14] != b"\x08\x00":
        return None
    proto = b[23]  # IPv4 头第 10 字节, 位置固定 (14+9)
    if proto not in (TCP, UDP):
        return None
    sip = ".".join(map(str, b[26:30]))
    dip = ".".join(map(str, b[30:34]))
    off = 14 + (b[14] & 0x0F) * 4
    if len(b) < off + 4:
        return None
    sport = int.from_bytes(b[off:off + 2], "big")
    dport = int.from_bytes(b[off + 2:off + 4], "big")
    return proto, sip, dip, sport, dport


def split_pcap(path):
    """流式读一个 pcap -> (flows, skipped)

    flows: { (sorted_addr_pair, proto): {"init": (ip, port), "peer": (ip, port),
                                         "pkts": [(ts, caplen, wirelen, direction, frame), ...]} }
    skipped: 非 IPv4/TCP/UDP 的包数 (统计用, 不应太多)
    """
    flows, skipped = {}, 0
    with RawPcapReader(path) as r:
        for raw, meta in r:
            info = parse_ipv4(bytes(raw))
            if info is None:
                skipped += 1
                continue
            proto, sip, dip, sport, dport = info
            a, b = (sip, sport), (dip, dport)
            key = (tuple(sorted((a, b))), proto)
            ts = meta.sec + meta.usec / 1e6
            if key not in flows:
                flows[key] = {"init": a, "peer": b, "pkts": []}
            f = flows[key]
            direction = 1 if a == f["init"] else 0
            f["pkts"].append((ts, meta.caplen, meta.wirelen, direction, bytes(raw)))
    return flows, skipped


def class_pcaps(root, entry):
    """类条目可能是目录(内含 pcap 分片, 如 SMB/)或单个 .pcap 文件; 返回 [(stem, path), ...]"""
    entry_path = os.path.join(root, entry)
    if os.path.isdir(entry_path):
        out = []
        for fn in sorted(os.listdir(entry_path)):
            if fn.lower().endswith(".pcap"):
                out.append((os.path.splitext(fn)[0], os.path.join(entry_path, fn)))
        return out
    if entry.lower().endswith(".pcap"):
        return [(os.path.splitext(entry)[0], entry_path)]
    return []


def discover_classes(root):
    """返回 [(split, class_name, [(stem, pcap_path), ...]), ...]; 跳过 README/LICENSE/.7z 等"""
    classes = []
    for split in ("Benign", "Malware"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for entry in sorted(os.listdir(split_dir)):
            pcaps = class_pcaps(split_dir, entry)
            if pcaps:
                # 扁平 pcap 类条目名带 .pcap 后缀 (如 "BitTorrent.pcap"), 与目录类 (SMB/Weibo) 统一去掉
                class_name = entry[:-5] if entry.lower().endswith(".pcap") else entry
                classes.append((split, class_name, pcaps))
    return classes


def flow_records(flows, split, class_name, stem):
    """把 split_pcap 的 flows 转成 flow_index 行 + {flow_id: pkts} 的打包数据"""
    rows, packets = [], {}
    for i, (key, f) in enumerate(sorted(flows.items())):
        _, proto = key
        pkts = f["pkts"]
        init_ip, init_port = f["init"]
        peer_ip, peer_port = f["peer"]
        fwd = sum(1 for p in pkts if p[3] == 1)
        bwd = len(pkts) - fwd
        tss = [p[0] for p in pkts]
        flow_id = f"{split}__{class_name}__{stem}__{i:06d}"
        rows.append({
            "flow_id": flow_id,
            "split": split,
            "class_name": class_name,
            "source_file": stem + ".pcap",
            "src_ip": init_ip, "src_port": init_port,
            "dst_ip": peer_ip, "dst_port": peer_port,
            "protocol": "TCP" if proto == TCP else "UDP",
            "packet_count": len(pkts),
            "forward_packets": fwd,
            "backward_packets": bwd,
            "start_time": min(tss), "end_time": max(tss),
            "duration_s": round(max(tss) - min(tss), 6),
            "total_bytes": sum(p[1] for p in pkts),
        })
        packets[flow_id] = pkts
    return rows, packets


def write_flows_pkl(packets, split, class_name, stem, skipped, out_dir):
    """每 (类, 源文件) 一个 pkl, 便于按 flow_id 直接取包级数据 (§6.4 验收)"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split}__{class_name}__{stem}.pkl")
    total_pkts = sum(len(v) for v in packets.values())
    with open(path, "wb") as fh:
        pickle.dump({
            "split": split,
            "class_name": class_name,
            "source_file": stem + ".pcap",
            "n_flows": len(packets),
            "n_packets": total_pkts,
            "skipped_packets": skipped,
            "packets": packets,  # flow_id -> [(ts, caplen, wirelen, direction, frame), ...]
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path
