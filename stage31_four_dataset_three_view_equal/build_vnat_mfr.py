#!/usr/bin/env python3
"""Recover YaTC MFR for one frozen VNAT Known Train/Val protocol."""
from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, IPv6, Raw
from scapy.utils import RawPcapReader

from preflight import OUT, PROJECT, sha
from test_unlock import ensure_all_heads_frozen

sys.path.insert(0, str(PROJECT / "stage23_closed_set_method_table" / "scripts"))
from build_yatc_mfr import author_mfr_packet  # noqa: E402
sys.path.insert(0, str(PROJECT / "stage14a_vnat_data_audit" / "scripts"))
from audit_vnat import canonical_tuple  # noqa: E402

MANIFEST = PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv"


def ipv6_compatible_mfr_packet(raw_bytes: bytes) -> bytes:
    """Stage31-only IPv6 adaptation of YaTC's fixed 80B header + 240B payload rule."""
    packet = IPv6(raw_bytes) if raw_bytes and raw_bytes[0] >> 4 == 6 else Ether(raw_bytes)
    if IPv6 not in packet:
        raise ValueError("frozen IPv6 flow has no decodable IPv6 layer")
    header = binascii.hexlify(bytes(packet[IPv6])).decode()
    try:
        payload = binascii.hexlify(bytes(packet[Raw])).decode()
        header = header.replace(payload, "")
    except (IndexError, KeyError):
        payload = ""
    return bytes.fromhex(header[:160].ljust(160, "0") + payload[:480].ljust(480, "0"))


def stream_refs(pcap: Path, wanted: set[str], limit: int = 5) -> dict[str, list[int]]:
    command = ["tshark", "-n", "-r", str(pcap), "-T", "fields",
               "-E", "separator=/t", "-E", "occurrence=f", "-e", "frame.number",
               "-e", "tcp.stream", "-e", "udp.stream", "-e", "ip.src",
               "-e", "ipv6.src", "-e", "ip.dst", "-e", "ipv6.dst",
               "-e", "ip.proto", "-e", "ipv6.nxt"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8")
    refs: dict[str, list[int]] = {name: [] for name in wanted}
    assert process.stdout is not None
    for line in process.stdout:
        cells = line.rstrip("\n").split("\t")
        if len(cells) < 9 or not cells[0].isdigit():
            continue
        if cells[1].isdigit():
            uid = "tcp:" + cells[1]
        elif cells[2].isdigit():
            uid = "udp:" + cells[2]
        else:
            src, dst = cells[3] or cells[4], cells[5] or cells[6]
            if not src or not dst:
                continue
            protocol = f"ip:{cells[7] or cells[8] or 'unknown'}"
            key = canonical_tuple(src, "", dst, "", protocol)
            uid = "other:" + hashlib.sha256(key.encode()).hexdigest()[:24]
        if uid in refs and len(refs[uid]) < limit:
            refs[uid].append(int(cells[0]))
    stderr = process.stderr.read() if process.stderr is not None else ""
    status = process.wait()
    if status:
        raise RuntimeError(f"tshark failed for {pcap}: exit {status}; {stderr[-600:]}")
    missing = [name for name, frames in refs.items() if not frames]
    if missing:
        raise RuntimeError(f"tshark could not locate {len(missing)} frozen streams in {pcap}: {missing[:8]}")
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("medium_seed2025", "medium_seed2026"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    if args.phase == "test":
        ensure_all_heads_frozen()
    protocol = args.protocol
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    base = "yatc_mfr" if args.phase == "trainval" else "yatc_mfr_test"
    suffix = base if args.attempt == 1 else f"{base}_attempt{args.attempt}"
    result = OUT / "input_caches" / "vnat" / protocol / suffix
    if result.exists():
        raise RuntimeError(f"refusing to overwrite cache: {result}")
    result.mkdir(parents=True)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        splits = ("train", "validation") if args.phase == "trainval" else ("test",)
        rows = [r for r in csv.DictReader(handle) if r["protocol_id"] == protocol and
                r["class_role"] == "known" and r["split"] in splits]
    target = {r["flow_uid"]: r for r in rows}
    if not target or len(target) != len(rows):
        raise RuntimeError("missing or duplicate frozen Known flows")
    by_pcap = defaultdict(dict)
    for uid, row in target.items():
        pcap = Path(row["source_pcap_path"])
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        stream = row["source_flow_id"]
        if stream in by_pcap[pcap]:
            raise RuntimeError(f"duplicate stream within capture: {pcap}/{stream}")
        by_pcap[pcap][stream] = uid
    arrays, indices = {}, {}
    for split in splits:
        ids = sorted(uid for uid, row in target.items() if row["split"] == split)
        indices[split] = {uid:i for i,uid in enumerate(ids)}
        np.save(result / f"{split}_flow_ids.npy", np.asarray(ids,dtype="U80"), allow_pickle=False)
        arrays[split] = np.lib.format.open_memmap(result / f"{split}_mfr.npy", mode="w+",
                                                 dtype=np.uint8, shape=(len(ids),40,40))
        arrays[split][:] = 0
    found = Counter()
    adapted_flows = set()
    adapted_packets = 0
    capture_audit = []
    for number, (pcap, wanted) in enumerate(sorted(by_pcap.items(), key=lambda x: str(x[0])),1):
        refs = stream_refs(pcap,set(wanted))
        packet_map = defaultdict(list)
        for stream, frames in refs.items():
            uid = wanted[stream]
            for slot, frame_number in enumerate(frames):
                packet_map[frame_number].append((uid, slot))
        last = max(packet_map)
        scanned = 0
        with RawPcapReader(str(pcap)) as reader:
            for frame_number,(raw,_) in enumerate(reader,1):
                scanned = frame_number
                for uid,slot in packet_map.get(frame_number,()):
                    split = target[uid]["split"]
                    try:
                        encoded = author_mfr_packet(raw)
                    except ValueError as exc:
                        if "requires an IPv4 IP layer" not in str(exc):
                            raise RuntimeError(f"MFR failed at {pcap} frame {frame_number} flow {uid}") from exc
                        encoded = ipv6_compatible_mfr_packet(raw)
                        adapted_flows.add(uid)
                        adapted_packets += 1
                    packet = np.frombuffer(encoded,dtype=np.uint8)
                    arrays[split][indices[split][uid]].reshape(-1)[slot*320:(slot+1)*320] = packet
                    found[uid] += 1
                if frame_number >= last:
                    break
        capture_audit.append({"pcap":str(pcap),"streams":len(wanted),"last_required_frame":last,
                              "last_read_frame":scanned})
        print(json.dumps({"protocol":protocol,"capture":number,"total_captures":len(by_pcap),
                          "flows_seen":len(found)}),flush=True)
    missing = [uid for uid,row in target.items() if found[uid] != min(5,int(row["packet_count"]))]
    if missing:
        (result/"failure.json").write_text(json.dumps({"incomplete":len(missing),"sample":missing[:30]},indent=2))
        raise RuntimeError(f"MFR coverage failed for {len(missing)} VNAT flows")
    for array in arrays.values():
        array.flush()
    audit = {"status":"PASS","dataset":"vnat","protocol":protocol,
             "roles":{role:len(index) for role,index in indices.items()},
             "split_manifest_sha256":sha(MANIFEST),"captures":capture_audit,
             "mfr_method":"YaTC author first-five 40x40 raw-packet rule",
             "ipv6_adaptation":"Stage31-only same 80-byte header/240-byte Raw payload truncation with IPv6 network layer; not strict official YaTC input",
             "ipv6_adapted_flow_count":len(adapted_flows),
             "ipv6_adapted_packet_count":adapted_packets,
             "ipv6_adapted_flow_ids":sorted(adapted_flows),
             "known_test_features_materialized":len(target) if args.phase == "test" else 0,
             "unknown_features_materialized":0,
             "raw_capture_streams_scanned_for_selected_flow_ids":True}
    (result/"cache_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":"PASS","protocol":protocol,"flows":len(target),
                      "capture_count":len(capture_audit)}),flush=True)


if __name__ == "__main__":
    main()
