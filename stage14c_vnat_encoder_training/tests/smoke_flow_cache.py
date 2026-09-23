#!/usr/bin/env python3
"""Project-local one-capture smoke check for flow-aligned packet extraction."""
from __future__ import annotations
import json, shutil, sys
from collections import Counter
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_flow_cache import load_clean_flow_rows, scan_capture  # noqa: E402
from common import FLOW_BYTES  # noqa: E402

def main():
    out=ROOT/"smoke_flow_cache"
    if out.exists(): shutil.rmtree(out)
    out.mkdir()
    rows=[r for r in load_clean_flow_rows() if r["capture_id"]=="nonvpn_skype-chat_capture1"]
    targets={r["source_flow_id"]:{"cache_index":i,"flow_uid":r["flow_uid"],"expected_packet_count":int(r["packet_count"])} for i,r in enumerate(rows)}
    images=open_memmap(out/"images.npy",mode="w+",dtype=np.uint8,shape=(len(rows),FLOW_BYTES)); images[:]=0
    used=open_memmap(out/"packets_used.npy",mode="w+",dtype=np.uint8,shape=(len(rows),)); used[:]=0
    counts=Counter(); failures=[]
    result=scan_capture(rows[0]["capture_id"],Path(rows[0]["source_pcap_path"]),targets,images,used,counts,failures,out/"tshark.log")
    mismatches=[]
    for r in rows:
        observed=counts[f"{r['capture_id']}|{r['source_flow_id']}"]
        if observed!=int(r["packet_count"]): mismatches.append([r["flow_uid"],r["packet_count"],observed])
    status={"status":"PASS" if not failures and not mismatches and np.all(used>0) else "FAIL","result":result,"flows":len(rows),"failures":failures,"mismatches":mismatches,"packets_used":used.tolist()}
    (out/"result.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status));
    if status["status"]!="PASS": raise SystemExit(1)

if __name__=="__main__": main()
