# -*- coding: utf-8 -*-
"""Stage 1 §7.4 Model A：z_t embedding 导出（本地 CPU / 服务器均可）。

从 finetuned_model.bin 加载 TrafficFormer 的 embedding+encoder，对
train/val/test 三个 TSV 流式逐批前向，导出 first-token 池化表示 z_t
（分类头之前，hidden=768）+ labels + flow_ids 三件套 npy。

flow_ids 对齐：task06 保证切分 TSV 行序 == 该 split 的 flow 按
tsv_line_number 升序；flow_map.csv 的 split 列是官方划分不可用，split
成员来自 data/splits/<name>/{train,val,test}.txt。本脚本逐行校验
(TSV label == flow_map label_id)，行数不符即报错。

用法（本地）：
  python scripts/task08_export_model_a_zt.py \
    --tf-code-dir D:/unknown_traffic_project/tf_runtime/code \
    --vocab_path  D:/unknown_traffic_project/tf_runtime/code/models/encryptd_vocab.txt \
    --pretrained_model_path D:/unknown_traffic_project/outputs/stage1/modelA/finetuned_model.bin \
    --config_path  D:/unknown_traffic_project/tf_runtime/code/models/bert/base_config.json \
    --train_path D:/unknown_traffic_project/data/splits/compatible_min1/train_dataset.tsv \
    --dev_path   D:/unknown_traffic_project/data/splits/compatible_min1/valid_dataset.tsv \
    --test_path  D:/unknown_traffic_project/data/splits/compatible_min1/test_dataset.tsv \
    --flow-map-path D:/unknown_traffic_project/data/trafficformer_input/compatible_min1/flow_map.csv \
    --split-dir D:/unknown_traffic_project/data/splits/compatible_min1 \
    --out-dir D:/unknown_traffic_project/outputs/stage1/modelA/embeddings \
    --seq_length 320 --batch_size 64 --device cpu
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EncoderOnly(nn.Module):
    """TrafficFormer embedding + encoder，输出 first-token 表示 z_t。"""

    def __init__(self, args):
        super().__init__()
        from uer.layers import str2embedding
        from uer.encoders import str2encoder
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        return output[:, 0, :]  # pooling == "first"，与训练一致


def _read_flow_map(flow_map_path, split_dir):
    """返回 {part: [(tsv_line_number, flow_id, label_id)]}，按行号升序。

    flow_map.csv 的 split 列是 task03 的官方划分，与 task06 重新分层切分
    不一致；因此用 data/splits/<name>/{train,val,test}.txt 的 flow_id 集合
    做成员过滤，再按 tsv_line_number 升序（task06 保证 == TSV 行序）。
    """
    by_fid = {}
    with open(flow_map_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_fid[r["flow_id"]] = (int(r["tsv_line_number"]), int(r["label_id"]))
    out = {}
    for part in ("train", "val", "test"):
        txt_path = Path(split_dir) / f"{part}.txt"
        fids = [ln.strip() for ln in txt_path.open(encoding="utf-8")
                if ln.strip()]
        missing = [f for f in fids if f not in by_fid]
        if missing:
            raise ValueError(f"{txt_path}: {len(missing)} 个 flow_id 不在 "
                             f"flow_map 中，例: {missing[:5]}")
        out[part] = sorted((by_fid[f][0], f, by_fid[f][1]) for f in fids)
    return out


def _stream_batches(args, tsv_path, flow_rows, batch_size):
    """逐行 tokenize（与 run_classifier.read_dataset 完全相同的处理），
    边读边校验 label 与 flow_map 一致，按批 yield。
    yield (src, tgt, seg, batch_fids)，最后一小批也产出。"""
    from uer.utils.constants import CLS_TOKEN
    src_buf, tgt_buf, seg_buf, fid_buf = [], [], [], []
    with open(tsv_path, encoding="utf-8") as fh:
        header = next(fh)
        assert header.strip().split("\t") == ["label", "text_a"], \
            f"TSV 表头不符合预期: {header!r}"
        for row_id, (line, (line_no, fid, map_label)) in enumerate(
                zip(fh, flow_rows)):
            label, text_a = line[:-1].split("\t", 1)
            label = int(label)
            assert label == map_label, \
                f"第 {row_id + 1} 行 label 不一致: TSV={label} flow_map={map_label}"
            src = args.tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + args.tokenizer.tokenize(text_a))
            seg = [1] * len(src)
            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            src_buf.append(src)
            tgt_buf.append(label)
            seg_buf.append(seg)
            fid_buf.append(fid)
            if len(src_buf) == batch_size:
                yield (torch.LongTensor(src_buf), torch.LongTensor(tgt_buf),
                       torch.LongTensor(seg_buf), list(fid_buf))
                src_buf, tgt_buf, seg_buf, fid_buf = [], [], [], []
        if src_buf:
            yield (torch.LongTensor(src_buf), torch.LongTensor(tgt_buf),
                   torch.LongTensor(seg_buf), list(fid_buf))


def main():
    # finetune_opts 需要 uer 在 sys.path（--tf-code-dir 尚未解析，先插默认值）
    sys.path.insert(0, "D:/unknown_traffic_project/tf_runtime/code")
    from uer.opts import finetune_opts
    parser = argparse.ArgumentParser(description="Model A z_t embedding 导出")
    finetune_opts(parser)
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"],
                        default="first", help="Pooling type.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"],
                        default="bert", help="Specify the tokenizer.")
    parser.add_argument("--tf-code-dir", default="D:/unknown_traffic_project/tf_runtime/code")
    parser.add_argument("--flow-map-path", required=True)
    parser.add_argument("--split-dir", required=True,
                        help="data/splits/<name>/（train.txt/val.txt/test.txt）")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0"])
    parser.add_argument("--max-flows", type=int, default=None,
                        help="每 split 最多处理前 N 流（冒烟用，按 TSV 行序截断）")
    # MoE 扩展参数（老师修改版 encoder 需要，与 run_classifier 一致）
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072,
                        help="Dim of expert,default is ffn.")
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    parser.add_argument("--moebert_route_method",
                        choices=["gate-token", "gate-sentence", "hash-random",
                                 "hash-balance", "proto"], default="hash-random",
                        help="moebert route method.")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str,
                        help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.0,
                        help="gate loss weight.")
    args = parser.parse_args()

    sys.path.insert(0, args.tf_code_dir)
    from uer.utils import str2tokenizer
    from uer.utils.config import load_hyperparam
    from uer.utils.constants import CLS_TOKEN

    args.labels_num = 20
    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer["bert"](args)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 构建 EncoderOnly "
          f"(hidden={args.hidden_size}, layers={args.layers_num}, vocab={len(args.tokenizer.vocab)})")

    model = EncoderOnly(args)
    state = torch.load(args.pretrained_model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"模型缺参数 {len(missing)} 个: {missing[:5]}")
    print(f"加载 {args.pretrained_model_path}: 匹配, "
          f"未用键 {len(unexpected)} 个（分类头，正常）")
    model.to(args.device).eval()

    flow_map = _read_flow_map(args.flow_map_path, args.split_dir)
    parts = [("train", args.train_path), ("val", args.dev_path),
             ("test", args.test_path)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "z_t_definition": "TrafficFormer first-token pooled (pooling=first), "
                          "before classification head",
        "model": args.pretrained_model_path,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "parts": {},
    }
    t_start = time.time()
    for part, tsv_path in parts:
        flow_rows = flow_map[part][:args.max_flows] if args.max_flows else flow_map[part]
        n_expected = len(flow_rows)
        Z, Y, F = [], [], []
        n_rows = 0
        t_part = time.time()
        for bi, (src_b, tgt_b, seg_b, fids) in enumerate(
                _stream_batches(args, tsv_path, flow_rows, args.batch_size)):
            src_b = src_b.to(args.device)
            seg_b = seg_b.to(args.device)
            with torch.no_grad():
                z = model(src_b, seg_b)
            Z.append(z.cpu().numpy().astype(np.float32))
            Y.append(tgt_b.numpy())
            F.extend(fids)
            n_rows += src_b.size(0)
            if (bi + 1) % 200 == 0:
                print(f"  {part}: {n_rows}/{n_expected} "
                      f"({time.time() - t_part:.0f}s)")
        if n_rows != n_expected:
            raise ValueError(f"{part}: TSV 行数 {n_rows} != flow_map {n_expected}")
        Z = np.concatenate(Z, axis=0)
        Y = np.concatenate(Y, axis=0)
        F = np.asarray(F)
        for name, arr in (("embedding", Z), ("labels", Y), ("flow_ids", F)):
            p = out_dir / f"{name}_{part}.npy"
            np.save(p, arr)
            summary["parts"].setdefault(part, {})[name] = {
                "shape": list(arr.shape), "dtype": str(arr.dtype),
                "sha256": _sha256_file(p),
            }
        print(f"{part}: Z{Z.shape} 导出完成 ({time.time() - t_part:.0f}s)")

    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"全部完成，总耗时 {(time.time() - t_start) / 60:.1f} 分钟 → {out_dir}")


if __name__ == "__main__":
    main()
