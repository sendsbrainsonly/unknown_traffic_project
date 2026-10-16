#!/usr/bin/env python3
"""Audit frozen chart data against source headers, manuscript tables and exports."""
import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'stage44_vnat_coarse_service_loso_open_set').is_dir())


def tables(text):
    blocks = re.findall(r'(?:^\|.*\n)+', text, re.M)
    result = []
    for block in blocks:
        rows = [[v.strip().replace('**', '') for v in line.strip().strip('|').split('|')]
                for line in block.splitlines()]
        result.append([dict(zip(rows[0], r, strict=True)) for r in rows[2:]])
    return result


def match(rows, **keys):
    selected = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(selected) == 1, (keys, len(selected))
    return selected[0]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def percent(v):
    return str((Decimal(str(v))*100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('render', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier verification receipts'
    payload = json.loads((args.render/'plot_data.json').read_text())
    manifest = json.loads((args.render/'manifest.json').read_text())
    records = payload['records']
    source_count = sd_count = paper_count = 0
    cache = {}
    for p, metadata in payload['sources'].items():
        path = ROOT/p
        assert digest(path) == metadata['sha256'], p
        cache[p] = list(csv.DictReader(path.open())) if path.suffix == '.csv' else tables(path.read_text())
    for r in records:
        src = cache[r['source_path']]
        f, s, m, metric = (r[k] for k in ['figure', 'setting', 'method', 'metric'])
        if f == 'fig4_1':
            row = match(src[0 if s == 'VPN6' else 1], Method=m)
            value, sd = row['Macro-F1 mean±SD'].split('±')
            assert Decimal(sd.strip()) == Decimal(str(r['sd']))
            sd_count += 1
        elif f == 'fig4_2':
            row = match(src[1], 数据集=s, 评价集='known_test', 历史对照=m if m != 'three_view' else 'E3_hard_remap')
            value = row['新 T0 Macro-F1' if m == 'three_view' else '历史 Macro-F1']
        elif f == 'fig4_3':
            row = match(src, fold=s, view='natural', threshold_percentile='95')
            value = row[f'{m}_{metric}']
            assert int(row['known_samples']) == r['n_known']
            assert int(row['unknown_samples']) == r['n_unknown']
        elif f == 'fig4_4':
            value = match(src, fold=s, method=m, view='natural', threshold_percentile='95')[metric]
        elif f == 'fig4_5':
            if 'stage42y_' in r['source_path']:
                row = match(src[0], **{'Unknown class':s})
                value = row[f'{m} {metric.upper()}']
                assert int(row['Unknown flows'].replace(',', '')) == r['n_unknown']
            else:
                row = match(src[0], Method=m)
                columns = [k for k in row if k in [metric.upper(), 'Natural '+metric.upper()]]
                assert len(columns) == 1, (r['source_path'], row.keys())
                value = row[columns[0]]
        elif f == 'fig4_6':
            row = match(src[0 if s.startswith('known') else 1], Setting=s)
            value = row[{'auroc':'AUROC mean', 'auprc':'AUPRC mean', 'ufar':'UFAR mean', 'known_frr':'Known FRR mean'}[metric]]
        elif f == 'fig4_7':
            row = match(src[0], Scenario=s)
            prefix = 'OD' if m == 'Open-Detect' else 'Ours'
            value = row[prefix+' '+{'auroc':'AUROC', 'binary_f1':'F1@P95', 'ufar':'UFAR'}[metric]]
        else:
            raise AssertionError(f)
        assert Decimal(value.strip()) == Decimal(str(r['value'])), r
        source_count += 1

    paper_path = ROOT/'paper/md/ch4/1.md'
    paper = paper_path.read_text()
    pt = tables(paper)
    method_names = {
        'TrafficFormer-pretrained':'TrafficFormer（预训练）', 'YaTC-official-pretrained-Stage20':'YaTC（预训练）',
        'Open-Detect corrected-paper':'Open-Detect（corrected-paper）', 'RoNeTC-runnable-reconstruction':'RoNeTC（重建版）',
        'TFE-GNN-8-UDP-short':'TFE-GNN-8-UDP-short（适配版）', 'Trident-early8-86D':'Trident-early8-86D（适配版）',
        'OURS-E3-T8-pretrained':'E3（本文早期双视图版本）'}
    services = dict(zip(['communication','file_transfer','remote_access','streaming'], ['通信','文件传输','远程访问','流媒体']))
    metrics = {'auroc':'AUROC ↑','binary_f1':'未知类 F1 ↑','ufar':'UFAR ↓','known_frr':'Known FRR ↓'}
    for r in records:
        f,s,m,metric = (r[k] for k in ['figure','setting','method','metric'])
        actual = None
        if f == 'fig4_1':
            row = match(pt[1], 方法=method_names[m])
            actual, sd = row[('VPN' if s == 'VPN6' else 'Tor')+' Macro-F1'].split(' ± ')
            assert sd == percent(r['sd'])
            paper_count += 1
        elif f == 'fig4_2':
            idx = {'E3_hard_remap':0,'E3_coarse_head':1,'YaTC_hard_remap_context':2,'three_view':3}[m]
            actual = pt[3][idx]['VPN 四类' if s == 'iscx_vpn' else 'Tor 五类']
        elif f == 'fig4_3':
            actual = match(pt[7], 未知服务=services[s])[metrics[metric]].split('／')[0 if m == 'od' else 1]
        elif f == 'fig4_5':
            row = match(pt[9], 未知攻击=s.replace(' - ', ' – '))
            actual = row[f'{m} {metric.upper()}'+(' ↓' if metric == 'ufar' else '')]
            assert int(row['未知流数'].replace(',', '')) == r['n_unknown']
        elif f == 'fig4_7':
            actual = match(pt[10], 配置=s)[metrics[metric]].split('／')[0 if m == 'Open-Detect' else 1]
        if actual is not None:
            assert actual == percent(r['value']), (r, actual, percent(r['value']))
            paper_count += 1
    for method, row in zip(['msp','energy','centroid','des_v1'], pt[8], strict=True):
        for metric in ['auroc','ufar','known_frr']:
            values = [Decimal(str(r['value'])) for r in records if r['figure']=='fig4_4' and r['method']==method and r['metric']==metric]
            assert len(values) == 4
            assert row[metrics[metric]] == percent(sum(values)/4)
            paper_count += 1

    export_checks = []
    for item in manifest['outputs']:
        p = args.render/item['path']
        assert digest(p) == item['sha256']
        info = {'path':p.name, 'sha256':digest(p)}
        if p.suffix == '.svg':
            root = ET.parse(p).getroot()
            assert not root.findall('.//{http://www.w3.org/2000/svg}image'), p
            assert root.findall('.//{http://www.w3.org/2000/svg}text'), p
            info.update(embedded_rasters=0, editable_text=True)
        elif p.suffix == '.pdf':
            data = p.read_bytes()
            assert data.startswith(b'%PDF-') and b'%%EOF' in data[-100:]
            assert not re.search(rb'/Subtype\s*/Image', data), p
            info['embedded_rasters'] = 0
        else:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                assert im.width > 2500 and im.height > 900
                assert abs(im.info['dpi'][0] - 300) < 1
                info.update(pixels=list(im.size), dpi=im.info['dpi'])
        export_checks.append(info)
    raw_links = re.findall(r'!\[图4-(\d+)[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)', paper)
    links = [(n, bracketed or plain) for n, bracketed, plain in raw_links]
    assert [int(n) for n,_ in links] == list(range(1,8))
    resolved_links = [(Path(p) if Path(p).is_absolute() else paper_path.parent / p).resolve() for _,p in links]
    assert all(p.is_file() and p.parent==args.render.resolve() for p in resolved_links)
    assert [int(n) for n in re.findall(r'^\*\*图 4-(\d+) ',paper,re.M)] == list(range(1,8))
    assert [int(n) for n in re.findall(r'^\*\*表 4-(\d+) ',paper,re.M)] == list(range(1,12))
    assert len(export_checks)==21 and len(records)==173 and sd_count==14
    receipt = {'status':'PASS', 'verified_utc':datetime.now(timezone.utc).isoformat(),
               'metric_source_checks':source_count, 'sd_source_checks':sd_count,
               'manuscript_numeric_checks':paper_count, 'source_files':len(cache),
               'figure_links':len(links), 'table_count':len(pt), 'exports':export_checks,
               'manuscript_sha256':digest(paper_path), 'verifier_sha256':digest(Path(__file__)),
               'scope':'Numeric/header, manuscript parity, file integrity and vector structure; visual review separately recorded.'}
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['exports','scope']},ensure_ascii=False))


if __name__ == '__main__':
    main()
