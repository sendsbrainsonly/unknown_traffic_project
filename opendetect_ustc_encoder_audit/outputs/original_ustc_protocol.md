# Original USTC Protocol Recovery

## Recovery Status

The historical USTC-TFC2016 protocol is uniquely recoverable from executed artifacts. This document records artifact facts, not a reconstruction from current defaults.

## Dataset and Sample Universe

- Dataset root recorded by the executed view manifest: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ustc-tfc2016/USTC-TFC2016（whole）/extracted/V1/1.DataSet(USTC-TFC2016)`
- View manifest: `data/raw_views/ustc_tfc2016_20class/view_manifest.json`
- Source PCAP files: 24
- Logical classes: 20
- TrafficFormer input policy: `compatible_min1`
- Retained flows: 489,101 of 489,101
- Stage 1 sampling cap: none (`max_per_class: null`)
- Short-flow rule: flows with at least one packet were retained; no synthetic packets were created.

## Actual 20-Class Mapping

The two executed label maps are byte-identical with SHA-256 `57a3b1a3abd2dfbd8a090de674ad513405e3921bd82a8747e6e0b6adc3342c59`:

- `data/fig_graph/all_flows/label_map.json`
- `data/trafficformer_input/compatible_min1/label_map.json`

| ID | Class |
|---:|---|
| 0 | BitTorrent |
| 1 | Cridex |
| 2 | FTP |
| 3 | Facetime |
| 4 | Geodo |
| 5 | Gmail |
| 6 | Htbot |
| 7 | Miuref |
| 8 | MySQL |
| 9 | Neris |
| 10 | Nsis-ay |
| 11 | Outlook |
| 12 | SMB |
| 13 | Shifu |
| 14 | Skype |
| 15 | Tinba |
| 16 | Virut |
| 17 | Weibo |
| 18 | WorldOfWarcraft |
| 19 | Zeus |

`SMB-1.pcap` and `SMB-2.pcap` both belong to logical class `SMB` (ID 12). They must not be split into separate labels in this audit. The four Weibo PCAPs likewise remain one logical `Weibo` class.

## Historical Split

- Split manifest: `data/splits/compatible_min1/split_summary.json`
- Membership files: `data/splits/compatible_min1/{train,val,test}.txt`
- First stratified split: `random_state=41`, train versus 20% remainder.
- Second stratified split: `random_state=42`, remainder divided equally into validation and test.
- Effective ratio: 80/10/10.

| Split | Flows | Classes |
|---|---:|---:|
| train | 391,280 | 20 |
| validation | 48,910 | 20 |
| test | 48,911 | 20 |
| total | 489,101 | 20 |

The exported Stage 1 arrays are:

- `outputs/stage1/modelA/embeddings/flow_ids_train.npy`
- `outputs/stage1/modelA/embeddings/flow_ids_val.npy`
- `outputs/stage1/modelA/embeddings/flow_ids_test.npy`
- matching `labels_{train,val,test}.npy`

Independent checks established that each NPY flow-ID set equals its corresponding text membership set, all IDs are unique within a split, and the three sets are pairwise disjoint. Their order is intentionally different: split text preserves randomized membership order, while embedding export uses TrafficFormer TSV line order. Every downstream join must therefore use `flow_id`, never row position across these two sources.

## Representative-Class Counts

| Class | ID | Train | Validation | Test |
|---|---:|---:|---:|---:|
| FTP | 2 | 80,830 | 10,103 | 10,104 |
| Cridex | 1 | 13,108 | 1,638 | 1,639 |
| Miuref | 7 | 10,782 | 1,348 | 1,348 |
| Outlook | 11 | 6,019 | 753 | 752 |

These counts were independently reproduced from the Stage 1 label arrays and match `split_summary.json` exactly.

## Stage 1 Representation Inputs Used by Stage 2.5

- TrafficFormer `z_t`: `outputs/stage1/modelA/embeddings`
- Graph branch `z_g` (seed 2): `outputs/stage1/modelB/seeds/seed2`
- Label map: `data/fig_graph/all_flows/label_map.json`
- Stage 2 alignment implementation: `scripts/task10_stage2_audit.py::_load_aligned`
- Stage 2 representation: `z_f = [standardized z_t; standardized z_g]`, with each branch standardized using train statistics only.
- Resulting `z_f` dimension: 896.
- Stage 2.5 loaded train and validation only; it did not load test embeddings.

## Stage 2.5 Controlled Comparison Protocol

Executed artifact: `outputs/stage2_5_covariance_diagnosis/covariance_diagnosis.csv` with metadata in `run_metadata.json`.

For the TrafficFormer comparison rows used by the new encoder audit:

- representation: train-standardized `z_f`, then PCA64;
- PCA solver: randomized;
- PCA fit: all known train only;
- validation use: transform only;
- PCA random state: 0;
- covariance type: full;
- K: 1, 2, 3 (read from the existing K=1..5 run; no TrafficFormer refit);
- `reg_covar`: 1e-3;
- `n_init`: 3;
- `max_iter`: 200;
- GMM seed: 0;
- class-specific GMM fit: train only;
- main metric: held-out validation average NLL.

The new Open-Detect audit will preserve dataset, mapping, split membership, representative classes, train-only preprocessing, and held-out NLL definition. Per the new protocol, Open-Detect GMMs will use seeds 0/1/2 and `max_iter >= 300`; this is a predeclared robustness extension and will be recorded in `protocol_diff.md` rather than retroactively changing the TrafficFormer results.

## Evidence Paths

- `data/raw_views/ustc_tfc2016_20class/view_manifest.json`
- `data/trafficformer_input/compatible_min1/generation_summary.json`
- `data/trafficformer_input/compatible_min1/flow_map.csv`
- `data/splits/compatible_min1/split_summary.json`
- `outputs/stage1/modelA/embeddings/export_summary.json`
- `outputs/stage2_5_covariance_diagnosis/run_metadata.json`
- `outputs/stage2_5_covariance_diagnosis/covariance_diagnosis.csv`
- `outputs/stage2_6_component_shortcut_audit/run_metadata.json`

## Protocol Gate

PASS. The historical mapping and split are uniquely identified. Step 3 may proceed, but no Open-Detect input generation or training is authorized until the read-only source audit and flow-alignment design are complete.
