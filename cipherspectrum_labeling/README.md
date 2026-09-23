# CipherSpectrum 非破坏性标签清单

本目录只在项目内生成标签清单，不改名、不移动、不覆盖原始 PCAP。

标签策略：

- 类别标签采用数据集的类别目录（发布说明中的 SNI/服务器域名分组）。
- 文件名中的网页域名保存为 `visited_domain`，不直接当类别标签。
- `outputs/cipherspectrum_official40_manifest.csv` 是保守的官方兼容 40 类视图。
- `outputs/cipherspectrum_label_manifest.csv` 是完整本地 41 类审计视图。
- `outputs/manual_review_inventory.csv` 保存本地额外 `getpocket.com` 和目录/密码套件异常。若实验只使用网站/SNI 类别，部分密码套件包装异常仍可保留；若实验依赖密码套件分组纯度，则必须过滤。
- `outputs/packaging_anomaly_summary.csv` 汇总外层目录与文件名记录的密码套件不一致情况。
- 后续划分必须按 `split_group_id` 分组，不能逐 PCAP 随机划分。

运行命令（必须在项目规定的固定 Conda 环境中执行）：

```bash
python -m cipherspectrum_labeling.scripts.build_label_manifest
python -m cipherspectrum_labeling.scripts.verify_label_manifest
```

本工具只生成标签映射和审计材料，不启动 TrafficFormer、训练、GMM 或数据集划分。

主要依据：

- CipherSpectrum 官方数据页：https://cgi.cse.unsw.edu.au/~cspectrum/
- 论文中的数据组成与 Traffic Labelling 说明：https://arxiv.org/html/2503.20093v4
