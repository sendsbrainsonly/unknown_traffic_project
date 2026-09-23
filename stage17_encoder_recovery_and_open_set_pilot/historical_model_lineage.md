# Historical model lineage

| Model | Verified lineage | Training state | Service-LOSO use |
|---|---|---|---|
| Model A / TrafficFormer | Official repository code pinned at `6d0ba64d82e74fb130c6c7301ef20885dbfbdf29`, plus project-local short-flow adapter | USTC-20 checkpoint complete | Architecture/input code recoverable; historical weights not reusable as five-class classifier |
| Model B / FIG-TAGCN | Project-local independent reconstruction of the published FIG/TAGCN description | USTC-20 seed2 checkpoint complete | Architecture/input code recoverable; must retrain |
| Model C / concat fusion | Project-local `task09_model_c_fusion.py` | Code exists; no historical checkpoint/result found | Recreate only by executing the recovered code on newly legal E1/E2 embeddings |
| E0 / corrected Open-Detect | Frozen Stage16S checkpoints | 18 Service-LOSO runs complete | Reused without retraining |

Model B is not the separate TFE-GNN paper implementation. Model C is a post-hoc probe, not an end-to-end joint encoder.
