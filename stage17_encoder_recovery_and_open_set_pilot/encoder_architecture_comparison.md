# Encoder architecture comparison

| ID | Input | Encoder | Representation | Historical status |
|---|---|---|---:|---|
| E0 | 32x32 byte image | corrected Open-Detect ResNet18 | 128 | frozen Service-LOSO available |
| E1 | 5 packets x 64 post-Ethernet bytes | TrafficFormer Transformer | 768 | USTC checkpoint only; retrain needed |
| E2 | per-flow packet/burst graph | K=2 TAGCN | 128 | USTC checkpoint only; retrain needed |
| E3 | E1 + E2 embeddings | train-zscore concat + linear probe | 896 | code only; retrain needed |
