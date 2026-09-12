# A-3 Strict Unknown-Free Leakage Audit

- Status: **PASS**
- Known classes: 15 — Gmail, FTP, Nsis-ay, Facetime, Weibo, Cridex, Zeus, SMB, BitTorrent, WorldOfWarcraft, Shifu, Outlook, Virut, MySQL, Skype
- Unknown classes: 5 — Geodo, Htbot, Tinba, Miuref, Neris
- Known train / validation / final-test: 308836 / 38604 / 38605
- Unknown final-test: 10306
- Excluded Unknown source-train/source-validation rows: 82444 / 10306
- Data manifest SHA-256: `8d52ab0d60e3e13a6504c7299ec23b0a1a67af78bc31b1649318a35a371522ea`

## Mandatory assertions

- Unknown `used_encoder_train` count = 0: **PASS**
- Unknown `used_encoder_val` count = 0: **PASS**
- Unknown `used_scaler_fit` count = 0: **PASS**
- Unknown `used_pca_fit` count = 0: **PASS**
- Unknown `used_density_fit` count = 0: **PASS**
- Unknown `used_threshold_calibration` count = 0: **PASS**
- Unknown is used only when `original_split=test` and `used_final_test=true`: **PASS**
- Fixed source split retained; no new random flow split: **PASS**
- Raw-byte input alignment gate: **PASS**

This audit permits GPU training for this setting. It does not evaluate any Unknown score.
