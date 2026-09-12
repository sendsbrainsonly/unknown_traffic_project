# A-2 Strict Unknown-Free Leakage Audit

- Status: **PASS**
- Known classes: 17 — Gmail, FTP, Nsis-ay, Facetime, Weibo, Cridex, Zeus, SMB, BitTorrent, WorldOfWarcraft, Shifu, Outlook, Virut, MySQL, Skype, Miuref, Neris
- Unknown classes: 3 — Geodo, Htbot, Tinba
- Known train / validation / final-test: 346651 / 43331 / 43332
- Unknown final-test: 5579
- Excluded Unknown source-train/source-validation rows: 44629 / 5579
- Data manifest SHA-256: `b730fa60e4c8f55c13d525c093d6f5831bace0d255138147f43dadb74f4710cb`

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
