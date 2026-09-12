# A-1 Strict Unknown-Free Leakage Audit

- Status: **PASS**
- Known classes: 19 — Gmail, FTP, Nsis-ay, Facetime, Weibo, Cridex, Zeus, SMB, BitTorrent, WorldOfWarcraft, Shifu, Outlook, Virut, Geodo, MySQL, Htbot, Skype, Miuref, Neris
- Unknown classes: 1 — Tinba
- Known train / validation / final-test: 384478 / 48059 / 48061
- Unknown final-test: 850
- Excluded Unknown source-train/source-validation rows: 6802 / 851
- Data manifest SHA-256: `32fc99ac29722b87523b8719c36761ff85858679bbfbfbb3808f3a08c1f67ffc`

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
