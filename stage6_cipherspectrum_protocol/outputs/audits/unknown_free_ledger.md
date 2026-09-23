# Strict Unknown-Free Ledger

`created_before_unknown_evaluation = true`

| Operation | Unknown classes/data allowed? |
|---|---|
| encoder train | NO |
| encoder validation | NO |
| checkpoint selection | NO |
| scaler fit | NO |
| PCA fit | NO |
| K1 fit | NO |
| K2 fit | NO |
| class boundary calibration | NO |
| component boundary calibration | NO |
| global threshold calibration | NO |
| hyperparameter tuning | NO |
| final test | YES — FINAL TEST ONLY |

Stage 6 used class identities, sample counts, cipher-source metadata, and `split_group_id` only. It did not import or run any encoder, checkpoint, scaler, PCA, Gaussian/GMM, score, prediction, UFAR, AUROC, or AUPRC procedure. Unknown class names were accessed solely to freeze protocol roles.

Future Low/Medium/High settings require independently trained Known-only Open-Detect encoders. A full-40 checkpoint and cross-setting warm starts are forbidden. The same legal architecture initialization source is allowed.

**UNKNOWN_FREE_AUDIT = PASS**
