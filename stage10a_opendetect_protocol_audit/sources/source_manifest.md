# Source Manifest

- Paper PDF: `/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/Open-Detect/paper/Meng 等 - 2025 - Detection of Unknown Attacks Through Encrypted Traffic A Gaussian Prototype-Aided Variational Autoe.pdf`
- Paper SHA-256: `21d4f1d09e73227cfdf7762aa43aeda178e34cb274b27543f4f8b627043a6efa`
- Extracted text: `sources/open_detect_paper.txt`
- Open-Detect parent commit: `f26ac911be4324110a6632912388cd19413353bd`
- Official nested code commit: `b50a18515a01799468c10f6c9b60c01f8a6a4e7c`
- Sibling USTC v6 checkpoints: `15` complete `model_best.pt` files; each SHA-256 is recorded in `outputs/ustc_reproduction/ustc_open_set_reproduction.csv`
- Sibling final matrix: `0` paper-exact, `8` local approximations, `1` blocked by missing data, `1` excluded by scope
- Stage9 result source: `stage9_cipherspectrum_final_test/outputs/summary/cross_setting_results.csv`

`Projects/Open-Detect` and `Projects/unknown_traffic_project` are sibling method projects.  This audit reads the former and writes only the latter.  No supplementary document was found locally.  The released code is sufficient for implementation auditing, but published materials do not contain the exact author folds/seeds and do not make an author-exact reproduction possible.  The nested official checkout has pre-existing tracked `__pycache__/*.pyc` changes; no tracked Python source change was observed or made by this audit.
