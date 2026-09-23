# DGSB Gate Contribution Audit

Only frozen USTC Known Validation and Known Test latent files were read. No USTC Unknown file was read, no estimator was fitted, and all Stage 5 thresholds were replayed unchanged.

## Gate Effectiveness

| Setting | Split | Global-exclusive rejection | Local-exclusive rejection | Both fail |
|---|---|---:|---:|---:|
| A-1 | Known Validation | 30 (0.0624%) | 1941 (4.0388%) | 432 (0.8989%) |
| A-1 | Known Test | 19 (0.0395%) | 1944 (4.0449%) | 393 (0.8177%) |
| A-2 | Known Validation | 32 (0.0739%) | 1835 (4.2348%) | 300 (0.6923%) |
| A-2 | Known Test | 24 (0.0554%) | 1867 (4.3086%) | 298 (0.6877%) |
| A-3 | Known Validation | 0 (0.0000%) | 1892 (4.9010%) | 59 (0.1528%) |
| A-3 | Known Test | 0 (0.0000%) | 1881 (4.8724%) | 52 (0.1347%) |

Global Gate has a nonzero independent effect in every setting/split cell: **NO**.

## Global-vs-Local Class Consistency

| Setting | Split | Different count | Different rate | Category |
|---|---|---:|---:|---|
| A-1 | Known Validation | 0 | 0.0000% | BASICALLY_SAME_CLASS |
| A-1 | Known Test | 0 | 0.0000% | BASICALLY_SAME_CLASS |
| A-2 | Known Validation | 0 | 0.0000% | BASICALLY_SAME_CLASS |
| A-2 | Known Test | 0 | 0.0000% | BASICALLY_SAME_CLASS |
| A-3 | Known Validation | 0 | 0.0000% | BASICALLY_SAME_CLASS |
| A-3 | Known Test | 0 | 0.0000% | BASICALLY_SAME_CLASS |

Maximum observed class mismatch: **0.0000%**.
