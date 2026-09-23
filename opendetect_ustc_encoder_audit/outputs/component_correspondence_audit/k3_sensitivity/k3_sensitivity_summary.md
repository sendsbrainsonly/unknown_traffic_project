# K=3 Cross-Encoder Correspondence Sensitivity

This is an appendix-only robustness check. K=2 remains the fixed main analysis and the main Gate is not reselected.

| class | NMI | AMI | ARI | Hungarian accuracy | label |
|---|---:|---:|---:|---:|---|
| FTP | 0.336017 | 0.335871 | 0.329931 | 0.715431 | MODERATE_CORRESPONDENCE |
| Cridex | 0.790774 | 0.790468 | 0.777831 | 0.832112 | MODERATE_CORRESPONDENCE |
| Miuref | 0.668105 | 0.667576 | 0.641185 | 0.811573 | MODERATE_CORRESPONDENCE |
| Outlook | 0.220940 | 0.213484 | 0.338466 | 0.895086 | MODERATE_CORRESPONDENCE |

K=3 does not establish semantic modes and is not used to alter the K=2 SIMPLE-STATISTICS PERSISTENCE Gate.
