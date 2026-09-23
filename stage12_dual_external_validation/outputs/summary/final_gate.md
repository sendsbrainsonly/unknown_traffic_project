# Stage 12 Final External Gate

## Decision: EXTERNAL_CONFIRMED

## Pre-registered conditions

- both_dataset_level_mean_delta_auroc_gt_0: `PASS`
- each_dataset_has_setting_mean_delta_auroc_ge_registered_minimum: `PASS`
- no_setting_mean_delta_auroc_at_or_below_registered_floor: `PASS`
- majority_paired_seeds_des_gt_od: `PASS`
- no_setting_mean_delta_known_frr_above_registered_ceiling: `PASS`
- no_setting_mean_delta_known_macro_f1_below_registered_floor: `PASS`

## Dataset-level mean Delta AUROC

- iscx_vpn: `+0.023452`
- iscx_tor: `+0.069734`

No method was modified after the one-shot Test opening. K1/K2, GMM, DAP, DGSB, and adaptive methods were not run.
