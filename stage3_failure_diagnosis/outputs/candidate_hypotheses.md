# Candidate Hypotheses

> These are post-hoc hypotheses, not a validated Adaptive-K method. No candidate was executed on the observed USTC Final Test.

## Candidate Rule R1 — Known-only mixture-validity gate

Predeclare a rule on a future development benchmark using only Known Train/Validation: require DeltaNLL2 above its Known-only median, min component weight at or above its 25th percentile, and train/validation TV and maximum covariance condition number at or below their 75th percentiles. Freeze both percentiles and action before touching the future test set.

## Candidate Rule R2 — support-expansion caution

Treat a high Known-validation upper-tail score shift or acceptance-margin shift as a caution signal rather than evidence that K2 is safe. A predeclared 75th-percentile flag may trigger likelihood/support calibration research, but must not select K on the observed USTC test.

## Candidate Rule R3 — metadata-associated component audit

When K2 assignment is strongly associated with packet/byte/duration statistics, label the mixture as metadata-associated and require independent boundary validation. Do not assume that a simple-statistics component either helps or hurts: the present directions differ by class.

## Known-only reference percentiles

- `delta_nll2`: P25=9.46338, P50=11.136, P75=15.6105.
- `min_component_weight`: P25=0.124924, P50=0.29244, P75=0.415846.
- `train_val_tv`: P25=0.00340237, P50=0.0080211, P75=0.0108476.
- `max_k2_cov_condition_number`: P25=1084.67, P50=2120.77, P75=4172.08.
- `val_own_score_shift_p95`: P25=13.4425, P50=17.1685, P75=25.3457.
- `val_global_margin_shift_p95`: P25=2.20579, P50=6.94339, P75=15.1414.

These values describe this Known-only sample only. They were not optimized against Unknown outcomes and must not be transported as validated universal cutoffs.
