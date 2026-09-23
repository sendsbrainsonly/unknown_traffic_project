# Frozen Dual-Gate Support Boundary (DGSB)

## Scope

This is a method freeze plus USTC Known-only sanity check. It contains no USTC
Unknown calibration or evaluation and no CSTNET training or inference.

## Frozen Representation and Density Model

- PCA dimension: `64`
- Per-class GMM: `K=2`, covariance=`full`,
  `reg_covar=0.001`
- Estimators are frozen Stage 3 assets; no fit operation is performed.

## Scores and Gates

`s_yk(z) = log(w_yk) + log N(z | mu_yk, Sigma_yk)`

`G(z) = max_y logsumexp_k s_yk(z)`

`(y*, k*) = argmax_y,k s_yk(z)` and `L(z) = s_y*k*(z)`.

The Local Gate is `L(z) >= tau_local_y*k*`. Each local threshold is the fixed
P05 of Known Validation samples assigned to that component under the true-class
frozen GMM. If the assigned validation count is below 30, the class P05 is used.

After the Local Gate is frozen, `tau_global_dual` is selected only on Known
Validation from the distinct finite `G` values of Local-Gate passing samples to
make final AND acceptance closest to 95%; ties choose the higher threshold.

Final decision:

`Known iff G(z) >= tau_global_dual AND L(z) >= tau_local_y*k*`.

The predicted class is `y*`, the class of the raw maximum component joint score.

## Freeze Boundary

- `created_before_cstnet_unknown_evaluation = true`
- Unknown samples used or accessed: `0`
- Rule JSON SHA-256: `959f0a0dba96bb3c8b0519cc2c15f3ca3a9520df4de9d08636924415fe70c6a0`
