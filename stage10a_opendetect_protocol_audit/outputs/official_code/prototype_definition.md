# Official Open-Detect Prototype Definition

## Exact structure

- There is exactly **one prototype mean per Known class**: `prototypes` has shape `n_classes x 128` (`model.py:13-19`).
- A prototype is **only** `mu_y`; there is no learned class covariance.  Its Gaussian is `N(mu_y, I)`.
- The query posterior is `q_x=N(mu_x, diag(exp(logvar_x)))`.
- The exact code KL is:

  `0.5 * (||mu_x-mu_y||^2 + sum(exp(logvar_x)-logvar_x-1))`.

  It therefore includes posterior variance and the log-determinant contribution; the prototype covariance is fixed identity.

## Lifecycle

1. Means are Kaiming-initialized trainable parameters.
2. The training discriminative term updates them through squared Euclidean distance to sampled `z`.
3. At epochs 50 and 80, `reset_prototype` switches to evaluation mode, computes deterministic encoder means, and replaces each prototype with its training-class mean.
4. The replacement is a new `nn.Parameter` after the optimizer was created.  The released optimizer is not rebuilt, so the reset parameter is no longer in that optimizer's parameter list.  This is an implementation hazard, not a paper-defined algorithmic step.

## Native detector

The released open-set detector does not use a GMM, Mahalanobis fit, or symmetric KL.  It uses the minimum forward KL from the query posterior to the fixed-identity class prototypes.  One component per class is therefore the precise Native structure.
