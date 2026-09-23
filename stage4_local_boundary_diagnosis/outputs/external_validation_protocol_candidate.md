# External Validation Protocol Candidate

> Candidate freeze only. No CSTNET training or evaluation was executed.

- Stage 4 Known-only diagnosis: **C — LOCAL BOUNDARY NECESSARY**.
- Proposed candidate for future untouched validation: **Component-P05**.
- Baselines: frozen Global rule and Class-P05; include Component-P05 when it is not the proposed candidate.
- Representation/model: training-split-only scaler, PCA64, full-covariance K2 per Known class; all frozen before test access.
- Class-P05: `tau_y = P05_{Known Validation, true class y}[log p_GMM(x|y)]`; accept iff `max_y(log p_GMM(x|y)-tau_y) >= 0`.
- Component-P05: assign each Known Validation sample within its true-class K2 model; `tau_yk=P05[log w_yk + log N(x|mu_yk,Sigma_yk)]`; accept iff `max_yk(s_yk-tau_yk) >= 0`.
- Fallback: if a component has fewer than 30 Known Validation samples, use its class threshold `tau_y`.
- Fixed quantile: P05 only. No P01/P02/P10, temperature, learned boundary, weighted quantile, or Unknown-aware selection.
- Target first independent benchmark: CSTNET-TLS1.3 under a new strict Unknown-Free split protocol.
- Freeze before CSTNET Unknown Test: class lists/splits, representation, K2, score definitions, P05, fallback, detector comparison, metrics, and decision rule.
- CipherSpectrum remains secondary validation only.
