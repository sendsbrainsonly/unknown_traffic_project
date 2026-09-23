# Stage 13A-1 — DES-v1 Local kNN Support Diagnosis

## Goal

Determine whether class-conditional local kNN support improves unknown detection over the frozen DES-v0 empirical-centroid readout.

## Development Scope

This is a USTC development experiment over A1/A2/A3 and seeds 2022–2026. It is not independent external validation.

## Frozen Inputs

The experiment reuses the 15 successful Stage11B B0 runs. Their Open-Detect checkpoints, USTC splits, deterministic `mu_x`, and class sets are read-only. Stage13A-1 consumes the frozen Stage11B `sample_outputs.npz` files and independently verifies both those files and their source checkpoint hashes.

## Methods

- `CENTROID`: DES-v0 minimum squared Euclidean distance to Known-Train class centroids.
- `KNN_5`: first select the nearest centroid class, then average Euclidean distance to the five nearest Known-Train embeddings in that class.
- `KNN_10`: the same readout with ten neighbours; this is the primary method.

No score fusion, log-variance term, margin, weighting, scaler, PCA, covariance model, adaptive k, or per-class threshold is used.

## Self-Neighbour Exclusion

Known-Train kNN diagnostics explicitly mask each query's own global train index before neighbour selection. Validation, Known Test, and Unknown Test are not members of the support set.

## Calibration and Evaluation

Each score has its own global Known Validation P95 threshold using `numpy.quantile(method="higher")`. Unknown is positive and larger scores mean more unknown. Metrics reuse the corrected USTC symmetric balanced 1:1 evaluation protocol from Stage11B.

## Primary Comparison

The only primary comparison is `KNN_10 - CENTROID`. `KNN_5` is a preregistered sensitivity check.

## Frozen Gate

`GO` requires at least two scenarios with mean ΔAUROC at least +0.02, no scenario at or below −0.02, at least 8/15 positive paired seeds, no scenario mean ΔFRR above +0.02, and lower UFAR in every materially improved scenario. `CONDITIONAL_GO` requires at least one material scenario with no clear harm and safe FRR. Otherwise the result is `NO_GO`.

## Prohibited External Test Use

ISCX-VPN Test, ISCXTor2016 Test, and CipherSpectrum Test are not inputs to this stage. No new encoder is trained and no frozen upstream experiment is modified.

## Outputs

Run evidence is stored under `artifacts/`; aggregate tables and the fixed gate are stored under `outputs/summary/`.
## Recorded Final Result

# Stage 13A-1 Final Gate

## Decision: CONDITIONAL_GO

## Primary kNN-10 versus centroid

| Scenario | Mean ΔAUROC | Positive seeds | Mean ΔUFAR | Mean ΔKnown FRR |
|---|---:|---:|---:|---:|
| A1 | -0.011600 | 0/5 | +0.014000 | +0.010000 |
| A2 | +0.036064 | 4/5 | -0.039333 | -0.004000 |
| A3 | +0.001138 | 3/5 | -0.003461 | -0.001483 |

## Gate conditions

- at_least_two_material_scenarios: `FAIL`
- no_clear_harm_scenario: `PASS`
- overall_majority_positive_seeds: `FAIL`
- known_frr_safe: `PASS`
- ufar_lower_in_each_material_scenario: `PASS`

- More stable sensitivity setting: `KNN_10`.
- Fusion study worthwhile: `True`.
- New encoder training: `NO`.
- External Test datasets read: `NO`.
- Frozen experiment modified: `NO`.

