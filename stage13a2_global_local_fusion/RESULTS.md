# Experiment results: stage13a2-global-local-fusion-20260916-v1

## Data and split

Fifteen frozen Stage13A-1 USTC score bundles: A1/A2/A3 crossed with seeds 2022–2026. Known Validation alone fits robust normalization and P95 thresholds; USTC Test is evaluation-only.

## Configuration and execution

- Status: `success`
- Experiment type: `method-development-evaluation`
- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`
- Fusion: fixed `0.5 * Zc + 0.5 * Zk`
- New encoder training/inference: `NO`
- External Test datasets read: `NO`
- Frozen experiments modified: `NO`

## Core results

- Final gate: **GO**
- A1: centroid/kNN-10/GL-Fusion AUROC `0.995170` / `0.983570` / `0.994475`.
- A2: centroid/kNN-10/GL-Fusion AUROC `0.927394` / `0.963458` / `0.953415`.
- A3: centroid/kNN-10/GL-Fusion AUROC `0.990382` / `0.991520` / `0.991972`.

## Preserved evidence

Per-run scores, normalization statistics, metrics, thresholds, hashes, RESULTS and manifests are under `artifacts/`; aggregate evidence is under `outputs/summary/`.

## Limitations

This is USTC method development, not independent external validation. It reuses cached scores from Stage13A-1.

## Conclusion and next step

The preregistered decision is GO. No next stage was started.
