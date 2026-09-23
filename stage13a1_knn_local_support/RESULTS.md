# Experiment results: stage13a1-knn-local-support-20260916-v1

## Data and split

Fifteen frozen Stage11B USTC B0 representation bundles: A1/A2/A3 crossed with seeds 2022–2026. Known Train supplies support, Known Validation supplies P95 thresholds, and USTC Test is evaluation-only.

## Configuration and execution

- Status: `success`
- Experiment type: `method-development-evaluation`
- Claim scope: `diagnostic`; `DEVELOPMENT_RESULT`
- New encoder training: `NO`
- External Test datasets read: `NO`
- Frozen experiments modified: `NO`

## Core results

- Final gate: **CONDITIONAL_GO**
- A1: centroid/kNN-5/kNN-10 AUROC `0.995170` / `0.981330` / `0.983570`.
- A2: centroid/kNN-5/kNN-10 AUROC `0.927394` / `0.962132` / `0.963458`.
- A3: centroid/kNN-5/kNN-10 AUROC `0.990382` / `0.991206` / `0.991520`.

## Preserved evidence

Per-run scores, metrics, thresholds, hashes, RESULTS and manifests are under `artifacts/`; aggregate CSV/JSON/Markdown evidence is under `outputs/summary/`.

## Limitations

This is USTC method development, not independent external validation. It reuses cached deterministic representations from Stage11B.

## Conclusion and next step

The preregistered decision is CONDITIONAL_GO. Fusion research recommendation is recorded, but no next stage was started.
