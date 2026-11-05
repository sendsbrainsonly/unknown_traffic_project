# Current-config A-2 Known/Unknown separability

- Status: success

## Data and split

A-2 exactly uses registered current_config4164Known/417Unknown. A-1/A-3 retain preceding balanced85/85 and1031/1031 samples. This combined figure intentionally has different Test sampling policies, explicitly labeled.

## Configuration and execution

CPU-only frozen CSV replay; reuse preceding KDE/plot helpers. Gaussian Scott KDE, full observed score range, fixed Known-Val P95. Monotonic asinh and linear display exports; no score changes, new selection or threshold adjustment.

## Core results

| Scenario | Known/Unknown | AUROC % | Unknown F1 % | UFAR % | Known FRR % |
|---|---:|---:|---:|---:|---:|
| A-1 | 85/85 | 97.69 | 96.05 | 0.00 | 8.24 |
| A-2 | 4164/417 | 99.57 | 86.93 | 11.51 | 1.51 |
| A-3 | 1031/1031 | 96.09 | 92.31 | 9.21 | 5.92 |

## Preserved evidence

The local full evidence bundle retains combined asinh/linear and singleA2 PNG/PDF/SVG;6813sample scores;3metric rows;density grids;source hashes;verification and manifest. The GitHub publication subset contains the figures, aggregate metrics, verification summary and readable documentation; per-sample scores and density-grid CSVs remain local. Previous balanced/full-natural control figures remain in a sibling local audit bundle.

## Limitations

A-2 current membership was selected from earlier Test errors (Known169/Geodo141 exclusions). It favors the earlier model and is a development result, not unbiased fullA2. A1/A3 use original balanced membership and historicalDES-v1; sampling and detector versions differ. No new exclusions in this redraw.

## Conclusion and next step

Deliver current-config figure. No new training, detector tuning or automatic push.
