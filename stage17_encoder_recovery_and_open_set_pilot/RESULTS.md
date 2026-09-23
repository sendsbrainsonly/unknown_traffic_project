# Stage 17 Historical Encoder Recovery and Open-Set Pilot

## Objective and status

Status: `success / diagnostic`. Recover the exact historical A/B/C lineage and execute the preregistered two-Service, three-seed Service-LOSO pilot without changing Stage16S.

## Data and split

The experiment reused the frozen Stage16S `loso_email` and `loso_streaming` memberships for seeds 2022--2024. Known Train/Validation alone fit encoders, normalization, support and P95 thresholds. Unknown use for those operations was zero.

## Configuration and execution

E0 reused frozen corrected Open-Detect artifacts. E1 used the recovered 12-layer TrafficFormer with strict random initialization because official pretraining exposure is unverified. E2 used the recovered 30-node, seven-feature K=2 TAGCN. E3 used Known-Train per-branch z-score, 768+128 concat and the recovered linear probe. All four representations used the fixed DES-v1 definition.

## Core results

- Email mean E0/E3 AUROC: `0.718684/0.801853`; E0/E3 UFAR: `0.948413/0.805556`.
- Streaming mean E0/E3 AUROC: `0.674952/0.720970`; E0/E3 UFAR: `0.883090/0.783925`.
- E3 exceeded E0 DES-v1 AUROC in `6/6` paired runs, but strict E1 underfit/collapsed and absolute UFAR remained high.
- Six formal protocol-seed runs, 18 new checkpoints, 20,028 prediction rows, and protected-asset hashes all completed.

## Preserved evidence

The bundle retains every checkpoint, training history, embedding, prediction, score, geometry summary, input/cache audit, failed tshark attempt reference, run manifest, SHA256 and completion check. Source datasets remain in their canonical read-only paths.

## Limitations

- Only two development Services were tested; this is not a six-Service or four-dataset result.
- E1 random-init performance does not estimate the unavailable strict-Unknown-Free pretrained TrafficFormer condition.
- E3 changes both representation and Known classifier/probe quality, so graph causality is not isolated.
- Labels retain Stage16S weak capture-activity and non-capture-disjoint limitations.

## Conclusion and next step

Historical A/B/C source lineage is recovered, but only A/B historical checkpoints exist and none is Service-LOSO reusable. E3 provides a strong pilot signal and improves Streaming as well as Email, yet promotion to four datasets is `NOT_YET_SUPPORTED`. Stop here; first resolve E1 exposure/training feasibility and preregister a broader controlled validation.

## Full report

See `stage17_report.md` for all 17 requested answers and per-Service tables.
