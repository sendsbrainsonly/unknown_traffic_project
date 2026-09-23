# Stage 5.5 — CSTNET External Validation Final Preflight Audit

## Purpose

Audit the frozen DGSB gate contribution, Global-vs-Local class consistency, actual Open-Detect byte-input leakage, and frozen CSTNET calibration sample sufficiency before any external-validation training.

## Frozen Inputs

The Stage 5 DGSB rule and CSTNET Low/Medium/High protocol are read-only inputs. Their SHA-256 manifests are revalidated before any audit data are processed.

## DGSB Gate Contribution

Global-exclusive rejection rates are very small for A-1/A-2 and exactly zero for A-3:

| Setting | Known Validation | Known Test |
|---|---:|---:|
| A-1 | 0.0624% | 0.0395% |
| A-2 | 0.0739% | 0.0554% |
| A-3 | 0.0000% | 0.0000% |

The Global Gate therefore does not retain an independent effect across all three settings. This is a preflight gate diagnosis, not an Unknown-detection result.

## Global-vs-Local Class Consistency

Across all six A-1/A-2/A-3 × Known Validation/Test cells, Global best class and Local winning-component class agree on every sample (`0/259,992` mismatches; 0%).

## CSTNET Raw-Byte Leakage Audit

The fixed sample uses seed `20260913`, at most 30 eligible classes, and at most 20 PCAPs per class. It reuses the audited Open-Detect adapter: first 8 packets, 80 header bytes plus 48 payload bytes per packet, IP source/destination zeroed, final 32×32 uint8 input.

## Domain/SNI Visibility

The fixed sample contains 30 eligible classes and 600/600 valid model inputs, with both Known and Unknown audit roles represented in every openness fold. Raw PCAP full-domain/SNI was recoverable in 52/600 (8.67%), but none of those strings survived into the actual Open-Detect input: full domain 0%, SNI 0%, domain/token union 0%. This is `LOW_DIRECT_DOMAIN_LEAKAGE`.

IP endpoint fields were masked in 4,800/4,800 encoded packets. TCP/UDP ports remained visible in 4,800/4,800 checked headers, so port visibility remains a warning rather than a demonstrated classifier shortcut.

## Calibration Sample Sufficiency

Known Validation counts have P05 values `11.6/11.0/12.0` for Low/Medium/High. The reliability category counts are:

| Fold | GOOD | ACCEPTABLE | WEAK | UNRELIABLE |
|---|---:|---:|---:|---:|
| Low | 27 | 54 | 14 | 18 |
| Medium | 20 | 52 | 10 | 19 |
| High | 11 | 53 | 9 | 16 |

## Fallback Risk

Only hypothetical 50/50, 80/20, 90/10, and 95/5 K2 component-weight stress is permitted. These are not observed GMM component results.

At 50/50, expected small-component `n<30` affects 109/113, 99/101, and 88/89 classes in Low/Medium/High. At 80/20, 90/10, and 95/5 it affects every Known class. The frozen minimum count remains unchanged, but the recommendation is `REVISE_BEFORE_TRAINING` under a separately frozen protocol v2.

## Final Preflight Gate

**NOT_READY**.

## Blocking Issues

The sole blocking condition is that the Global Gate has zero independent rejection in A-3 on both Known Validation and Known Test. Hash, group split, mismatch, and direct-domain leakage gates pass.

## Next Step

Stop after the preflight. Before training, separately decide whether to freeze a protocol v2 that either raises class eligibility or enlarges the Known-only calibration pool; this audit does not choose between them. CSTNET training, formal latent extraction, density fitting, boundary calibration, Unknown inference, and open-set metrics remain unexecuted.
