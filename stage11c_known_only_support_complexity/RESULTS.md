# Experiment results: stage11c-known-only-support-complexity-20260915-v1

## Configuration and execution

- Status: `success`
- Scope: `EXPLORATORY_ONLY_POST_HOC_DEVELOPMENT_DIAGNOSIS`
- New training/detector fitting/threshold fitting: `NO/NO/NO`
- Formal runs: `15/15`
- Final gate: **WEAK_SIGNAL**
- Mechanism: **C4**

## Data and split

All criterion features use frozen Stage11B Known Train/Validation arrays and models. Outcomes are isolated post-hoc. No Known/Unknown Test representation is a feature.

## Core results

- A1: DeltaNLL `85.510456`, Delta validation Macro-F1 `-0.012012`, instability `0.111365`
- A2: DeltaNLL `86.317375`, Delta validation Macro-F1 `-0.005346`, instability `0.102138`
- A3: DeltaNLL `86.122883`, Delta validation Macro-F1 `-0.005047`, instability `0.096008`

## Limitations

Only 15 USTC development runs are available. Correlations and selector outcomes are exploratory and retrospective, not external validation.

## Preserved evidence

All run-level inputs-by-reference, Known-only statistics, outcomes, rule decisions, LOSO results, diagnoses, commands, hashes and manifests are retained in this directory.

## Conclusion and next step

Fixed result: WEAK_SIGNAL / C4. No adaptive detector or next stage was started.
