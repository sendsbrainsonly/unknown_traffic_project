# Stage 3 Primary Execution Plan Freeze

- Status: **FROZEN — execution-level supplement**
- Created UTC: `2026-09-12T15:11:35Z`
- Created before Unknown evaluation: **yes**
- Project Git commit at freeze: `25cd43c98878075fce13b9d5e5ed956170c89f2b`
- Original protocol canonical SHA-256: `1fff4ed33211de0b123cf4c0ec6b203cc33be683ac315b610ffd8f0194550ae1`
- Original protocol raw-file SHA-256: `86718b1930c71ef6a904f16f37a199b8ceeea6600d3ceba65e7592048f54677a`

## Primary settings

`A-1`, `A-2`, and `A-3` are all primary benchmarks, not secondary
sensitivity analyses. They must be executed and reported in the fixed order:

1. `A-1`
2. `A-2`
3. `A-3`

A favorable result in an earlier setting does not permit stopping. All three
settings are trained independently from the same released initialization
semantics, with no cross-setting warm start and no reuse of the existing
20-class diagnosis checkpoint.

## Scope

This file only resolves the execution-level question of which settings are
primary. It does not alter any frozen Known/Unknown class list, seed, source
train/validation/test split, metric, threshold rule, detector, or utility gate.
Final-test evaluation is permitted only after the setting-specific checkpoint,
StandardScaler, PCA64, K1/K2 density models, and Known-Validation-only
thresholds have been frozen and hashed.
