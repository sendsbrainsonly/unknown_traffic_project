# Stage 16 dataset parity audit

The frozen data side passes even though the method-identity side does not.

| Protocol | Train | Validation | Label space | Status |
|---|---:|---:|---|---|
| A | 2,730 | 335 | 6 Services | PASS |
| C | 1,367 | 184 | 6 Services | PASS_WITH_LOW_VALIDATION_SUPPORT |

Canonical hashes use sorted UTF-8 records and a terminal newline:

- dataset manifest: `bcea0ae3de1c0d5305528ae26861b1874fb8659e78b0f3ed43e61dff295b8dbe`
- Train membership: `b52e7dd17bc12561025e262a5d49960f3c048cb72fc7961dcafdef7aad1b27df`
- Validation membership: `95de3f2e2088ea4ed87c261de136c7cb20a0bedce888eeae2be89e92df519afd`
- Service labels: `d46449f1962f4a976705f797791269455484c0af2900ef9b3d00419a6f2864e4`

Every A flow ID is unique; A and C retain all six Service labels. All labels
remain `WEAK_CAPTURE_LABEL`. No Known Test or Unknown Test feature was read.
Because no second method was run, cross-method runtime parity is not claimed.
Machine-readable A/C hashes are in `dataset_parity_audit.csv`.
