# Stage 14C-4 — Our-Method Training Mechanism Audit & Cleanup

This experiment audits and cleans the **F2 own-method pipeline**. F2 retains the
Stage3 model wrapper, the F2 flow-statistics representation, the existing loss,
and the prototype mechanism. Native Open-Detect is read-only reference evidence
and is not used as the cleaned implementation.

Only frozen VNAT Known Train and Known Validation inputs are visible. Known
Test, Unknown Test, DES, and Stage 14B mutation are forbidden.

The cleanup is pre-registered in `cleanup_plan.json`. It changes only mechanisms
with direct evidence or an implementation-invariant defect:

1. batch size 512 -> 128;
2. patience-5 early stopping -> disabled, exactly 100 epochs;
3. prototype reset assignment -> in-place reset with optimizer state cleared,
   preserving optimizer ownership of the prototype parameter.

No hyperparameter search is performed. The primary runs keep each protocol's
frozen seed. Fixed-seed-2022 runs are diagnostic controls only.
