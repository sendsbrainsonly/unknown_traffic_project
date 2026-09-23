# Stage 14C-4 — Our-Method Training Mechanism Audit & Cleanup

## Final conclusion

**`NEEDS_FURTHER_DIAGNOSIS`**

This cleanup retains the F2 representation, Stage3 model, existing loss,
composite checkpoint rule, scheduler, protocol-seed primary policy, logvar
guard, and prototype mechanism. Native Open-Detect is reference only.

## Selected-checkpoint results

| Protocol | Version | Epochs | Best epoch | Val loss | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| medium_seed2025 | original_f2 | 30 | 25 | 0.230403 | 0.893367 | 0.653313 | 0.864173 |
| medium_seed2025 | cleaned_primary | 100 | 93 | 0.186996 | 0.892857 | 0.727247 | 0.864414 |
| medium_seed2025 | native_opendetect_reference | 100 | 53 | — | 0.896939 | 0.739755 | 0.862802 |
| medium_seed2026 | original_f2 | 10 | 5 | 0.160869 | 0.975406 | 0.732555 | 0.972495 |
| medium_seed2026 | cleaned_primary | 100 | 53 | 0.045150 | 0.994767 | 0.980668 | 0.994803 |
| medium_seed2026 | native_opendetect_reference | 100 | 58 | — | 0.995290 | 0.983318 | 0.995223 |

## Deterministic Known split metrics

| Protocol | Version | Split | Loss | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---:|---:|---:|---:|
| medium_seed2025 | original_f2 | train | 0.226732 | 0.891875 | 0.690109 | 0.864371 |
| medium_seed2025 | original_f2 | validation | 0.230400 | 0.893367 | 0.653313 | 0.864173 |
| medium_seed2025 | cleaned_primary | train | 0.177977 | 0.895568 | 0.748663 | 0.867859 |
| medium_seed2025 | cleaned_primary | validation | 0.186996 | 0.892857 | 0.727247 | 0.864414 |
| medium_seed2026 | original_f2 | train | 0.151128 | 0.978732 | 0.723205 | 0.975102 |
| medium_seed2026 | original_f2 | validation | 0.160873 | 0.975406 | 0.732555 | 0.972495 |
| medium_seed2026 | cleaned_primary | train | 0.028631 | 0.997651 | 0.978087 | 0.997659 |
| medium_seed2026 | cleaned_primary | validation | 0.045150 | 0.994767 | 0.980668 | 0.994803 |

## Gate evidence

- Both primary runs completed exactly 100 epochs without NaN/crash: **True**.
- Weak Medium-2025 Macro-F1 recovery: **+0.073934** (gate >= +0.10).
- Normal Medium-2026 Macro-F1 change: **+0.248113** (gate >= -0.03).
- Seed sensitivity reduced on both protocols: **True**.
- Known Test / Unknown Test used: **0 / 0**.
- DES executed: **false**.

## Mechanism decisions

1. **Removed:** patience-5 early stopping. It caused every original F0/F2 run
   to stop before the epoch-50 scheduler/prototype phase.
2. **Changed:** batch size 512 to 128, based on the prior one-factor D1 result.
3. **Fixed:** prototype resets keep the same epochs and class-mean values but
   now update the existing Parameter in place and clear its Adam state. The old
   object replacement left Adam attached to the obsolete Parameter.
4. **Retained:** F2 feature, Stage3 model, loss, composite checkpoint selection,
   scheduler, protocol seed, initialization and logvar guard.
5. **Not added:** gradient clipping, class weights, weighted sampler, warmup,
   LR search, weight decay, dropout, or any new regularizer, because no direct
   evidence justifies them.

## Required answers

1. Unreasonable mechanisms: premature patience-5 stopping, batch 512 update
   starvation, and prototype Parameter replacement that breaks optimizer link.
2. Deleted/modified: early stopping removed, batch set to 128, prototype reset
   made optimizer-safe. The evidence is recorded in `training_mechanism_audit.csv`.
3. Required mechanisms retained: model architecture, F2 feature design, loss,
   representation/prototype schedule, scheduler, composite Known-Val checkpoint
   rule, initialization and numerical guard.
4. Under-training solved: **True**.
5. Weak run recovery: **+0.073934 Macro-F1**.
6. Stability: see `seed_sensitivity.csv`; reduced on both protocols =
   **True**.
7. Formal 15-run recommendation follows the `NEEDS_FURTHER_DIAGNOSIS` gate; only
   `CLEANUP_PASS` supports immediate freeze and rerun.

## Limitations

- This is a two-protocol representative audit, not the formal 15-run result.
- Fixed seed 2022 is a diagnostic control and is not selected as the production
  seed policy.
- No Unknown performance was observed, so no open-set claim is made.

Generated at `2026-09-18T04:04:09.844681+00:00`.
