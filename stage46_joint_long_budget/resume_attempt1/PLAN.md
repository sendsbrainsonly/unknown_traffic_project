# Stage46 Resume Attempt 1 — Continue from verified epoch-75 snapshots

## Objective

Complete the interrupted Stage46 original-initialization joint 100-epoch runs for ISCX-VPN and ISCXTor without overwriting the first attempt. Continue from each complete epoch-75 recovery checkpoint, replay epochs 76–100, freeze the existing 20/50 selections plus resumed 100-epoch selection for both datasets, then run the originally planned Known Test evaluation and independent replay.

## Frozen scope

- Same Stage20 frozen flow membership, coarse service labels, seed 2022, original pretrained initialization, three-view architecture, optimizer/schedule/loss, effective batch 64 and microbatch 32.
- Existing Stage46 code and `frozen_sources.json` must remain unchanged.
- Training and checkpoint selection use Known Train/Validation only. No Unknown samples; no Test values before all six checkpoints are frozen.
- The first Stage46 attempt and its `KeyboardInterrupt`/failure snapshots are retained as a separate interruption record. This attempt writes only under `resume_attempt1/` until it creates the new 100-epoch checkpoint and summary required by the existing freeze barrier.
- Resume begins from the last complete epoch boundary at epoch 75, not from the mid-epoch `failure_state.pt`. This restores model, all optimizer states, scheduler, GradScaler, best Known-Val state, Python/NumPy/CPU/CUDA RNG states, and DataLoader generator state.

## Recovery validation

Before GPU use, validate source locks, checkpoint schema/configuration, epoch/optimizer-step totals, best epoch against the first 75 Known-Val history rows, split disjointness, and exact next-epoch sample-order replay against the retained original `order_epoch076.npy`. Any mismatch stops the attempt before training.

## Execution

- Continue epochs 76–100, saving immutable recovery snapshots at 85, 95 and 100; save a distinct 100-epoch best-validation checkpoint in this attempt directory.
- Keep the original Stage46 partial history, per-epoch outputs, 20/50 checkpoints, and failure snapshots intact.
- Only after both datasets pass training and hash checks, write the two 100-epoch train summaries and invoke the original Stage46 all-six checkpoint freeze barrier.
- Evaluate with the original Stage46 evaluator only after that barrier; then use its independent metric/identity replay. Test is not used to select checkpoints or tune anything.
- If interrupted again, preserve the attempt as partial and resume from its last complete recovery checkpoint in a new attempt directory. Do not restart inside this directory.

## Acceptance

Both resumed runs reach epoch 100 with the registered optimizer-step totals, Known-Val-only selection, all source locks intact, and the original first-attempt artifacts unchanged. Then all six prefix checkpoints are frozen before any Test evaluation; evaluation and independent replay must pass. No automatic next-stage work.
