# Stage49 user-selected first TrafficFormer epoch

- Status: success / epoch1 replay and promotion verified

## Data and split

Same frozen Stage49 eleven-class seed2022 members; Train99922/Val8438/Test11688. No role/label/group changes.

## Configuration and execution

User requested epoch1. Original job held its best state only in memory, so an isolated one-epoch replay retained the original20-epoch scheduler horizon, all data ordering, initialization, optimizer and seed. Full-attempt logs/config/history are preserved. Original orchestrator stopped before replay; original TF worker stopped only after a verified first-epoch checkpoint/features existed. Original source files remain unchanged.

## Core results

Selected epoch1 Validation Accuracy=98.98080114%, Macro-F1=96.81137865%. All stored original epoch1 metrics match within registered tolerances. This verifies metric parity of a deterministic replay; original in-memory parameters were not directly exported.

## Preserved evidence

PLAN.md, reference_epoch1.json, epoch1_replay_verification.json, source-code hashes, promotion.json, original full-attempt directory/history, original and replay tmux logs; canonical selected checkpoint/features in parent experiment.

## Limitations

Training-budget amendment is user-requested after viewing Known Validation. It is a development checkpoint choice; no Test results used. Original parameter-by-parameter equality cannot be verified against the in-memory-only original snapshot.

## Conclusion and next step

Continue the original adapter/head recipe, frozen Test extraction/evaluation and independent replay using the promoted epoch1 branch. Parent experiment remains running until its own verification completes.
