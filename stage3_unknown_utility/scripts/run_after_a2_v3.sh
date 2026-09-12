#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project
SELECTOR=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/using-superpowers/scripts/select_gpu.py
BUNDLE_TOOLS=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/experiment-data-preservation/scripts
ALLOWED_GPUS=0,3,4,5,6,7
SESSION_NAME=stage3_after_a2_v3

cd "$PROJECT_ROOT"

run_gpu() {
  python "$SELECTOR" --min-free-gb 8 --count 1 --max-utilization 60 \
    --allowed "$ALLOWED_GPUS" -- "$@"
}

finish_setting() {
  local setting=$1
  run_gpu python stage3_unknown_utility/scripts/extract_latents.py \
    --setting "$setting" --phase fit --batch-size 1024 --workers 0
  python stage3_unknown_utility/scripts/fit_support_models.py --setting "$setting"
  run_gpu python stage3_unknown_utility/scripts/extract_latents.py \
    --setting "$setting" --phase final --batch-size 1024 --workers 0
  python stage3_unknown_utility/scripts/evaluate_final.py \
    --setting "$setting" --bootstrap-resamples 1000 --bootstrap-seed 0 \
    --tmux-session "$SESSION_NAME"
}

# A-2 formal v3 is run in its own tmux task and must already be complete.
finish_setting A-2

python stage3_unknown_utility/scripts/prepare_setting.py --setting A-3
run_gpu python stage3_unknown_utility/scripts/train_known_only.py \
  --setting A-3 --mode smoke --epochs 3 --batch-size 512 --workers 0 \
  --early-stopping-patience 5
run_gpu python stage3_unknown_utility/scripts/train_known_only.py \
  --setting A-3 --mode formal --epochs 100 --batch-size 512 --workers 0 \
  --early-stopping-patience 5
finish_setting A-3

python stage3_unknown_utility/scripts/summarize_primary.py
for setting in A-1 A-2 A-3; do
  python "$BUNDLE_TOOLS/refresh_artifact_manifest.py" \
    "stage3_unknown_utility/outputs/$setting"
  python "$BUNDLE_TOOLS/validate_experiment_bundle.py" \
    "stage3_unknown_utility/outputs/$setting" --verify-hashes
done
python stage3_unknown_utility/scripts/verify_stage3.py
