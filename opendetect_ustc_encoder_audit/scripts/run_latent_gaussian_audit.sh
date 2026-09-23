#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project
AUDIT="$PROJECT/opendetect_ustc_encoder_audit"
TRAIN_STATUS="$PROJECT/.tmux-task/od_formal_composite_earlystop_gpu3/exit.status"
SELECT_GPU=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/using-superpowers/scripts/select_gpu.py

while [[ ! -f "$TRAIN_STATUS" ]]; do
  sleep 30
done

if [[ "$(tr -d '[:space:]' < "$TRAIN_STATUS")" != "0" ]]; then
  echo "formal training failed; latent Gaussian audit aborted" >&2
  exit 1
fi

python "$AUDIT/scripts/freeze_best_checkpoint.py" \
  --patience 5 --max-epochs 100

python "$SELECT_GPU" \
  --min-free-gb 4 --count 1 --max-utilization 30 --allowed 0,3,4 -- \
  python "$AUDIT/scripts/extract_mu_and_compactness.py" \
    --batch-size 512 --workers 0

python "$AUDIT/scripts/run_gaussian_audit.py" \
  --reg-covar 1e-3 --n-init 3 --max-iter 300

python "$AUDIT/scripts/build_encoder_comparison.py"

python "$AUDIT/scripts/verify_latent_gaussian_audit.py"

echo "latent_gaussian_audit_complete"
