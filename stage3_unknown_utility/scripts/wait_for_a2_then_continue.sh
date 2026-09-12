#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project
A2_STATUS="$PROJECT_ROOT/.tmux-task/stage3_a2_formal_v3/exit.status"

cd "$PROJECT_ROOT"
while [[ ! -f "$A2_STATUS" ]]; do
  sleep 30
done

if [[ "$(tr -d '[:space:]' < "$A2_STATUS")" != "0" ]]; then
  echo "A-2 formal v3 did not complete successfully; refusing downstream execution."
  exit 1
fi

bash stage3_unknown_utility/scripts/run_after_a2_v3.sh
