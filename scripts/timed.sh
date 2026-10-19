#!/usr/bin/env bash
# Run a command and append its wall-clock time to runs/stage_times.jsonl, so GPU-hours per stage come from
# measurements. Usage: bash scripts/timed.sh <stage-name> <command...>
# One line per run: {"stage", "seconds", "gpu_hours", "gpu", "exit_code", "started"}. gpu_hours = wall hours x GPUs
# visible to the command (1 under the default single-GPU launch; the DDP step sets 2 via GPUS=2).
set -uo pipefail
stage=$1; shift
mkdir -p runs
start=$(date +%s); started=$(date -u +%FT%TZ)
"$@"; code=$?
secs=$(( $(date +%s) - start ))
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
python - "$stage" "$secs" "${GPUS:-1}" "$gpu" "$code" "$started" <<'PY'
import json, sys
stage, secs, gpus, gpu, code, started = sys.argv[1:7]
row = {"stage": stage, "seconds": int(secs), "gpu_hours": round(int(secs) / 3600 * int(gpus), 4),
       "gpu": gpu or None, "exit_code": int(code), "started": started}
with open("runs/stage_times.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print("[timed]", json.dumps(row))
PY
exit "$code"
