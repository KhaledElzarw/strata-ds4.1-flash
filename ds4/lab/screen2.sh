#!/usr/bin/env bash
# Knob screen at 256K: one engine load per arm, bench_dsv41.py's 7 prompts x 2 rounds (round 1 warms the RAM store),
# split attention on (the launcher's default), frozen expert profile.  Arms are TAG "ENV=.." "ENGINE ARGS" triples.
# usage: bash screen2.sh NAME TAG "ENVS" "ARGS" [TAG "ENVS" "ARGS" ...]
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"; name="$1"; shift
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin:$PATH"
OUT="$R/ds4/lab/runs/screen"; mkdir -p "$OUT"
while [ $# -ge 3 ]; do
  tag="$1"; envs="$2"; xargs="$3"; shift 3
  echo "=== $name-$tag: ${envs} ${xargs}  $(date +%H:%M:%S)"
  env DS4_ATTN_SPLIT=1 $envs "$R/.venv/Scripts/python.exe" -I "$R/ds4/lab/bench_dsv41.py" --exe "$R/build-dsv41/strata-dsv41.exe" \
     --pack "$R/models/pack" --out "$OUT/$name-$tag.json" --rounds 2 -- --gpus 0 --max-context 262144 --vision off --dspark auto \
     --expert-profile "$R/ds4/lab/runs/profile-1339.bin" --profile-readonly $xargs 2>&1 | grep -E "^round|READY|exited|stopped"
  grep -E "RAM store: up to|store:" "$OUT/$name-$tag.log" | head -3
  powershell -NoProfile -Command '$o = Get-CimInstance Win32_OperatingSystem; "free RAM at the end: {0:N1} GB" -f ($o.FreePhysicalMemory/1MB)'
done
