#!/usr/bin/env bash
# Screening A/B for strata-dsv41 env knobs: one engine load per arm, 2 rounds (round 1 warms the RAM store).
# usage: bash screen.sh TAG "ENV=1 ENV2=2" [more "TAG|ENVS" pairs as TAG ENVS TAG ENVS ...]
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin:$PATH"
OUT="$R/ds4/lab/runs"; mkdir -p "$OUT"
while [ $# -ge 2 ]; do
  tag="$1"; envs="$2"; shift 2
  echo "=== $tag: ${envs:-defaults}  $(date +%H:%M:%S)"
  env $envs "$R/.venv/Scripts/python.exe" -I "$R/ds4/lab/bench_dsv41.py" --exe "$R/build-dsv41/strata-dsv41.exe" \
     --pack "$R/models/pack" --out "$OUT/$tag.json" --rounds 2 -- --gpus 0 --max-context 65536 --dspark auto 2>&1 | grep -E "^round|READY|exited|stopped"
done
