#!/usr/bin/env bash
# Long-context A/B for strata-dsv41: bench_dsv41.py --long prompts (this repo's sources cut to N tokens) per arm,
# arms interleaved over REPEATS (A B C / C B A / ...), every arm with the frozen expert profile (see idcheck.sh).
# usage: bash longab.sh NAME CTX SIZES REPEATS TAG "EXE" "ENV=.." [TAG "EXE" "ENV" ...]
#   e.g. bash longab.sh l256 262144 100000,200000 2 base build-dsv41-base/strata-dsv41.exe "" iq1s build-dsv41/strata-dsv41.exe ""
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"; name="$1"; ctx="$2"; sizes="$3"; reps="$4"; shift 4
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin:$PATH"
OUT="$R/ds4/lab/runs/long"; mkdir -p "$OUT"
PROF="$(cygpath -m "$R/ds4/lab/runs/profile-1339.bin")"
tags=(); exes=(); envs=()
while [ $# -ge 3 ]; do tags+=("$1"); exes+=("$2"); envs+=("$3"); shift 3; done
for ((k = 0; k < reps; k++)); do
  idx=($(seq 0 $((${#tags[@]} - 1)))); [ $((k % 2)) -eq 1 ] && idx=($(printf '%s\n' "${idx[@]}" | tac))
  for i in "${idx[@]}"; do
    exe="${exes[$i]}"; [[ "$exe" = /* || "$exe" = [A-Za-z]:* ]] || exe="$R/$exe"
    echo "=== $name-${tags[$i]}-k$k: ${envs[$i]:-defaults}  $(date +%H:%M:%S)"
    env ${envs[$i]} "$R/.venv/Scripts/python.exe" -I "$R/ds4/lab/bench_dsv41.py" --exe "$exe" --pack "$R/models/pack" \
       --out "$OUT/$name-${tags[$i]}-k$k.json" --rounds 2 --long "$sizes" -- --gpus 0 --max-context "$ctx" --dspark auto \
       --expert-profile "$PROF" --profile-readonly 2>&1 | grep -E "^r[0-9]|^round|READY|exited|stopped"
  done
done
"$R/.venv/Scripts/python.exe" -I - "$OUT" "$name" "$reps" "${tags[@]}" <<'PY'
import json, statistics, sys
from pathlib import Path
out, name, reps, tags = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4:]
for t in tags:
    rows = [r for k in range(reps) for r in json.load(open(out / f"{name}-{t}-k{k}.json"))["rows"]]
    for pid in sorted({r["id"] for r in rows}):
        d = [r["decode_tps"] for r in rows if r["id"] == pid]
        p = [r["prompt_tps"] for r in rows if r["id"] == pid and r["round"] == 1]
        print(f"{t:12s} {pid:10s} decode median {statistics.median(d):6.2f} tok/s (n={len(d)}, {min(d):.2f}-{max(d):.2f})"
              f"  prompt read {statistics.median(p):6.1f} tok/s (round 1)")
PY
