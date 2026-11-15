#!/usr/bin/env bash
# Tier-A exactness check for strata-dsv41: greedy decode of bench_dsv41.py's 7 prompts (128 tokens each, one round)
# per arm, then every arm's token ids compared with the first arm's.  Run an A/A pair first: if the same exe twice
# already differs, the check needs pins before it can judge a change.  Extra engine args: XARGS="...".
# usage: bash idcheck.sh NAME TAG "EXE" "ENV=.." [TAG "EXE" "ENV" ...]
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"; name="$1"; shift
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin:$PATH"
OUT="$R/ds4/lab/runs/id"; mkdir -p "$OUT"; tags=()
while [ $# -ge 3 ]; do
  tag="$1"; exe="$2"; envs="$3"; shift 3; tags+=("$tag")
  echo "=== $name-$tag: $exe ${envs:-defaults}  $(date +%H:%M:%S)"
  env $envs "$R/.venv/Scripts/python.exe" -I "$R/ds4/lab/bench_dsv41.py" --exe "$exe" --pack "$R/models/pack" \
     --out "$OUT/$name-$tag.json" --rounds 1 --max-tokens 128 -- --gpus 0 --max-context 65536 --dspark auto ${XARGS:-} 2>&1 | grep -E "^round|exited|stopped"
done
"$R/.venv/Scripts/python.exe" -I - "$OUT" "$name" "${tags[@]}" <<'PY'
import json, sys
out, name, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
runs = {t: {r["id"]: r["ids"] for r in json.load(open(f"{out}/{name}-{t}.json"))["rows"]} for t in tags}
a = runs[tags[0]]
for t in tags[1:]:
    same = sum(runs[t].get(k) == v for k, v in a.items())
    first = [next((i for i, (x, y) in enumerate(zip(v, runs[t].get(k, []))) if x != y), None) for k, v in a.items()]
    print(f"{tags[0]} vs {t}: {same}/{len(a)} prompts identical; first differing token per prompt: {first}")
PY
