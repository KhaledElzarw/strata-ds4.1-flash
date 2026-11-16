#!/usr/bin/env bash
# The CPU lane's speed by experts a call and rows a job (DS4_LANE_BENCH=1): one engine load at 256K with the frozen
# profile, the store filled, the bench printed at load, then the engine stopped.  Output: runs/lane/TAG.log.
# usage: bash lanebench.sh TAG [EXE] [ENV=..]
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"; tag="$1"; exe="${2:-$R/build-dsv41/strata-dsv41.exe}"; envs="${3:-}"
OUT="$R/ds4/lab/runs/lane"; mkdir -p "$OUT"; log="$OUT/$tag.log"
if tasklist //FI "IMAGENAME eq strata-dsv41.exe" | grep -qi strata-dsv41; then echo "a strata-dsv41 is already running: not started"; exit 1; fi
env DS4_LANE_BENCH=1 DS4_ATTN_SPLIT=1 DS4_STORE_HEADROOM_GB=6 $envs "$exe" --pack "$R/models/pack" --gpus 0 \
    --max-context 262144 --vision off --dspark auto --expert-profile "$R/ds4/lab/runs/profile-1339.bin" \
    --profile-readonly -v --serve > "$log" 2>&1 &
pid=$!
for _ in $(seq 1 900); do
  if grep -q "lane bench: k 16 rb 256\|lane bench: too few\|lane bench: layer .* not" "$log"; then break; fi
  if ! kill -0 $pid 2>/dev/null; then echo "engine exited early"; break; fi
  sleep 1
done
kill $pid 2>/dev/null; sleep 3; taskkill //F //IM strata-dsv41.exe > /dev/null 2>&1
grep "lane bench\|CPU lane:" "$log"
