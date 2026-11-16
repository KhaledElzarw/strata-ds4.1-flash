#!/usr/bin/env bash
# Timing packet for the design council: one 38-token prompt, then N greedy decode tokens (CLI mode, no EOS stop),
# frozen expert profile, --max-context 262144.  Arms: host = the DS4_DIAG build's host timers (DS4_CPU_LOG, DS4_TLOG2,
# DS4_PHASES); prof / prof0 = DS4_PROF GPU stage times with / without the split attention; nsys = 200 tokens traced.
# usage: [ARMS="host prof"] bash tp.sh NAME [N]   (ARMS: any of host prof prof0 nsys; all by default)
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"; name="$1"; n="${2:-768}"
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin:$PATH"
OUT="$R/ds4/lab/runs/tp"; mkdir -p "$OUT"
ids="$(cat "$OUT/prose_moe.ids")"
args=(--pack "$R/models/pack" --gpus 0 --max-context 262144 --vision off --dspark auto --expert-profile "$R/ds4/lab/runs/profile-1339.bin" --profile-readonly --ids "$ids")
run() { local tag="$1" exe="$2" envs="$3" k="$4"; echo "=== $name-$tag ${envs}  $(date +%H:%M:%S)"
  env $envs "$exe" "${args[@]}" --n "$k" > "$OUT/$name-$tag.out" 2> "$OUT/$name-$tag.log"; echo "exit $?"
  grep -E "^decode|GPU per token|tlog2|cpu_experts_iq|phases|store:" "$OUT/$name-$tag.log"; }
arm() { case " ${ARMS:-host prof prof0 nsys} " in *" $1 "*) return 0;; esac; return 1; }
arm host && run host "$R/build-dsv41-diag/strata-dsv41.exe" "DS4_ATTN_SPLIT=1 DS4_CPU_LOG=1 DS4_TLOG2=1 DS4_PHASES=1" "$n"
arm prof && run prof "$R/build-dsv41/strata-dsv41.exe" "DS4_ATTN_SPLIT=1 DS4_PROF=1" "$n"
arm prof0 && run prof0 "$R/build-dsv41/strata-dsv41.exe" "DS4_PROF=1" "$n"
arm nsys || exit 0
echo "=== $name-nsys  $(date +%H:%M:%S)"
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe"
env DS4_ATTN_SPLIT=1 "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o "$OUT/$name-nsys" \
  "$R/build-dsv41/strata-dsv41.exe" "${args[@]}" --n 200 > "$OUT/$name-nsys.out" 2> "$OUT/$name-nsys.log"; echo "exit $?"
grep -E "^decode" "$OUT/$name-nsys.log"
