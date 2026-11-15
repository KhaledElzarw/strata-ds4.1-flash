#!/usr/bin/env bash
# Rounding-level quality check for strata-dsv41: teacher-forced scoring of the quant_quality v1 text (5 segments x 1000
# DeepSeek tokens) read in decode-sized windows (--window W, default 4: the decode / short-verify attention path), one
# scored TSV per arm, then tools/quant_quality.py compare of every arm against the first.  Cross the arms: a stock
# engine under two placements gives the engine's own run-to-run noise; a candidate must stay inside it.
# usage: bash qcheck.sh NAME TAG "EXE" "ENV=.." "EXTRA ARGS" [TAG "EXE" "ENV" "ARGS" ...]
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"; name="$1"; shift
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin:$PATH"
OUT="$R/ds4/lab/runs/qq"; mkdir -p "$OUT"; IDS="$OUT/dsv41-v1.ids.json"
if [ ! -f "$IDS" ]; then
  "$R/.venv/Scripts/python.exe" -I - "$R" "$IDS" <<'PY'
import json, sys
from pathlib import Path
R, out = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(R / "tools" / "dsv41")); sys.path.insert(0, str(R / "tools"))
import smoke
tok = smoke.tokenizer(R / "models" / "pack")
segs = []
for n in ["code", "docs", "english", "multilingual", "agent"]:
    ids = tok.encode("<｜begin▁of▁sentence｜>", parse_special=True) + tok.encode(
        (R / "data" / "quant_quality" / "v1" / f"{n}.txt").read_text(encoding="utf-8"), parse_special=False)
    segs.append({"name": n, "ids": ids[:1000]})
json.dump({"segments": segs}, open(out, "w"))
print("ids:", [len(s["ids"]) for s in segs])
PY
fi
runs=()
while [ $# -ge 4 ]; do
  tag="$1"; exe="$2"; envs="$3"; xargs="$4"; shift 4
  [[ "$exe" = /* || "$exe" = [A-Za-z]:* ]] || exe="$R/$exe"
  echo "=== $name-$tag: ${envs:-defaults} $xargs  $(date +%H:%M:%S)"
  env $envs "$exe" --pack "$R/models/pack" --gpus 0 --max-context 65536 --vision off --dspark off --window "${W:-4}" $xargs \
     --score "$(cygpath -m "$IDS")" --score-out "$(cygpath -m "$OUT/$name-$tag.tsv")" 2>&1 | grep -E "^score|error|strata-dsv41:"
  runs+=("$OUT/$name-$tag.tsv")
done
"$R/.venv/Scripts/python.exe" -I "$R/tools/quant_quality.py" compare "${runs[@]}"
