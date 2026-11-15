"""Decode/prefill benchmark for strata-dsv41 (DeepSeek-V4.1-Flash) on this PC.

    python bench_dsv41.py --exe EXE --pack PACK --out OUT.json [--rounds 2] [--max-tokens 384] -- [engine args...]

Loads the engine once in --serve mode, sends every prompt (non-thinking chat template, greedy) for each round,
and records prompt ms, decode ms and tok/s from the engine's DONE line. Engine stderr goes to OUT.log.
Round 1 warms the RAM expert store; later rounds are the measured ones.
"""
import argparse, json, statistics, subprocess, sys, time, threading
from pathlib import Path

LAB = Path(__file__).resolve().parent
TOOLS = Path(__file__).resolve().parents[2] / "tools" / "dsv41"   # this repo's tools (tokenizer, smoke)
# the prompt texts below still read files at their 2026-11-15 baseline paths, so runs stay comparable with that baseline
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))

BOS, USER, ASSIST = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜></think>"


def read(p, limit=None):
    t = Path(p).read_text(encoding="utf-8", errors="replace")
    return t if limit is None else t[:limit]


def prompts():
    ab = read(r"D:\Projects\local_ai\strata-lab\ab.py", 9000)
    gw = read(r"D:\Projects\local_ai\strata-lab\gpuwatch.py", 5000)
    big = "".join(read(p, 20000) for p in [
        r"C:\src\Strata\tools\dsv41\pack_gguf.py", r"C:\src\Strata\tools\dsv41\tune.py",
        r"C:\src\Strata\tools\dsv41\smoke.py", r"D:\Projects\local_ai\strata-lab\ab.py"])
    return [
        ("code_lru", "Implement an LRU cache in C++17 with O(1) get and put. Include a short usage example in main()."),
        ("code_parse", "Write a Python function that parses one line of an nginx combined access log into a dict "
                       "(ip, time as datetime, method, path, status, bytes, referrer, user agent). Handle malformed lines. Add three pytest tests."),
        ("agent_review", "You are a coding agent. Review this file and list concrete bugs or risky spots with line references, "
                         "then propose a patch for the most serious one.\n\n```python\n" + ab + "\n```"),
        ("agent_rewrite", "Rewrite this script with full type hints and docstrings. Keep the behaviour identical. Output only the new code.\n\n```python\n"
                          + gw + "\n```"),
        ("prose_moe", "Explain to a smart beginner how a mixture-of-experts language model decides which experts handle each token, "
                      "and why that makes local inference memory-bound. About 300 words."),
        ("prose_story", "Write a short story (about 300 words) about a lighthouse keeper who discovers the light has been signalling to someone."),
        ("long_summary", "Summarize what each of these files does, how they connect, and the three riskiest assumptions they make.\n\n" + big),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=384)
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    extra = argv[cut + 1:]

    import smoke
    tok = smoke.tokenizer(Path(a.pack))
    items = [(name, tok.encode(BOS + USER + text + ASSIST, parse_special=True)) for name, text in prompts()]
    out = Path(a.out)
    log = open(out.with_suffix(".log"), "w", encoding="utf-8", errors="replace")
    cmd = [a.exe, "--pack", a.pack, *extra, "--serve"]
    log.write("CMD " + " ".join(cmd) + "\n"); log.flush()
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    threading.Thread(target=lambda: [log.write(l) or log.flush() for l in p.stderr], daemon=True).start()
    while True:
        line = p.stdout.readline()
        if not line:
            print("engine exited before READY (see %s)" % out.with_suffix(".log"), flush=True)
            sys.exit(2)
        if line.startswith("READY"):
            break
    load_s = time.time() - t0
    print("READY after %.0f s" % load_s, flush=True)
    rows = []
    for rnd in range(1, a.rounds + 1):
        for name, ids in items:
            mx = 256 if name == "long_summary" else a.max_tokens
            log.write("=== round %d %s prompt %d\n" % (rnd, name, len(ids))); log.flush()
            p.stdin.write("GEN %d %s\n" % (mx, ",".join(map(str, ids))))
            p.stdin.flush()
            gen, done, ids_out = 0, None, []
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                if line.startswith("T "):
                    gen += 1
                    ids_out.append(int(line.split()[-1]))
                elif line.startswith("DONE"):
                    done = line.split()
                    break
            if done is None:
                print("engine stopped during %s" % name, flush=True)
                sys.exit(3)
            # DONE <produced> <prompt tokens> <prompt ms> <decode ms> <finish> ...
            gen = int(done[1]) if done[1].lstrip("-").isdigit() else gen
            pms, dms = float(done[3]), float(done[4])
            r = {"round": rnd, "id": name, "prompt_tokens": len(ids), "gen": gen, "prompt_ms": pms, "decode_ms": dms,
                 "decode_tps": gen * 1000.0 / dms if dms > 0 else 0.0, "prompt_tps": len(ids) * 1000.0 / pms if pms > 0 else 0.0,
                 "done": " ".join(done), "ids": ids_out}
            rows.append(r)
            print("r%d %-14s prompt %6d (%7.1f tok/s)  gen %3d  decode %6.2f tok/s" % (
                rnd, name, len(ids), r["prompt_tps"], gen, r["decode_tps"]), flush=True)
            json.dump({"cmd": cmd, "load_s": load_s, "rows": rows}, open(out, "w"), indent=1)
    p.stdin.write("QUIT\n"); p.stdin.flush()
    try:
        p.wait(timeout=120)
    except Exception:
        p.terminate()
    for rnd in range(1, a.rounds + 1):
        d = [r["decode_tps"] for r in rows if r["round"] == rnd]
        print("round %d: median decode %.2f tok/s (min %.2f, max %.2f)" % (rnd, statistics.median(d), min(d), max(d)), flush=True)


if __name__ == "__main__":
    main()
