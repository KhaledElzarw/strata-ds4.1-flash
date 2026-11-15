"""Union multiplier rho(T) from a DS4_TRACE dump: CPU-lane experts a T-token window reads (picks missing the LRU cache
as it stood at the window's start) per window, over the same for single tokens.

    python window_union.py D_trace.bin 2059
"""
import sys
from collections import OrderedDict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_sim as cs
recs = cs.load(sys.argv[1]); N = int(sys.argv[2])

def run(T):
    c = OrderedDict(); win = []; tok = []; cpu = allp = nwin = 0
    def touch(k):
        if k in c: c.move_to_end(k)
        else:
            c[k] = 1
            if len(c) > N: c.popitem(last=False)
    def flush():
        nonlocal cpu, allp, nwin, win
        if not win: return
        for l in range(cs.L):
            u = set().union(*[w[l] for w in win if l < len(w)])
            allp += len(u); cpu += sum(1 for k in u if k not in c)
        for w in win:
            for s in w:
                for k in s: touch(k)
        nwin += 1; win = []
    for dec, lay, ids in recs:
        ks = set(lay * cs.E + e for e in ids)
        if not dec:
            flush()
            for k in ks: touch(k)
            continue
        tok.append(ks)
        if len(tok) == cs.L:
            win.append(tok); tok = []
            if len(win) == T: flush()
    flush()
    return cpu / nwin, allp / nwin
c1, a1 = run(1)
print("T=1: %.1f CPU-lane experts a token, %.1f distinct picks" % (c1, a1))
for T in range(2, 6):
    c, a = run(T)
    print("T=%d: CPU-lane %.2fx one token, all picks %.2fx" % (T, c / c1, a / a1))
