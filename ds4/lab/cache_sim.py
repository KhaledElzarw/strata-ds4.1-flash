"""Expert-cache simulator over a strata-dsv41 DS4_TRACE routing dump.

Decode records: u8 layer (<0x80), u8 T, T x 6 int16 expert ids.
Prompt-chunk records: u8 (layer | 0x80), u16 T, T x 6 int16 ids.
Reports the decode-time VRAM hit rate for caches of N experts under several policies,
and the CPU-lane picks per token that follow.
"""
import struct, sys, heapq
from collections import OrderedDict, defaultdict
from pathlib import Path

K, E, L = 6, 384, 40


def load(path):
    b = Path(path).read_bytes()
    i, recs = 0, []
    while i < len(b):
        lay = b[i]
        if lay & 0x80:
            T = b[i + 1] | (b[i + 2] << 8); i += 3; dec = False; lay &= 0x7F
        else:
            T = b[i + 1]; i += 2; dec = True
            if T & 0x80:
                raise SystemExit("trace has lookahead data; unsupported")
        n = T * K
        ids = struct.unpack_from("<%dh" % n, b, i); i += 2 * n
        for t in range(T):
            recs.append((dec, lay, ids[t * K:(t + 1) * K]))
    return recs


def profile_rank(path):
    b = Path(path).read_bytes()
    u = struct.unpack("<%dI" % (len(b) // 4), b)
    keys = [(u[l * E + e], l * E + e) for l in range(L) for e in range(E)]
    return [k for _, k in sorted(keys, reverse=True)]


def run(recs, N, policy, rank=None, half_life=4000, every=64):
    hits = picks = 0
    if policy == "static":
        cache = set(rank[:N])
        for dec, lay, ids in recs:
            if dec:
                for e in ids:
                    picks += 1; hits += (lay * E + e) in cache
        return hits / picks
    if policy == "lru":
        c = OrderedDict()
        for dec, lay, ids in recs:
            for e in ids:
                k = lay * E + e
                if k in c:
                    c.move_to_end(k); hits += dec
                else:
                    c[k] = 1
                    if len(c) > N:
                        c.popitem(last=False)
                picks += dec
        return hits / picks
    if policy == "lfu_decay":   # decayed counts, profile-seeded; the cache = top-N, rebuilt every `every` routing steps
        score = defaultdict(float)
        for r, k in enumerate(rank):
            score[k] = 1.0 / (1 + r / 2000.0)   # seed: the profile order, weak
        cache = set(rank[:N])
        decay = 0.5 ** (1.0 / half_life)
        w, step = 1.0, 0
        for dec, lay, ids in recs:
            for e in ids:
                k = lay * E + e
                if dec:
                    picks += 1; hits += k in cache
                score[k] += w
            step += 1
            w /= decay
            if w > 1e30:   # renormalise
                for kk in score: score[kk] /= w
                w = 1.0
            if step % every == 0:
                cache = set(heapq.nlargest(N, score, key=score.get))
        return hits / picks
    raise ValueError(policy)


def main():
    recs = load(sys.argv[1])
    rank = profile_rank(sys.argv[2])
    ndec = sum(1 for d, _, _ in recs if d)
    print("routing steps: %d decode (%d tokens), %d prompt" % (ndec, ndec // L, len(recs) - ndec))
    configs = [("1x 3090 (measured slots)", 2059), ("2x 3090", 4950), ("3x 3090", 7850), ("4x 3090", 10750)]
    print("%-26s %6s %8s %8s %8s %8s" % ("config", "N", "static", "LRU", "LFU-dec", "picks/tok"))
    for name, N in configs:
        s = run(recs, N, "static", rank)
        r = run(recs, N, "lru")
        f = run(recs, N, "lfu_decay", rank)
        best = max(s, r, f)
        print("%-26s %6d %7.1f%% %7.1f%% %7.1f%% %8.1f" % (name, N, 100 * s, 100 * r, 100 * f, 240 * (1 - best)), flush=True)


if __name__ == "__main__":
    main()
