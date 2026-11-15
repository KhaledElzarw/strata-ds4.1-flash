"""SHA-256 check of the IQ1_M copy on NVMe: shards 1-5 and the drafter against setup.py's pins,
shard 6 (the Engram, no pin in setup.py) against the original on D:."""
import hashlib, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DST = Path(r"C:\models\DSV41-IQ1_M")
SRC = Path(r"D:\models\DSV41")
PINS = {
    "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf": (7976914240, "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf": (5780704, "3e8ddd11441edd4bc19bacaef5dbe38818af2cbdf0820fb70f1d2d851fcc192a"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00006.gguf": (39340012896, "be4ceb1caf1bed28e886860cf6a52adc8f5ca9eb0d2098ade39bb4179d48b323"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00006.gguf": (38698805888, "758c66ba65236e8ce0820c21a7b6f9b2947681df8cc509f672a77722e8191b9a"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00006.gguf": (39707055008, "04a5bea2b38eb576f49efc19c0a677768d6e2a8ea5772d72fb6cb79e1733a212"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00005-of-00006.gguf": (2969539840, "7c8dbee99890f05dbc420eba6f44393f5228d06248fe61e1ae8ed002449c7549"),
}
ENGRAM = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00006-of-00006.gguf"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb", buffering=0) as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    bad = 0
    for name, (size, want) in PINS.items():
        p = DST / name
        got_size = p.stat().st_size
        got = sha(p)
        ok = got_size == size and got == want
        bad += not ok
        print(("OK  " if ok else "BAD ") + name, got_size, got[:16], flush=True)
    if "--no-engram" not in sys.argv:
        with ThreadPoolExecutor(2) as ex:
            a, b = ex.map(sha, [DST / ENGRAM, SRC / ENGRAM])
        ok = a == b
        bad += not ok
        print(("OK  " if ok else "BAD ") + ENGRAM + " copy == original", a[:16], b[:16], flush=True)
    print("ALL OK" if not bad else f"{bad} BAD", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
