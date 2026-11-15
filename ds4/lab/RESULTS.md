# DeepSeek-V4.1-Flash IQ1_M on Strata's DeepSeek engine, one RTX 3090 (2026-11-15)

Engine: `strata-dsv41 0.1.58 build 9f0c981a5`, built for sm_86 from our fork (`C:\src\Strata\build-dsv41`).
Model: ISTA-DAS IQ1_M six-shard set + drafter, copied to NVMe (`C:\models\DSV41-IQ1_M`, SHA-256 checked), packed with
setup.py's recipe (`pack/`). Smoke test: "The capital of France is Paris." Context 65,536. Qwen server stopped during runs.

## Measured (bench_dsv41.py, 7 prompts, greedy, non-thinking)

| Arm | Round 1 | Round 2 | Round 3 | Experts in VRAM |
|---|---|---|---|---|
| A: defaults (`--dspark auto` = off) | 22.13 | 22.83 | 23.19 tok/s median | 2,059 of 15,360 (15.7 GB) |
| B: `--dspark on --dspark-conf 0.65` | 19.81 | 21.02 | 20.98 | 394 (the drafter takes the room) |

Prompt reading (A, warm): 14.5K tokens 811 tok/s, 2.8K 350 tok/s, 947 202 tok/s.
GPU: 93% SM busy, memory controller 12% busy (dmon, arm B); peak 71 C, up to 387 W.

## Where a token's 40.3 ms goes (DS4_PROF, 768-token decode, 24.2 tok/s)

waiting for the CPU lane 20.85 | attention 11.26 | routing + shared expert 4.28 | experts gate/up 1.91 | down + post 1.93 | between layers 0.06.
CPU lane: 63,271 of 184,080 expert picks (34.4%), 82 a token, ~29 GB/s of RAM read (DDR5-5600 gives 60-80).
VRAM hit rate 65.6% with 13% of the experts resident (static profile alone predicts 34%: routing is sticky within a conversation).

## Simulated hit rates (cache_sim.py over D_trace.bin, 2,435 decode tokens)

| Config | Experts in VRAM | static | LRU | LFU-decay | CPU picks/token |
|---|---|---|---|---|---|
| 1x 3090 | 2,059 | 33.9% | 74.7% | 70.5% | 61 (measured 82) |
| 2x 3090 | ~4,950 | 59.1% | 89.9% | 89.0% | 24 |
| 3x 3090 | ~7,850 | 77.8% | 94.3% | 94.9% | 12 |
| 4x 3090 | ~10,750 | 90.9% | 96.5% | 97.7% | 5.5 |

LRU ignores upload cost (optimistic); the sim's LFU is ~5 points above the engine's measured rate at 1x.

## Knob screen (screen.sh, 2026-11-15 12:07-12:35, one load per arm, 7 prompts x 2 rounds)

| Arm | Env | Round 2 median decode | vs base |
|---|---|---|---|
| base | DS4_IO_STATS=1 | 22.78 | - |
| stream | DS4_STREAM=1 | 23.43 | +2.9% |
| attnsplit | DS4_ATTN_SPLIT=1 | 23.36 | +2.5% |
| cache_eager | DS4_CACHE_MARGIN=1.0 DS4_CACHE_BONUS=0 | 21.54 | -5.4% |
| recent16 | DS4_RECENT=16 | 20.05 | -12.0% |

Per-prompt spread inside an arm is 19-26 tok/s, so +2.5-3% is inside the noise: stream and attnsplit need an interleaved
A/B with more rounds before either is called a win. With DS4_STREAM=1 the engine times streaming against the CPU lane in
decode and stops streaming requests whose experts are all in RAM (2.91 vs 0.57 ms, +414%), so it mostly turns itself off.
The serve line's "CPU lane x% of the experts" is cumulative since load (prompt picks included), not per request.
DS4_IO_STATS prints totals at exit for load + prompts + decode together, so it can't isolate decode-time disk reads.

## Draft windows: what a T-token window reads (window_union.py over D_trace.bin, LRU cache of 2,059)

| T | CPU-lane experts vs one token | all distinct picks vs one token |
|---|---|---|
| 1 | 1.00x (60.1) | 1.00x (240) |
| 2 | 1.97x | 1.69x |
| 3 | 2.92x | 2.28x |
| 4 | 3.85x | 2.82x |
| 5 | 4.77x | 3.31x |

Consecutive tokens share their hot experts (already in VRAM) but almost never their cold ones, so every drafted token
costs close to a full token of CPU-lane reads. Prompt-lookup drafting can only pay where drafts are nearly always right
(verbatim rewrites), and only with a gate that stops drafting after misses.

## Files

bench_dsv41.py (driver), A_default.*, B_dspark.*, C_prof.log (profile), D_trace.bin (routing), cache_sim.py/.out, window_union.py, screen.sh, verify_copy.py/.log.
