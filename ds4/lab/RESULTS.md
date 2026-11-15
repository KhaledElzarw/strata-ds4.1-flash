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

## Files

bench_dsv41.py (driver), A_default.*, B_dspark.*, C_prof.log (profile), D_trace.bin (routing), cache_sim.py/.out, verify_copy.py/.log.
