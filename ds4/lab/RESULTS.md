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

## Real-turn A/B at 256K context (ab.py runs/ab/exp-r2-256k.json, 2026-11-15 14:00-17:26)

30 recorded Hermes agent turns (replayset hermes-v1), 3 interleaved rounds, every arm `--max-context 262144` and the
frozen profile (`--expert-profile runs/profile-1339.bin --profile-readonly`), GPU gate 5% SM.  Paired geo-mean of
per-turn decode tok/s, 95% bootstrap CI, turns another GPU program overlapped left out (ab_clean.py --pmon 5):

| Arm | Change | Median tok/s by round | vs base (old exe) | vs iq1s (same exe) |
|---|---|---|---|---|
| base | build 8d41c6f08 | 26.0, 26.1, 26.1 | - | - |
| iq1s | build a5d9cc516 (AVX-512 IQ1_S rows) | 26.0, 26.8, 27.2 | +1.5% [0.6, 2.4] | - |
| attnsplit | iq1s + DS4_ATTN_SPLIT=1 | 27.0, 27.7, 28.4 | +5.1% [3.7, 6.4] | +3.6% [2.2, 4.8] |
| cache_eager | iq1s + DS4_CACHE_MARGIN=1.0 DS4_CACHE_BONUS=0 | 26.6, 26.6, 26.4 | +1.0% [-0.1, 2.2] | -0.5% [-1.7, 0.7] |
| stream | iq1s + DS4_STREAM=1 | 24.2, 27.0, 27.4 | -0.7% [-2.4, 0.9] | -2.2% [-3.5, -0.9] |
| recent16 | iq1s + DS4_RECENT=16 (1,393 slots) | 25.2, 25.4, 26.2 | -2.3% [-3.4, -1.2] | -3.8% [-4.7, -2.8] |

iq1s also cut the paired prompt time to 0.47x of base (geo-mean, CI 0.36-0.59): the short prompt reads of an agent
turn run their routed experts on the CPU rows.  The knob screen's +2.9% for stream was noise.

## attnsplit's quality (qcheck.sh, 2026-11-15 17:32-17:48)

DS4_ATTN_SPLIT=1 merges the split-KV chunks in another order (4e-7 against a double reference), so greedy tokens differ
from iq1s on 6 of 7 bench prompts (idcheck.sh as1: first differences at tokens 4-92; iq1s A/A with the same pins is 7/7).
Check: the quant_quality v1 text (5 segments x 1,000 DeepSeek tokens) scored teacher-forced with `--score --window 4`
(the decode attention path the split serves), crossed against the stock engine under other placements:

| Arm | vs s1: top-1 agreement | KL | PPL |
|---|---|---|---|
| s1 stock, profile, cache pinned (reference) | - | - | 3.319 |
| c1 attnsplit, same pins as s1 | 97.48% | 0.00458 | 3.337 |
| s2 stock, cache adapting (CACHE_BONUS=0) | 97.48% | 0.00469 | 3.329 |
| c2 attnsplit, cache adapting | 97.50% | 0.00473 | 3.327 |
| s3 stock, no profile, cache pinned | 97.60% | 0.00492 | 3.335 |

The stock engine disagrees with itself across placements as much as attnsplit disagrees with it.  Pooled text effect
(mean of the attnsplit arms minus mean of the stock arms, per-token NLL, 50-token block bootstrap): +0.0014 nats/token
[-0.0010, +0.0038], PPL x1.0014; stock-vs-stock differences reach +0.0049.  Verdict: rounding-level, inside the
engine's own noise: PASS.

## Long prompts (longab.sh, 2026-11-15 17:50-19:13)

The 100K- and 200K-token prompts (this repo's sources, then one question; 256 tokens generated, greedy) at
`--max-context 262144`, and the 100K one at 131072.  One load per arm, arms in ABCCBA order, two rounds a load, no
expert profile; no other program on the GPU (pmon).  Paired geo-mean of decode tok/s over matching (load, round, prompt):

| Context | Prompt | base | iq1s | attnsplit | attnsplit vs iq1s | attnsplit vs base |
|---|---|---|---|---|---|---|
| 262144 | 100K | 21.91 | 22.67 | 23.94 median | +5.1% over both prompts, 8 of 8 pairs faster (+2.1 to +9.7%) | +7.3%, 8 of 8 |
| 262144 | 200K | 23.45 | 23.79 | 25.25 | | |
| 131072 | 100K | 23.19 | 23.47 | 24.89 | +6.1%, 4 of 4 (+3.1 to +9.1%) | +7.0%, 4 of 4 |

Prompt reading is ~985 tok/s for every arm (100K in ~102 s, 200K in ~203 s): the split serves decode only, and iq1s,
which halved the short prompts of agent turns, does not change a prompt this long.  At 131072 the second round reused
the whole 100K prefix (99,999 tokens kept), so its prompt figure is not a read speed.  The gain at 200K tokens of
context shows the split still runs there (it falls back to the scalar kernel above its row cap).  Adopted: `launch-ds4.ps1` sets DS4_ATTN_SPLIT=1 (a shell
DS4_ATTN_SPLIT=0 still turns it off); checked by starting the server through it and reading the engine's environment.

## Where a token goes now (tp.sh, 2026-11-15 19:38-19:42)

Build 8de3bc34 (iq1s rows + split attention), `--max-context 262144`, frozen profile, one 38-token prompt then greedy
decode in CLI mode (it ended at 270-295 tokens: the store is still warming, so these runs are slower than a long
conversation's).  DS4_PROF (GPU stage time a token): 46.5 ms with the split, 45.1 without; attention 10.60 vs 11.07 ms,
waiting for the CPU lane 27.4 vs 25.5 (the lane varies more than the split saves at this length).

The DS4_DIAG build's host timers over 270 tokens (cpu_experts_iq phases, ms a token): gate+up 13.5, down 7.6, activation
quantize 0.64 + 0.38, prefetch hints 0.04 = 22.2 ms, of which the RAM store's SSD reads are 8.8 ms a token (3.9 of the
lane's 85 experts a token miss the store: it holds 9,513 of the 13,301 lane experts at 256K, 82.8 GB, sized as free RAM
minus 12 GB).  DS4_TLOG2's host spans around the expert stage are 0.01-0.03 ms (the host only queues there).

Nsight Systems, 200 tokens (tokens 21-200 measured, 46.97 ms a token under the tracer):

| | a token |
|---|---|
| kernels | 1,511 launches, one stream; GPU busy 42.65 ms (91%), idle 4.32 ms |
| idle gaps | 1,443 under 5 us (2.47 ms), 44 of 5-20 us (0.39), 22 of 20-100 us (0.81), 2 of 0.1-1 ms (0.62) |
| wait_seq_kernel (the GPU spinning for the CPU lane) | 26.89 ms, 40 launches |
| mv_kernel (weight matvecs) | 6.81 ms, 203 launches |
| gate_up / down (VRAM experts) | 1.60 / 1.09 ms |
| attn_part (the split attention itself) | 0.44 ms |
| host | cudaStreamSynchronize 33.4 ms (14.8 calls), cudaLaunchKernel 15.6 ms (1,511), cudaMemcpyAsync 7.3 ms (73) |

So the CPU lane is 27 of 47 ms: ~13 ms of row compute (~45 GB/s), ~9 ms of SSD reads for store misses, ~5 ms where the GPU
waits but the lane has not started or finished its quantize/hand-off.  CUDA graphs (#5) can win at most the 4.3 ms of idle
gaps; the sparse attention kernel is no longer a lever (0.44 ms), its stage's time is the projections' matvecs.

## The RAM store's size (screen2.sh 2026-11-15 19:47-20:07, ab.py runs/ab/exp-r3-256k.json 20:09-21:23)

The auto store is the free RAM after the GPU's weights less 12 GB: 86-89 GB at 256K on this PC (9,900-10,300 slots of the
13,301 CPU-lane experts).  Screen (7 prompts, round 2, paired vs the default): `--ssd-cache 92` +2.7% (7 of 7 prompts
faster), `--ssd-policy lfu` -1.5%, DS4_STORE_THREADS=16 +0.4%.  Real-turn A/B (30 Hermes turns x 3 rounds, split on,
frozen profile, ab_clean.py --pmon 5):

| Arm | Median tok/s by round | vs split, clean pairs | all pairs |
|---|---|---|---|
| split (auto store, 86 GB) | 26.8, 27.4, 27.0 | - | - |
| ssd92 (92 GB, 10,569 slots) | 28.1, 28.3, 27.8 | +3.5% [1.1, 6.0], n 58 | +4.7% [2.1, 8.0], n 88 |

Prompt time unchanged (paired 1.004).  Free RAM (logged every 30 s) never went below 11.6 GB.  Adopted as a 6 GB headroom
rather than a fixed size, so the store still shrinks when other programs hold RAM: DS4_STORE_HEADROOM_GB (new, default 12)
set to 6 by launch-ds4.ps1.

Correction to the timing packet above (found by the design council, Codex gpt-6.1-sol and Gemini 4 Argon, checked in
the code): the store's SSD waits happen in StoreHold, outside cpu_experts_iq's phases, so the lane's row compute is the
whole ~21 ms of gate+up+down (about 30 GB/s), not ~13 ms.  Both reviewers put the store's slot size first: every slot is
sized for the largest layer's blob (8,704,000 bytes), while 17 of the 40 layers need 6,912,000 and the mean is 7,458,048,
so exact-size slots hold ~16% more experts in the same 92 GB.  Screened and dropped at 256K with the 6 GB headroom (7
prompts, round 2, paired): DS4_MOE_TOKEN=0 (VRAM down rows before the lane wait) +1.4% (one prompt +8%, the rest ~0);
DS4_AHEAD=1 (the router lookahead's guesses read into the store) -4.5%, 7 of 7 slower.

## Size-class store slots (2026-11-15/16, ab.py runs/ab/exp-r4-256k.json and exp-r5-256k.json)

ExpertStore now cuts its arena into one class of uniform slots per distinct expert size (six in this pack, 6.9-8.7 MB),
the budget split so every class holds the same share of its experts (written by Codex gpt-6.1-sol, reviewed and
rebuilt by the lead; DS4_STORE_CLASSES=0 restores the old uniform slots).  Same bytes in a slot: token identity vs the
previous exe 7/7, legacy mode 7/7 (idcheck sc1).  Capacity: 11,748 slots in 88.1 GB against 10,466 in 91.1 GB (+16% a GB).
Real-turn A/B at a fixed `--ssd-cache 88` for both arms (30 Hermes turns x 3 rounds, twice, split on, frozen profile,
ab_clean.py --pmon 5):

| Run | legacy medians by round | classes medians by round | classes vs legacy, clean pairs | all pairs |
|---|---|---|---|---|
| r4 | 26.6, 25.4, 26.9 | 27.5, 28.2, 29.4 | +3.6% [-2.1, 9.0], n 60 | +4.3% [0.0, 8.3], n 89 |
| r5 | 27.7, 27.4, 27.6 | 29.5, 28.8, 28.1 | +4.0% [2.0, 5.8], n 72 | +4.5% [2.7, 6.2], n 88 |

Prompt time: no consistent effect (paired 0.82x in r4, 1.50x in r5, both from the small prompts' medians; the big
prompts are equal or faster with classes; pooled mean prompt time 1,866 vs 1,853 ms).  Blender and Unity ran during
r4's first round: GPU contention, and with the fixed 88 GB store free RAM fell to ~4 GB (so the launcher keeps sizing
the store by the 6 GB headroom at load, not a fixed size).  Adopted: classes on by default.

## The CPU lane's layout (screen2.sh s3 and tune_lane, 2026-11-16 01:38-02:09)

A 256K knob screen (base, `STRATA_DS4_CPU_LANE=smt`, `STRATA_DS4_LANE_FREE=0`) could not judge it: on this CPU the lane
takes 16 of 32 logical CPUs, so LANE_FREE does nothing and base vs free0 is an A/A pair, yet their round-2 medians
differ by +20% (Unity took ~1.7 cores at times).  The engine's own load-time timer (tune_lane: the same 4-expert calls
on each layout, best of 5 interleaved rounds), opened to the by-hand layouts with `STRATA_DS4_LANE_MEASURE_ALL=1`:
cores (16 threads) 0.93 ms, smt (32) 1.10 ms - the siblings make the lane 18% slower.  The lane stays a thread a core.

## Small exact CPU-lane items, batched (issue #7; ab.py runs/ab/exp-r6-256k.json, 2026-11-16 02:10-03:05)

Four changes, each likely under 1 ms a token, A/B'd together: (1) `cancel_stale` returns before locking the store when
no guess was ever started (guesses are off by default; it scanned all ~12,000 slots under the store lock every layer);
(2) at one token the request's publish is done by the routing kernel itself, one launch less a layer; (3) when none of
the store-read experts is in, the lane waits for whichever lands first, not the first in its list; (4) the second
activation quantize runs one row a job on the lane's workers (DS4_SERIAL_QUANT2=1: serial).  Store test 13/13; token
identity vs the previous exe 7/7 (idcheck b7).  Real turns at 256K, fixed `--ssd-cache 88`, split on, frozen profile:

| Arm | medians by round | vs head, clean pairs | all pairs |
|---|---|---|---|
| head (1e4e082a) | 28.7, 29.5, 28.8 | | |
| batch | 29.8, 29.6, 29.8 | +3.2% [1.8, 4.9], n 61 | +2.3% [0.9, 3.7], n 88 |

Prompt time unchanged (paired 0.98x [0.94, 1.02]).  Adopted.

## Files

bench_dsv41.py (driver), qcheck.sh (quality), idcheck.sh (identity), longab.sh (long prompts), ab.py/ab_clean.py (A/B), A_default.*, B_dspark.*, C_prof.log (profile), D_trace.bin (routing), cache_sim.py/.out, window_union.py, screen.sh, verify_copy.py/.log.
