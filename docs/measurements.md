# Measurement notes

All numbers are from the OpenAI-compatible `/v1/completions` endpoint on one RTX PRO 6000 Blackwell Max-Q, greedy
(`temperature: 0`), no BOS token, with `bench/` scripts. "c1..c8" are 1..8 concurrent requests of 512 generated
tokens on a code prompt; aggregate t/s. Decode rows force 2048 tokens (`min_tokens`, loop detection off) from a
short natural prompt.

## Method

- Speed comparisons alternate server boots A, A, B, B (each config loaded twice), three runs per boot, medians.
  The card is power-capped; run-to-run spread on a single boot reached ±3.5 % (c1-c8) and ±7.5 % (prose/chat).
- Output identity: 8 fixed prompts, 128 tokens each, compared as text and as token ids.
- Next-token drift: top-20 log-probabilities on 16 short and 9 long prompts (8.7k, 35k, 157k tokens), KL against a
  reference run. Two runs of the same config give exactly 0 at one request at a time; batched (8 at once) 0.0015.
  Compare runs that executed the same requests in the same order (see "Server history" in the README).
- Accuracy: `bench/accuracy.py`, 24 exact-answer prompts, 768-token budget, graded after any `</think>`.
- Long context: a planted fact retrieved from a 520k-token prompt.

## KV cache modes at the largest pool that keeps 1.5 GiB free

|mode|KiB/token|pool|accuracy|KL vs 8,8, long context (mean)|
|---|---|---|---|---|
|8,8|16.5|2,573,312|22/24|-|
|8,6|15.0|2,833,920|21/24|0.0013|
|6,6|13.5|3,142,656 → 3,121,152 with the final engine (2048-token chunks); shipped: 2,999,808 with 8192-token chunks|21-22/24|0.0065|
|8,4|13.5|2,883,584|20/24, `2+2=` → 5|rejected|

## Per-change A/B (each against the step before)

|change|c1|c2|c4|c8|other|
|---|---|---|---|---|---|
|coop MoE k-loop (0001)|+4.7 %|+8.9 %|+8.6 %|+4.3 %|math +6.4 %|
|no expert-count readback (0003)|neutral|+20-23 %|+8-9 %|+3-12 %|bit-identical|
|verify-only wide coop rows (0004)|+3.0 %| | |+20.0 %|code −0.2 %|
|drafter CUDA graph (0005)|+2.8 %| | |+4.0 %|code +5.3 %|
|draft confidence 0.45|+5.5 %| | |+3.9 %| |
|shared expert side stream (0007)|+6.1 %| | |+12.0 %|code +7.4 %|
|engram lock|not slower| | | |cold 32k worst case 16.5 s → 8.65 s|

## Rejected

|candidate|result|
|---|---|
|built-in MTP drafter|fits only at a 2,560,000-token pool; prose +11.5 %, chat +2.4 %, code −15 %, math −15.6 %|
|dense DGEMV (0002, `EXL3_DGEMV=1`)|greedy identity 2/8; c1 −5.5 %, c8 +5.4 %|
|adaptive prefill chunks (0006)|fresh server: 32k/128k 4,820/4,853 t/s vs ~4,200; after decode traffic the chunks shrink back: 128k +3.4 %, c1 −2.5 %, c8 −3.1 %; long-context KL 0.0128|
|piecewise decode CUDA graphs|correct after excluding warm-up from capture; +3 % at c8 for +474 MiB|
|draft confidence 0.30|prose +4.4 %, code +5.6 %, c4 −4.1 %; greedy identity 7/8|
|split-D GatedResidual kernel|bit-faithful, but latency-bound: 25.6 → 27.7 µs per call at 8 rows|
|larger fused-MoE row limit|+9-14 % at c2/c8 for 160-260 MB VRAM; the readback removal (0003) does better for free|

## Prefill

Cold prompts (a unique prefix per request, so no prefix-cache hits), one generated token, wall time from request to
response.

|prompt|v1 (2048-token chunks)|V2, 2048-token chunks, 3.12M pool|V2 shipped, 8192-token chunks, 3M pool|
|---|---|---|---|
|~2k tokens, prose / code / chat / math|3,044-3,268 t/s|3,453-3,507 t/s|3,475 t/s|
|32k|-|4,200-4,262 t/s|5,690 t/s (5.8 s)|
|128k|-|4,146-4,240 t/s|5,647 t/s (23.2 s)|
|520k (needle)|130.2 s, ~3,990 t/s|129.8-132.0 s, ~3,940-3,990 t/s|98.3 s, 5,294 t/s, found|
|cold 32k, slowest of 5|16.5 s|8.65 s (engram lock)|-|

The v1 and 2048-chunk V2 columns come from different sessions; at 520k they agree. The shipped column is one boot
(two runs per prompt, median), measured right after an 8-user run.

Chunk-size experiments:

|chunking|pool|result|
|---|---|---|
|`chunk_size: 8192`|3,137,536|+39-40 % cold prefill at 32k and 128k; peak free VRAM 649 MiB, below the 1.5 GiB floor|
|`chunk_size: 8192` (shipped)|2,999,808, `max_seq_len` 655,360|+33 % at 32k, +36 % at 128k, 520k 131 → 98 s; peak free 1,953 MiB; c1 305.3 / c8 668.8 t/s; greedy identity 8/8 text and tokens, KL short 0 / long 0.0019 mean, 0.0053 max vs 2048-token chunks|
|`chunk_size: 16384`|3,121,152|does not boot (the loader sizes its memory check by the chunk)|
|`chunk_size: 16384`|2,999,808|boots, then runs out of GPU memory on a cold 32k prefill (641 MiB free)|
|adaptive chunks (patch 0006, `EXL3_PREFILL_ADAPTIVE=1`, `prefill_chunk_size: 16384`)|3,121,152|fresh server: 4,820 / 4,853 t/s at 32k / 128k; after decode traffic chunks shrink back to ~2,000-2,300 rows: +3.4-3.7 % at 128k, long-context KL 0.0128|

`deploy/tabbyapi-prefill-chunk-size.patch` adds the separate `prefill_chunk_size` setting that adaptive chunks need.
To favor context over prefill speed instead, set `chunk_size: 2048` and raise `cache_size` to 3,121,152 and
`max_seq_len` to 1,048,576.

## Prefix cache under an agent workload

Live server (3M pool, 640k per request, 8192-token chunks, `max_batch_size` 8), 4,612 finished requests over 23 h
from coding agents through an OpenAI-compatible router.

|period|in flight|requests with a cache hit|prompt tokens from cache|queue median|
|---|---|---|---|---|
|09:00-18:59 (≤12 conversations in flight)|p90 5, p99 10|96-99 %|91-99 %|0 s|
|19:00-08:00 (15-22 in flight, 2.0-2.8M prompt tokens)|15-22|0-70 %|2-10 %|501 s (p90 626 s)|
|first 10 min after the fix|≤8|80 %|77 % (cold cache)|2 s (max 58 s)|

Cause: the recurrent-checkpoint budget (`memory.sysmem_recurrent_cache` 4096 MB ≈ 39 checkpoints of ~104 MiB: 36
Gated DeltaNet layers × 48 heads × 128 × 128 fp16 ≈ 54 MiB, plus the drafter's state) and KV-page eviction under
too many concurrent long prompts. Each miss re-prefilled 50-500k tokens at ~640 t/s while sharing the GPU with 8
decoding requests, queue time passed the client's 15-minute timeout, and resent requests kept the load up.
Fix: `sysmem_recurrent_cache: 65536`, plus client-side admission (≤8 in flight, ≤2.4M prompt tokens) and a
no-progress timeout instead of a fixed total.

## Prefill V3 candidates (2026-11-15)

All MEASURED on one RTX PRO 6000 Max-Q with the V2 release config (6,6 KV, 2,999,808-token pool, 8192-token
chunks, bs 8, DFlash2 n7 at confidence 0.45) unless stated. "Shotgun" = one server boot per arm, 3 runs per
suite, median with the run spread in brackets; "A,A,B,B" = four boots in that order, median of the two arm
medians. Greedy identity = the 8 fixed prompts, 128 tokens, text and token sequence. KL = next-token KL over the
top-20 on 16 short prompts and 9 long prompts (8.7k / 35k / 157k tokens).

### MoE fused-tier row tiles (`EXL3_MOE_MTILE=1`), shotgun

Tier trace on one cold 32k prefill (188 MoE layer passes, 1,540,096 rows, 10 experts per row):

|MTILE|fused tier capacity|rows fused, 16-row tile|32-row tile|64-row tile|batched reconstruct|
|---|---|---|---|---|---|
|0 (V2)|128 rows|1,590,002 (10 %)|—|—|13,810,958 (90 %)|
|1|256 rows|112,790|165,049|3,445,459 (24 % total fused)|11,677,662 (76 %)|

|arm|cold 2k t/s|32k|128k|c1 t/s|c8 t/s|peak MiB|identity vs V2|KL short / long (mean, max)|
|---|---|---|---|---|---|---|---|---|
|MTILE 0 (V2)|3,590 [561-601 ms]|5,761|5,603|297|653|95,918|—|—|
|MTILE 1|3,945 (+9.9 %)|5,954 (+3.4 %)|5,843 (+4.3 %)|299 [269-299]|653|95,990|text 4/8, token 7/8|0.004 / 0.007, max 0.057|
|MTILE 1, fused rows 384|3,978|5,977|5,830|298|654|95,908|text 5/8, token 7/8|—|
|MTILE 1, fused rows 512|3,926|5,912|5,768|295|653|95,926|token 5/8|—|

Not bit-identical (the 64-row tile accumulates in a different order); top-1 agreement 25/25. Not shipped: it
changes greedy output.

### Prefill while other requests decode (`prefill_cooldown` / `shared_chunk_size`, TabbyAPI patch v3), shotgun

Four users stream forced 16k-token decodes; after 20 s one cold 131k-token prompt arrives. TTFT is that prompt's
wall time; slowdown is the decoders' per-user rate during its prefill against the 15 s before (≈100 t/s each).

|cooldown|shared chunk|TTFT s [3 runs]|decoder slowdown|
|---|---|---|---|
|16|2048|86.0 [86.0-86.1]|30 %|
|16|4096|53.6 [53.5-54.4]|54 %|
|8|2048|60.7 [60.2-60.8]|59 %|
|8|4096|41.4 [41.2-41.6]|71 %|
|4|2048|48.0 [47.8-48.1]|73 %|
|**4 (V2)**|**4096**|**35.0 [33.9-35.2]**|**80 %**|
|2|4096|32.0 [31.9-32.0]|87 %|
|1|4096|30.4 [30.3-30.4]|91 %|
|0|4096|28.8 [28.8-28.9]|96 %|
|4|8192|27.7 [27.6-27.8]|87 %|
|2|8192|27.0 [26.2-43.8]|92 %|
|1|8192|25.4 [25.2-25.5]|94 %|
|0|8192|24.6 [24.5-24.6]|96 %|

With 7 decoders (default setting): TTFT 37.9 s, slowdown 73 %. With 8 decoders all batch slots are taken and
the new prompt waits for a slot (≈180 s here). No setting keeps the slowdown under 15 %: each prefill chunk
stalls every decoder for its full duration, so the trade is linear in how often chunks run. The defaults stay.

### Adaptive chunks at 16384 with the pool preallocated (patch 0006 + limiter diagnostics), shotgun

At load the 3M-token pool takes the card: allocator active = allocated = 92.8 GiB, inactive split blocks 0, ~2.2 GiB
of room. The limiter cuts every 16k chunk to 7.9-9.0k rows; releasing torch's cached blocks first changes nothing.
Measured rise per chunk 1.63-1.85 GiB at 7.9-9.0k rows. Cold prefill: 32k 5,611 t/s, 128k 5,690 t/s (V2 5,761 /
5,603). Memory is held by the pool, not fragmented; see the growable pool below.

### Growable KV pool (candidate patches 0010-0012, `EXL3_KV_GROW=1`)

The pool is reserved as virtual address space and mapped as pages are used, so an empty pool leaves its VRAM free
for prefill. Shotgun, `prefill_chunk_size: 16384`, `EXL3_PREFILL_ADAPTIVE=1`:

|stage|VRAM used|chunks the limiter chose|result|
|---|---|---|---|
|after load|54.5 GiB (V2: 93.9)|—|—|
|cold 131k on the empty pool|—|16384 × 8|21.2 s, 6,196 t/s (V2 8192-chunk: 5,603)|
|filling to 2.5M tokens (5 × 500k cold prompts)|→ 88 GiB|16384, last chunk of each prompt 8.6k|86 s per 500k (≈5,800 t/s)|
|cold 131k on the 2.5M-token pool|94.2 GiB|4096 × 32 (one 8192 step at 2.8M)|21.5 s, 6,095 t/s|
|131k arriving while 4 users decode, pool full|—|—|TTFT 34.8 s, decoders −81 % (V2 default 35.0 s, −80 %)|

No out-of-memory and no failed requests across the fill. Every prompt was cold (no cached tokens): at this depth
4096-row chunks prefill as fast as 16384-row ones, so on this model the growable pool's gain comes from the
first chunks of a prompt, not from chunk size later on. Greedy identity 8/8 against the same engine with the pool
preallocated; c1 and c8 below.

Growable pool + 16k chunks against V2 (same boot order, back to back): cold 2k 3,556 vs 3,689 t/s (−3.6 %), 32k
6,248 vs 5,953 (+5.0 %), 128k 6,168 vs 5,776 (+6.8 %); runs within 1 %. KL: short 0.0000, long mean 0.0016, max
0.0056 (V2 run-to-run floor 0.0000; the 3M/8192 V2 change measured 0.0019 / 0.0053). Greedy identity 8/8.

Decode, A,A,B,B (A = pool preallocated, B = growable + 16k adaptive), c1 307.1 [307.0-307.1] vs 305.9 [305.9-305.9]
(−0.4 %), c8 681.9 [672.2-691.5] vs 657.4 [649.6-665.2] (−3.6 %, just outside the ±3.5 % band; start loads differed
by 3.2, past the 3.0 rule). Load-matched re-run, A,A,B,B: c1 304.4 [304.0, 304.8] vs 304.3 [303.9, 304.7] (0.0 %),
c8 680.1 [676.6, 683.6] vs 673.4 [672.6, 674.2] (−1.0 %, inside the ±3.5 % band), decode 2048 code 301.1 vs 303.0
(+0.6 %). Decode is unchanged. Not shipped by default: off until approved.

### Growable pool, round 2 (candidate patches 0013-0015): shrink-back, 2k, MTILE, admission, soak

Engine = V2 + 0010-0015, `EXL3_KV_GROW=1 EXL3_PREFILL_ADAPTIVE=1`, `prefill_chunk_size: 16384`. Patch 0013 releases
mapped pages from the top when the queue drains, until 10 GiB are free (`EXL3_KV_GROW_DRAIN_FREE_MIB`), never below
the highest page in use or the first 262,144 tokens, which are mapped at start (`EXL3_KV_GROW_PREMAP_TOKENS`).

**Shrink-back** (shotgun): 5 concurrent cold 500k prompts hold 2.5M tokens at once (424 s); all finish; the pool
shrinks 9,984 → 9,688 pages in 19 ms; a cold 131k prompt then runs 16384-row chunks throughout, 21.05 s,
6,229 t/s. With the earlier 6 GiB drain target the same prompt only got 8192-row chunks (23.1 s).

**2k prefill**: still 1.6-4.6 % slower with 262k tokens premapped and the limiter no longer releasing torch's cache
on every chunk (shotgun 3,600 vs 3,774 t/s; A,A,B,B −1.6 %, inside the ±3.5 % band). Not first-use mapping cost:
the 2k prompt never maps a page. The remaining gap is the per-chunk free-memory query and the larger load-time
16k measuring pass. 2k TTFT moves from 0.56 s to 0.58 s.

**MTILE=1 on the growable pool** (shotgun):

|arm|cold 2k t/s|32k|128k|greedy identity|KL short / long|
|---|---|---|---|---|---|
|growable|3,600 [562-580 ms]|6,310 [5,181-5,236 ms]|6,208 [21,043-21,157 ms]|—|—|
|growable + MTILE 1|3,948 (+9.7 %) [512-533 ms]|6,401 (+1.4 %)|6,319 (+1.8 %)|text 6/8, token 7/8|0.003 / 0.003, max 0.009|

Per-position KL along 48-token greedy continuations of the 9 long prompts (276 positions compared up to the first
token divergence): mean 0.010, 34 positions above 0.01, in 8 of 9 prompts; greedy text diverges within 48 tokens in
7 of 9. The 157k prompts carry the largest values (0.37-0.50 at single positions, on near-ties such as `<think>`
0.77 vs 0.37). After a divergence the continuations are different texts (KL 6-19 at the next positions, not
counted above). A pattern, not one isolated token: the 64-row tile changes the summation order of every
prefill row, so long contexts drift. The MTILE gain on top of the growable pool is small at 32k/128k (+1.4/+1.8 %).

**Admission past 8 users** (growable pool, shotgun):

|`max_batch_size`|VRAM after load|8 users decoding + cold 131k: TTFT|8 decoders during|c1 / c8|
|---|---|---|---|---|
|8|57.7 GiB|180.3 s (waits for a slot)|73 t/s (unaffected)|—|
|9|57.9 GiB (+148 MiB)|29.4 s|13.4 t/s (−80 %)|306.6 / 673.6|
|10|58.1 GiB (+348 MiB)|29.4 s|13.3 t/s (−80 %)|303.8 / 678.0|

A ninth slot costs ~150 MiB (one more recurrent state and verify scratch) and turns a 3-minute wait into 29 s,
paid by ~29 s at a fifth of the speed for the eight running streams. c1/c8 unchanged.

**Soak** (90 min, 15 cycles): each cycle 4 concurrent cold 400k prompts + 2 decoders fill the pool to 87 GiB, all
finish, then c1, c8 and a cold 131k. 0 errors, 0 illegal memory accesses, 0 out-of-memory, 63 shrinks. c1 median
293.4 (first three 303.1 / 300.2 / 289.1, last three 293.6 / 292.6 / 296.6), c8 median 689.8 (first 694.8 /
664.1 / 695.7, last 709.8 / 689.8 / 656.2), cold 131k 20.9-21.4 s throughout. Greedy identity at the end vs the
start 6/8: both diverging prompts change at a near-tie after ~200 characters, the server-history effect
described in the README (it also occurs on V2).

### Final A,A,B,B: approved set vs V2 (2026-11-16 00:12-00:44)

A = V2 (live). B = V2 + candidate patches 0010-0015, TabbyAPI patch v3, `prefill_chunk_size: 16384`,
`EXL3_KV_GROW=1`, `EXL3_PREFILL_ADAPTIVE=1`. 3 runs per suite per boot; start loads 0.4-1.7 (VALID).

|metric|A (A1 / A2)|B (B1 / B2)|Δ|
|---|---|---|---|
|c1 t/s|305.3 (305.8 / 304.8)|304.3 (304.4 / 304.2)|−0.3 %|
|c8 t/s|673.7 (673.2 / 674.1)|667.7 (669.1 / 666.2)|−0.9 %|
|decode 2048 code|302.1 (303.3 / 301.0)|299.1 (299.6 / 298.6)|−1.0 %|
|cold 2k t/s|3,585 (3,597 / 3,572)|3,527 (3,537 / 3,516)|−1.6 %|
|cold 32k t/s|5,745 (5,766 / 5,724)|6,203 (6,201 / 6,204)|**+8.0 %**|
|cold 128k t/s|5,689 (5,691 / 5,687)|6,184 (6,184 / 6,183)|**+8.7 %**|
|VRAM peak (whole run)|96,024 / 95,990 MiB|72,004 / 71,964 MiB|−23.5 GiB|
|greedy identity|8/8|8/8 (text and token)|—|
|KL short / long|0 / 0 (floor)|0.0000 / 0.0016, max 0.0056|—|

Every run of every suite lies within 1 % of its arm median except one c8 run in A1 (650.0) and one code run per B
boot (284). Under load (4 users decoding + cold 131k) the B arms ran 27.3 s TTFT with decoders at 12.9 t/s against
34.3 s / 20.2 t/s: with `prefill_chunk_size: 16384` TabbyAPI's shared chunk defaults to half of it, 8192. Pinned to
4096 (`shared_chunk_size: 4096`, one boot): 34.6 s [34.3-34.7], decoders 20.0 t/s, identity 8/8, c1 305.3 / c8
669.3, i.e. V2's behaviour under load. Pin `shared_chunk_size: 4096` with this configuration unless the faster
first token for new prompts (and slower running streams) is wanted.

### Decode ports from the QMesh fork (candidate patch 0008), A,A,B,B

B = all three switches on. Start loads 7.0-12.3 (spread 5.3; arms interleaved).

|metric|A|B|Δ|
|---|---|---|---|
|c1 t/s|300.4 [299.3, 301.5]|297.6 [295.2, 300.1]|−0.9 %|
|c8 t/s|655.8 [649.7, 661.9]|652.6 [652.3, 653.0]|−0.5 %|
|decode 2048 code|295.0|294.3|−0.3 %|
|greedy identity|8/8|8/8|—|
|KL short / long|0 / 0|0 / 0|—|

Single-switch shotgun arms: defer-confidence c1 304.6 / c8 661.6, batch readback 300.0 / 656.8, window batch
301.0 / 654.1, off 299.2 / 652.9. Lossless but no measurable gain on this stack: the sync the ports remove is
short next to the verify forward. Not shipped.

### Where cold prefill time goes (nsys, one cold 32k prefill, 8192-token chunks, V2 engine)

Kernel time by NVTX range of the launch (5.51 s of GPU work, 4.9 % idle in the span):

|component|share|
|---|---|
|routed MoE (fused + batched reconstruct + gather)|41.5 %|
|layer glue (gated residual mix/apply, norms, copies)|18.7 %|
|sparse-attention QSA indexer (`_qsa_sparse_split_kernel` alone 15.8 %)|17.4 %|
|Gated DeltaNet layers (projections + chunked delta rule)|16.8 %|
|full attention|3.9 %|
|shared expert|1.4 %|
|engram / embeddings|<0.1 %|
|DFlash2 drafter|<0.1 % (detached: 32k 5,808 vs 5,761 t/s attached, −2.3 GiB VRAM)|
