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
