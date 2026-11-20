# Measurement notes

All numbers are from the OpenAI-compatible `/v1/completions` endpoint on one RTX PRO 6000 Blackwell Max-Q, no BOS
token, with the `bench/` scripts. "cN" is N concurrent requests. V3 numbers use `bench/final.py` (streamed, decode
t/s from each stream's first token, 512 forced tokens per request, greedy unless marked "preset"); the V2 section
uses the V2 harnesses (end-to-end timing; decode rows force 2048 tokens from a short natural prompt).

## V3 release run (2026-11-19 / 20)

Harness: `bench/final.py` (streamed requests; decode t/s from each stream's first token, end-to-end t/s, time to first
token; see `bench/README.md`). A = V2 as released (tag v2.0: `launch.sh`, `config.yml`, exllamav3 v1.6.4 +
patches 0001-0007, TabbyAPI + prefill-chunk patch). B = V3 (21 patches, `deploy/config.yml`, `deploy/launch.sh`).
**Both at YaRN factor 4** (V3 ships factor 2; its effect is measured separately below).
Boots A, A, B, B, three runs per cell per boot; a cell's value is the median
of the two boot medians. Every arm ran at the card's 300 W cap ("SW Power Cap" throttle during decode and prefill,
SM 1,640-2,350 MHz, memory 13,365 MHz, 85-88 °C at the end of each section). Host load at arm start 2.2-7.7.

Decode, aggregate t/s from the first token, geometric mean over prose / chat / code / math, 512 forced tokens per
request, YaRN factor 4 (per-cell tables in the README):

|users|greedy A|greedy B|Δ|preset A|preset B|Δ|per-user greedy A → B|TTFT B|
|---|---|---|---|---|---|---|---|---|
|1|207.5|215.3|+3.8 %|196.0|194.7|−0.6 %|207.5 → 215.3|0.12 s|
|2|302.5|348.4|+15.2 %|269.9|323.8|+20.0 %|152.8 → 177.9|0.21 s|
|4|409.6|453.2|+10.7 %|370.8|444.7|+19.9 %|105.8 → 128.1|0.33 s|
|8|472.2|586.0|+24.1 %|439.4|548.7|+24.9 %|67.2 → 84.8|0.71 s|
|12|381.1|661.5|+73.6 %|358.2|607.5|+69.6 %|47.8 → 68.6|1.36 s|

The 1-2 user rows and the 4-12 user rows are two separate A,A,B,B blocks (start loads 4.4-5.4 and 4.3-5.3), both
measured after the verify-row-cap fix below.

Prefill, cold prompt → 1 token (A,A,B,B, 3 runs): 2k 3,541 → 3,284 t/s (−7.3 %), 32k 5,660 → 5,990 (+5.8 %), 128k
5,683 → 5,949 (+4.7 %). Needle: A 520,163 tokens in 96.6 s, found; B 520,164 tokens in 91.0 s, found.

Prompt cache (B), ~100k-token prompt: cold 16.8 s; resent at once 0.20 s; after 96 distinct 32k prompts (3.1M tokens,
more than the GPU pool) 0.44 s with 99,840 of 99,996 tokens cached (RAM tier); after a server restart 0.87 s (disk
tier).

Quality (`bench/identity_kl.py`, 16 short + 9 long prompts; `bench/accuracy.py`):

|pair|top-1|KL short mean (max)|KL long mean (max)|greedy identity|
|---|---|---|---|---|
|A1 vs A2 (floor)|25/25|0.0004 (0.0065)|0.0000 (0.0001)|7/8|
|B1 vs B2 (floor)|25/25|0.0003 (0.0026)|0.0000 (0.0003)|6/8|
|A vs B (three pairings)|25/25|0.0010-0.0014 (0.0087)|0.0017-0.0018 (0.0066)|3-5/8|

Accuracy, raw-completion probe (now `bench/accuracy_raw.py`): A 21/24 and 22/24, B 20/24 on both boots (YaRN factor 4);
V3 as shipped (factor 2): 22/24. Replaced as the quality probe by the chat probe below.

Memory: peak VRAM A 96.8 GiB (pool preallocated), B 76.4-91.2 GiB (pool mapped as used; 60.7 GiB after load).
Server RSS A 52-58 GiB (48.3 locked engram), B 244-268 GiB (192 GiB RAM tier + engram + checkpoints); host RAM in
use A 69-75 GiB, B 262-287 GiB.

### Verify-row cap: draft buffer height (fixed in 0021)

The first V3 run lost 6-23 % at 1-2 users on math and chat. Teacher-forced replay (A's own 512-token greedy outputs
forced through each build, so both decode identical text) separated drafting from text drift: on the same math text
A needed 93 rounds (419 accepted, 214 rejected drafts), V3 124 rounds (388 / 99), i.e. ~3.9 drafts proposed per
round instead of ~6.8. Overlap off, hybrid off and `hybrid_draft_tokens: 0` gave identical numbers; only
`max_batch_size: 9` moved it. Cause: the row cap took the batch size from the draft tensor's height, which is the
drafter's pinned buffer (all `max_batch_size` rows), so with 12 slots every round was cut to 64 // 12 − 1 = 4 drafts
even with one user. Fixed to count decoding rows; on the same text V3 then needs 93 rounds for math (= V2).

### YaRN factor 4 → 2 (A,A,B,B)

A = V3 at factor 4 (model config.json factor 4.0, `max_position_embeddings` 1,048,576, `max_seq_len` 655,360),
B = V3 at factor 2 (as released). Same engine, launcher and everything else. Greedy, 512 forced tokens per request, 3
runs per boot; t/s = median of the two boot medians [boot 1 / boot 2]; rounds = decode rounds per 512 tokens per
request (tokens − accepted drafts), median over all six runs. Host load at arm start 9.4 / 10.6 / 7.4 / 10.6
(another process on the host).

|users|prompt|factor 4 t/s|factor 2 t/s|Δ t/s|factor 4 rounds/512|factor 2 rounds/512|Δ rounds|
|---|---|---|---|---|---|---|---|
|1|prose|139.3 [142.3 / 136.3]|142.7 [142.4 / 143.0]|+2.4 %|242|240|-1.0 %|
|1|chat|182.7 [187.6 / 177.9]|182.2 [184.2 / 180.1]|-0.3 %|180|182|+0.8 %|
|1|code|250.4 [251.6 / 249.2]|259.9 [244.9 / 274.9]|+3.8 %|118|116|-1.3 %|
|1|math|327.2 [336.7 / 317.8]|316.3 [330.4 / 302.2]|-3.3 %|86|92|+7.6 %|
|2|prose|235.9 [241.3 / 230.4]|226.9 [232.3 / 221.4]|-3.8 %|228|223|-2.2 %|
|2|chat|293.4 [292.4 / 294.4]|298.9 [297.1 / 300.8]|+1.9 %|168|168|-0.4 %|
|2|code|391.1 [394.6 / 387.6]|421.4 [419.6 / 423.2]|+7.7 %|120|108|-9.6 %|
|2|math|489.8 [429.9 / 549.7]|534.1 [547.4 / 520.8]|+9.0 %|90|88|-3.0 %|
|8|prose|409.1 [404.2 / 414.0]|418.4 [420.4 / 416.4]|+2.3 %|210|207|-1.4 %|
|8|chat|491.3 [490.2 / 492.4]|498.9 [497.9 / 499.9]|+1.6 %|163|161|-1.3 %|
|8|code|676.7 [657.0 / 696.4]|700.6 [696.8 / 704.5]|+3.5 %|115|109|-5.6 %|
|8|math|783.7 [697.1 / 870.2]|857.8 [877.4 / 838.2]|+9.5 %|88|87|-1.8 %|

geomean c1: 213.7 -> 215.0 (+0.6 %)
geomean c2: 339.3 -> 351.5 (+3.6 %)
geomean c8: 571.4 -> 595.1 (+4.2 %)

The earlier one-boot shotgun of the same change (1 user math +22.8 %, code +14.4 %, prose +8.9 %) did not hold: its
factor-4 math median was a low run (266-335 across three runs), and on rounds math did not improve. At one user the
factor is neutral; the gain is mostly code and grows with concurrency. Short-prompt next-token KL factor 2 vs factor
4: 0.0151 mean, 0.0812 max, top-1 16/16. Needle at factor 2: 500,195 tokens, 86.1 s (5,808 t/s), found.

### V2 against V3 as shipped (YaRN factor 2), 4 and 12 users (A,A,B,B)

A = V2 as released, B = V3 as released (YaRN factor 2). Greedy, 512 forced tokens per request, 3 runs per boot; agg =
aggregate decode t/s from the first token, median of the two boot medians [boot 1 / boot 2]. Host load at arm start
2.6 / 9.1 / 12.5 / 5.7 (other processes on the host).

|users|tag|V2 agg|V3 agg|Δ|V2 per-user|V3 per-user|V2 e2e|V3 e2e|V2 TTFT s|V3 TTFT s|
|---|---|---|---|---|---|---|---|---|---|---|
|4|prose|284.3 [292.9 / 275.6]|329.4 [340.9 / 317.9]|+15.9 %|74.7|89.5|271.1|313.8|0.39|0.34|
|4|chat|344.3 [347.5 / 341.1]|418.5 [417.5 / 419.4]|+21.5 %|87.4|107.0|326.0|390.1|0.36|0.35|
|4|code|455.2 [455.3 / 455.1]|575.6 [578.7 / 572.4]|+26.4 %|118.6|152.8|423.0|518.6|0.34|0.36|
|4|math|533.2 [545.1 / 521.3]|705.6 [653.7 / 757.4]|+32.3 %|156.6|193.0|495.4|641.2|0.32|0.32|
|12|prose|268.0 [271.0 / 264.9]|486.1 [478.7 / 493.4]|+81.4 %|32.9|50.6|263.5|476.6|1.39|1.50|
|12|chat|318.2 [317.0 / 319.3]|573.6 [574.5 / 572.8]|+80.3 %|39.2|58.6|313.4|553.2|1.44|1.40|
|12|code|423.3 [422.9 / 423.7]|740.3 [723.3 / 757.3]|+74.9 %|53.1|79.7|415.0|710.4|1.42|1.37|
|12|math|553.5 [548.0 / 559.1]|843.1 [871.5 / 814.7]|+52.3 %|70.5|94.1|534.3|810.1|1.27|1.37|

Geometric mean over the four prompts: 4 users 392.6 → 486.4 t/s (+23.9 %), 12 users 375.9 → 645.9 t/s (+71.8 %);
per user 104.9 → 129.6 and 46.9 → 68.7 t/s.

### V3 as shipped (YaRN factor 2): README sections, one boot per release (2026-11-20)

Decode, 2048 forced tokens from a short prompt (`bench/decode_natural.py`, median of three repeats, one run), t/s and
accepted / rejected drafts; host load 3.9 (V2) / 4.6 (V3):

|prompt|V2 greedy|V3 greedy|V3 `qwen38_thinking` preset|
|---|---|---|---|
|prose|127.3 (981 / 1,074)|127.4 (942 / 1,004)|123.4 (883 / 934)|
|code|289.1 (1,677 / 589)|306.8 (1,712 / 583)|276.4 (1,662 / 837)|
|chat|155.8 (1,208 / 1,112)|152.1 (1,183 / 1,212)|152.3 (1,179 / 1,231)|
|math|319.8 (1,727 / 436)|329.3 (1,727 / 438)|321.2 (1,720 / 482)|

Five cold 32k prompts in a row (`bench/coldttft.py --n 5`): 5.21, 5.17, 5.61, 5.19, 5.16 s; server RSS 246.8-247.6
GiB, 48.3 GiB locked. `bench/accuracy_raw.py` (the old probe): 22/24. 520k needle (fresh nonce, 0 tokens cached): 520,174 tokens,
87.9 s (5,915 t/s), found; peak VRAM 65.9 GiB. A first needle run without the nonce line reused 44 % of its prompt
from the disk prompt cache (an earlier needle with the same filler) and took 54.2 s; `bench/needle.py` now starts
every prompt with a fresh nonce.

`bench/matrix.py` method (512-token greedy replies, send to last token, total t/s, median of 3; one boot each, host
load 1.4 / 1.6): code c1 / c8 / c12 313.0 / 708.4 / 573.1 → 313.5 / 881.5 / 901.3 t/s; math 297.6 / 684.7 /
559.8 → 297.5 / 805.2 / 844.5; cold 32k (`coldttft.py`, 1 run) 5.53 → 5.31 s.

### Exact-answer probe, chat form (`bench/accuracy.py`, 2026-11-20)

50 questions, each asking for one answer in a stated form, sent as chat messages (the model's template, thinking on),
greedy, max 4096 tokens; graded by exact match on the text after the thinking (trimmed; one pair of quotes and a
trailing period removed). V3 as shipped (live server, load ~3): 50/50; every reply finished its thinking (98-331
characters) and answered with the bare value. V2 as released (one boot, load 2.7): 50/50. The raw-completion probe's
misses (`1 kilometre equals` continued as "how many miles?", `sum(range(10)) in Python is` continued as prose) were
continuation-format artefacts: asked as questions, both releases answer `1000` and `45`.

### Shotgun results behind the V3 switches (one boot per change)

|change|result|
|---|---|
|drafter / accept overlap (`EXL3_SD_DRAFT_OVERLAP`)|8 users prose 456.7 → 477.9, code 772.0 → 781.5 t/s; 8-user greedy output identical to the build without it (8/8)|
|hybrid drafting with one active request (`EXL3_HYBRID_SOLO`, 16 tokens)|1 user rewriting a ~1.5k-token file with one change: 261.6 → 399.9 t/s (load 2.5), 355.9 after the row-cap fix (load 5.0); 8 users unchanged (accepted / rejected drafts 6,827 / 65 vs 6,823 / 61): stays off|
|counting-sort run table (`EXL3_MOE_COOP_FAST_RUNS`)|rotation launch 51 / 59 → 8.0 / 9.1 µs at 56 / 64 verify rows; outputs bit-identical|
|split down-stage reduction (`EXL3_MOE_COOP_SPLIT_B`)|kernel faster in isolation, slower end to end at 8 users (−1..−8 %): off|
|8-warp coop blocks|down stage −24 % in isolation, 8 users −4..−8 % end to end: not used|
|MoE row tiles (`EXL3_MOE_MTILE=1`) on V3|cold 2k +12 %, 32k +0.9 %; short-prompt KL 0.0054 vs floor 0.0032: off|
|prompt-cache tiers|see above; c8 decode unchanged within one run; short-prompt KL 0 (cached pages are bit-identical)|

## V2 measurements

Measured with the V2 harnesses (`matrix.py` / `decode_natural.py`, end-to-end timing). Patches called "candidates" here
are 0008-0018 of the V3 series.

### Method

- Speed comparisons alternate server boots A, A, B, B (each config loaded twice), three runs per boot, medians.
  The card is power-capped; run-to-run spread on a single boot reached ±3.5 % (c1-c8) and ±7.5 % (prose/chat).
- Output identity: 8 fixed prompts, 128 tokens each, compared as text and as token ids.
- Next-token drift: top-20 log-probabilities on 16 short and 9 long prompts (8.7k, 35k, 157k tokens), KL against a
  reference run. Two runs of the same config give exactly 0 at one request at a time; batched (8 at once) 0.0015.
  Compare runs that executed the same requests in the same order (see "Server history" in the README).
- Accuracy: `bench/accuracy_raw.py` (then `accuracy.py`), 24 exact-answer prompts, 768-token budget, graded after any `</think>`.
- Long context: a planted fact retrieved from a 520k-token prompt.

### KV cache modes at the largest pool that keeps 1.5 GiB free

|mode|KiB/token|pool|accuracy|KL vs 8,8, long context (mean)|
|---|---|---|---|---|
|8,8|16.5|2,573,312|22/24|-|
|8,6|15.0|2,833,920|21/24|0.0013|
|6,6|13.5|3,142,656 → 3,121,152 with the final engine (2048-token chunks); shipped: 2,999,808 with 8192-token chunks|21-22/24|0.0065|
|8,4|13.5|2,883,584|20/24, `2+2=` → 5|rejected|

### Per-change A/B (each against the step before)

|change|c1|c2|c4|c8|other|
|---|---|---|---|---|---|
|coop MoE k-loop (0001)|+4.7 %|+8.9 %|+8.6 %|+4.3 %|math +6.4 %|
|no expert-count readback (0003)|neutral|+20-23 %|+8-9 %|+3-12 %|bit-identical|
|verify-only wide coop rows (0004)|+3.0 %| | |+20.0 %|code −0.2 %|
|drafter CUDA graph (0005)|+2.8 %| | |+4.0 %|code +5.3 %|
|draft confidence 0.45|+5.5 %| | |+3.9 %| |
|shared expert side stream (0007)|+6.1 %| | |+12.0 %|code +7.4 %|
|engram lock|not slower| | | |cold 32k worst case 16.5 s → 8.65 s|

### Rejected

|candidate|result|
|---|---|
|built-in MTP drafter|fits only at a 2,560,000-token pool; prose +11.5 %, chat +2.4 %, code −15 %, math −15.6 %|
|dense DGEMV (0002, `EXL3_DGEMV=1`)|greedy identity 2/8; c1 −5.5 %, c8 +5.4 %|
|adaptive prefill chunks (0006)|fresh server: 32k/128k 4,820/4,853 t/s vs ~4,200; after decode traffic the chunks shrink back: 128k +3.4 %, c1 −2.5 %, c8 −3.1 %; long-context KL 0.0128|
|piecewise decode CUDA graphs|correct after excluding warm-up from capture; +3 % at c8 for +474 MiB|
|draft confidence 0.30|prose +4.4 %, code +5.6 %, c4 −4.1 %; greedy identity 7/8|
|split-D GatedResidual kernel|bit-faithful, but latency-bound: 25.6 → 27.7 µs per call at 8 rows|
|larger fused-MoE row limit|+9-14 % at c2/c8 for 160-260 MB VRAM; the readback removal (0003) does better for free|

### Prefill

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

### Prefix cache under an agent workload

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

### Prefill V3 candidates (2026-11-15)

All MEASURED on one RTX PRO 6000 Max-Q with the V2 release config (6,6 KV, 2,999,808-token pool, 8192-token
chunks, bs 8, DFlash2 n7 at confidence 0.45) unless stated. "Shotgun" = one server boot per arm, 3 runs per
suite, median with the run spread in brackets; "A,A,B,B" = four boots in that order, median of the two arm
medians. Greedy identity = the 8 fixed prompts, 128 tokens, text and token sequence. KL = next-token KL over the
top-20 on 16 short prompts and 9 long prompts (8.7k / 35k / 157k tokens).

#### MoE fused-tier row tiles (`EXL3_MOE_MTILE=1`), shotgun

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

#### Prefill while other requests decode (`prefill_cooldown` / `shared_chunk_size`, TabbyAPI patch v3), shotgun

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

#### Adaptive chunks at 16384 with the pool preallocated (patch 0006 + limiter diagnostics), shotgun

At load the 3M-token pool takes the card: allocator active = allocated = 92.8 GiB, inactive split blocks 0, ~2.2 GiB
of room. The limiter cuts every 16k chunk to 7.9-9.0k rows; releasing torch's cached blocks first changes nothing.
Measured rise per chunk 1.63-1.85 GiB at 7.9-9.0k rows. Cold prefill: 32k 5,611 t/s, 128k 5,690 t/s (V2 5,761 /
5,603). Memory is held by the pool, not fragmented; see the growable pool below.

#### Growable KV pool (candidate patches 0010-0012, `EXL3_KV_GROW=1`)

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

#### Growable pool, round 2 (candidate patches 0013-0015): shrink-back, 2k, MTILE, admission, soak

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

#### Final A,A,B,B: approved set vs V2 (2026-11-16 00:12-00:44)

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

#### Decode ports from the QMesh fork (candidate patch 0008), A,A,B,B

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

#### Where cold prefill time goes (nsys, one cold 32k prefill, 8192-token chunks, V2 engine)

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
