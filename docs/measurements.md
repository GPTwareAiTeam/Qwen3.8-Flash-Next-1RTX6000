# Measurement notes

All numbers are from the OpenAI-compatible `/v1/completions` endpoint on one RTX PRO 6000 Blackwell Max-Q, no BOS
token, with the `bench/` scripts. "cN" is N concurrent requests. V3 numbers use `bench/final.py` (streamed, decode
t/s from each stream's first token, 512 forced tokens per request, greedy unless marked "preset"); the V2 section
uses the V2 harnesses (end-to-end timing; decode rows force 2048 tokens from a short natural prompt).

## README run (2026-11-20): V2 against V3 as shipped

Source of the README's headline numbers unless a README line names another section. One server boot per release
(shotgun), `bench/final.py`, greedy, 512 forced tokens per request, tokens/s from each stream's first token, 3 runs
per cell, medians; runs column = min-max of the three. A = V2 as released (tag v2.0), B = V3 as released (YaRN factor
2). Host load at boot 1.3 (A) / 1.6 (B); both at the 300 W cap during decode ("SW Power Cap", SM 2,090-2,350 MHz).

|users|prompt|V2 total|V3 total|Δ|V2 per user|V3 per user|V2 runs|V3 runs|
|---|---|---|---|---|---|---|---|---|
|1|prose|144.6|149.2|+3.2 %|144.6|149.2|138-147|147-151|
|1|chat|175.0|189.4|+8.2 %|175.0|189.4|171-178|176-194|
|1|code|241.3|256.9|+6.5 %|241.3|256.9|236-357|254-270|
|1|math|323.0|328.4|+1.7 %|323.0|328.4|315-324|286-335|
|2|prose|211.8|237.6|+12.2 %|106.8|120.6|195-212|236-249|
|2|chat|256.5|289.8|+13.0 %|129.6|147.5|243-284|272-292|
|2|code|331.4|393.3|+18.7 %|170.4|206.4|320-338|381-413|
|2|math|461.7|546.5|+18.4 %|235.1|273.4|428-466|492-598|
|4|prose|289.4|341.1|+17.9 %|72.8|89.7|264-296|334-341|
|4|chat|365.5|391.9|+7.2 %|92.4|105.4|356-368|381-401|
|4|code|455.0|575.1|+26.4 %|118.3|154.6|378-460|554-587|
|4|math|601.1|674.2|+12.2 %|157.7|190.5|600-635|480-745|
|8|prose|339.3|419.4|+23.6 %|46.9|58.2|324-348|408-425|
|8|chat|402.1|487.5|+21.2 %|55.5|69.1|394-405|474-497|
|8|code|544.7|664.6|+22.0 %|78.5|103.8|536-555|637-701|
|8|math|743.0|819.5|+10.3 %|105.5|133.2|667-748|661-895|
|12|prose|273.4|510.5|+86.7 %|33.7|50.7|263-289|500-531|
|12|chat|324.2|587.2|+81.1 %|39.8|58.9|321-325|567-593|
|12|code|425.5|768.0|+80.5 %|52.6|80.9|411-438|727-809|
|12|math|525.6|823.5|+56.7 %|71.9|95.0|525-578|772-925|

Geometric mean over the four prompts, total (per user) t/s, V2 → V3: 1 user 210.7 → 221.0 (+4.9 %), 2 users 302.0 →
348.8 (+15.5 %; 153.4 → 178.0 per user), 4 users 412.4 → 477.2 (+15.7 %; 105.8 → 129.2), 8 users 484.8 → 577.7
(+19.2 %; 68.2 → 86.3), 12 users 375.2 → 659.9 (+75.9 %; 47.4 → 69.2). V2 serves at most 9 requests at once; at 12
users, 3 wait in its queue.

V3 only, same boot: cold prefill (fresh nonce, 3 runs) ~2k tokens 3,438 t/s (0.59 s), 128k 6,038 t/s (21.7 s).
Prompt cache, ~100k-token prompt (`bench/prompt_cache.py`): cold 16.75 s; resent at once 0.18 s; after 96 distinct
32k prompts (3.1M tokens, more than the GPU pool) 0.43 s, 99,840 of 99,997 tokens cached (RAM tier); after a server
restart (second boot, load 3.7) 0.76 s, 99,840 cached (disk tier). V2 has no RAM or disk tier: an evicted prompt, or
any prompt after a restart, is prefilled again (the cold time). VRAM after load and warm-up: V2 91.8 GiB, V3 59.2 GiB;
peak over the boot (decode to 12 users; V3 also prefill to 128k and the 3.1M-token fill): V2 93.0 GiB, V3 88.7 GiB,
of 95.6 GiB (free at peak 2.6 / 6.9 GiB).

Ratios quoted in the README's "V3 at a glance", computed from the numbers in this file: evicted resend 16.75 / 0.43 =
39×, after restart 16.75 / 0.76 = 22×; `matrix.py` code 12 users 901.3 / 573.1 = 1.57×, 8 users 881.5 / 708.4 =
1.24× (+24.4 %), math 8 users 805.2 / 684.7 = +17.6 %; VRAM after load 91.8 − 59.2 = 32.6 GiB less; 520k 96.6 / 87.9
s = 1.10× (+9.8 % in t/s); prefill V2 → V3 in t/s, 2k 3,541 → 3,438 (−2.9 %, "−3 %"), 32k 5,695 → 5,933 (+4.2 %), 128k
5,683 → 6,038 (+6.2 %), 520k 5,385 → 5,915 (520,163 tokens in 96.6 s; 520,174 in 87.9 s); `matrix.py` 12 users code
+57.3 %, math +50.9 %; since v1, 8 users code 881.5 / 453.7 = 1.94×, 1 user 313.5 / 231.1 = 1.36×.

### Current-build re-measure (2026-11-20 15:14-15:30)

Every drafting-sensitive V3 number in the README comes from the current build (the shipped V3, on the reference
server since 2026-11-19 22:32, after the verify-row-cap fix) or from this section. One boot per arm, the shipped
launcher and config with one setting changed, 3 runs each, fixed nonces (identical prompts across arms). Host load
recorded per arm (another process on the host), not voided.

**File rewrite, 1 user** (one chat request: rewrite a ~1.5k-token Python file with one rename, greedy, 1,536 max tokens;
1,068 tokens generated in every run; output text byte-identical across all arms, V2 included):

|arm|t/s, 3 runs|median|accepted drafts|load|
|---|---|---|---|---|
|V3, hybrid drafting on (live config, `hybrid_draft_tokens: 16`)|405.2 / 463.3 / 463.6|**463.3**|991|10.5-10.7|
|V3, hybrid drafting off (`hybrid_draft_tokens: 0`)|376.8 / 385.6 / 385.0|385.0|929|10.3-12.4|
|V2 (tag v2.0, one boot, one run, 14:50)|337.2|337.2|930|11.4|

Hybrid drafting on → off: 463.3 / 385.0 = **+20.3 %**. V3 against V2: 463.3 / 337.2 = **+37.4 %** (+20.2 % at V3's
slowest run, 405.2). V2's single run was at the same host load as V3's.

**8 users, prose and code** (8 concurrent copies of the `final.py` prompt, 512 forced tokens each, greedy,
total t/s send to last token), aggregate t/s, 3 runs and median:

|arm|prose|median|code|median|load|
|---|---|---|---|---|---|
|all on (live)|411.8 / 505.6 / 427.7|427.7|725.6 / 757.9 / 757.9|757.9|11.9-12.8|
|drafter / accept overlap off (`EXL3_SD_DRAFT_OVERLAP=0`)|419.7 / 418.3 / 402.9|418.3|764.8 / 749.4 / 762.4|762.4|6.3-9.9|
|hyper-connection tables off (`EXL3_HC_TABLE_CACHE=0`)|389.5 / 395.5 / 374.2|389.5|695.4 / 703.8 / 693.7|695.4|5.9-6.5|
|wide shared expert off (`EXL3_MOE_SHARED_WIDE=0`)|388.6 / 366.9 / 370.5|370.5|698.4 / 715.7 / 760.4|715.7|8.7-9.1|
|batched accept off (`EXL3_SD_BATCH_ACCEPT=0`, overlap off as it needs it)|393.3 / 403.8 / 432.2|403.8|715.9 / 690.5 / 712.6|712.6|10.5-12.1|

Effect of each switch (on against off, medians): overlap prose 427.7 / 418.3 = +2.2 %, code 757.9 / 762.4 = −0.6 %
(within the run spread: no measurable effect at 8 users); hyper-connection tables (0016) prose +9.8 %, code +9.0 %;
wide shared expert (0017) prose +15.4 %, code +5.9 %; batched accept (0018, against the overlap-off arm) prose +3.6 %,
code +7.0 %. Single boots with prose runs spreading up to ±12 %, so these are approximate.

### Draft acceptance, plain prompt against chat form (live V3, 2026-11-20)

The `final.py` code and math prompts, 512 forced tokens, greedy, 3 runs each, sent once as plain completions and once
as chat messages (the model's template, thinking on); accepted drafts per decode round = accepted / (tokens −
accepted). Host load 10.4-11.1.

|prompt|plain|chat, thinking on|
|---|---|---|
|code|4.39 / 4.39 / 4.45 (265-284 t/s)|1.63 / 1.75 / 1.94 (157-167 t/s)|
|math|4.63 / 4.69 / 4.57 (296-303 t/s)|4.45 / 4.75 / 4.75 (235-290 t/s)|

Charts in the README (`docs/img/*.svg`) are drawn from this section by `docs/img/make_charts.py` from
`docs/img/chart_data.json`.

### Earlier V1 numbers used in the README's "since v1" line

v1 = stock exllamav3 1.6.4 with 8-bit KV, the same model, drafter and TabbyAPI. Measured 2026-11-14 with
`bench/matrix.py` (code prompt, 512-token greedy replies, send to last token, total t/s, median of 3): 1 user 231.1,
8 users 453.7 t/s. V3 on the same harness and prompt, 2026-11-20 (section "V3 as shipped ... README sections",
`matrix.py` paragraph): 1 user 313.5, 8 users 881.5 t/s. Separate sessions; same card and harness.

## README number audit (2026-11-20)

Every number in the README, with when and how it was measured. "Current build" = the shipped V3, live since
2026-11-19 22:32 after the verify-row-cap fix. The fix changed how many drafts each round verifies, so a speed number measured from the moment the cap went live (2026-11-19 10:21) until 22:32 is
drafting-sensitive and stale. Numbers that do not depend on drafting (prefill, VRAM, host memory, kernel
microbenchmarks, quality) are valid from any build that ships the code they describe. V2 = tag v2.0. Hardware and
install facts (card, RAM sizing, revisions, config values, patch counts) are configuration, not measurements, and are
not listed.

|README number(s)|section|measured (EST)|build|harness|drafting-sensitive?|
|---|---|---|---|---|---|
|matrix.py 1 / 8 / 12 users, code and math, V2 and V3 (313.0-901.3 t/s, 1.24×, 1.57×, +57.3 / +50.9 %)|at a glance, Measured|11-20 03:06-03:13|current build; V2|`bench/matrix.py`, one boot per release, median of 3|yes, current build|
|Per prompt type, 1-12 users, V2 and V3 total and per user; geomeans (+4.9 … +75.9 %); per-user 69 vs 47 at 12 users|Measured, Trade-offs, Details (admission)|11-20 10:34-11:03|current build; V2|`bench/final.py` decode, one boot per release, median of 3|yes, current build|
|File rewrite, 1 user: V3 463.3 t/s, V2 337.2 t/s (+37 %); hybrid on/off +20 % (385.0 → 463.3); accepted 991 / 929 of 1,068|at a glance, What's new, Details (drafting)|11-20 15:14-15:18 (V3), 14:50 (V2)|current build; V2|file-rewrite request (fixed prompt), one boot per arm, 3 runs (V2 1 run)|yes, current build|
|8 users: overlap within noise (+2.2 / −0.6 %), 0016 +9-10 %, 0017 +6-15 %, 0018 +4-7 %|What's new|11-20 15:17-15:30|current build|8 concurrent `final.py` prompts, one boot per arm, 3 runs|yes, current build|
|Draft acceptance 4-5 per round plain code/math, 1-2 prose/chat; chat form code 1.6-1.9, math 4.5-4.8|Details (drafting)|11-20 10:42 (plain, README run) and 14:56 (chat form)|current build|`bench/final.py` acceptance counts; chat/completions probe, 3 runs|yes, current build|
|YaRN factor 4 → 2: 1 / 2 / 8 users +0.6 / +3.6 / +4.2 %; "~4 % lower at 2-8 users"|What's new, Trade-offs|11-20 01:17-01:38|current build at factor 4 vs 2|`bench/final.py`, A,A,B,B, 3 runs per boot|yes, current build|
|One request, 2048 tokens, V2 and V3, greedy and preset (123.4-329.3 t/s)|Measured|11-20 03:32-03:44|current build; V2|`bench/decode_natural.py`, one boot per release, median of 3|yes, current build|
|Since v1: 8 users 453.7 → 881.5 (1.94×), 1 user 231.1 → 313.5 (1.36×)|at a glance|v1 11-14; V3 11-20 03:09|v1 recipe; current build|`bench/matrix.py`|yes: V3 side current build; v1 side is the v1 recipe (no row cap)|
|Prefill V3 2k 0.59 s / 3,438 t/s, 128k 21.7 s / 6,038 t/s|Measured|11-20 10:42-11:00|current build|`bench/final.py` prefill, fresh nonce, 3 runs|no (prefill)|
|Prefill V2 2k 0.57 s / 3,541 t/s, 128k 23.1 s / 5,683 t/s|Measured (footnote 6), Trade-offs|11-19 19:00-20:42|V2|`bench/final.py` prefill, A,A,B,B|no (prefill, V2)|
|Prefill 32k V2 5.53 s / V3 5.31 s (+4.2 %); worst of 5 cold 32k 5.61 s (median 5.19 s)|Measured|11-20 03:06-03:13 and 03:32-03:44|current build; V2|`bench/coldttft.py`|no (prefill)|
|520k cold prompt V2 96.6 s / V3 87.9 s (1.10×, +9.8 %); needle retrieved|at a glance, Measured, Quality|V2 11-19 19:00-20:42; V3 11-20 03:55-04:00|V2; current build|`bench/needle.py`|no (prefill)|
|Cold 2k prefill 0.62 s (release run, "3-7 % slower")|Trade-offs|11-19 19:48-20:42|V3 bug-window build|`bench/final.py` prefill, A,A,B,B|no (prefill: the row cap acts on decode only)|
|Prompt cache 100k: cold 16.75 s, hot 0.18 s, evicted 0.43 s, after restart 0.76 s; 99,840 of 99,997 cached; 39× / 22×|at a glance, Measured, What's new|11-20 10:42-11:03|current build|`bench/prompt_cache.py`|no (prefill / cache)|
|VRAM after load V2 91.8 / V3 59.2 GiB; free at peak 2.6 / 6.9 GiB|at a glance, What's new|11-20 10:34-11:03|current build; V2|`nvidia-smi` sampling in the run|no (memory)|
|Host RAM ~250 GiB (RSS 244-268 GiB), V2 ~60-120 GiB; start ~80-90 s, V2 ~45 s|What it takes, Upgrading, Trade-offs|11-19 19:00-20:42 and every boot since|V3 bug-window build and current build; V2|server RSS, boot logs|no (memory, start time)|
|Exact-answer probe 50/50 both releases|at a glance, Quality|11-20 08:18-08:21|current build; V2|`bench/accuracy.py`|no (quality; drafting is lossless)|
|Top-1 25/25, KL 0.0010-0.0014 short / 0.0018 long, floor 0.0004 / 0.0000; factor 2 vs 4 top-1 16/16, KL 0.015|Quality, Trade-offs|11-19 19:00-20:42; 11-20 01:17-01:38|V3 bug-window build; current build|`bench/identity_kl.py`|no (quality: next-token distributions do not depend on how many drafts are verified)|
|Adaptive prefill chunks: cold 128k +5-9 %, 2k −3..−7 %|What's new|11-16 00:12-00:44 (A,A,B,B against V2) and 11-19 19:00-20:42|V3 candidate build; bug-window build|`bench/final.py` prefill|no (prefill)|
|Counting-sort run table: rotation launch 50-60 → 8-9 µs|What's new|11-19 11:00-12:40|microbenchmark|kernel microbenchmark|no (kernel microbenchmark)|
|Prefill while others decode: 35 s / −80 %, 86 / 41 / 25 s, −30 / −71 / −96 %|Details|11-15|V2-era build (labelled)|`bench/prefill_under_load.py`|yes, but labelled as V2-era; predates every V3 decode change, not the bug window|
|Prefix cache under agent load: 39 / ~640 checkpoints, 104 MiB, 93-99 % / 2-10 % / 77 % from cache, 2 s median queue; 197 checkpoints after the fill|Details|11-14 (V2 live traffic, before and after the checkpoint-budget fix); 197 on 11-19 17:28-17:42|V2; V3 tier build|server journal; fill test|no (cache capacity and hit rates)|
|Server history: first ~74 tokens, batched KL 0.0015|Details|11-12 (V2.1 era)|V2|`bench/identity_kl.py` probes|no (numerics of the drafter/target, not the draft width)|
|Model facts: 48 layers, 512 experts, 262,144 positions; KV 13.5 / 3.75 KiB per token; 6.3M tokens for 12 full requests; engram 48.3 GiB, ~4 GiB streamed|Details, What it takes|model config; computed|—|—|no (configuration / arithmetic)|

Drafting-sensitive numbers from the bug window (2026-11-19 10:21-22:32) left in the README: **none**. The ones it had
(file-rewrite baseline 261.6 and 399.9 t/s, the hybrid +36-53 %, overlap +4.6 %, 0016 +9-11 %, 0017 +7 %, 0018 +6-8 %,
the fix's 254 → 335 t/s) are replaced by the current-build re-measure above or removed.

## Trade-offs, in detail

- **Host RAM.** The server holds ~250 GiB: the 192 GiB pinned RAM tier of the prompt cache (`memory.sysmem_kv_cache`,
  committed at start), the 48.3 GiB engram table locked in RAM, and up to 64 GiB of recurrent-state checkpoints.
  `sysmem_kv_cache` is in MB; 192 GiB holds ~15M tokens of evicted KV at 13.5 KiB per token. Set it lower to fit a
  smaller host, or to 0 to drop both cache tiers (the disk tier needs the RAM tier); the server then needs ~60-120 GiB
  as in V2. Measured RSS in the release run: 244-268 GiB (V2: 52-58 GiB).
- **Rounding-level drift.** Patch 0017 and the adaptive prefill chunks change floating-point summation order. Greedy
  text differs from V2 on 3-5 of the 8 identity prompts at near-tie tokens (V2 against itself: 7/8; V3 against
  itself: 6/8); next-token top-1 stays 25/25 and KL 0.001-0.002 ("V3 release run", quality table).
- **YaRN factor 2.** 512k per request instead of 640k. The model was trained at 262,144 positions and the drafter
  without YaRN, likely why the smaller factor drafts slightly better ("YaRN factor 4 → 2"). Factor 4 against factor 2
  there: 1 user 213.7 / 215.0 = −0.6 % (within run-to-run spread), 2 users 339.3 / 351.5 = −3.5 %, 8 users 571.4 /
  595.1 = −4.0 %; the README's "~4 % lower at 2-8 users". For longer single requests, edit
  the patched model config.json to `"factor": 4.0` and `"max_position_embeddings": 1048576`, and raise `max_seq_len`
  to 655,360 or more (V2's setting).
- **Cold 2k prefill.** 0.57 s (V2) → 0.62 s (release run, A, A, B, B) / 0.59 s (README run): the adaptive limiter
  queries free memory before each chunk. Prompts from 32k up gain 4-10 %.
- **Above 8 users.** The 64-row verify cap shares the draft budget: 8 requests keep 7 drafts per round, 9-12 get
  6-4, so per-user speed falls (README run: 8 users 86.3, 12 users 69.2 t/s per user; V2 at 12 users 47.4, with 3 of
  the 12 waiting in its queue) while total throughput rises.
- **Start time.** ~80-90 s from launch to ready (V2 ~45 s): the 192 GiB RAM tier is committed and page-locked at load.

## V3 release run (2026-11-19 / 20)

Harness: `bench/final.py` (streamed requests; decode t/s from each stream's first token, end-to-end t/s, time to first
token; see `bench/README.md`). A = V2 as released (tag v2.0: `launch.sh`, `config.yml`, exllamav3 v1.6.4 +
patches 0001-0007, TabbyAPI + prefill-chunk patch). B = V3 (21 patches, `deploy/config.yml`, `deploy/launch.sh`).
**Both at YaRN factor 4** (V3 ships factor 2; its effect is measured separately below).
Boots A, A, B, B, three runs per cell per boot; a cell's value is the median
of the two boot medians. Every arm ran at the card's 300 W cap ("SW Power Cap" throttle during decode and prefill,
SM 1,640-2,350 MHz, memory 13,365 MHz, 85-88 °C at the end of each section). Host load at arm start 2.2-7.7.

Decode, aggregate t/s from the first token, geometric mean over prose / chat / code / math, 512 forced tokens per
request, YaRN factor 4 (per cell below):

|users|greedy A|greedy B|Δ|preset A|preset B|Δ|per-user greedy A → B|TTFT B|
|---|---|---|---|---|---|---|---|---|
|1|207.5|215.3|+3.8 %|196.0|194.7|−0.6 %|207.5 → 215.3|0.12 s|
|2|302.5|348.4|+15.2 %|269.9|323.8|+20.0 %|152.8 → 177.9|0.21 s|
|4|409.6|453.2|+10.7 %|370.8|444.7|+19.9 %|105.8 → 128.1|0.33 s|
|8|472.2|586.0|+24.1 %|439.4|548.7|+24.9 %|67.2 → 84.8|0.71 s|
|12|381.1|661.5|+73.6 %|358.2|607.5|+69.6 %|47.8 → 68.6|1.36 s|

The 1-2 user rows and the 4-12 user rows are two separate A,A,B,B blocks (start loads 4.4-5.4 and 4.3-5.3), both
measured after the verify-row-cap fix below.

Per cell (aggregate decode t/s from the first token, median of the two boot medians):

|users|prompt|sampling|V2|V3|Δ|
|---|---|---|---|---|---|
|1|prose|greedy|139.0|146.8|+5.6 %|
|1|chat|greedy|173.6|183.6|+5.7 %|
|1|code|greedy|240.4|235.8|-1.9 %|
|1|math|greedy|319.3|338.1|+5.9 %|
|2|prose|greedy|209.2|234.3|+12.0 %|
|2|chat|greedy|256.1|296.5|+15.8 %|
|2|code|greedy|336.2|389.1|+15.7 %|
|2|math|greedy|465.0|545.1|+17.2 %|
|4|prose|greedy|289.6|337.1|+16.4 %|
|4|chat|greedy|350.9|396.8|+13.1 %|
|4|code|greedy|452.8|542.1|+19.7 %|
|4|math|greedy|611.6|582.1|-4.8 %|
|8|prose|greedy|338.8|419.4|+23.8 %|
|8|chat|greedy|407.0|492.2|+20.9 %|
|8|code|greedy|524.6|670.3|+27.8 %|
|8|math|greedy|687.1|852.3|+24.1 %|
|12|prose|greedy|273.0|497.3|+82.2 %|
|12|chat|greedy|320.0|587.1|+83.5 %|
|12|code|greedy|438.4|729.4|+66.4 %|
|12|math|greedy|550.8|899.0|+63.2 %|
|1|prose|preset|130.8|126.1|-3.6 %|
|1|chat|preset|172.0|171.7|-0.2 %|
|1|code|preset|236.5|241.1|+1.9 %|
|1|math|preset|277.5|275.8|-0.6 %|
|2|prose|preset|197.2|214.6|+8.8 %|
|2|chat|preset|234.1|279.4|+19.3 %|
|2|code|preset|322.1|361.7|+12.3 %|
|2|math|preset|356.7|506.9|+42.1 %|
|4|prose|preset|264.6|299.8|+13.3 %|
|4|chat|preset|320.5|375.7|+17.2 %|
|4|code|preset|452.6|520.6|+15.0 %|
|4|math|preset|492.4|667.4|+35.5 %|
|8|prose|preset|321.3|380.3|+18.4 %|
|8|chat|preset|383.8|469.7|+22.4 %|
|8|code|preset|528.9|634.1|+19.9 %|
|8|math|preset|571.6|800.2|+40.0 %|
|12|prose|preset|264.4|466.2|+76.3 %|
|12|chat|preset|301.0|543.0|+80.4 %|
|12|code|preset|420.7|711.0|+69.0 %|
|12|math|preset|492.0|756.6|+53.8 %|

Prefill, cold prompt → 1 token (A,A,B,B, 3 runs): 2k 3,541 → 3,284 t/s (−7.3 %; 0.57 → 0.62 s), 32k 5,660 → 5,990
(+5.8 %; 5.79 → 5.47 s), 128k 5,683 → 5,949 (+4.7 %; 23.07 → 22.04 s). Needle: A 520,163 tokens in 96.6 s, found; B 520,164 tokens in 91.0 s, found.

Prompt cache (B), ~100k-token prompt: cold 16.8 s; resent at once 0.20 s; after 96 distinct 32k prompts (3.1M tokens,
more than the GPU pool) 0.44 s with 99,840 of 99,996 tokens cached (RAM tier); after a server restart 0.87 s (disk
tier).

Quality (`bench/identity_kl.py`, 16 short + 9 long prompts; the release run's accuracy probe was `bench/accuracy_raw.py`):

|pair|top-1|KL short mean (max)|KL long mean (max)|greedy identity|
|---|---|---|---|---|
|A1 vs A2 (floor)|25/25|0.0004 (0.0065)|0.0000 (0.0001)|7/8|
|B1 vs B2 (floor)|25/25|0.0003 (0.0026)|0.0000 (0.0003)|6/8|
|A vs B (three pairings)|25/25|0.0010-0.0014 (0.0087)|0.0017-0.0018 (0.0066)|3-5/8|

Accuracy, raw-completion probe (now `bench/accuracy_raw.py`): A 21/24 and 22/24, B 20/24 on both boots (YaRN factor 4);
V3 as shipped (factor 2): 22/24. Replaced as the quality probe by the chat probe ("Exact-answer probe, chat form": 50/50
on both releases).

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
|drafter / accept overlap (`EXL3_SD_DRAFT_OVERLAP`)|8-user greedy output identical to the build without it (8/8); speed on the current build: "Current-build re-measure" (within noise)|
|hybrid drafting with one active request (`EXL3_HYBRID_SOLO`, 16 tokens)|off with 2+ requests: 8-user accepted / rejected drafts unchanged (6,827 / 65 vs 6,823 / 61); 1-user speed on the current build: "Current-build re-measure"|
|counting-sort run table (`EXL3_MOE_COOP_FAST_RUNS`)|rotation launch 51 / 59 → 8.0 / 9.1 µs at 56 / 64 verify rows; outputs bit-identical|
|split down-stage reduction (`EXL3_MOE_COOP_SPLIT_B`)|kernel faster in isolation, slower end to end at 8 users (−1..−8 %): off|
|8-warp coop blocks|down stage −24 % in isolation, 8 users −4..−8 % end to end: not used|
|MoE row tiles (`EXL3_MOE_MTILE=1`) on V3|cold 2k +12 %, 32k +0.9 %; short-prompt KL 0.0054 vs floor 0.0032: off|
|prompt-cache tiers|see above; c8 decode unchanged within one run; short-prompt KL 0 (cached pages are bit-identical)|

### What's new in V3, per item

Every improvement over V2, with the patch or setting that ships it and its measured effect. Unless noted, effects are
from an A/B of that change alone against the build before it (A,A,B,B or one boot per arm where marked "one run");
the headline tables measure the whole set against V2. "8 users" = 8 concurrent requests, aggregate decode t/s.

|Improvement|Where|Measured effect|Numerics|
|---|---|---|---|
|Hyper-connection int8 tables built once at load instead of on every call of the tiled GatedResidual mix (above 8 rows)|0016, `EXL3_HC_TABLE_CACHE=1`|8 users prose +9.8 %, code +9.0 % (current build, one boot per arm); +1.3 GiB VRAM|bit-identical|
|Shared expert of a wide verify window (more than 8 rows) as one launch instead of 8-row graph pieces|0017, `EXL3_MOE_SHARED_WIDE=1`|8 users prose +15.4 %, code +5.9 % (current build, one boot per arm)|rounding-level (one GEMM tiles differently: max abs diff 4e-5..2e-4 in the MoE output)|
|Batched speculative accept: one readback per round for all requests, recurrent-state rewinds settled in one batch. Includes the sampler fix: TabbyAPI's no-op penalty steps marked every sampler as needing past tokens, which had kept the batched readback (0008) from ever engaging; the flag now comes from the steps that survive simplification|0018, `EXL3_SD_BATCH_ACCEPT=1`|8 users prose +3.6 %, code +7.0 % (current build, one boot per arm)|lossless|
|Drafter's next block launched under the host accept loop; checked next round and rolled back when the round ended differently|0021, `EXL3_SD_DRAFT_OVERLAP=1`|8 users prose +2.2 %, code −0.6 % (current build, one boot per arm): within noise; 8-user greedy output identical with and without it|lossless|
|Counting-sort build of the run table for wide verify windows (replaces a per-slot rank loop)|0020, `EXL3_MOE_COOP_FAST_RUNS=1`|rotation launch 50-60 → 8-9 µs at 56-64 verify rows (~2 ms per 8-user round); end-to-end within noise|bit-identical|
|12 concurrent requests: batched-cache decode graphs for 12 jobs, a 64-row cap on verify rows per round so 9-12 requests stay on the captured 64-row paths, and the cap's fix (it had read the batch size from the draft buffer's height, which cut every round to 4 drafts at any load)|0019, 0021, `max_batch_size: 12`, `EXL3_DRAFT_MAX_ROWS=64`|12 users 1.57× V2 on `matrix.py` code, +75.9 % on the four-prompt mean (README run, current build)|lossless (draft width only)|
|Hybrid (suffix-match) drafting of 16 tokens while exactly one request is active (a request repeating its own context drafts from it)|0021, `hybrid_draft_tokens: 16`, `EXL3_HYBRID_SOLO=1`|1 user rewriting a ~1.5k-token file with one change: +20.3 % against V3 with it off (385.0 → 463.3 t/s), +37.4 % against V2 (337.2 t/s); off with 2+ requests (8-user draft counts unchanged)|lossless|
|Growable KV pool: the 3M-token pool is reserved as CUDA virtual memory, mapped in 256-page steps as requests use it and released from the top when the queue drains|0010-0015, `EXL3_KV_GROW=1`|VRAM after load 91.8 → 59.2 GiB (README run; 93.9 → 60.7 GiB in the release run); the free memory goes to prefill (next row)|lossless (same pages, mapped on demand)|
|Adaptive prefill chunks up to 16k tokens, sized to the free memory the growable pool leaves; TabbyAPI patch v3 adds `prefill_chunk_size`, `shared_chunk_size` (4096 while other requests decode) and `prefill_cooldown` (pacing; default kept)|0006 (V2, now on), 0010, 0013, 0014, `EXL3_PREFILL_ADAPTIVE=1`, `prefill_chunk_size: 16384`, `shared_chunk_size: 4096`, `deploy/tabbyapi-v3.patch`|cold 32k +6-8 %, 128k +5-9 %; cold 2k −2..−7 %|rounding-level (chunk boundaries move with free memory); long-context KL 0.0016|
|Prompt-cache tiers: evicted prompt pages kept in 192 GiB of pinned RAM and written through to NVMe, where a restarted server finds them|config `memory.sysmem_kv_cache`, `disk_cache_dir`, `disk_cache_size` (stock TabbyAPI / exllamav3 settings)|100k-token prompt evicted from VRAM: 16.75 s → 0.43 s; after a restart: 0.76 s (README run; 0.44 / 0.87 s in the release run)|bit-identical (cached pages)|
|YaRN factor 4 → 2 (524,288 positions); per-request limit 655,360 → 524,032|`deploy/model-config-yarn2.patch`, `max_seq_len: 524032`|1 user +0.6 %, 2 users +3.6 %, 8 users +4.2 % (A,A,B,B), mostly on code (−10 % decode rounds at 2 users)|changes the position encoding: short-prompt KL 0.015 mean vs factor 4|
|Bench: streamed first-token timing, the release benchmark, needle and prompt-cache harnesses; gate policy (next-token top-1 + KL against the run-to-run noise floor; greedy identity reported, not a gate)|`bench/timing.py`, `final.py`, `needle.py`, `prompt_cache.py`, `bench/README.md`|decode t/s from the first token instead of end to end; every V3 number uses them|—|

In the series but not enabled by `launch.sh` (measured, no gain on this stack, or diagnostics only): the speculative
host readbacks of 0008 (`EXL3_SD_DEFER_DRAFT_CONF`, `EXL3_SD_BATCH_READBACK`, `EXL3_SD_WINDOW_BATCH`; subsumed by
0018), the split down-stage reduction of 0020 (`EXL3_MOE_COOP_SPLIT_B`, slower end to end) and the diagnostics of
0009. Shipped in V2 and unchanged: 0001-0007 (the dense DGEMV of 0002 stays off).

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
