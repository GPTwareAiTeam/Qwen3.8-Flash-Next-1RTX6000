# Qwen3.8-Flash-Next on one RTX PRO 6000 — V3

Serving recipe for `Qwen/Qwen3.8-Flash-Next` in EXL3 `3.18bpw_h6_ng8`
([turboderp/Qwen3.8-Flash-Next-exl3, branch 3.18bpw_h6_ng8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.18bpw_h6_ng8))
on a single 96 GiB RTX PRO 6000 Blackwell (SM120), with TabbyAPI, a patched exllamav3, the DFlash2 drafter, a
3M-token shared KV pool, up to 512k tokens per request, 12 concurrent requests, and a prompt cache that spills
evicted prompts to system RAM and NVMe.

This is V3. It improves on V2, the previous release of this repo, which is the baseline in the tables below. All
numbers are for V3 as shipped (YaRN factor 2) unless a row says "YaRN factor 4".

|Users|V2|V3|Change|V3 per user|
|---|---|---|---|---|
|1|207.5 t/s|**216.6 t/s**¹|+4.4 %|216.6 t/s|
|2|302.5|**360.9**¹|+19.3 %|184.3|
|4|392.6|**486.4**|+23.9 %|129.6|
|8|472.2|**610.6**¹|+29.3 %|88.4|
|12|375.9|**645.9**|**+71.8 %**|68.7|
|KV cache (shared pool)|2,999,808 tokens, allocated at load|**2,999,808 tokens, mapped as it fills**|VRAM after load 93.9 → 60.7 GiB|
|Max per request|655,360|524,032|up to 12 requests share the pool|

Aggregate decode throughput, geometric mean over prose, chat, code and math prompts; 512 generated tokens per
request, greedy, tokens/s measured from each stream's first token (`bench/final.py`). Every comparison alternates
server boots A, A, B, B on the same card, three runs per boot, medians. 4 and 12 users: V2 against V3 as shipped.
¹ 1, 2 and 8 users: a V2-against-V3 run with V3 at YaRN factor 4 (215.3 / 348.4 / 586.0 t/s), times the measured
change from factor 4 to factor 2 (a separate A, A, B, B: +0.6 / +3.6 / +4.2 %). Per-prompt cells: "Decode, all
cells".

Measured the way the V2 release was (`bench/matrix.py`: 512-token greedy replies, time from sending to the last
token, total t/s across users, median of three runs; one server boot per release, 2026-11-20):

|`matrix.py`|V2|V3|Change|
|---|---|---|---|
|code, 1 user|313.0 t/s|313.5 t/s|+0.2 %|
|code, 8 users|708.4|**881.5**|+24.4 %|
|code, 12 users|573.1²|**901.3**|+57.3 %|
|math, 1 user|297.6|297.5|−0.0 %|
|math, 8 users|684.7|**805.2**|+17.6 %|
|math, 12 users|559.8²|**844.5**|+50.9 %|

² V2 serves at most 9 requests at once; with 12 users, 3 wait in its queue (as measured).

|Decode, 2048 forced tokens from a short prompt|V2, greedy|V3, greedy|Change|V3, `qwen38_thinking` preset|
|---|---|---|---|---|
|prose|127.3 t/s|127.4 t/s|+0.1 %|123.4 t/s|
|code|289.1|**306.8**|+6.1 %|276.4|
|chat|155.8|152.1|−2.4 %|152.3|
|math|319.8|**329.3**|+3.0 %|321.2|

One request at a time, so most of V3's gains (which are in batching) don't show; prose and chat are within run-to-run
noise. `bench/decode_natural.py`, one run of three repeats, one server boot per release; accepted drafts per
request are in `docs/measurements.md`. The preset column uses the shipped `qwen38_thinking` preset: temperature 0.6, top-k 20, top-p 0.95. That
is this repo's choice (Qwen recommends temperature 1.0 for thinking mode); clients that send their own sampling
settings keep them. Sampled tokens accept fewer drafts, so the preset is 0-10 % slower than greedy.

|Prefill, cold prompt → 1 token|V2|V3|V3 time to first token|
|---|---|---|---|
|~2k tokens|3,541 t/s|3,284 t/s (YaRN factor 4)|0.62 s|
|32k|5,660 t/s|**5,990 t/s** (YaRN factor 4); 5,933 t/s (factor 2)|5.3-5.5 s|
|128k|5,683 t/s|**5,949 t/s** (YaRN factor 4)|22.0 s|
|520k|5,385 t/s (96.6 s)|**5,915 t/s (87.9 s)**|88 s|
|worst of 5 cold 32k requests|—|5.61 s (median 5.19 s)||

2k/32k/128k: A, A, B, B of V2 against V3 at YaRN factor 4, three runs per boot. The factor-2 32k figure is one run
(`coldttft.py`, V2 in the same session: 5,695 t/s). 520k: one cold request each,
same prompt construction; V3 at factor 2. Worst of 5: five fresh 32k prompts in a row on V3 as shipped
(`coldttft.py --n 5`: 5.21, 5.17, 5.61, 5.19, 5.16 s). Long prompts gain because the growable pool leaves room for
16k-token prefill chunks; short prompts pay ~0.05 s for the per-chunk memory check.

|Prompt cache, ~100k-token prompt (YaRN factor 4)|V2|V3|
|---|---|---|
|resent while its pages are in VRAM|0.2 s|0.2 s|
|resent after 3.1M tokens of other prompts evicted it from VRAM|full prefill (~17 s)|**0.44 s** (RAM tier)|
|resent after a server restart|full prefill (~17 s)|**0.87 s** (disk tier)|

Quality, V3 as shipped unless noted:
- Exact-answer probe (`bench/accuracy.py`): **50/50**, V2 50/50 (one run each, same session). 50 unambiguous
  questions (arithmetic, capitals, chemistry, number bases, Python, units) sent as chat messages with thinking on,
  each asking for one answer in a stated form ("How many metres are in one kilometre? Reply with only the number."),
  graded by exact match on the final answer after the thinking. This replaces V2's raw-completion probe (`accuracy_raw.py`,
  24 prompts continued as plain text and searched for the answer), whose misses were continuations like `1 kilometre
  equals` → "how many miles?" rather than wrong answers; it is kept for comparison with older results only.
- A passphrase planted halfway into a 520k-token prompt is retrieved (87.9 s prefill).
- Next-token agreement with V2 (YaRN factor 4, 16 short and 9 long prompts up to 157k tokens): top-1 25/25; KL
  0.0010-0.0014 mean at short context and 0.0018 at long context, against a V2-vs-V2 floor of 0.0004 / 0.0000
  (rounding-level drift, see "Trade-offs").
- YaRN factor 2 against factor 4: top-1 16/16 on short prompts, KL 0.015 mean (0.081 max): the position encoding
  itself changes, as intended.
- VRAM free at peak: 6.6 GiB in the release benchmark (decode to 12 users, prefill to 128k, a 520k prompt; peak
  89.0 of 95.6 GiB), against 1.0 GiB for V2 (its pool is allocated at load). A cold 520k prompt alone peaks at
  65.9 GiB.

## Decode, all cells

YaRN factor 4 (both releases). Aggregate decode t/s from the first token (V2 / V3), per-user decode t/s, V3
end-to-end t/s (first request sent to last stream finished) and V3 time to first token. Each cell: median of two
boots per release, three runs each. The factor-2 deltas per cell are in "Context length".

**Greedy** (`temperature: 0`):

|Users|Prompt|V2|V3|Change|V2 per user|V3 per user|V3 end-to-end|V3 time to first token|
|---|---|---|---|---|---|---|---|---|
|1|prose|139.0|**146.8**|+5.6 %|139.0|146.8|141.7|0.13 s|
|1|chat|173.6|**183.6**|+5.7 %|173.6|183.6|175.4|0.13 s|
|1|code|240.4|**235.8**|-1.9 %|240.4|235.8|224.0|0.12 s|
|1|math|319.3|**338.1**|+5.9 %|319.3|338.1|315.1|0.11 s|
|2|prose|209.2|**234.3**|+12.0 %|105.6|118.4|223.7|0.21 s|
|2|chat|256.1|**296.5**|+15.8 %|128.9|150.5|278.4|0.23 s|
|2|code|336.2|**389.1**|+15.7 %|171.0|201.4|361.9|0.20 s|
|2|math|465.0|**545.1**|+17.2 %|234.1|278.9|494.0|0.20 s|
|4|prose|289.6|**337.1**|+16.4 %|74.4|89.2|320.7|0.34 s|
|4|chat|350.9|**396.8**|+13.1 %|88.8|107.3|374.1|0.34 s|
|4|code|452.8|**542.1**|+19.7 %|117.7|145.6|497.3|0.35 s|
|4|math|611.6|**582.1**|-4.8 %|160.8|193.5|538.9|0.31 s|
|8|prose|338.8|**419.4**|+23.8 %|46.2|57.8|402.6|0.76 s|
|8|chat|407.0|**492.2**|+20.9 %|56.3|69.2|473.7|0.79 s|
|8|code|524.6|**670.3**|+27.8 %|77.2|98.6|633.6|0.66 s|
|8|math|687.1|**852.3**|+24.1 %|101.4|131.2|765.1|0.64 s|
|12|prose|273.0|**497.3**|+82.2 %|33.8|49.2|479.9|1.37 s|
|12|chat|320.0|**587.1**|+83.5 %|39.4|59.4|555.5|1.53 s|
|12|code|438.4|**729.4**|+66.4 %|53.5|79.2|708.0|1.35 s|
|12|math|550.8|**899.0**|+63.2 %|73.5|95.6|850.5|1.22 s|

**`qwen38_thinking` preset** (temperature 0.6, top-k 20, top-p 0.95):

|Users|Prompt|V2|V3|Change|V2 per user|V3 per user|V3 end-to-end|V3 time to first token|
|---|---|---|---|---|---|---|---|---|
|1|prose|130.8|**126.1**|-3.6 %|130.8|126.1|122.5|0.12 s|
|1|chat|172.0|**171.7**|-0.2 %|172.0|171.7|164.7|0.13 s|
|1|code|236.5|**241.1**|+1.9 %|236.5|241.1|228.7|0.12 s|
|1|math|277.5|**275.8**|-0.6 %|277.5|275.8|260.2|0.11 s|
|2|prose|197.2|**214.6**|+8.8 %|104.4|113.0|205.3|0.22 s|
|2|chat|234.1|**279.4**|+19.3 %|120.7|143.2|263.0|0.22 s|
|2|code|322.1|**361.7**|+12.3 %|168.6|194.3|338.2|0.20 s|
|2|math|356.7|**506.9**|+42.1 %|192.8|257.6|461.4|0.20 s|
|4|prose|264.6|**299.8**|+13.3 %|70.7|82.3|283.9|0.38 s|
|4|chat|320.5|**375.7**|+17.2 %|82.0|96.8|354.7|0.39 s|
|4|code|452.6|**520.6**|+15.0 %|121.5|139.0|478.4|0.36 s|
|4|math|492.4|**667.4**|+35.5 %|144.3|180.5|602.3|0.32 s|
|8|prose|321.3|**380.3**|+18.4 %|45.5|55.0|370.0|0.79 s|
|8|chat|383.8|**469.7**|+22.4 %|52.5|64.1|448.3|0.69 s|
|8|code|528.9|**634.1**|+19.9 %|77.0|97.1|598.0|0.67 s|
|8|math|571.6|**800.2**|+40.0 %|89.4|121.1|741.7|0.68 s|
|12|prose|264.4|**466.2**|+76.3 %|33.1|47.0|455.5|1.40 s|
|12|chat|301.0|**543.0**|+80.4 %|37.6|56.2|528.0|1.45 s|
|12|code|420.7|**711.0**|+69.0 %|54.0|77.3|687.0|1.38 s|
|12|math|492.0|**756.6**|+53.8 %|64.5|84.9|727.1|1.33 s|

Single-user math and code depend most on the drafter (about 4-5 accepted drafts per round); prose and chat accept
1-2. Spread between the two boots of a release is up to ±4 % on greedy cells and up to ±12 % on single-user preset
cells (sampled text accepts a different number of drafts each run).

## What it takes

- One RTX PRO 6000 Blackwell (96 GiB). Measured on the Max-Q (300 W) edition; every run here sat at the power cap
  ("SW Power Cap" throttle) during decode and prefill.
- Host RAM: **plan for 320 GB or more.** The server holds ~250 GiB: the 192 GiB pinned RAM tier of the prompt cache
  (`memory.sysmem_kv_cache`, committed at start), the 48.3 GiB engram table locked in RAM (memlock ≥ 50 GiB;
  `launch.sh` checks it), and up to 64 GiB of recurrent-state checkpoints (`memory.sysmem_recurrent_cache`). Without
  the RAM tier (`sysmem_kv_cache: 0`, and no disk tier, which needs it) the server needs ~60-120 GiB as in V2.
- Disk: ~100 GiB for the model, plus up to 200 GB for the disk tier of the prompt cache (`memory.disk_cache_dir`,
  `disk_cache_size`) on a fast NVMe.
- The model: [`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.18bpw_h6_ng8),
  branch `3.18bpw_h6_ng8` (measured at revision `2bfc9dade6be0b5b42e2b381d7a6cc160c1d409c`). About 100 GiB on disk,
  half of it the engram (n-gram embedding) table.
- Python 3.12, PyTorch 2.11 with CUDA 13 (`cu130` wheels), flash-linear-attention 0.6.0, the CUDA toolkit
  (`nvcc`, for the extension's JIT build) and `ninja`. Use an environment without a pip-installed exllamav3: its
  prebuilt extension would shadow the patched build (`launch.sh` refuses to start in that case).
- [exllamav3](https://github.com/turboderp-org/exllamav3) `v1.6.4` + `engine/patches` (21 patches).
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) `564f8fb` + `deploy/tabbyapi-v3.patch`.
- Drafter: [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
  (measured at revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`).

## Setup

```bash
# environment
python3.12 -m venv venv && . venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# engine
git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3
git checkout v1.6.4 && git am /path/to/repo/engine/patches/*.patch
pip install -r requirements.txt flash-linear-attention==0.6.0 && cd ..

# server
git clone https://github.com/theroyallab/tabbyAPI && cd tabbyAPI
git checkout 564f8fb && git apply /path/to/repo/deploy/tabbyapi-v3.patch
pip install .
cp /path/to/repo/deploy/sampler_overrides/qwen38_thinking.yml sampler_overrides/ && cd ..

# model and drafter
# (the revisions every number here was measured on; the model's branch is 3.18bpw_h6_ng8)
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 2bfc9dade6be0b5b42e2b381d7a6cc160c1d409c \
  --local-dir /path/to/models/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8
hf download incoai/Qwen3.8-Flash-Next-DFlash2 --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6 \
  --local-dir /path/to/drafts/Qwen3.8-Flash-Next-DFlash2

# long context (needed for the 512k limit): apply the YaRN factor-2 change to the model's config.json
(cd /path/to/models/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 && patch -p1 < /path/to/repo/deploy/model-config-yarn2.patch)

# edit deploy/config.yml: model_dir, draft_model_dir, memory.disk_cache_dir, network.port, auth; then
ulimit -l 52428800      # memlock for the engram lock (or LimitMEMLOCK=52428800K in a systemd unit)
TABBY=$PWD/tabbyAPI ENGINE=$PWD/exllamav3 GPU=0 ./deploy/launch.sh
```

`launch.sh` sets the engine switches, waits for the model and sends one 128-token warm-up request before serving
(see "Server history"). Startup takes ~80-90 s: the 192 GiB RAM tier is committed and page-locked at load.

## What's new in V3

Every improvement over V2, with the patch or setting that ships it and its measured effect. Unless noted, effects are
from an A/B of that change alone against the build before it (A,A,B,B or one boot per arm where marked "one run");
the headline tables measure the whole set against V2. "8 users" = 8 concurrent requests, aggregate decode t/s.

|Improvement|Where|Measured effect|Numerics|
|---|---|---|---|
|Hyper-connection int8 tables built once at load instead of on every call of the tiled GatedResidual mix (above 8 rows)|0016, `EXL3_HC_TABLE_CACHE=1`|8 users +9-11 %; +1.3 GiB VRAM|bit-identical|
|Shared expert of a wide verify window (more than 8 rows) as one launch instead of 8-row graph pieces|0017, `EXL3_MOE_SHARED_WIDE=1`|8 users +7 %|rounding-level (one GEMM tiles differently: max abs diff 4e-5..2e-4 in the MoE output)|
|Batched speculative accept: one readback per round for all requests, recurrent-state rewinds settled in one batch. Includes the sampler fix: TabbyAPI's no-op penalty steps marked every sampler as needing past tokens, which had kept the batched readback (0008) from ever engaging; the flag now comes from the steps that survive simplification|0018, `EXL3_SD_BATCH_ACCEPT=1`|8 users +6-8 %|lossless|
|Drafter's next block launched under the host accept loop; checked next round and rolled back when the round ended differently|0021, `EXL3_SD_DRAFT_OVERLAP=1`|8 users prose +4.6 %, code +1.2 % (one run); 8-user greedy output identical with and without it|lossless|
|Counting-sort build of the run table for wide verify windows (replaces a per-slot rank loop)|0020, `EXL3_MOE_COOP_FAST_RUNS=1`|rotation launch 50-60 → 8-9 µs at 56-64 verify rows (~2 ms per 8-user round); end-to-end within noise|bit-identical|
|12 concurrent requests: batched-cache decode graphs for 12 jobs, a 64-row cap on verify rows per round so 9-12 requests stay on the captured 64-row paths, and the cap's fix (it had read the batch size from the draft buffer's height, which cut every round to 4 drafts at any load)|0019, 0021, `max_batch_size: 12`, `EXL3_DRAFT_MAX_ROWS=64`|9 users with the cap against 8 slots + 1 queued: code +14-18 %, prose +21 %; 12 users 1.7× V2 (headline); the fix: 1 user math 254 → 335 t/s (one run each; 93 instead of 124 rounds per 512 tokens on identical text)|lossless (draft width only)|
|Hybrid (suffix-match) drafting of 16 tokens while exactly one request is active (a request repeating its own context drafts from it)|0021, `hybrid_draft_tokens: 16`, `EXL3_HYBRID_SOLO=1`|1 user rewriting a ~1.5k-token file with one change: +36-53 % (one run each, 262 → 356-400 t/s); off with 2+ requests (8-user draft counts unchanged)|lossless|
|Growable KV pool: the 3M-token pool is reserved as CUDA virtual memory, mapped in 256-page steps as requests use it and released from the top when the queue drains|0010-0015, `EXL3_KV_GROW=1`|VRAM after load 93.9 → 60.7 GiB; the free memory goes to prefill (next row)|lossless (same pages, mapped on demand)|
|Adaptive prefill chunks up to 16k tokens, sized to the free memory the growable pool leaves; TabbyAPI patch v3 adds `prefill_chunk_size`, `shared_chunk_size` (4096 while other requests decode) and `prefill_cooldown` (pacing; default kept)|0006 (V2, now on), 0010, 0013, 0014, `EXL3_PREFILL_ADAPTIVE=1`, `prefill_chunk_size: 16384`, `shared_chunk_size: 4096`, `deploy/tabbyapi-v3.patch`|cold 32k +6-8 %, 128k +5-9 %; cold 2k −2..−7 %|rounding-level (chunk boundaries move with free memory); long-context KL 0.0016|
|Prompt-cache tiers: evicted prompt pages kept in 192 GiB of pinned RAM and written through to NVMe, where a restarted server finds them|config `memory.sysmem_kv_cache`, `disk_cache_dir`, `disk_cache_size` (stock TabbyAPI / exllamav3 settings)|100k-token prompt evicted from VRAM: 17 s → 0.44 s; after a restart: 0.87 s|bit-identical (cached pages)|
|YaRN factor 4 → 2 (524,288 positions); per-request limit 655,360 → 524,032|`deploy/model-config-yarn2.patch`, `max_seq_len: 524032`|1 user +0.6 %, 2 users +3.6 %, 8 users +4.2 % (A,A,B,B), mostly on code (−10 % decode rounds at 2 users)|changes the position encoding: short-prompt KL 0.015 mean vs factor 4|
|Bench: streamed first-token timing, the release benchmark, needle and prompt-cache harnesses; gate policy (next-token top-1 + KL against the run-to-run noise floor; greedy identity reported, not a gate)|`bench/timing.py`, `final.py`, `needle.py`, `prompt_cache.py`, `bench/README.md`|decode t/s from the first token instead of end to end; every V3 number uses them|—|

In the series but not enabled by `launch.sh` (measured, no gain on this stack, or diagnostics only): the speculative
host readbacks of 0008 (`EXL3_SD_DEFER_DRAFT_CONF`, `EXL3_SD_BATCH_READBACK`, `EXL3_SD_WINDOW_BATCH`; subsumed by
0018), the split down-stage reduction of 0020 (`EXL3_MOE_COOP_SPLIT_B`, slower end to end) and the diagnostics of
0009. Shipped in V2 and unchanged: 0001-0007 (the dense DGEMV of 0002 stays off).

## Trade-offs

- **Host RAM ~250 GiB.** The prompt-cache RAM tier dominates. Lower `sysmem_kv_cache` to fit a smaller host (it is
  sized in MB; 192 GiB holds ~15M tokens of evicted KV at 13.5 KiB per token), or set it to 0 to drop both tiers.
- **Rounding-level drift.** Patch 0017 and the adaptive prefill chunks change floating-point summation order. Greedy
  text differs from V2 on 3-5 of the 8 identity prompts at near-tie tokens (V2 against itself: 7/8; V3 against
  itself: 6/8); next-token top-1 agreement stays 25/25 and KL stays at 0.001-0.002. The exact-answer probe is 50/50 on
  both releases.
- **YaRN factor 2.** 512k per request instead of 640k; the model was trained at 262,144 positions and the drafter
  without YaRN, so the smaller factor drafts slightly better (+0.6 % at 1 user, +3.6 % at 2, +4.2 % at 8, mostly on
  code). Next-token KL against
  factor 4 is 0.015 mean at short context: the position encoding itself changes, as intended. For longer single
  requests, edit the patched config.json to `"factor": 4.0` and `"max_position_embeddings": 1048576`, and raise
  `max_seq_len` to 655,360 or more (V2's setting).
- **Cold 2k prefill −7 %.** The adaptive limiter queries free memory per chunk; short prompts pay for it (0.57 →
  0.62 s). Long prompts gain.
- **12 users share the verify budget.** Above 8 concurrent requests each gets fewer drafts per round (7 → 6-4), so
  per-user speed falls (8 users 84.8, 12 users 68.6 t/s) while total throughput rises.
- **Slower start.** ~80-90 s to ready (V2 ~45 s): the RAM tier is page-locked at load.

## Details

**Model.** Hidden 2560, 48 layers (12 sparse full-attention + 36 Gated DeltaNet linear-attention), 512 experts
with 10 routed + 1 shared active per token, hyper-connections, vocab 248,320. The checkpoint is trained at 262,144
positions.

**Context length.** `deploy/model-config-yarn2.patch` applies static YaRN factor 2 (524,288 positions);
`max_seq_len: 524032` leaves one 256-token page for the draft and verify positions of a request at the limit
(drafts are up to 16 tokens with hybrid drafting). V2 used factor 4 (1,048,576 positions, 640k per request). On this
stack the factor matters for speed because the DFlash2 drafter was trained without YaRN: the closer the target's
position encoding is to native, the more drafts it accepts. A,A,B,B (V3 at factor 4 against V3 at factor 2, everything
else equal; 3 runs per boot; greedy, 512 tokens per request), aggregate decode t/s and decode rounds per 512 tokens
per request (fewer rounds = more accepted drafts):

|users|prompt|factor 4|factor 2|Δ|rounds, factor 4 → 2|
|---|---|---|---|---|---|
|1|prose|139.3|142.7|+2.4 %|242 → 240|
|1|chat|182.7|182.2|−0.3 %|180 → 182|
|1|code|250.4|259.9|+3.8 %|118 → 116|
|1|math|327.2|316.3|−3.3 %|86 → 92|
|2|prose|235.9|226.9|−3.8 %|228 → 223|
|2|chat|293.4|298.9|+1.9 %|168 → 168|
|2|code|391.1|421.4|+7.7 %|120 → 108|
|2|math|489.8|534.1|+9.0 %|90 → 88|
|8|prose|409.1|418.4|+2.3 %|210 → 207|
|8|chat|491.3|498.9|+1.6 %|163 → 161|
|8|code|676.7|700.6|+3.5 %|115 → 109|
|8|math|783.7|857.8|+9.5 %|88 → 87|

Geometric mean: 1 user +0.6 %, 2 users +3.6 %, 8 users +4.2 %. Host load was 7-11 during this run (another
process on the host), so single cells move by a few percent between boots; the round counts are the steadier signal.
A passphrase 500k tokens deep is retrieved at factor 2 (86 s prefill).

**KV capacity.** 6-bit K/V costs 13.5 KiB per token; 3.75 KiB of that is fixed fp16 planes for the sparse-attention
indexer. The 3M-token pool (2,999,808, a multiple of 256) is reserved as virtual memory at load and mapped in
256-page steps as requests use it; when the queue drains, mapped pages are released from the top until 10 GiB are
free, so prefill gets the room for 16k-token chunks back. Each request can use up to 524,032 tokens. Twelve
full-length requests would need 6.3M tokens, more than the pool: the generator reserves pages for a request's prompt
plus its next output chunk, so when the pool runs short a new or continuing request waits for pages rather than
failing.

**Prompt cache.** Three tiers: the GPU pool, a 192 GiB pinned-RAM tier and a 200 GB disk tier. A prompt page evicted
from the GPU pool moves to RAM; pages entering RAM, and the recurrent checkpoints, are written through to disk
(`disk_cache_dir`), where a later server process finds them. The disk tier is keyed by the model's config and
weights, so a config change (such as the YaRN factor) starts a fresh subdirectory. Each tier needs the recurrent
checkpoint at the prefix end as well (see below); 197 checkpoints were held after a 3.1M-token fill test, well inside
the 64 GiB budget.

**Drafting.** DFlash2 drafts 8-token blocks: the first position of a block is the anchor (the last accepted token),
so a block carries 7 drafts, and `draft_num_tokens: 7` uses all of them. Draft acceptance drives single-stream
speed: about 4-5 accepted drafts per round on code and math, 1-2 on prose and chat. With one active request, a
16-token suffix match against the request's own context (hybrid drafting) replaces the DFlash2 block when the match
is long and agrees with it: rewriting a file with one small change accepts ~990 of 1,068 tokens.

**Engram table.** 48.3 GiB of n-gram embeddings read per token. Locked in RAM it costs 48.3 GiB of host memory
and removes disk reads from the critical path. Streamed from disk (`EXL3_NGRAM_LOCK=0`) it uses ~4 GiB RSS, but a
cold request can stall when the page cache is under pressure.

**Prefix cache under load.** Reusing a cached prompt prefix on this model needs two things: the prefix's KV pages
still in the GPU pool, and a saved state for the 36 linear-attention layers at that point (a "recurrent checkpoint",
~104 MiB each, kept in system RAM). TabbyAPI's default checkpoint budget (`memory.sysmem_recurrent_cache: 4096` MB)
holds only ~39. Under a coding-agent workload (15-22 concurrent conversations of 50-500k tokens) that budget ran out:
checkpoints of conversations waiting for their next turn were evicted, every turn re-prefilled from scratch, queues
passed client timeouts, and timed-out clients resent the same long prompts. Measured on the live server: 93-99 % of
prompt tokens served from cache with ≤12 conversations in flight, 2-10 % once that pattern set in. `deploy/config.yml`
therefore sets 65536 MB (~640 checkpoints); right after the change, 77 % of prompt tokens came from cache with a cold
cache, median queue 2 s and no timeouts.

If you front the server with a router or gateway:
- keep requests in flight at or below `max_batch_size` (12) and their combined prompts well below the KV pool (we use
  ~80 %, 2.4M tokens); send the overflow to another backend instead of letting it queue in this server, where every
  queued long prompt evicts other conversations' cached prefixes;
- time out on lack of progress (no streamed chunk for N minutes) rather than a fixed total, so long answers are not
  cut off and then resent in full.

**Admission.** `max_batch_size` is 12. The batched-cache decode graphs cover up to 12 jobs (patch 0019) and the
verify-row cap (`EXL3_DRAFT_MAX_ROWS=64`) trims each round's draft width so 9-12 jobs stay inside the 64-row
captured paths: 8 jobs keep the full 7 drafts, 9-12 jobs get 6-4. With 12 users the GPU serves 1.7× V2's total
throughput, and each user still gets ~65-70 t/s. VRAM after load is 60.7 GiB with the growable KV pool.

**Server history.** With the drafter attached, a freshly started server's first ~74 decoded tokens can differ
slightly from later requests; for example a raw `2+2=` as the very first request can complete with 5 instead of 4.
The launcher's 128-token warm-up absorbs this. More generally, greedy text on this stack can depend slightly on
what the server has decoded since start (near-tie tokens; next-token distributions differ only at noise level (batched KL 0.0015)). v1
behaves the same way.

**Prefill while others decode.** A new prompt shares the GPU with running decodes: by default it takes one
4096-token chunk every fifth generator step. With four users decoding, a cold 131k prompt reaches its first token
in 35 s and the decoders run at about a fifth of their speed meanwhile. TabbyAPI patch v3
(`deploy/tabbyapi-v3.patch`) exposes the two knobs: `prefill_cooldown` (steps between chunks)
and `shared_chunk_size`. Measured, 4 decoders + one cold 131k prompt:

|`prefill_cooldown` / `shared_chunk_size`|time to first token|decoders during the prefill|
|---|---|---|
|16 / 2048|86 s|−30 %|
|8 / 4096|41 s|−71 %|
|4 / 4096 (default)|35 s|−80 %|
|0 / 8192|25 s|−96 %|

No setting keeps the decoders within 15 %: every chunk stalls all decoders for its whole duration, so faster
first tokens cost decode speed linearly. Lower the cooldown for agent traffic where the new prompt's user is
waiting; raise it where streaming users must stay smooth.

## Repository

```
deploy/   launch.sh, config.yml, YaRN config patch, TabbyAPI patch v3
engine/   exllamav3 patch series (21 patches) and its README
bench/    the harnesses behind the tables: final.py (decode / prefill with first-token timing), prompt_cache.py,
          needle.py, identity_kl.py, accuracy.py (chat, exact answer), accuracy_raw.py (V2's probe), prefill under load, the fill/drain soak, the nsys prefill profiler
docs/     measurements.md: per-change A/B results and how they were measured
```

Run the benchmarks against a running server, e.g.
`BASE_URL=http://127.0.0.1:5000 API_KEY=<api_key from tabbyAPI/api_tokens.yml> python bench/final.py --label run1 --out run1.jsonl`.
See `bench/README.md` for each script, the timing definitions and the gate policy for changes.

## Credits

- **Qwen3.8-Flash-Next**: Qwen team, Alibaba. Qwen Community License 1.0; weights are not redistributed here.
- **exllamav3**: turboderp and contributors, MIT. EXL3 format, kernels, DFlash2 support; V2's patches 0004, 0005
  and 0007 extend its coop MoE kernels, CUDA-graph machinery and DFlash2 generator.
- **TabbyAPI**: theroyallab and contributors, AGPL-3.0.
- **DFlash2 drafter**: Inco AI / z-lab, Apache-2.0.
- **flash-linear-attention**: Songlin Yang et al., MIT.
- **QMesh R5** exllamav3 fork (GPTwareAiTeam, MIT), on which V2 builds: the cooperative MoE k-loop (0001), the dense
  decode GEMV (0002), the readback-free fused MoE dispatch (0003) and the adaptive prefill limiter (0006) are ported
  from it, as is the A, A, B, B / next-token-KL measurement method used for every number here. The candidate
  growable KV pool (patches 0010-0015) ports its growable cache: `cache/native/vmm_pool.cpp` and the `VmmRange`
  wrapper on CUDA virtual memory management.
  <https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10>

### New in V3

V2's credits above are unchanged. For the code new in V3, per item, where it comes from and how it was used. Items
not listed have no outside source.

|V3 item|source|how it was used|
|---|---|---|
|0008 speculative-decoding host readbacks (`EXL3_SD_DEFER_DRAFT_CONF`, `EXL3_SD_BATCH_READBACK`, `EXL3_SD_WINDOW_BATCH`)|[QMesh R5](https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10) exllamav3 fork, GPTwareAiTeam, MIT|base, modified: same three switches, the launch-all-samplers / one-synchronize design, the batchable-window conditions, `window_batch_safe` and the RNG replay. Changed: all off by default (R5: on), conditions factored into `_window_batchable`, the deferred readback moved to the draft-confidence calibrator, one pinned row per job. R5's rank-collective agree and device-side accept were not ported|
|0009 MoE tier trace (`EXL3_MOE_TIER_TRACE`)|QMesh R5, MIT|idea only: the switch name and purpose; the per-prefill aggregate counters and the allocator fields are new code|
|0010 growable KV pool: `exllamav3/cache/native/vmm_pool.cpp`|QMesh R5, MIT|copied exactly (CUDA virtual memory: reserve, map, unmap, view)|
|0010 growable KV pool: `VmmRange` (`exllamav3/cache/vmm.py`) and the `ensure_pages` / `page_bytes` / backed-page hooks|QMesh R5, MIT|base, modified: docstrings rewritten; generalized into a `GrowablePool` mixin (`cache/grow.py`) and applied to the quantized K/V layers and the sparse-attention plane pool (R5: its DeepSeek attention cache only). Not carried over: R5's slot lender, fixed KV budget and page-at-a-time growth|
|0013 release mapped pages on drain: `trim_pages` / `VmmRange.trim`|QMesh R5, MIT|base, modified: the trim helper is R5's; the drain policy (trigger on an empty queue, never below the highest referenced page or the premapped region, a free-memory target) is new|
|0020 counting-sort run table|QMesh R5 (the run-table code it replaces, carried in through 0001 / 0004), MIT|new code on an R5-derived base: replaces R5's per-slot rank loop|
|0005 `exllamav3/util/segment.py` (CUDA-graph segments, shipped in V2 and used again by V3's graphs)|QMesh R5, MIT|base, modified: R5's `Segment` class and capture-stream handling; added labels, parameter signatures, a second warm call and the `EXL3_SEG_CHECK` / `EXL3_SEG_MEMLOG` checks. Not named in V2's credits|
|0004 wide coop rows on verify windows (`EXL3_MOE_COOP_ROWS`, shipped in V2)|QMesh R5, MIT|base, modified: R5's switch (8-48 rows) raised to 64 rows / 640 slots, verify forwards only. Not named in V2's credits|

No code from TensorFold, MiaAI-Lab or z-lab is in the patches. Hybrid (suffix-match) drafting, which 0021 limits to
a single active request, is upstream exllamav3. DFlash2 drafter code is upstream exllamav3 as well; 0005 only wraps
its layer stack in a CUDA graph. `deploy/tabbyapi-v3.patch` and the new bench scripts are this repository's own work.

## License

The recipe, scripts and `engine/patches` are MIT (the engine patches modify MIT-licensed exllamav3).
`deploy/tabbyapi-v3.patch` modifies AGPL-3.0 TabbyAPI and is therefore AGPL-3.0. The linked
components keep their own licenses.
