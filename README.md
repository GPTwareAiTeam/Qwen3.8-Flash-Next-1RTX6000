# Qwen3.8-Flash-Next on one RTX PRO 6000 — V3

Serving recipe for `Qwen/Qwen3.8-Flash-Next` in EXL3 `3.18bpw_h6_ng8`
([turboderp/Qwen3.8-Flash-Next-exl3, branch 3.18bpw_h6_ng8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.18bpw_h6_ng8))
on a single 96 GiB RTX PRO 6000 Blackwell (SM120), with TabbyAPI, a 21-patch exllamav3, the DFlash2 drafter, a
3M-token shared KV pool, up to 512k tokens per request and 12 concurrent requests.

This is V3. It improves on V2, the previous release of this repo, the baseline in the tables below. All numbers are
V3 as shipped (YaRN factor 2) unless a row says "YaRN factor 4".

|Users|V2|V3|Change|
|---|---|---|---|
|1|207.5 t/s|**216.6 t/s**¹|+4.4 %|
|2|302.5|**360.9**¹|+19.3 %|
|4|392.6|**486.4**|+23.9 %|
|8|472.2|**610.6**¹|+29.3 %|
|12|375.9|**645.9**|**+71.8 %**|
|KV cache (shared pool)|2,999,808 tokens, allocated at load|**2,999,808 tokens, mapped as it fills**|VRAM after load 93.9 → 60.7 GiB|
|Max per request|655,360|524,032|12 requests share the pool|

Aggregate decode, geometric mean over prose, chat, code and math prompts; 512 generated tokens per request, greedy,
tokens/s from each stream's first token (`bench/final.py`). Every comparison alternates server boots A, A, B, B on
the same card, three runs per boot, medians. The 4- and 12-user rows are V2 against V3 as shipped, measured directly.
¹ 1, 2 and 8 users come from a V2-against-V3 run with V3 at YaRN factor 4 (215.3 / 348.4 / 586.0 t/s) times the
measured factor-4 → factor-2 change (+0.6 / +3.6 / +4.2 %, its own A, A, B, B): projections, not direct end-to-end
runs.

|Decode, 2048 forced tokens from a short prompt|V2, greedy|V3, greedy|Change|V3, `qwen38_thinking` preset|
|---|---|---|---|---|
|prose|127.3 t/s|127.4 t/s|+0.1 %|123.4 t/s|
|code|289.1|**306.8**|+6.1 %|276.4|
|chat|155.8|152.1|−2.4 %|152.3|
|math|319.8|**329.3**|+3.0 %|321.2|

One request at a time, so most of V3's gains (which are in batching) don't show; prose and chat are within run-to-run
noise. `bench/decode_natural.py`, one run of three repeats per release. The preset is the shipped
`qwen38_thinking` sampling (temperature 0.6, top-k 20, top-p 0.95): sampled tokens accept fewer drafts, so it is
0-10 % slower than greedy.

|Prefill, cold prompt → 1 token|V2|V3|V3 time to first token|
|---|---|---|---|
|~2k tokens|3,541 t/s|3,284 t/s (YaRN factor 4)|0.62 s|
|32k|5,660 t/s|**5,990 t/s** (factor 4); 5,933 t/s (factor 2)|5.3-5.5 s|
|128k|5,683 t/s|**5,949 t/s** (factor 4)|22.0 s|
|520k|5,385 t/s (96.6 s)|**5,915 t/s (87.9 s)**|88 s|
|worst of 5 cold 32k requests|—|5.61 s (median 5.19 s)||

2k/32k/128k: A, A, B, B of V2 against V3 at YaRN factor 4, three runs per boot; the factor-2 32k figure is one run
with V2 in the same session. 520k: one cold request each, V3 at factor 2. Long prompts gain because the growable
pool leaves room for 16k-token chunks; short prompts pay ~0.05 s for the per-chunk memory check.

Quality, V3 as shipped unless noted: **50/50** on the exact-answer probe (`bench/accuracy.py`, V2 50/50); a passphrase
planted halfway into a 520k-token prompt is retrieved; next-token agreement with V2 (YaRN factor 4, 16 short and 9
long prompts up to 157k) is top-1 25/25 with KL 0.0010-0.0014 short and 0.0018 long, against a V2-vs-V2 floor of
0.0004 / 0.0000; an evicted 100k prompt returns in **0.44 s** (RAM tier, V2: a full ~17 s prefill) and in **0.87 s**
after a server restart (NVMe tier); 6.6 GiB of VRAM stays free at peak (V2: 1.0 GiB). Factor 2 against factor 4:
top-1 16/16 and KL 0.015 mean — the position encoding itself changes, as intended. Details and per-cell tables:
`docs/measurements.md`.

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

## What changed against V2

|Change|Where|Measured effect|Numerics|
|---|---|---|---|
|12 concurrent requests: decode graphs for 12 jobs, a 64-row cap on verify rows per round, and the fix for the cap reading the batch size from the draft buffer (it had cut every round to 4 drafts)|0019, 0021, `max_batch_size: 12`, `EXL3_DRAFT_MAX_ROWS=64`|12 users 1.7× V2; cap-fix alone: 1 user math 254 → 335 t/s|lossless (draft width only)|
|Growable KV pool: the 3M-token pool reserved as CUDA virtual memory, mapped in 256-page steps, released from the top when the queue drains|0010-0015, `EXL3_KV_GROW=1`|VRAM after load 93.9 → 60.7 GiB; the freed room goes to prefill|lossless (same pages, mapped on demand)|
|Adaptive prefill chunks up to 16k sized to free memory; TabbyAPI patch v3 adds `prefill_chunk_size`, `shared_chunk_size`, `prefill_cooldown`|0006 (now on), 0010, 0013, 0014, `prefill_chunk_size: 16384`|cold 32k +6-8 %, 128k +5-9 %, 2k −2..−7 %|rounding-level (chunk boundaries move with free memory)|
|Prompt-cache RAM (192 GiB pinned) and NVMe tiers|`memory.sysmem_kv_cache`, `disk_cache_dir`, `disk_cache_size`|evicted 100k prompt: 17 s → **0.44 s**; after a restart **0.87 s**|bit-identical (cached pages)|
|Hyper-connection int8 tables built once at load instead of per call (above 8 rows)|0016, `EXL3_HC_TABLE_CACHE=1`|8 users +9-11 %; +1.3 GiB VRAM|bit-identical|
|Shared expert of a wide verify window as one launch|0017, `EXL3_MOE_SHARED_WIDE=1`|8 users +7 %|rounding-level|
|Batched speculative accept: one readback per round, rewinds settled in one batch (incl. the TabbyAPI sampler flag that had kept it from engaging)|0018, `EXL3_SD_BATCH_ACCEPT=1`|8 users +6-8 %|lossless|
|Drafter's next block launched under the host accept loop; hybrid (suffix-match) drafting of 16 tokens while exactly one request is active|0021, `EXL3_SD_DRAFT_OVERLAP=1`, `hybrid_draft_tokens: 16`|8 users prose +4.6 %; 1 user rewriting a ~1.5k-token file +36-53 %|lossless|
|Counting-sort run table for wide verify windows|0020, `EXL3_MOE_COOP_FAST_RUNS=1`|rotation launch 50-60 → 8-9 µs at 56-64 rows|bit-identical|
|YaRN factor 4 → 2 (524,288 positions); per-request limit 655,360 → 524,032|`deploy/model-config-yarn2.patch`, `max_seq_len: 524032`|1/2/8 users +0.6/+3.6/+4.2 % (A, A, B, B), mostly on code|changes the position encoding: short-prompt KL 0.015 vs factor 4|
|Bench: first-token timing, the release benchmark, needle and prompt-cache harnesses; gate policy = next-token top-1 + KL against the run-to-run noise floor|`bench/timing.py`, `final.py`, `needle.py`, `prompt_cache.py`|every V3 number uses them|—|

In the series but not enabled by `launch.sh` (measured, no gain on this stack, or diagnostics only): the speculative
host readbacks of 0008 (`EXL3_SD_DEFER_DRAFT_CONF`, `EXL3_SD_BATCH_READBACK`, `EXL3_SD_WINDOW_BATCH`; subsumed by
0018), the split down-stage reduction of 0020 (`EXL3_MOE_COOP_SPLIT_B`, slower end to end) and the diagnostics of
0009. Shipped in V2 and unchanged: 0001-0007 (the dense DGEMV of 0002 stays off). Per-change A/Bs, one boot per arm
where marked, and the numbers behind each row: `docs/measurements.md`.

## Trade-offs

- **Host RAM ~250 GiB.** The prompt-cache RAM tier dominates. Lower `sysmem_kv_cache` (sized in MB; 192 GiB holds
  ~15M tokens of evicted KV at 13.5 KiB per token) to fit a smaller host, or set it to 0 to drop both cache tiers.
- **Rounding-level drift.** Patch 0017 and the adaptive chunks change floating-point summation order: greedy text
  differs from V2 on 3-5 of the 8 identity prompts at near-tie tokens (V2 against itself: 7/8; V3 against itself:
  6/8). Next-token top-1 stays 25/25, KL stays 0.001-0.002, the exact-answer probe is 50/50 on both releases.
- **YaRN factor 2.** 512k per request instead of 640k. The model was trained at 262,144 positions and the drafter
  without YaRN, so the smaller factor drafts slightly better. For longer single requests, set `"factor": 4.0`,
  `"max_position_embeddings": 1048576` and `max_seq_len` ≥ 655,360 (V2's setting).
- **Cold 2k prefill −7 %**: the adaptive limiter queries free memory per chunk (0.57 → 0.62 s). Long prompts gain.
- **12 users share the verify budget**: 7 → 6-4 drafts per round, so per-user speed falls (8 users 84.8, 12 users
  68.7 t/s) while total throughput rises.
- **Slower start**: ~80-90 s to ready (V2 ~45 s), because the RAM tier is page-locked at load.

## Details

**Model.** Hidden 2560, 48 layers (12 sparse full-attention + 36 Gated DeltaNet linear-attention), 512 experts
with 10 routed + 1 shared active per token, hyper-connections, vocab 248,320. The checkpoint is trained at 262,144
positions.

**Context length.** `deploy/model-config-yarn2.patch` applies static YaRN factor 2 (524,288 positions);
`max_seq_len: 524032` leaves one 256-token page for the draft and verify positions of a request at the limit
(drafts are up to 16 tokens with hybrid drafting). V2 used factor 4 (1,048,576 positions, 640k per request). The
per-cell factor-4 → factor-2 comparison is in `docs/measurements.md`.

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
weights, so a config change (such as the YaRN factor) starts a fresh subdirectory. 197 checkpoints were held after a
3.1M-token fill test, well inside the 64 GiB budget.

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
what the server has decoded since start (near-tie tokens; batched KL 0.0015). v1 behaves the same way.

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
docs/     measurements.md: per-change A/B results, per-cell tables and how they were measured;
          provenance.md: where each V3 item comes from
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
  from it, as is the A, A, B, B / next-token-KL measurement method used for every number here. The growable KV pool
  (patches 0010-0015) ports its growable cache: `cache/native/vmm_pool.cpp` and the `VmmRange` wrapper on CUDA
  virtual memory management.
  <https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10>

V3's own items and what each owes to that fork (0008, 0009, 0010-0015, 0020, and 0004/0005, which V2's credits did
not name): `docs/provenance.md`. No code from TensorFold, MiaAI-Lab or z-lab is in the patches; hybrid drafting and
the DFlash2 drafter code are upstream exllamav3, and `deploy/tabbyapi-v3.patch` plus the new bench scripts are this
repository's own work.

## License

The recipe, scripts and `engine/patches` are MIT (the engine patches modify MIT-licensed exllamav3).
`deploy/tabbyapi-v3.patch` modifies AGPL-3.0 TabbyAPI and is therefore AGPL-3.0. The linked
components keep their own licenses.
