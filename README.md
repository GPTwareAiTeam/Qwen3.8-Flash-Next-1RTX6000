# Qwen3.8-Flash-Next on one RTX PRO 6000 — V2

Serving recipe for `Qwen/Qwen3.8-Flash-Next` in EXL3 `3.18bpw_h6_ng8`
([turboderp/Qwen3.8-Flash-Next-exl3, branch 3.18bpw_h6_ng8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.18bpw_h6_ng8))
on a single 96 GiB RTX PRO 6000 Blackwell
(SM120), with TabbyAPI, a patched exllamav3, the DFlash2 drafter, a 3M-token KV pool and up to 640k tokens per
request.

This is V2. It improves on the original v1 recipe (stock exllamav3 1.6.4, 8-bit KV), the baseline in the tables below.

|Users|v1|V2|Change|
|---|---|---|---|
|1|231.1 t/s|**299.4 t/s**|+29.5 %|
|2|240.1|**398.0**|+65.8 %|
|4|344.5|**560.9**|+62.8 %|
|8|453.7|**676.7**|+49.1 %|
|KV cache (shared pool)|2,555,904 tokens|**2,999,808 tokens**|+17.4 %|
|Max per request|1,048,576|655,360|up to 8 requests share the pool|

v1: stock exllamav3 1.6.4 with 8-bit KV and the same model, drafter and TabbyAPI. Both measured on the same
card, alternating A, A, B, B server boots, three runs per boot, medians; 512 generated tokens per request, greedy.
The user rows were measured with V2 at a 3,121,152-token pool and 2048-token prefill chunks; the shipped config
(3M pool, 8192-token chunks) measured 305.3 t/s at 1 user and 668.8 t/s at 8 users in a single run, the same within
noise.

|Decode, 2048 forced tokens from a short prompt|v1, greedy|V2, greedy|Change|V2, `qwen38_thinking` preset|
|---|---|---|---|---|
|prose|104.2 t/s|125.8 t/s|+20.7 %|117.8 t/s|
|code|249.6|287.3|+15.1 %|263.6|
|chat|122.8|156.1|+27.1 %|152.6|
|math|260.9|313.8|+20.3 %|312.0|

Greedy shown for comparison; the shipped qwen38_thinking preset (Qwen's recommended thinking-mode sampling) is 0–8 %
slower because sampled tokens accept fewer drafts. The preset column samples at temperature 0.6, top-k 20, top-p 0.95
(`bench/decode_natural.py` with `SAMPLING=preset`, one run of three repeats).

|Prefill, cold prompt → 1 token|v1 (2048-token chunks)|V2 (8192-token chunks)|V2 time to first token|
|---|---|---|---|
|~2k tokens|3,044-3,268 t/s|3,475 t/s|0.6 s|
|32k|-|**5,690 t/s**|5.8 s|
|128k|-|**5,647 t/s**|23 s|
|520k|3,990 t/s (130.2 s)|**5,294 t/s (98.3 s)**|98 s|
|worst of 5 cold 32k requests|16.5 s (stall)|8.65 s (measured with 2048 chunks)||

Long prompts prefill 33-36 % faster than with 2048-token chunks (32k: 4,262 → 5,690 t/s; 128k: 4,146 → 5,647 t/s;
520k: 131 → 98 s). The larger chunks fit because the pool is 3M tokens instead of 3.12M. With the engram table locked
in RAM, a cold first request also no longer stalls on disk reads.

Quality: 22/24 on the exact-answer probe (`bench/accuracy.py`, v1 22/24; measured with 2048-token chunks); a planted
fact 520k tokens deep is retrieved; 1.9 GiB of VRAM stays free at peak (8-user run plus a 520k prefill). Against
2048-token chunks, 8192-token chunks give identical greedy output on the 8 identity prompts and identical next-token
distributions at short context; at long context KL is 0.0019 mean / 0.0053 max (the 6-bit KV cache itself: 0.0065).

## What it takes

- One RTX PRO 6000 Blackwell (96 GiB). Measured on the Max-Q (300 W) edition.
- 50 GiB of lockable host RAM for the engram table (memlock ≥ 50 GiB; `launch.sh` checks it), or stream the table
  from a fast NVMe instead (see below).
- The model: [`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.18bpw_h6_ng8),
  branch `3.18bpw_h6_ng8` (measured at revision `2bfc9dade6be0b5b42e2b381d7a6cc160c1d409c`). About 100 GiB on disk,
  half of it the engram (n-gram embedding) table.
- Python 3.12, PyTorch 2.11 with CUDA 13 (`cu130` wheels), flash-linear-attention 0.6.0, the CUDA toolkit
  (`nvcc`, for the extension's JIT build) and `ninja`. Use an environment without a pip-installed exllamav3: its
  prebuilt extension would shadow the patched build (`launch.sh` refuses to start in that case).
- [exllamav3](https://github.com/turboderp-org/exllamav3) `v1.6.4` + `engine/patches`.
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) `564f8fb` + `deploy/tabbyapi-prefill-chunk-size.patch`.
- Drafter: [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).

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
git checkout 564f8fb && git apply /path/to/repo/deploy/tabbyapi-prefill-chunk-size.patch
pip install .
cp /path/to/repo/deploy/sampler_overrides/qwen38_thinking.yml sampler_overrides/ && cd ..

# model and drafter
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.18bpw_h6_ng8 \
  --local-dir /path/to/models/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8
hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir /path/to/drafts/Qwen3.8-Flash-Next-DFlash2

# long context (needed for the 640k limit): apply the YaRN factor-4 change to the model's config.json
(cd /path/to/models/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 && patch -p1 < /path/to/repo/deploy/model-config-yarn4.patch)

# edit deploy/config.yml: model_dir, draft_model_dir, network.port, auth; then
ulimit -l 52428800      # memlock for the engram lock (or LimitMEMLOCK=52428800K in a systemd unit)
TABBY=$PWD/tabbyAPI ENGINE=$PWD/exllamav3 GPU=0 ./deploy/launch.sh
```

`launch.sh` sets the engine switches, waits for the model and sends one 128-token warm-up request before serving
(see "Server history").

## What changed against v1

|Change|Where|Effect|Numerics|
|---|---|---|---|
|KV cache 8-bit → 6-bit (`cache_mode: '6,6'`), pool 2,555,904 → 2,999,808 tokens|config|+17 % context capacity, speed unchanged|slightly lossy: top-20 KL 0.018 short / 0.0065 long context, accuracy unchanged|
|Prefill chunks 2048 → 8192 (`chunk_size: 8192`), per-request limit 640k|config|cold prefill +33 % at 32k, +36 % at 128k, 520k in 98 s instead of 131 s|identical greedy output on the identity prompts; long-context KL 0.0019 mean / 0.0053 max|
|Cooperative MoE k-loop for 3-bit experts|patch 0001|1 user +4.7 %, 2-4 users +9 %|bit-identical|
|No per-layer expert-count readback to the host in the fused MoE tier|patch 0003|2 users +20-23 %, 4 users +8 %|bit-identical|
|Verify windows up to 64 rows through the coop MoE kernels|patch 0004, `EXL3_MOE_COOP_ROWS=64`|8 users +20 %|same next-token distributions|
|Shared expert on a side stream, overlapping the routed experts|patch 0007, `EXL3_MOE_SHARED_SIDE=1`|1 user +6 %, 8 users +12 %|bit-identical|
|DFlash2 drafter layers as one CUDA graph|patch 0005, `EXL3_DFLASH_GRAPH=1`|8 users +4 %, code +5 %|lossless|
|Draft confidence 0.30 → 0.45|config|1 user +5.5 %, 8 users +3.9 %|changes greedy text on near-ties; same next-token distributions|
|Engram table locked in RAM (`EXL3_NGRAM_LOCK=1`)|launch.sh|removes a 16 s first-request stall on cold 32k prompts (8.7 s worst case after); boot +15 s|lossless|

Each row was measured as an A/B against the step before it. Ports from the
[QMesh R5](https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10) exllamav3 fork are noted in
`engine/README.md`.

Tried and left out: the built-in MTP drafter (needs 18 % less KV to fit; prose +11.5 % but code and math −15 %),
dense DGEMV for the 6/7-bit dense projections (changes greedy output, net neutral), adaptive prefill chunk sizing
(+3.4 % at 128k, slower decode; in the tree as `EXL3_PREFILL_ADAPTIVE`, off), 16384-token prefill chunks (out of
memory on a 32k prompt at the 3M pool), piecewise decode CUDA graphs (+3 % at 8 users for 474 MiB of VRAM), draft
confidence 0.30 (+4-6 % single-stream decode, −4 % at 4 users, changes greedy text).

## Details

**Model.** Hidden 2560, 48 layers (12 sparse full-attention + 36 Gated DeltaNet linear-attention), 512 experts
with 10 routed + 1 shared active per token, hyper-connections, vocab 248,320. The checkpoint is trained at 262,144
positions; `deploy/model-config-yarn4.patch` applies static YaRN factor 4 for 1,048,576. In an early comparison
against the native 262k context, YaRN factor 4 cost speed on code-like continuations (lower draft acceptance) and
nothing on prose.

**KV capacity.** 6-bit K/V costs 13.5 KiB per token; 3.75 KiB of that is fixed fp16 planes for the sparse-attention
indexer. The pool is allocated at load, so `cache_size` is the hard ceiling. 2,999,808 tokens (3M rounded to a
multiple of 256) with 8192-token prefill chunks keeps 1.9 GiB free after an 8-user run and a 520k-token prefill.
Each request can use up to 655,360 tokens (`max_seq_len`). Eight full-length requests would need 5.2M tokens, more
than the pool: the generator reserves pages for a request's prompt plus its next output chunk, so when the pool runs
short a new or continuing request waits for pages rather than failing. In practice requests rarely all grow to 640k
at once. The largest pool with 2048-token chunks is 3,121,152 tokens; the model supports 1,048,576 tokens per
request with the YaRN patch, so `max_seq_len` can be raised again for longer single requests.

**Drafting.** DFlash2 drafts 8-token blocks; the verify window is capped at 7 drafts because the fused MoE path
handles at most 8 rows per sequence. Draft acceptance drives single-stream speed: about 75 % on code and math,
about 35 % on prose and chat.

**Engram table.** 48.3 GiB of n-gram embeddings read per token. Locked in RAM it costs 48.3 GiB of host memory
and removes disk reads from the critical path. Streamed from disk (`EXL3_NGRAM_LOCK=0`) it uses ~4 GiB RSS, but a
cold request can stall when the page cache is under pressure.

**Prefill.** See the prefill table at the top. With 8192-token chunks, throughput is ~3,500 t/s at 2k and ~5,300-5,700
t/s from 32k to 520k; with 2048-token chunks it is ~4,000-4,250 t/s at 32k-520k. The KV cache mode does not change it
(130-133 s for the 520k needle at 2048-token chunks in every mode tested).

**Server history.** With the drafter attached, a freshly started server's first ~74 decoded tokens can differ
slightly from later requests; for example a raw `2+2=` as the very first request can complete with 5 instead of 4.
The launcher's 128-token warm-up absorbs this. More generally, greedy text on this stack can depend slightly on
what the server has decoded since start (near-tie tokens; next-token distributions are unaffected). v1
behaves the same way.

## Repository

```
deploy/   launch.sh, config.yml, YaRN config patch, TabbyAPI patch
engine/   exllamav3 patch series and its README
bench/    the harnesses behind the tables (matrix, decode, accuracy, cold time-to-first-token)
```

Run the benchmarks against a running server, e.g.
`BASE_URL=http://127.0.0.1:5000 API_KEY=<api_key from tabbyAPI/api_tokens.yml> python bench/matrix.py`.
`matrix.py` takes `CONC`, `REPS` and `SECTIONS` (e.g. `CONC=1 REPS=1 SECTIONS=concurrency`); `decode_natural.py`
takes `SAMPLING=preset` to use the server's sampling preset instead of greedy.

## Credits

- **Qwen3.8-Flash-Next**: Qwen team, Alibaba. Qwen Community License 1.0; weights are not redistributed here.
- **exllamav3**: turboderp and contributors, MIT. EXL3 format, kernels, DFlash2 support; V2's patches 0004, 0005
  and 0007 extend its coop MoE kernels, CUDA-graph machinery and DFlash2 generator.
- **TabbyAPI**: theroyallab and contributors, AGPL-3.0.
- **DFlash2 drafter**: Inco AI / z-lab, Apache-2.0.
- **flash-linear-attention**: Songlin Yang et al., MIT.
- **QMesh R5** exllamav3 fork (GPTwareAiTeam, MIT), on which V2 builds: the cooperative MoE k-loop (0001), the dense
  decode GEMV (0002), the readback-free fused MoE dispatch (0003) and the adaptive prefill limiter (0006) are ported
  from it, as is the A, A, B, B / next-token-KL measurement method used for every number here.
  <https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10>

## License

The recipe, scripts and `engine/patches` are MIT (the engine patches modify MIT-licensed exllamav3).
`deploy/tabbyapi-prefill-chunk-size.patch` modifies AGPL-3.0 TabbyAPI and is therefore AGPL-3.0. The linked
components keep their own licenses.
