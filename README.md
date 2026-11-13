# Qwen3.8-Flash-Next on one RTX PRO 6000 — V2

Serving recipe for `Qwen/Qwen3.8-Flash-Next` (EXL3 `3.18bpw_h6_ng8`) on a single 96 GiB RTX PRO 6000 Blackwell
(SM120), with TabbyAPI, a patched exllamav3, the DFlash2 drafter and a 1M-token context.

This is V2. It improves on the original v1 recipe (stock exllamav3 1.6.4, 8-bit KV), the baseline in the tables below.

|Users|v1|V2|Change|
|---|---|---|---|
|1|231.1 t/s|**299.4 t/s**|+29.5 %|
|2|240.1|**398.0**|+65.8 %|
|4|344.5|**560.9**|+62.8 %|
|8|453.7|**676.7**|+49.1 %|
|KV cache|2,555,904 tokens|**3,121,152 tokens**|+22.1 %|

v1: stock exllamav3 1.6.4 with 8-bit KV and the same model, drafter and TabbyAPI. Both measured on the same
card, alternating A, A, B, B server boots, three runs per boot, medians; 512 generated tokens per request, greedy.

|Decode, 2048 forced tokens from a short prompt|v1, greedy|V2, greedy|Change|V2, `qwen38_thinking` preset|
|---|---|---|---|---|
|prose|104.2 t/s|125.8 t/s|+20.7 %|117.8 t/s|
|code|249.6|287.3|+15.1 %|263.6|
|chat|122.8|156.1|+27.1 %|152.6|
|math|260.9|313.8|+20.3 %|312.0|

Greedy shown for comparison; the shipped qwen38_thinking preset (Qwen's recommended thinking-mode sampling) is 0–8 %
slower because sampled tokens accept fewer drafts. The preset column samples at temperature 0.6, top-k 20, top-p 0.95
(`bench/decode_natural.py` with `SAMPLING=preset`, one run of three repeats).

|Prefill, cold prompt → 1 token|v1|V2|V2 time to first token|
|---|---|---|---|
|~2k tokens|3,044-3,268 t/s|3,453-3,507 t/s|0.6-0.8 s|
|32k|-|4,200-4,260 t/s|7.3-7.5 s|
|128k|-|4,146-4,240 t/s|~30 s|
|520k|3,990 t/s (130.2 s)|3,940-3,990 t/s|130-132 s|
|worst of 5 cold 32k requests|16.5 s (stall)|8.65 s||

Prefill throughput is unchanged: V2's patches target decode, and both use 2048-token chunks (the 2k gap comes from
separate sessions, not an A/B). What V2 improves is the worst case: with the engram table locked in RAM, a cold
first request no longer stalls on disk reads. Larger chunks would help long prompts (chunk 8192: +39-40 % at 32k and
128k) but do not fit next to the full KV pool; see `docs/measurements.md`.

Quality: 22/24 on the exact-answer probe (`bench/accuracy.py`, v1 22/24); a planted fact 520k tokens deep is
retrieved; 1.6 GiB of VRAM stays free at peak.

## What it takes

- One RTX PRO 6000 Blackwell (96 GiB). Measured on the Max-Q (300 W) edition.
- 50 GiB of lockable host RAM for the engram table (memlock ≥ 50 GiB; `launch.sh` checks it), or stream the table
  from a fast NVMe instead (see below).
- The model, about 100 GiB on disk, half of it the engram (n-gram embedding) table.
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

# 1M context: apply the YaRN factor-4 change to the model's config.json
cd /path/to/models/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 && patch -p1 < /path/to/repo/deploy/model-config-yarn4.patch

# edit deploy/config.yml: model_dir, draft_model_dir, network.port, auth; then
ulimit -l 52428800      # memlock for the engram lock (or LimitMEMLOCK=52428800K in a systemd unit)
TABBY=$PWD/tabbyAPI ENGINE=$PWD/exllamav3 GPU=0 ./deploy/launch.sh
```

`launch.sh` sets the engine switches, waits for the model and sends one 128-token warm-up request before serving
(see "Server history").

## What changed against v1

|Change|Where|Effect|Numerics|
|---|---|---|---|
|KV cache 8-bit → 6-bit (`cache_mode: '6,6'`), pool 2,555,904 → 3,121,152 tokens|config|+22 % context capacity, speed unchanged|slightly lossy: top-20 KL 0.018 short / 0.0065 long context, accuracy unchanged|
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
(+3.4 % at 128k, slower decode; in the tree as `EXL3_PREFILL_ADAPTIVE`, off), piecewise decode CUDA graphs
(+3 % at 8 users for 474 MiB of VRAM), draft confidence 0.30 (+4-6 % single-stream decode, −4 % at 4 users,
changes greedy text).

## Details

**Model.** Hidden 2560, 48 layers (12 sparse full-attention + 36 Gated DeltaNet linear-attention), 512 experts
with 10 routed + 1 shared active per token, hyper-connections, vocab 248,320. The checkpoint is trained at 262,144
positions; `deploy/model-config-yarn4.patch` applies static YaRN factor 4 for 1,048,576. In an early comparison
against the native 262k context, YaRN factor 4 cost speed on code-like continuations (lower draft acceptance) and
nothing on prose.

**KV capacity.** 6-bit K/V costs 13.5 KiB per token; 3.75 KiB of that is fixed fp16 planes for the sparse-attention
indexer. The pool is allocated at load, so `cache_size` is the hard ceiling: 3,121,152 tokens is the largest that
keeps 1.5 GiB free after an 8-user run and a 520k-token prefill. Any request can use up to 1M tokens of it.

**Drafting.** DFlash2 drafts 8-token blocks; the verify window is capped at 7 drafts because the fused MoE path
handles at most 8 rows per sequence. Draft acceptance drives single-stream speed: about 75 % on code and math,
about 35 % on prose and chat.

**Engram table.** 48.3 GiB of n-gram embeddings read per token. Locked in RAM it costs 48.3 GiB of host memory
and removes disk reads from the critical path. Streamed from disk (`EXL3_NGRAM_LOCK=0`) it uses ~4 GiB RSS, but a
cold request can stall when the page cache is under pressure.

**Prefill.** See the prefill table at the top. Throughput rises with prompt length from ~3,500 t/s at 2k to ~4,200
t/s at 32k-128k and eases to ~4,000 t/s at 520k. The KV cache mode does not change it (130-133 s for the 520k needle
in every mode tested).

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
