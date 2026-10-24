# Qwen3.8-Flash-Next EXL3 3.18 bpw on one RTX PRO 6000 (Blackwell, SM120)

Deployment recipe and measured results for serving `Qwen/Qwen3.8-Flash-Next` (EXL3 `3.18bpw_h6_ng8`)
through TabbyAPI + exllamav3 on a **single** 96 GiB RTX PRO 6000 Max-Q, with a **DFlash2 block-diffusion
drafter**, **1M-token YaRN context**, and the engram (PLE) table streamed from NVMe instead of loaded.

Everything below was measured on this host on 2026-10-23. Numbers are medians unless stated; the
harness scripts that produced them are in `bench/`.

---

## Hardware and software

|Item|Value|
|---|---|
|GPU|NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB (GPU1 only; GPU0 hosts an unrelated vLLM)|
|CPU / disk|model on a Gen5 NVMe (`/models/dsv41`), engram table streamed per forward|
|Runtime venv|`python 3.12`, `torch 2.11.0+cu130`|
|exllamav3|**1.6.4+cu132.torch2.11.0** (MIT)|
|TabbyAPI|`theroyallab/tabbyAPI` @ `564f8fb` (AGPL-3.0)|
|flash-linear-attention|0.6.0|
|Target model|`Qwen3.8-Flash-Next` EXL3 `3.18bpw_h6_ng8`, 99.17 GiB on disk (48.28 GiB of it is the streamed n-gram/engram table)|
|Drafter|`incoai/Qwen3.8-Flash-Next-DFlash2` (Apache-2.0), 603M params BF16, 1.2 GiB|

Model geometry: hidden 2560, 48 layers (12 full attention + 36 linear attention), 24 Q heads / 2 KV heads,
head_dim 256, partial rotary 0.25, `full_attention_interval 4`, 512 experts with 10 active,
`moe_intermediate_size 640`, vocab 248,320, built-in MTP component (1 layer) that this deployment does
**not** use — DFlash2 drafts instead.

---

## Deployed configuration

`deploy/config.yml` (excerpt):

```yaml
model:
  model_name: Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8
  max_seq_len: 1048576     # 1M per stream (YaRN factor 4)
  cache_size: 2555904      # 2.44M-token KV pool
  cache_mode: 8,8          # K 8-bit, V 8-bit
  gpu_split: [99]
  max_batch_size: 8
  chunk_size: 2048
  ngram_ram: false         # engram table stays on disk
  vision: true

draft_model:
  draft_mode: model
  draft_model_dir: /models/dsv41/drafts
  draft_model_name: Qwen3.8-Flash-Next-DFlash2
  draft_num_tokens: 7
  dynamic_draft: true
  draft_confidence: 0.30
```

`deploy/launch.sh` pins `CUDA_VISIBLE_DEVICES=1` and sets the decode-path selectors:

```bash
export EXL3_NGRAM_STREAM=1        # engram rows gathered from disk per forward
export EXL3_NGRAM_PREFETCH=1
export EXL3_MOE_MTILE=0           # one 16-row fused MoE launch instead of 32/64-row tiles
export EXL3_MOE_SHARED_COOP=0     # keep the three-launch shared-expert graph
export EXL3_MOE_COOP_WIDE=1       # wide coop tile for routed experts
```

Those three MoE selectors were worth 207 -> 254 t/s at the time they were tuned (MTP drafting);
they are content-independent kernel-shape choices and were kept after the drafter swap.

### 1M context

The checkpoint is trained at 262,144 positions. The model card's own recipe is static YaRN, applied by
patching the model's `config.json` (`deploy/model-config-yarn4.patch`):

```json
"max_position_embeddings": 1048576,
"rope_parameters": { "mrope_interleaved": true, "mrope_section": [11,11,10],
                     "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                     "rope_type": "yarn", "factor": 4.0,
                     "original_max_position_embeddings": 262144 }
```

Verified: a planted string at 267,907 / 401,845 / 535,783 / 669,721 tokens was retrieved correctly at the
lower three, prefill 4,099-4,700 tok/s. Cost of static YaRN: roughly 10% on code-like continuations
(draft acceptance falls), unchanged on prose.

---

## Measured results

All figures come from the OpenAI-compatible endpoint (`/v1/completions`, greedy, `temperature: 0`,
`add_bos_token` left at the model default of **false**). Medians of 3 where noted.

### Single stream, drafters compared (512 new tokens)

|prompt|MTP, dyn 0.4|DFlash2, dyn 0.30|
|---|---|---|
|code rewrite|147.6 / 177.9|**249.6 / 267.2**|
|"reverse a linked list"|208.4 / 209.6|**244.8 / 246.7**|
|prose (sky blue)|163.7 / 163.4|160.6 / 148.5|
|chat endpoint|173.3 / 174.8|**188.6 / 190.0**|
|batch 8 x 256 aggregate|323.5|**342.9**|

DFlash2 wins code, chat and long-form instructions; MTP is still better on open-ended prose
(164-174 vs 109-160 t/s) because block-8 drafting wastes verify slots there (rejections ~1,000-1,250 per
1024 tokens vs ~400).

### Concurrency (512 new tokens, code prompt, median of 3)

|concurrency|aggregate|per stream|
|---|---|---|
|1|233.8 t/s|233.9|
|2|243.3 t/s|121.7|
|4|352.2 t/s|91.7|
|8|**467.7 t/s**|62.2|

### Prefill ~2048 tokens -> 1 output token (median of 3)

|context type|prompt tokens|wall|prefill|
|---|---|---|---|
|prose|1,717|530 ms|3,242 tok/s|
|code|2,613|821 ms|3,184 tok/s|
|chat / instruction|1,670|536 ms|3,115 tok/s|
|math / derivation|2,444|800 ms|3,054 tok/s|

Longer prompts: 40k -> 4,550-4,700 tok/s; 535k -> 4,099 tok/s cold.

### ~128 prompt tokens -> 2048 decode (natural prompts, `min_tokens` forced, loop detect off)

|context type|decode|draft accepted / rejected|
|---|---|---|
|prose (essay)|108.9 t/s|1,106 / 2,204|
|code (LRU module)|**254.6 t/s**|1,709 / 583|
|chat (support reply)|125.6 t/s|1,262 / 2,367|
|math (derivation)|264.4 t/s|1,723 / 486|

Acceptance is the whole story: prose/chat accept ~35%, code/math ~75%. Kernel time is nearly constant.

### Knobs measured and rejected

|knob|effect|
|---|---|
|`EXL3_QT_OPTIMIZED=1`|186 vs 190 t/s|
|`EXL3_GEMV=0`|174 vs 190 t/s|
|`EXL3_FA75=1`|neutral decode, prefill 4,548 vs 4,694 tok/s|
|`EXL3_SDPA_PREFILL=0`|174 t/s, prefill 4,158 tok/s (worse)|
|`EXL3_HGEMM_F16ACC=1`, `EXL3_MOE_COOP_KSPLIT=2`, `EXL3_MOE_POOL_PRIME=1`, `EXL3_MOE_MTILE=0 + FUSED_DET=0`|all neutral or worse|
|draft confidence 0.20 / 0.45|116 / 184 vs 190 t/s at 0.30|
|`dynamic_draft: false` (fixed 7)|code 168, live 232, prose 145|
|draft depth 8 (9 verify positions)|falls off the `MAX_BSZN=8` fused path: 250 -> 165 t/s|
|KV `8,4`|no speed gain at short context; changes argmax (`2+2=` -> 5) so not shipped|

### Cache sizing and capacity

At `8,8` the pool costs **17.9 KB/token** (12 full-attention layers at 1 byte per element plus fp16
recurrent state), so:

|pool|VRAM|free|
|---|---|---|
|2,097,152|81.7 GiB (at 8,4)|-|
|2,555,904|95.2 GiB|-2.7 GiB|
|2,883,584 (at 8,4)|92.0 GiB|-5.8 GiB|

Shipped: 2,555,904 tokens -> **1M per stream, 2.44M total**, i.e. two full 1M streams or eight streams
averaging ~305k. Five to six million tokens of total window is not reachable on one 96 GiB card: 5M would
need ~90 GiB of KV on top of ~50 GiB of resident weights. Pages allocate on demand, so the pool is a
ceiling rather than a reservation.

---

## Operation

```bash
# start (GPU1 pinned inside launch.sh)
cd /path/to/tabbyAPI && ./launch.sh

# health and model identity
curl -s localhost:5000/v1/models | jq '.data[] | select(.id|startswith("Qwen"))'

# a completion
curl -s localhost:5000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8","prompt":"def add(a, b):\n    return","max_tokens":64,"temperature":0}'
```

Endpoint contract: OpenAI-compatible at `http://127.0.0.1:5000/v1`; model id
`Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8`; `max_model_len` 1,048,576; `/v1/models` lists every directory
under the model dir, only one of which is loaded.

Pitfalls learned the hard way:

- Start the server through `launch.sh`. Running `main.py` directly without `CUDA_VISIBLE_DEVICES=1`
  takes GPU0 (this host shares it with another engine).
- `gpu_split: [99]` is required; `[95]` raises `Insufficient VRAM in split for model and cache` once the
  drafter and recurrent rings are added.
- exllamav3 preallocates the KV pool at load; a pool that does not fit fails at boot, not at request time.
- Draft depth must stay <= `MAX_BSZN - 1` (7). `MAX_BSZN = 8` lives in
  `exllamav3/modules/block_sparse_mlp.py` and its C++ twin `libtorch/blocksparse_mlp.h`.
- The stack is **not bit-reproducible run to run**: identical config and prompt produced different
  continuations in 3 of 12 probes. Benchmark with medians.

---

## Credits

This repository is a deployment recipe only; it contains no model weights and no upstream source.

- **Qwen3.8-Flash-Next** — Alibaba Qwen team. Weights under the Qwen Community License 1.0; not
  redistributed here. <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- **exllamav3** — Turboderp / exllamav3 contributors, MIT. The EXL3 quantization format, the MoE and
  block-sparse kernels, YaRN/MRoPE handling, and the DFlash/DFlash2 draft-model support
  (`architecture/dflash2.py`) all come from this project. <https://github.com/turboderp-org/exllamav3>
- **TabbyAPI** — theroyallab and contributors, AGPL-3.0. OpenAI-compatible server, draft-model plumbing
  (`draft_mode`, `dynamic_draft`, `draft_confidence`), cache and generator wiring.
  <https://github.com/theroyallab/tabbyAPI>
- **DFlash 2** — Inco AI / the z-lab group. Apache-2.0 drafter
  `incoai/Qwen3.8-Flash-Next-DFlash2`; architecture "block-diffusion drafter that predicts a whole block of
  tokens in a single pass and keeps the top candidates at every position", lossless under greedy
  verification. <https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2> ·
  <https://github.com/z-lab/dflash>
- **flash-linear-attention** — Songlin Yang et al., MIT; GDN kernels used by the linear-attention layers.
- The EXL3 quantisation of the target checkpoint follows the standard exllamav3 converter
  (`3.18bpw_h6_ng8` = 3.18 bpw, 6-bit head/lm_head, n-gram table quantised at 8 bits).

Related harness in this repo: `docs/model-router-multi-endpoint.md` describes the multi-endpoint provider
change made to the local Model Router so that the DeepSeek and Qwen endpoints appear under one provider.

---

## Repository layout

```
deploy/    launch.sh, config.yml as deployed, YaRN model-config patch
bench/     the harnesses used for every number above
docs/      measurement log, restore runbook, router change
```

## License

This repository contains a deployment recipe, configuration, and benchmark harnesses — no model weights and
no upstream source code. Those files are released under the MIT license. The components credited above keep
their own licenses: exllamav3 (MIT), TabbyAPI (AGPL-3.0), the DFlash 2 drafter (Apache-2.0), and the
Qwen3.8-Flash-Next weights (Qwen Community License 1.0, not redistributed here).
