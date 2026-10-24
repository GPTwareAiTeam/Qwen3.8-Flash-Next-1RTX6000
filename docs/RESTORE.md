# Restore this deployment

State as of 2026-10-23 (server stopped, GPU1 freed for another workload).
Everything below was verified working; the numbers are from the README.

## 1. What must exist on disk

|Path|Contents|
|---|---|
|`/models/dsv41/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8`|EXL3 `3.18bpw_h6_ng8` target, 99.17 GiB (48.28 GiB of that is the streamed engram table). `config.json.orig-262k` is the untouched 256k config; `config.json` carries the 1M YaRN patch in `deploy/model-config-yarn4.patch`.|
|`/models/dsv41/drafts/Qwen3.8-Flash-Next-DFlash2`|`incoai/Qwen3.8-Flash-Next-DFlash2` drafter, 1.2 GiB BF16 (`hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir Qwen3.8-Flash-Next-DFlash2`).|
|`/models/dsv41/venvs/tabby`|venv: torch 2.11.0+cu130, exllamav3 1.6.4+cu132.torch2.11.0, flash-linear-attention 0.6.0|
|`/models/dsv41/qwen38-src/tabbyAPI`|TabbyAPI @ `564f8fb`, with `launch.sh` and `config.yml` from `deploy/` copied in|

## 2. Restore

```bash
# 1. target config must have the 1M YaRN patch
cd /models/dsv41/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8
patch -p1 < /path/to/repo/deploy/model-config-yarn4.patch     # or copy config.json by hand

# 2. server files
cp /path/to/repo/deploy/launch.sh  /models/dsv41/qwen38-src/tabbyAPI/launch.sh
cp /path/to/repo/deploy/config.yml /models/dsv41/qwen38-src/tabbyAPI/config.yml

# 3. start (launch.sh pins CUDA_VISIBLE_DEVICES=1)
cd /models/dsv41/qwen38-src/tabbyAPI && ./launch.sh

# 4. verify
curl -s localhost:5000/v1/models | python3 -c "import json,sys;[print(m['id'],m.get('max_model_len')) for m in json.load(sys.stdin)['data'] if m['id'].startswith('Qwen')]"
# expect: Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 1048576
```

Boot takes ~12-21 s (load) + 2 s warmup. Expected footprint: GPU1 95.2 GiB used, ~2.0 GiB free.

## 3. The recipe in one paragraph

TabbyAPI + exllamav3 on one 96 GiB Blackwell card. Target at 3.18 bpw with the engram table left on
NVMe (`ngram_ram: false`, `EXL3_NGRAM_STREAM=1`, `EXL3_NGRAM_PREFETCH=1`), RSS ~4 GiB. Decode-path
selectors `EXL3_MOE_MTILE=0 EXL3_MOE_SHARED_COOP=0 EXL3_MOE_COOP_WIDE=1` (worth 207 -> 254 t/s).
Drafting is the external DFlash2 block-diffusion drafter (`draft_mode: model`, 7 drafts per round,
`dynamic_draft: true`, `draft_confidence: 0.30`) — not the built-in MTP component. Context 1M per stream
via the model card's static YaRN factor 4, KV `8,8`, pool `2,555,904` tokens (2.44M total, 16.66 KiB per
token, ~40.6 GiB). `gpu_split: [99]` is mandatory; anything lower fails at load.

## 4. Results to compare against after a restore

|measure|value|
|---|---|
|single stream, code, 512 tok|233.8 t/s|
|single stream, 2048 tok: code / math / chat / prose|254.6 / 264.4 / 125.6 / 108.9 t/s|
|concurrency 1 / 2 / 4 / 8 (512 tok)|233.8 / 243.3 / 352.2 / 467.7 t/s aggregate|
|prefill ~2k tokens|3,050-3,242 tok/s|
|prefill 40k / cold 535k|4,550-4,700 / 4,099 tok/s|
|1M context needle @535k|passes|

## 5. Things that will bite

- Starting `main.py` directly without `CUDA_VISIBLE_DEVICES=1` grabs GPU0, which belongs to another engine.
- The KV pool is preallocated at load; a pool that does not fit raises
  `Insufficient VRAM in split for model and cache` at boot, not at request time.
- Draft depth must stay <= 7 (9 verify positions leave the `MAX_BSZN = 8` fused path: 250 -> 165 t/s).
- Static YaRN costs ~10% on code-like continuations; drop the patch if 256k is enough.
- Do not switch V to 4 bits: it is cheaper (13.5 KiB/token, 2.75M pool) but changed the argmax on trivial
  arithmetic (`2+2=` -> 5).
- The stack is not bit-reproducible run to run; benchmark medians, not single runs.
