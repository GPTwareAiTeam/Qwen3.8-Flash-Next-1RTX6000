# Candidate patches (not part of the V2 release)

These apply on top of `../patches/0001-0007` (`git am ../patches/*.patch *.patch`). Every switch defaults
to off, so with no new environment variables the engine behaves exactly like V2. Measurements are in
`docs/measurements.md`, section "Prefill V3 candidates".

|patch|change|switch|result|
|---|---|---|---|
|0008|Speculative decoding host readbacks: draft confidence read back once per round (`EXL3_SD_DEFER_DRAFT_CONF`), verify-window samplers launched up front with one synchronize when no per-position constraint applies (`EXL3_SD_BATCH_READBACK`), one fused greedy launch over the window (`EXL3_SD_WINDOW_BATCH`)|the three variables, `=1`|greedy identity 8/8 and KL 0 for each; c1/c8/code within noise (A,A,B,B)|
|0009|Diagnostics: per-tier MoE row counts on prefill (`EXL3_MOE_TIER_TRACE=1`), allocator statistics in the adaptive-prefill log, `EXL3_PREFILL_ADAPTIVE_EMPTY_CACHE`|diagnostics only|—|
|0010-0015|Growable KV pool (`EXL3_KV_GROW=1`): the quantized K/V and QSA plane pools are reserved as CUDA virtual address ranges and mapped in 2 MiB granules as the page table hands out pages (lowest unmapped page first, in steps of `EXL3_KV_GROW_STEP_PAGES`, default 256); the first `EXL3_KV_GROW_PREMAP_TOKENS` (262,144) are mapped at start; when the queue drains, pages are released from the top until `EXL3_KV_GROW_DRAIN_FREE_MIB` (10,240) are free. With `EXL3_PREFILL_ADAPTIVE=1` the prefill limiter counts only memory the device reports free and steps the chunk down planned → 8192 → 4096 → 2048 as the pool fills|`EXL3_KV_GROW=1` with `EXL3_PREFILL_ADAPTIVE=1`, `prefill_chunk_size: 16384` and `shared_chunk_size: 4096` (TabbyAPI patch v3)|A,A,B,B vs V2: cold 32k +8.0 %, 128k +8.7 %, 2k −1.6 %; c1 −0.3 %, c8 −0.9 %; greedy identity 8/8; long-context KL 0.0016; VRAM peak −23.5 GiB; 90-min fill/drain soak clean|

`deploy/tabbyapi-prefill-chunk-size.patch.v3` replaces the V2 TabbyAPI patch and adds the `prefill_cooldown` and
`shared_chunk_size` model settings (prefill pacing while other requests decode).

The growable pool ports the growable KV cache of the exllamav3 fork behind
[QMesh](https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10) (MIT): `exllamav3/cache/native/vmm_pool.cpp`
and the `VmmRange` wrapper are taken from it, adapted to the quantized and QSA cache layers used here.

## Running the growable pool

This is the configuration our own server has run since 2026-11-16:

```bash
git am ../patches/*.patch *.patch                     # on exllamav3 v1.6.4
cd tabbyAPI && git apply /path/to/repo/deploy/tabbyapi-prefill-chunk-size.patch.v3
```

`deploy/config.yml`, `model:` section:

```yaml
  chunk_size: 8192
  prefill_chunk_size: 16384
  shared_chunk_size: 4096
```

and before `deploy/launch.sh` starts the server: `export EXL3_KV_GROW=1 EXL3_PREFILL_ADAPTIVE=1`. The CUDA virtual memory
module builds on first start (C++ only, a few seconds) and links `libcuda`.
