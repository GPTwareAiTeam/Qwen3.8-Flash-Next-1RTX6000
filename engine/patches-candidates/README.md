# Candidate patches (not part of the V2 release)

These apply on top of `../patches/0001-0007` (`git am ../patches/*.patch *.patch`). Every switch defaults
to off, so with no new environment variables the engine behaves exactly like V2. Measurements are in
`docs/measurements.md`, section "Prefill V3 candidates".

|patch|change|switch|result|
|---|---|---|---|
|0008|Speculative decoding host readbacks: draft confidence read back once per round (`EXL3_SD_DEFER_DRAFT_CONF`), verify-window samplers launched up front with one synchronize when no per-position constraint applies (`EXL3_SD_BATCH_READBACK`), one fused greedy launch over the window (`EXL3_SD_WINDOW_BATCH`)|the three variables, `=1`|greedy identity 8/8 and KL 0 for each; c1/c8/code within noise (A,A,B,B)|
|0009|Diagnostics: per-tier MoE row counts on prefill (`EXL3_MOE_TIER_TRACE=1`), allocator statistics in the adaptive-prefill log, `EXL3_PREFILL_ADAPTIVE_EMPTY_CACHE`|diagnostics only|—|
|0010-0012|Growable KV pool (`EXL3_KV_GROW=1`): the quantized K/V and QSA plane pools are reserved as CUDA virtual address ranges and mapped in 2 MiB granules as the page table hands out pages (lowest unmapped page first, in steps of `EXL3_KV_GROW_STEP_PAGES`, default 256). With `EXL3_PREFILL_ADAPTIVE=1` the prefill limiter counts only memory the device reports free and steps the chunk down planned → 8192 → 4096 → 2048 as the pool fills|`EXL3_KV_GROW=1` with `EXL3_PREFILL_ADAPTIVE=1` and `prefill_chunk_size: 16384` (TabbyAPI patch v3)|16384-token chunks on an empty pool, shrinking to 8192 and then 4096 as the pool fills to 2.5M tokens, no out-of-memory; cold 32k +5.0 %, 128k +6.8 %; greedy identity 8/8; long-context KL 0.0016; c1/c8 unchanged (A,A,B,B: 0.0 % / −1.0 %)|

`deploy/tabbyapi-prefill-chunk-size.patch.v3` replaces the V2 TabbyAPI patch and adds the `prefill_cooldown` and
`shared_chunk_size` model settings (prefill pacing while other requests decode).

The growable pool ports the growable KV cache of the exllamav3 fork behind
[QMesh](https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10) (MIT): `exllamav3/cache/native/vmm_pool.cpp`
and the `VmmRange` wrapper are taken from it, adapted to the quantized and QSA cache layers used here.
