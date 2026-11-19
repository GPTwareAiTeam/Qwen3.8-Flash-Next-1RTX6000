# Candidate patches (not part of the V2 release)

These apply on top of `../patches/0001-0007` (`git am ../patches/*.patch *.patch`). Every switch defaults
to off, so with no new environment variables the engine behaves exactly like V2. Measurements are in
`docs/measurements.md`, section "Prefill V3 candidates".

|patch|change|switch|result|
|---|---|---|---|
|0008|Speculative decoding host readbacks: draft confidence read back once per round (`EXL3_SD_DEFER_DRAFT_CONF`), verify-window samplers launched up front with one synchronize when no per-position constraint applies (`EXL3_SD_BATCH_READBACK`), one fused greedy launch over the window (`EXL3_SD_WINDOW_BATCH`)|the three variables, `=1`|greedy identity 8/8 and KL 0 for each; c1/c8/code within noise (A,A,B,B)|
|0009|Diagnostics: per-tier MoE row counts on prefill (`EXL3_MOE_TIER_TRACE=1`), allocator statistics in the adaptive-prefill log, `EXL3_PREFILL_ADAPTIVE_EMPTY_CACHE`|diagnostics only|—|
|0010-0015|Growable KV pool (`EXL3_KV_GROW=1`): the quantized K/V and QSA plane pools are reserved as CUDA virtual address ranges and mapped in 2 MiB granules as the page table hands out pages (lowest unmapped page first, in steps of `EXL3_KV_GROW_STEP_PAGES`, default 256); the first `EXL3_KV_GROW_PREMAP_TOKENS` (262,144) are mapped at start; when the queue drains, pages are released from the top until `EXL3_KV_GROW_DRAIN_FREE_MIB` (10,240) are free. With `EXL3_PREFILL_ADAPTIVE=1` the prefill limiter counts only memory the device reports free and steps the chunk down planned → 8192 → 4096 → 2048 as the pool fills|`EXL3_KV_GROW=1` with `EXL3_PREFILL_ADAPTIVE=1`, `prefill_chunk_size: 16384` and `shared_chunk_size: 4096` (TabbyAPI patch v3)|A,A,B,B vs V2: cold 32k +8.0 %, 128k +8.7 %, 2k −1.6 %; c1 −0.3 %, c8 −0.9 %; greedy identity 8/8; long-context KL 0.0016; VRAM peak −23.5 GiB; 90-min fill/drain soak clean|
|0016|Hyper-connection tables: the tiled GatedResidual mix (above 8 rows) builds its int8 tables once at load instead of on every call|`EXL3_HC_TABLE_CACHE=1`|bit-identical; +1.3 GiB VRAM; see "Decode fixes" below|
|0017|Shared expert of wide verify windows (more than 8 rows) as one pass instead of 8-row graph pieces|`EXL3_MOE_SHARED_WIDE=1`|rounding-level (not bit-identical); see below|
|0018|Batched speculative accept: one readback per round for all users, the round's recurrent-state rewinds settled in one batch; under this switch the sampler "needs past tokens" flag comes from the steps that survive simplification (TabbyAPI's no-op penalty steps had set it on every stack, so 0008's batched readback never engaged)|`EXL3_SD_BATCH_ACCEPT=1`|top-1 and KL at the floor; see below|

`deploy/tabbyapi-v3.patch` replaces the V2 TabbyAPI patch and adds the `prefill_cooldown` and
`shared_chunk_size` model settings (prefill pacing while other requests decode).

The growable pool ports the growable KV cache of the exllamav3 fork behind
[QMesh](https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10) (MIT): `exllamav3/cache/native/vmm_pool.cpp`
and the `VmmRange` wrapper are taken from it, adapted to the quantized and QSA cache layers used here.

## Running the growable pool

This is the configuration our own server has run since 2026-11-16:

```bash
git am ../patches/*.patch *.patch                     # on exllamav3 v1.6.4
cd tabbyAPI && git apply /path/to/repo/deploy/tabbyapi-v3.patch
```

`deploy/config.yml`, `model:` section:

```yaml
  chunk_size: 8192
  prefill_chunk_size: 16384
  shared_chunk_size: 4096
```

and before `deploy/launch.sh` starts the server: `export EXL3_KV_GROW=1 EXL3_PREFILL_ADAPTIVE=1`. The CUDA virtual memory
module builds on first start (C++ only, a few seconds) and links `libcuda`.

## Decode fixes (0016-0018)

Our server runs all three since 2026-11-19. A,A,B,B on the same server build and config (DFlash2 n7 dynamic 0.45,
6,6 KV, growable pool), A = 0001-0015, B = + 0016-0018 with the three switches on, aggregate tok/s, 512 forced tokens
per stream, host load within 1.1 across the four arms:

|metric|A1 / A2|B1 / B2|change|
|---|---|---|---|
|1 user, code|278.2 / 279.6|284.1 / 283.6|+2.1 / +1.9 %|
|1 user, prose|133.0 / 133.2|134.5 / 129.2|+1.1 / -2.9 % (noise)|
|8 users, code|635.9 / 637.1|792.1 / 784.9|**+24.6 / +23.4 %**|
|8 users, prose|345.8 / 330.7|396.2 / 400.4|**+14.6 / +15.8 %**|
|peak VRAM|61,516 MiB|63,072 MiB|+1,556 MiB|

Gate (see `bench/README.md`): next-token top-1 agreement 25/25 on the short and long (8.7k / 35k / 157k) probes; KL
0 at long context, 0.0005-0.0019 mean at short context (max 0.022 on one short prompt), inside the run-to-run floor.
Drift: 0017 is not bit-identical (one GEMM over all rows tiles differently from the 8-row pieces: max |diff|
4e-5..2e-4 in the MoE output), so the combined set changes greedy output on 2 of the 8 identity prompts by one word
each ("closed-form expression" -> "closed-form formula"; "visible spectrum of light" -> "... of electromagnetic
radiation"). These are near-ties: the reference build itself flips the same two prompts between sessions. 0016 alone
and 0018 alone match the reference 8/8. Each fix alone, 8 users: 0016 +9..11 %, 0017 +7 %, 0018 +6..8 %.

Enable with `export EXL3_HC_TABLE_CACHE=1 EXL3_MOE_SHARED_WIDE=1 EXL3_SD_BATCH_ACCEPT=1` before `deploy/launch.sh`.
