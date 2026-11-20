# Engine: exllamav3 v1.6.4 + 21 patches

`patches/` is a `git format-patch` series for
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) tag `v1.6.4`
(`da68d57de61781e5b74ad68698ac4caf36408d7a`). The patched tree is the engine every V3 number in the README was
measured on.

```bash
git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3
git checkout v1.6.4
git am /path/to/this/repo/engine/patches/*.patch
pip install -r requirements.txt      # torch with CUDA 13 for SM120
# the C++/CUDA extension builds on first import (JIT); set TORCH_CUDA_ARCH_LIST=12.0 to build for SM120 only.
# The growable KV pool (0010) adds a small C++ module that links libcuda and builds on first start.
```

Every patch puts its change behind a switch that defaults to the stock behaviour, so the patched engine without
environment variables behaves like exllamav3 1.6.4 apart from the bit-identical changes (0001, 0003). `deploy/launch.sh`
sets the switches in the last column.

|patch|change|switch|`launch.sh`|
|---|---|---|---|
|0001|Cooperative MoE decode k-loop for 3-bit experts (steady-state loop without bounds checks, late decode from the ring, one MMA per tile). Bit-identical.|always on|on|
|0002|Dense decode GEMV for 1-48 rows, K 6/7 instances|`EXL3_DGEMV`|off: not bit-identical, no net gain on this model|
|0003|Fused MoE tier without the per-layer expert-count readback to the host. Bit-identical.|`EXL3_MOE_NO_READBACK`|on (default)|
|0004|Speculative-verify windows up to N rows through the coop MoE kernels; prompt prefill keeps the fused tier|`EXL3_MOE_COOP_ROWS`|64|
|0005|DFlash2 drafter layer stack captured as a CUDA graph|`EXL3_DFLASH_GRAPH`|1|
|0006|Adaptive prefill chunks: each chunk shrunk to fit free device memory|`EXL3_PREFILL_ADAPTIVE`|1|
|0007|Shared expert on a side stream, joined before the routed down projection (bit-identical); DGEMV workspace per device|`EXL3_MOE_SHARED_SIDE`|1|
|0008|Speculative-decoding host readbacks: draft confidence read once per round, verify-window samplers launched up front, fused greedy window|`EXL3_SD_DEFER_DRAFT_CONF`, `EXL3_SD_BATCH_READBACK`, `EXL3_SD_WINDOW_BATCH`|off: lossless, no gain on this stack|
|0009|Diagnostics: per-tier MoE row counts on prefill, allocator statistics in the adaptive-prefill log|`EXL3_MOE_TIER_TRACE`, `EXL3_PREFILL_ADAPTIVE_LOG`|off|
|0010-0015|Growable KV pool: the quantized K/V and sparse-attention plane pools are reserved as CUDA virtual address ranges and mapped in 2 MiB granules as pages are handed out (in steps of `EXL3_KV_GROW_STEP_PAGES`, default 256); the first `EXL3_KV_GROW_PREMAP_TOKENS` (262,144) are mapped at start; when the queue drains, pages are released from the top until `EXL3_KV_GROW_DRAIN_FREE_MIB` (10,240) are free. The adaptive limiter (0006) then counts only memory the device reports free.|`EXL3_KV_GROW`|1|
|0016|Hyper-connection tables: the tiled GatedResidual mix (above 8 rows) builds its int8 tables once at load instead of on every call. Bit-identical, +1.3 GiB VRAM.|`EXL3_HC_TABLE_CACHE`|1|
|0017|Shared expert of a wide verify window (more than 8 rows) as one pass instead of 8-row graph pieces. Rounding-level (one GEMM tiles differently: max abs diff 4e-5..2e-4 in the MoE output).|`EXL3_MOE_SHARED_WIDE`|1|
|0018|Batched speculative accept: one readback per round for all users, the round's recurrent-state rewinds settled in one batch|`EXL3_SD_BATCH_ACCEPT`|1|
|0019|Batched-cache graph slots for up to 12 jobs (attention, Gated DeltaNet, MLA); above the limit the decode falls back to eager kernels|always on (bound)|—|
|0020|MoE coop: counting-sort run table for wide verify windows (rotation launch 50-60 µs → 8-9 µs at 56-64 rows, bit-identical); down-stage reduction as its own launch (`EXL3_MOE_COOP_SPLIT_B`, no end-to-end gain)|`EXL3_MOE_COOP_FAST_RUNS`|1 (`SPLIT_B` off)|
|0021|Generator: total verify-row cap (`EXL3_DRAFT_MAX_ROWS`: jobs × (drafts + 1) ≤ N, so 9-12 jobs stay on the 64-row paths); drafter/accept overlap (`EXL3_SD_DRAFT_OVERLAP`: the drafter's K/V update and next block are launched before the host accept loop and rolled back when the round ends differently); hybrid (suffix-match) drafting only with one active job (`EXL3_HYBRID_SOLO`)|the three variables|64, 1, 1|

Patches 0001, 0002, 0003, 0004, 0006, 0008 and the growable pool (0010, 0013) port or adapt work from the exllamav3
fork behind [QMesh R5](https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10) (MIT), as do the graph segments of
0005 and the tier-trace switch of 0009; `exllamav3/cache/native/vmm_pool.cpp` is taken from it unchanged. Per-item
detail (copied, modified or idea only) is in the README's Credits. The patches are MIT, like exllamav3.

## Numerics

Bit-identical to stock: 0001, 0003, 0007, 0016, 0020. Lossless (accepted tokens are the target's own samples, the
drafter only proposes): 0005, 0018, 0021. Rounding-level: 0017 (one GEMM over all rows instead of 8-row pieces) and
the growable pool with adaptive chunks (prefill chunk boundaries move with free memory). Together they change greedy
output at near-tie tokens; next-token top-1 and KL stay at the run-to-run noise floor. See the README, "Quality".
