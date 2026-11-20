# V3 provenance

The README's "Credits" cover v1, V2 and V3, with the public recipes and benchmarks we compared against listed
separately as reference points. This file details the code new in V3: per item, where it comes from and how it was
used. Items not listed have no outside source.

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

No code from TensorFold, Mia's AI Lab, bertholomus or z-lab is in the patches (the first three are reference points
in the README, z-lab's DFlash is credited for the drafter model). Hybrid (suffix-match) drafting, which 0021 limits to
a single active request, is upstream exllamav3. DFlash2 drafter code is upstream exllamav3 as well; 0005 only wraps
its layer stack in a CUDA graph. `deploy/tabbyapi-v3.patch` and the new bench scripts are this repository's own work.
