# Engine: exllamav3 v1.6.4 + 7 patches

`patches/` is a `git format-patch` series for
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) tag `v1.6.4`
(`da68d57de61781e5b74ad68698ac4caf36408d7a`). The patched tree is the engine every V2/V2.1 number in the README
was measured on.

```bash
git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3
git checkout v1.6.4
git am /path/to/this/repo/engine/patches/*.patch
pip install -r requirements.txt      # torch with CUDA 13 for SM120
# the C++/CUDA extension builds on first import (JIT); set TORCH_CUDA_ARCH_LIST=12.0 to build for SM120 only
```

`deploy/launch.sh` puts this tree first on `PYTHONPATH` and sets the switches below.

|patch|change|switch|default in this repo|
|---|---|---|---|
|0001|Cooperative MoE decode k-loop for 3-bit experts (steady-state loop without bounds checks, late decode from the ring, one MMA per tile). Bit-identical.|always on|on|
|0002|Dense decode GEMV for 1-48 rows, K 6/7 instances|`EXL3_DGEMV`|off: not bit-identical, no net gain on this model|
|0003|Fused MoE tier without the per-layer expert-count readback to the host. Bit-identical.|`EXL3_MOE_NO_READBACK`|on|
|0004|Speculative-verify windows up to N rows through the coop MoE kernels; prompt prefill keeps the fused tier|`EXL3_MOE_COOP_ROWS`|64|
|0005|DFlash2 drafter layer stack captured as a CUDA graph|`EXL3_DFLASH_GRAPH`|1|
|0006|Adaptive prefill chunks: each chunk shrunk to fit free device memory|`EXL3_PREFILL_ADAPTIVE`|off|
|0007|Shared expert on a side stream, joined before the routed down projection (bit-identical); DGEMV workspace per device|`EXL3_MOE_SHARED_SIDE`|1|

Patches 0001, 0002, 0003 and 0006 port work from the exllamav3 1.6.5 fork behind
[QMesh R5](https://github.com/GPTwareAiTeam/QMesh-1xRTXPro6000-1xGB10) (MIT), adapted to Qwen3.8-Flash-Next.
The patches are MIT, like exllamav3.
