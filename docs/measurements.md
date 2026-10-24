# Measurement log — 2026-10-23

Chronological record of what was tried on the RTX PRO 6000 (GPU1) serving
`Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8`. Kept because several early conclusions were later corrected;
the corrections are part of the data.

## 1. Kernel-shape sweep (MTP drafting, 256 greedy tokens / 8x128 batch)

|variant|single stream|batch-8 aggregate|
|---|---|---|
|base|175.7|362.3|
|`HGEMM_F16ACC=1`|178.4|361.2|
|`MOE_COOP_WIDE=1`|154.1|363.7|
|`MOE_COOP_KSPLIT=2`|176.1|361.2|
|`MOE_POOL_PRIME=1`|175.3|360.3|
|`MOE_FUSED_DET=0`|201.4|310.9|
|**`MOE_MTILE=0`**|**207.3**|**453.5**|
|`QKV_SLICE=0`|166.0|360.1|
|`MTILE=0 + FUSED_DET=0`|149.3|307.0|
|`MTILE=0 + F16ACC`|205.2|454.4|
|`MTILE=0 + ROWS=64`|205.7|461.7|
|`MTILE=0 + MGEMM_K_THRESHOLD=8`|204.8|453.6|
|`MTILE=0 + QT_OPTIMIZED=1`, depth 7|241.2|493.2|
|`MTILE=0 + GEMV=0`, depth 7|242.9|493.6|
|`MTILE=0 + BC_ATTN=0`, depth 6|116.5|323.2|

Depth sweep at MTILE=0: 3 -> 188.8, 4 -> 207.3, 5 -> 218.3, 6 -> 236.4, 7 -> 253.3-254.6,
8 -> 165.0, 10 -> 111.6, 12 -> 103.5, 16 -> 94.1. The cliff at 8 is `MAX_BSZN = 8`: nine verify
positions leave the fused path.

## 2. A correction: the "+BOS" measurement artifact

An in-process harness that passed `add_bos=True` reported 253-254 t/s and appeared to beat the server by
12-15%. The model's `config.json` has `bos_token_id = 248044` (the same id as one of its EOS tokens) and
`generation_config.json` does not set `add_bos_token`, so TabbyAPI encodes without it: 85 prompt tokens
instead of 86. That single token changed the continuation into an echo of the prompt and raised MTP draft
acceptance from 4.6 to 7.9 tokens/round.

Matched encoding (no BOS), 1024 tokens, same prompt:

|path|t/s|
|---|---|
|in-process, 4 passes|150.8 / 192.5 / 184.8 / 197.2|
|server `/v1/completions`|214.2 / 219.6 / 223.6|
|server with `add_bos_token: true`|279-355, but `finish_reason=stop` at exactly 800 tokens (loop detector)|

Conclusion: the server is *faster* than the naive harness on identical encoding; there is no per-token
server overhead worth chasing. All published numbers use the server default (no BOS).

## 3. Drafter change: MTP -> DFlash2

`incoai/Qwen3.8-Flash-Next-DFlash2` (603M params, BF16, `block_size 8`, taps `[5,15,25,35,45]`,
hidden 2560, vocab 248320) loads as a normal draft model (`draft_mode: model`) because exllamav3 1.6.4
ships `architecture/dflash2.py` plus `dflash2_selector_walk` / `dflash2_dynconv` / `dflash2_topk` kernels.

Confidence sweep at `draft_num_tokens: 7` (dyn):

|conf|code|live|prose|chat|
|---|---|---|---|---|
|0.20|235.1|220.8-225.0|134.0-135.3|166.8-170.9|
|0.25|240.1|228.9-229.5|145.3-148.8|174.4-174.8|
|**0.30**|**239.4**|**228.6-229.0**|**149.1-150.4**|**172.6-174.2**|
|0.40|236.1|209.7-222.1|149.3-149.5|173.6-184.6|
|0.55|231.6|223.6-224.4|134.4-152.8|173.4-177.5|
|fixed 7 (dyn off)|167.9|231.5-232.2|144.3-146.0|173.2-173.7|

## 4. Context extension experiments

YaRN factor 4 (1M) versus factor 2 (512k) versus the base 256k, same 512-token bench:

|arm|code|live|prose|mean|
|---|---|---|---|---|
|base 262k, no YaRN|218.9|208.1|151.9|193.0|
|YaRN f=2, 524k|251.9|195.6|149.6|199.0|
|YaRN f=4, 1M|146.9-183.7|207.4-209.4|152.0-155.8|169.4-189.9|

Shipped: factor 4 at 1M, per operator decision.

## 5. KV cache quantisation A/B

24-item exact-answer probe (`bench/accuracy.py`), greedy:

- `8,8` (2.62M pool): 22/24; `2+2=` -> "4" on every run.
- `8,4` (2.75M pool): 20/24; `2+2=` -> "5" on every run.

Other differences between the arms were punctuation-level, and the stack is not bit-reproducible
(3 of 12 identical-config repeats differed), so only the repeated `2+2=` shift is treated as evidence.
Decision: keep `8,8`; the extra 0.3M tokens of 4-bit-V budget does not reach the 5-6M target anyway.

## 6. Final verification

- 1M context live: `max_model_len 1048576`.
- Needle at 535,785 tokens: planted string returned, 130.7 s cold prefill (4,099 tok/s).
- Engram streaming: idle `read_bytes` growth 0 KiB over 6 s.
- VRAM after the matrix runs: 96,070 MiB on GPU1 of 97,887, GPU0 untouched.
