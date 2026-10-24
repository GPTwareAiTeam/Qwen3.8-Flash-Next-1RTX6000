#!/usr/bin/env bash
# Qwen3.8-Flash-Next EXL3 3.18bpw_h6_ng8 on one RTX PRO 6000 (96 GiB).
# The n-gram table (48.28 GiB) is streamed from the Gen5 copy of the checkpoint and is
# never resident: ngram_ram false, RSS stays ~4 GiB, rows are gathered per forward.
set -euo pipefail

VENV=/models/dsv41/venvs/tabby
APP=/models/dsv41/qwen38-src/tabbyAPI

# GPU1 (index 1) is the only free 96 GiB card; GPU0 runs the DSV4.1 vLLM engine.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export EXL3_NGRAM_STREAM=1        # table stays on disk (default, stated explicitly)
export EXL3_NGRAM_PREFETCH=1      # worker-thread row prefetch (1.6.4 knob; not PREFETCH2)

# Decode-path selectors, measured on this card (c1 176 -> 254 t/s at 3.18 bpw, MTP depth 7):
#   MTILE=0        one 16-row fused MoE launch; the 32/64-row tile instances only add
#                  launches, because decode puts 1-2 rows on each of 10 hot experts
#   SHARED_COOP=0  keep the three-launch shared-expert graph; the coop single launch is slower here
#   COOP_WIDE=1    wide coop tile for the routed experts
export EXL3_MOE_MTILE=0
export EXL3_MOE_SHARED_COOP=0
export EXL3_MOE_COOP_WIDE=1

cd "$APP"
exec "$VENV/bin/python" main.py
