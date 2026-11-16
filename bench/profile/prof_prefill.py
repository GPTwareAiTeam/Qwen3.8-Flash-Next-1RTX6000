"""
Cold-prefill module split (the "Where cold prefill time goes" table in docs/measurements.md).

Loads the model in-process with the V2 serving settings (6,6 KV, 8192-token chunks, batch 8, DFlash2 attached unless
--no-draft), runs one warm-up prefill, then one cold ~32k prompt bracketed by cudaProfilerStart/Stop, with NVTX
ranges per module: gdn (Gated DeltaNet), qsa_attn (full attention), qsa_indexer (sparse-attention indexer), moe
(routed experts), moe_shared (shared expert), embed (embeddings + engram), hc (gated residual), lm_head,
drafter_fwd / draft. Instrumentation only: the NVTX wrappers add host time, so its wall time is not a speed number.

    # with the patched exllamav3 first on PYTHONPATH and the same EXL3_* switches deploy/launch.sh sets
    CUDA_VISIBLE_DEVICES=0 nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi -o pf32k \
        python bench/profile/prof_prefill.py --model /path/to/models/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 \
            --draft /path/to/drafts/Qwen3.8-Flash-Next-DFlash2 [--no-draft] [--ctx 32768]
    nsys export --type sqlite -o pf32k.sqlite pf32k.nsys-rep
    python bench/profile/analyze.py pf32k.sqlite > pf32k_table.md

Run it with the server stopped: it needs the whole GPU.
"""
import argparse
import functools
import json
import time

import torch
import torch.cuda.nvtx as nvtx

from exllamav3 import Model, Config, Cache, CacheLayer_quant, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3 import modules as M

PROSE = ("The estuary narrows as the tide turns, and the sandbars that were dry at noon disappear under a "
         "sheet of moving water. Gulls follow the current line, and the fisherman reads it the way a driver reads "
         "a road: the colour change marks the channel, the smooth patch marks the shoal. ")


def wrap(cls, meth, label):
    f = getattr(cls, meth)

    @functools.wraps(f)
    def g(*a, **k):
        nvtx.range_push(label)
        try:
            return f(*a, **k)
        finally:
            nvtx.range_pop()
    setattr(cls, meth, g)


def wrap_instance(obj, label):
    f = obj.forward

    def g(*a, _f = f, **k):
        nvtx.range_push(label)
        try:
            return _f(*a, **k)
        finally:
            nvtx.range_pop()
    obj.forward = g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--draft", default = None)
    ap.add_argument("--no-draft", action = "store_true")
    ap.add_argument("--ctx", type = int, default = 32768)
    ap.add_argument("--chunk", type = int, default = 8192)
    ap.add_argument("--cache", type = int, default = 131072)
    a = ap.parse_args()

    cfg = Config.from_directory(a.model)
    model = Model.from_config(cfg)
    tok = Tokenizer.from_config(cfg)
    kw = dict(max_batch_size = 8, max_history = 7, recurrent_dtype = torch.half)
    cache = Cache(model, max_num_tokens = a.cache, layer_type = CacheLayer_quant, k_bits = 6, v_bits = 6, **kw)
    draft = dcache = None
    if not a.no_draft:
        assert a.draft, "--draft is required unless --no-draft"
        draft = Model.from_config(Config.from_directory(a.draft))
        dcache = Cache(draft, max_num_tokens = a.cache, **kw)
        draft.load(max_chunk_size = a.chunk, max_batch_size = 8)
    model.load(max_chunk_size = a.chunk, max_batch_size = 8)
    model.warmup(cache = cache, max_chunk_size = a.chunk, max_batch_size = 8, max_q_len = 8)
    gen = Generator(model = model, cache = cache, draft_model = draft, draft_cache = dcache, tokenizer = tok,
                    max_batch_size = 8, max_chunk_size = a.chunk,
                    **(dict(num_draft_tokens = 7, dynamic_draft_tokens = True, draft_confidence = 0.45) if draft else {}))

    from exllamav3.modules import qsa_indexer as qsa_mod
    wrap(M.GatedDeltaNet, "forward", "gdn")
    wrap(M.Attention, "forward", "qsa_attn")
    for m in ("project", "pool_keys", "update_planes", "sparse_attend"):
        if hasattr(qsa_mod.QSAIndexer, m):
            wrap(qsa_mod.QSAIndexer, m, "qsa_indexer")
    wrap(M.BlockSparseMLP, "forward", "moe")
    for mod in model.modules:
        for sub in [mod] + list(getattr(mod, "modules", []) or []):
            se = getattr(sub, "shared_experts", None)
            if se is not None and not getattr(se, "_nvtx", False):
                wrap_instance(se, "moe_shared")
                se._nvtx = True
            if getattr(sub, "key", None) == "lm_head":
                wrap_instance(sub, "lm_head")
    wrap(M.Embedding, "forward", "embed")
    wrap(M.NGramEmbedding, "forward", "embed")
    wrap(M.GatedResidual, "forward", "hc")
    if draft is not None:
        wrap(type(draft), "forward", "drafter_fwd")
        wrap(Generator, "iterate_draftmodel_dflash_gen", "draft")

    def one(tag):
        ids = tok.encode(f"<!--{tag}-{time.time_ns()}-->\n" + PROSE * int(a.ctx * 4.43 / len(PROSE)), add_bos = False)
        gen.enqueue(Job(input_ids = ids, max_new_tokens = 1, sampler = ArgmaxSampler(), stop_conditions = None))
        t = time.perf_counter()
        while gen.num_remaining_jobs():
            gen.iterate()
        torch.cuda.synchronize()
        return ids.shape[-1], time.perf_counter() - t

    n, w = one("warm")
    print(f"warm prefill {n} tokens {w:.2f} s", flush = True)
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push("round/prefill")
    n, w = one("cold")
    nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps(dict(draft = draft is not None, prompt_tokens = n, wall_s = round(w, 2))), flush = True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
