"""
Prompt-cache tiers: time to first token of a ~100k-token prompt when its KV pages are no longer in VRAM.

    # evicted: send prompt KEY, fill the GPU pool with distinct prompts, resend KEY
    BASE_URL=... API_KEY=... python bench/prompt_cache.py evict --key run1 [--fill 96 --fill-tokens 32768]
    # after a restart: resend KEY (same --key as before the restart)
    BASE_URL=... API_KEY=... python bench/prompt_cache.py send --key run1

The prompt is built from `--key` alone, so the same key gives the same prompt in every process. The fill sends
`--fill` prompts of `--fill-tokens` tokens each, all distinct; the default (96 x 32k = 3.1M tokens) is more than the
2,999,808-token GPU pool of deploy/config.yml. One JSON line per send: time to first token, prompt tokens and how many
of them the server reports as cached. With `memory.sysmem_kv_cache` and `memory.disk_cache_dir` set, the evicted
prompt comes back from the RAM tier and, after a restart, from the disk tier; without them it is prefilled again.
"""
import argparse
import hashlib
import json
import os

from common import SEED
from timing import stream_post


def prompt(key: str, tokens: int) -> str:
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"<!--{tag}-->\n" + SEED["prose"] * int(tokens * 4.43 / len(SEED["prose"]))


def send(what: str, key: str, tokens: int):
    r = stream_post({"prompt": prompt(key, tokens), "max_tokens": 1, "loop_detect_window": 0})
    print(json.dumps(dict(what = what, key = key, ptok = r["ptok"], cached = r["cached"], ttft_s = round(r["ttft_s"], 3),
                          load1 = os.getloadavg()[0])), flush = True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices = ("evict", "send"))
    ap.add_argument("--key", required = True)
    ap.add_argument("--tokens", type = int, default = 100_000)
    ap.add_argument("--fill", type = int, default = 96)
    ap.add_argument("--fill-tokens", type = int, default = 32768)
    a = ap.parse_args()
    if a.cmd == "send":
        send("resend", a.key, a.tokens)
        return
    send("cold", a.key, a.tokens)
    send("hot", a.key, a.tokens)
    for i in range(a.fill):
        stream_post({"prompt": prompt(f"{a.key}-fill-{i}", a.fill_tokens), "max_tokens": 1, "loop_detect_window": 0})
    print(json.dumps(dict(what = "fill", n = a.fill, tokens_each = a.fill_tokens)), flush = True)
    send("evicted", a.key, a.tokens)


if __name__ == "__main__":
    main()
