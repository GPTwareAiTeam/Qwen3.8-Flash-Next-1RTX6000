"""
Cold long-prompt probe: time to first token for fresh long prompts (used to measure the engram-lock effect).

Sends N fresh long prompts one at a time to the running TabbyAPI (OAI completions, max_tokens=1, temperature 0). Every
prompt starts with a unique nonce, so no prefix pages are shared with earlier requests (cold prefill every time).
Reports per request: wall time (time to the first and only generated token, i.e. the prefill plus one step), the
prompt token count from usage, and the server's VmRSS and VmLck (locked bytes) from /proc.

    python bench/coldttft.py --label lock --tokens 32768 --n 5 --pid <server pid>

Output: one JSON line per request and a summary line with the slowest wall time (the stall metric) and the median.
Each prompt gets a uuid4 nonce, so reruns never hit the prefix cache.
"""
import argparse
import json
import statistics
import time
import urllib.request
import uuid

import os
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:5000")
FILLER = ("The estuary narrows as the tide turns, and the sandbars that were dry at noon disappear under water while "
          "gulls circle the pilings and a fisherman mends a net by the slipway. ")


def proc_mem(pid):
    out = {}
    if not pid:
        return out
    for line in open(f"/proc/{pid}/status"):
        if line.startswith(("VmRSS:", "VmLck:")):
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


def prompt_for(tokens, nonce):
    # ~4 characters per token in this filler; the server's usage.prompt_tokens is the authoritative count
    body = FILLER * (tokens * 4 // len(FILLER) + 1)
    return f"[{nonce}] Summarize the following in one word.\n{body[: tokens * 4]}"


def post(prompt, model):
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {})},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return time.monotonic() - t0, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for i in range(a.n):
        p = prompt_for(a.tokens, uuid.uuid4().hex)
        wall, d = post(p, a.model)
        mem = proc_mem(a.pid)
        row = dict(label=a.label, i=i, tokens=a.tokens, ptok=d["usage"]["prompt_tokens"], wall_s=round(wall, 3),
                   rss_gib=round(mem.get("VmRSS", 0) / 2**30, 2), lck_gib=round(mem.get("VmLck", 0) / 2**30, 2))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
    walls = [r["wall_s"] for r in rows]
    print(json.dumps(dict(label=a.label, summary=True, slowest_s=max(walls), median_s=round(statistics.median(walls), 3),
                          rss_gib_last=rows[-1]["rss_gib"], lck_gib_last=rows[-1]["lck_gib"])), flush=True)


if __name__ == "__main__":
    main()
