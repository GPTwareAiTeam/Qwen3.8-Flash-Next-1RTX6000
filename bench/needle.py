"""
Long-context retrieval: a passphrase planted halfway into a ~500k-token prose prompt, then asked for at the end.

    BASE_URL=http://127.0.0.1:5000 API_KEY=... python bench/needle.py [--tokens 500000]

Prints one JSON line: prompt tokens, how many were served from the prompt cache (0: the prompt starts with a fresh
nonce), time to first token (prefill), prefill tokens/s and whether the passphrase is in the answer. The model opens a <think> block before answering, so up to 400 tokens are generated (greedy).
Needs max_seq_len above the prompt size (524,032 in deploy/config.yml; use --tokens 500000 there).
"""
import argparse
import json

from common import SEED, nonce
from timing import stream_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type = int, default = 500000)
    a = ap.parse_args()
    key = f"NEEDLE-{nonce()}"
    reps = int(a.tokens * 4.43 / len(SEED["prose"]))
    half = reps // 2
    # a fresh nonce first, so no prefix is shared with earlier prompts (cold prefill, no prompt-cache hit)
    p = (f"<!--{nonce()}-->\n" + SEED["prose"] * half + f"\nThe secret passphrase is {key}. Remember it.\n" + SEED["prose"] * (reps - half)
         + "\nQuestion: what is the secret passphrase? Answer: The secret passphrase is")
    r = stream_post({"prompt": p, "max_tokens": 400, "loop_detect_window": 0})
    print(json.dumps(dict(ptok = r["ptok"], cached = r["cached"], ttft_s = round(r["ttft_s"], 1), prefill_tps = round(r["ptok"] / r["ttft_s"]),
                          found = key in r["text"], tail = r["text"][-120:])), flush = True)


if __name__ == "__main__":
    main()
