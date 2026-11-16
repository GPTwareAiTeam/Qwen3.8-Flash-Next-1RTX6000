"""Shared fixtures for the bench scripts: server access (BASE_URL, API_KEY, MODEL) and the fixed prompt sets.

Every A/B in the README used these prompts unchanged. Long prompts are built by repeating fixed seed texts, so
anyone can regenerate them byte for byte; `nonce()` makes a prompt cold (no shared prefix with earlier requests).
"""
import json
import os
import random
import string
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:5000")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8")
HDR = {"Content-Type": "application/json",
       **({"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {})}


def post(body: dict, timeout: float = 3600):
    """POST /v1/completions (temperature 0 unless the body sets it); returns (wall seconds, response json)"""
    body = {"model": MODEL, "temperature": 0, **body}
    r = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), HDR)
    t = time.time()
    d = json.load(urllib.request.urlopen(r, timeout = timeout))
    return time.time() - t, d


def forced(prompt: str, n: int) -> dict:
    """Request body for exactly n generated tokens (no early stop, no loop detector)"""
    return {"prompt": prompt, "max_tokens": n, "min_tokens": n, "loop_detect_window": 0}


def nonce() -> str:
    return "".join(random.choices(string.hexdigits.lower(), k = 10))


CODE_P = "Write a Python implementation of a binary heap with push, pop, and heapify, plus a test harness."

NATURAL = {
    "prose": ("Write a long, detailed essay about how a coastal town's economy changed after the fishing fleet "
              "shrank: who stayed, who left, what the harbour is used for now, and how the town's self-image adapted. "
              "Use concrete detail throughout and do not summarise."),
    "code": ("Write a complete Python module implementing an LRU cache with a doubly linked list and a dict, "
             "including type hints, docstrings, a thread lock, and a pytest test file that covers eviction order, "
             "capacity one, and concurrent access. Output code only."),
    "chat": ("You are a support agent. A customer says their laptop battery drains from full to 20% in ninety minutes "
             "and they have already tried a restart. Write the full troubleshooting reply: diagnostics to run, what each "
             "result means, when to replace the battery, and what to do if the problem persists."),
    "math": ("Derive the closed form for the sum of the first n squares, prove it by induction, then use it to "
             "compute the sum from 1 to 2500 and verify the result with a second method. Show every step."),
}

SEED = {
    "prose": ("The estuary narrows as the tide turns, and the sandbars that were dry at noon disappear under a "
              "sheet of moving water. Gulls follow the current line, and the fisherman reads it the way a driver reads "
              "a road: the colour change marks the channel, the smooth patch marks the shoal. "),
    "code": ("def merge(a, b):\n    out = []\n    while a and b:\n        if a[0] <= b[0]:\n            out.append(a.pop(0))\n"
             "        else:\n            out.append(b.pop(0))\n    return out + a + b\n\n# usage:\n"),
    "chat": ("User: I need to move about forty gigabytes of photos from an old external drive to a NAS, and I "
             "keep losing track of which folders are already copied.\nAssistant: "),
    "math": ("Solve for x: 4x^2 - 9x + 2 = 0. Step 1: identify a, b and c. Step 2: compute the discriminant. "
             "Step 3: substitute into the quadratic formula and simplify both roots. "),
}

# The 8 greedy identity prompts (128 tokens each)
IDENTITY = [NATURAL["prose"], NATURAL["code"], NATURAL["chat"], NATURAL["math"], CODE_P,
            "Explain why the sky is blue, starting from Rayleigh scattering and going through the perception side.",
            "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n",
            "List the planets of the solar system in order and give one fact about each."]

# Long-context endings: mid-sentence, so the next token is not EOS (TabbyAPI returns no logprobs for an EOS token)
LONG_ENDINGS = ("\nThe fisherman reads the current line the way a driver reads a",
                "\nFor 4x^2 - 9x + 2 = 0 the discriminant b^2 - 4ac equals 81 - 32 =",
                "\ndef merge_sorted(a, b):\n    out = []\n    while a and")
LONG_TARGETS = (8192, 32768, 131072)   # ~8.7k / ~35k / ~157k tokens with this tokenizer


def short_kl_prompts() -> list[str]:
    """16 short prompts: the identity prompts NATURAL x4 + IDENTITY[4:] x4, each cut at half and at full length"""
    base = [NATURAL[k] for k in NATURAL] + IDENTITY[4:]
    return [p[: max(8, int(len(p) * f))] for p in base for f in (0.5, 1.0)]


def long_kl_prompts() -> list[str]:
    """9 long prompts: the four seed texts (x40 each) repeated to ~target tokens, plus each of the three endings"""
    text = "".join(SEED[t] * 40 for t in SEED)
    out = []
    for target in LONG_TARGETS:
        body = text * max(1, int(target * 4.43 / len(text)))
        out += [body + q for q in LONG_ENDINGS]
    return out


def cold_prose(tokens: int) -> str:
    """A cold prose prompt of about `tokens` tokens (fresh nonce header)"""
    return f"<!--{nonce()}-->\n" + SEED["prose"] * int(tokens * 4.43 / len(SEED["prose"]))
