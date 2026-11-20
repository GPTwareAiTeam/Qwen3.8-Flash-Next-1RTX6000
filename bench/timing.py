"""
Streamed completion with first-token timing, shared by the decode and prefill harnesses.

`stream_post(body)` sends POST /v1/completions with `stream: true` and `stream_options.include_usage`, and returns

    ttft_s       seconds from sending the request to the first chunk that carries generated text
    wall_s       seconds from sending the request to the end of the stream
    decode_tps   generated tokens after the first, divided by (wall_s - ttft_s): decode speed from the first token
    e2e_tps      generated tokens divided by wall_s: end-to-end speed, prefill included
    usage        the final usage chunk (prompt/completion tokens, cached tokens, accepted/rejected drafts)
    text         the generated text

The first token arrives with the prefill, so `decode_tps` excludes prefill and queueing and `e2e_tps` includes
them. With speculative decoding several tokens can arrive in one chunk; token counts come from the usage chunk,
not from counting chunks.
"""
import json
import time
import urllib.request

from common import BASE, HDR, MODEL


def stream_post(body: dict, greedy: bool = True, timeout: float = 7200) -> dict:
    """greedy=True sends temperature 0; greedy=False sends no sampling fields, so the server's preset applies"""
    body = {"model": MODEL, **({"temperature": 0} if greedy else {}), **body,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), HDR)
    t0 = time.monotonic()
    first = None
    usage = {}
    text = []
    with urllib.request.urlopen(req, timeout = timeout) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices") or []:
                if ch.get("text"):
                    if first is None:
                        first = time.monotonic()
                    text.append(ch["text"])
    end = time.monotonic()
    n = usage.get("completion_tokens", 0)
    ttft = (first or end) - t0
    det = usage.get("completion_tokens_details") or {}
    return dict(
        ttft_s = ttft,
        wall_s = end - t0,
        decode_tps = (n - 1) / (end - first) if first and n > 1 and end > first else None,
        e2e_tps = n / (end - t0) if n else None,
        ptok = usage.get("prompt_tokens", 0),
        ctok = n,
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        acc = det.get("accepted_prediction_tokens") or 0,
        rej = det.get("rejected_prediction_tokens") or 0,
        text = "".join(text),
    )
