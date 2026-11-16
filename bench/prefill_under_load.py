"""
Prefill under decode load: time to first token for a cold long prompt while other users are decoding, and how much
those users slow down meanwhile (the "Prefill while others decode" and admission tables in the README).

    BASE_URL=http://127.0.0.1:5000 API_KEY=... python bench/prefill_under_load.py --users 4 --tokens 131072 --reps 3

Per repetition: `users` streams decode forced 16,384-token answers (the four NATURAL prompts in turn); after 20 s
one cold prompt of ~`tokens` tokens arrives (max_tokens 1). Reported: its wall time (TTFT) and each decoder's
streamed tokens/s in the 15 s before it arrived (base) and during its prefill; then the decoders are stopped.
With users >= max_batch_size the prompt also waits for a free slot.
"""
import argparse
import json
import statistics as st
import threading
import time
import urllib.request

from common import BASE, HDR, MODEL, NATURAL, cold_prose, forced, post


def stream(body: dict, marks: list, stop: threading.Event):
    body = {"model": MODEL, "temperature": 0, "stream": True, "logprobs": 1, **body}
    r = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), HDR)
    n = 0
    with urllib.request.urlopen(r, timeout = 3600) as f:
        for line in f:
            if not line.startswith(b"data: {"):
                continue
            c = json.loads(line[6:])["choices"][0]
            n += len((c.get("logprobs") or {}).get("tokens") or [])
            marks.append((time.time(), n))
            if stop.is_set():
                break


def rate(m: list, a: float, b: float) -> float:
    pts = [x for x in m if a <= x[0] <= b]
    return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]) if len(pts) > 1 and pts[-1][0] > pts[0][0] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type = int, default = 4)
    ap.add_argument("--tokens", type = int, default = 131072)
    ap.add_argument("--reps", type = int, default = 3)
    a = ap.parse_args()
    keys = list(NATURAL)
    rows = []
    for _ in range(a.reps):
        marks = [[] for _ in range(a.users)]
        stop = threading.Event()
        ts = [threading.Thread(target = stream, daemon = True,
                               args = (forced(NATURAL[keys[i % 4]], 16384), marks[i], stop)) for i in range(a.users)]
        for t in ts:
            t.start()
        time.sleep(20)
        t0 = time.time()
        _, d = post({"prompt": cold_prose(a.tokens), "max_tokens": 1, "loop_detect_window": 0})
        t1 = time.time()
        stop.set()
        for t in ts:
            t.join(timeout = 60)
        base = [rate(m, t0 - 15, t0) for m in marks]
        during = [rate(m, t0, t1) for m in marks]
        row = dict(ttft_s = round(t1 - t0, 2), prompt_tokens = d["usage"]["prompt_tokens"],
                   decoder_base_tps = round(st.median(base), 1), decoder_during_tps = round(st.median(during), 1))
        print(json.dumps(row), flush = True)
        rows.append(row)
        time.sleep(3)
    print(json.dumps(dict(users = a.users, ttft_median_s = st.median(r["ttft_s"] for r in rows),
                          ttft_runs = [r["ttft_s"] for r in rows],
                          decoder_slowdown_pct = round(100 * (1 - st.median(
                              r["decoder_during_tps"] / max(r["decoder_base_tps"], 1e-9) for r in rows)), 1))))


if __name__ == "__main__":
    main()
