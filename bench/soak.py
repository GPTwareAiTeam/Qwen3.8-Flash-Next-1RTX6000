"""
Fill/drain soak: repeatedly fills the KV pool with concurrent cold long prompts, lets everything finish, then
probes decode speed and a cold long prompt (the growable KV pool soak in docs/measurements.md).

    BASE_URL=http://127.0.0.1:5000 API_KEY=... python bench/soak.py --minutes 90 --users 4 --tokens 400000

Each cycle: `users` concurrent cold prompts of ~`tokens` tokens (max_tokens 1) plus 2 forced 2048-token decodes;
when all finish, 1-user and 8-user decode rates (512 forced tokens) and the wall time of one cold ~131k prompt.
One JSON line per cycle; any request error is recorded in that cycle's `errors` and the soak continues. Watch the
server log for out-of-memory or illegal-memory-access errors alongside.
"""
import argparse
import concurrent.futures as cf
import json
import time

from common import CODE_P, NATURAL, cold_prose, forced, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type = float, default = 90)
    ap.add_argument("--users", type = int, default = 4)
    ap.add_argument("--tokens", type = int, default = 400_000)
    a = ap.parse_args()
    t_end = time.time() + a.minutes * 60
    cyc = 0
    while time.time() < t_end:
        cyc += 1
        rec = dict(cycle = cyc, t = time.strftime("%H:%M:%S"), errors = [])
        try:
            t = time.time()
            with cf.ThreadPoolExecutor(a.users + 2) as ex:
                futs = [ex.submit(post, {"prompt": cold_prose(a.tokens), "max_tokens": 1, "loop_detect_window": 0}, 7200)
                        for _ in range(a.users)]
                futs += [ex.submit(post, forced(NATURAL["code"], 2048)) for _ in range(2)]
                for f in futs:
                    f.result()
            rec["fill_s"] = round(time.time() - t, 1)
            for conc in (1, 8):
                t = time.time()
                with cf.ThreadPoolExecutor(conc) as ex:
                    out = list(ex.map(lambda _: post(forced(CODE_P, 512)), range(conc)))
                rec[f"c{conc}_tps"] = round(sum(d["usage"]["completion_tokens"] for _, d in out) / (time.time() - t), 1)
            w, _ = post({"prompt": cold_prose(131072), "max_tokens": 1, "loop_detect_window": 0})
            rec["cold131k_s"] = round(w, 2)
        except Exception as e:
            rec["errors"].append(repr(e)[:200])
        print(json.dumps(rec), flush = True)


if __name__ == "__main__":
    main()
