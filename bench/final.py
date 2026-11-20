"""
Release benchmark: decode at 1/2/4/8/12 users x prose/chat/code/math x greedy/preset, cold prefill at 2k/32k/128k,
all with first-token timing (bench/timing.py). One JSON line per cell on stdout and in --out.

    BASE_URL=http://127.0.0.1:5000 API_KEY=... python bench/final.py --label B1 --out B1.jsonl \
        [--sections decode,prefill] [--conc 1,2,4,8,12] [--reps 3] [--tokens 512]

Decode cell: `conc` concurrent requests of the same NATURAL prompt (each with its own nonce line, so no request
reuses another's prefix), `--tokens` forced tokens each, `--reps` repetitions. Per repetition it records
  agg_decode_tps   sum of tokens after the first / time from the first request's first token to the last stream's end
  agg_e2e_tps      sum of tokens / wall time of the repetition
  user_decode_tps  median over streams of the per-stream decode speed from its first token
  ttft_s           median over streams of the time to first token
and the cell reports the median over repetitions with the min and max (spread).

Prefill cell: one cold prompt of ~N tokens (fresh nonce), 1 generated token, `--reps` repetitions; ttft_s and
prompt tokens / ttft_s.

Each section starts and ends with a GPU snapshot (`nvidia-smi -q -d PERFORMANCE` throttle reasons, power, SM / memory
clocks) and the host load average, emitted as "env" lines.
"""
import argparse
import concurrent.futures as cf
import json
import os
import statistics as st
import subprocess
import time

from common import NATURAL, SEED, nonce
from timing import stream_post

OUT = None


def emit(rec: dict):
    print(json.dumps(rec), flush = True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")


def gpu_env(gpu: str) -> dict:
    q = "power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,memory.used"
    v = subprocess.run(["nvidia-smi", "-i", gpu, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output = True, text = True).stdout.strip().split(", ")
    perf = subprocess.run(["nvidia-smi", "-i", gpu, "-q", "-d", "PERFORMANCE"], capture_output = True, text = True).stdout
    active = [l.split(":")[0].strip() for l in perf.splitlines() if l.strip().endswith(": Active")]
    keys = ["power_w", "power_limit_w", "sm_mhz", "mem_mhz", "temp_c", "vram_mib"]
    return dict(zip(keys, v), throttle_active = active, load1 = os.getloadavg()[0])


def spread(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    return dict(med = round(st.median(xs), 2), min = round(min(xs), 2), max = round(max(xs), 2)) if xs else None


def decode_cell(label, conc, tag, greedy, tokens, reps):
    runs = []
    for r in range(reps):
        def one(i):
            p = f"<!--{nonce()}-->\n" + NATURAL[tag]
            t_send = time.monotonic()
            res = stream_post({"prompt": p, "max_tokens": tokens, "min_tokens": tokens, "loop_detect_window": 0},
                              greedy = greedy)
            res["t_first"] = t_send + res["ttft_s"]
            res["t_end"] = t_send + res["wall_s"]
            return res
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(conc) as ex:
            out = list(ex.map(one, range(conc)))
        wall = time.monotonic() - t0
        n = sum(o["ctok"] for o in out)
        first = min(o["t_first"] for o in out)
        last = max(o["t_end"] for o in out)
        runs.append(dict(
            agg_decode = (n - conc) / (last - first) if last > first else None,
            agg_e2e = n / wall,
            user_decode = st.median(o["decode_tps"] for o in out if o["decode_tps"]),
            ttft = st.median(o["ttft_s"] for o in out),
            acc = sum(o["acc"] for o in out), rej = sum(o["rej"] for o in out)))
    emit(dict(label = label, section = "decode", conc = conc, tag = tag, sampling = "greedy" if greedy else "preset",
              tokens = tokens, reps = reps,
              agg_decode_tps = spread([x["agg_decode"] for x in runs]),
              agg_e2e_tps = spread([x["agg_e2e"] for x in runs]),
              user_decode_tps = spread([x["user_decode"] for x in runs]),
              ttft_s = spread([x["ttft"] for x in runs]),
              acc = [x["acc"] for x in runs], rej = [x["rej"] for x in runs], load1 = os.getloadavg()[0]))


def prefill_cell(label, target, reps):
    runs = []
    for _ in range(reps):
        p = f"<!--{nonce()}-->\n" + SEED["prose"] * int(target * 4.43 / len(SEED["prose"]))
        res = stream_post({"prompt": p, "max_tokens": 1, "loop_detect_window": 0})
        runs.append(res)
    emit(dict(label = label, section = "prefill", target = target, reps = reps, ptok = runs[0]["ptok"],
              cached = [r["cached"] for r in runs],
              ttft_s = spread([r["ttft_s"] for r in runs]),
              prefill_tps = spread([r["ptok"] / r["ttft_s"] for r in runs]), load1 = os.getloadavg()[0]))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required = True)
    ap.add_argument("--out")
    ap.add_argument("--sections", default = "decode,prefill")
    ap.add_argument("--conc", default = "1,2,4,8,12")
    ap.add_argument("--tags", default = "prose,chat,code,math")
    ap.add_argument("--sampling", default = "greedy,preset")
    ap.add_argument("--tokens", type = int, default = 512)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--prefill", default = "2048,32768,131072")
    ap.add_argument("--gpu", default = os.environ.get("GPU", "0"))
    a = ap.parse_args()
    OUT = a.out
    # warm-up: autotune and the drafter's confidence calibrator
    stream_post({"prompt": "Warmup. " * 20, "max_tokens": 128})
    for sec in a.sections.split(","):
        emit(dict(label = a.label, section = "env", at = f"{sec}_start", **gpu_env(a.gpu)))
        if sec == "decode":
            for sampling in a.sampling.split(","):
                for conc in [int(x) for x in a.conc.split(",")]:
                    for tag in a.tags.split(","):
                        decode_cell(a.label, conc, tag, sampling == "greedy", a.tokens, a.reps)
        elif sec == "prefill":
            for target in [int(x) for x in a.prefill.split(",")]:
                prefill_cell(a.label, target, a.reps)
        emit(dict(label = a.label, section = "env", at = f"{sec}_end", **gpu_env(a.gpu)))


if __name__ == "__main__":
    main()
