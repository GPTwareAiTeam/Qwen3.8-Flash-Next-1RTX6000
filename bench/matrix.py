import json, time, urllib.request, concurrent.futures as cf, random, string, statistics as st
U="http://127.0.0.1:5000/v1/completions"; M="Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8"
def post(body, timeout=1800):
    r=urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r, timeout=timeout)); return time.time()-t, d
def nonce(): return "".join(random.choices(string.hexdigits.lower(), k=10))
SEED={
 "prose":("The estuary narrows as the tide turns, and the sandbars that were dry at noon disappear under a "
   "sheet of moving water. Gulls follow the current line, and the fisherman reads it the way a driver reads "
   "a road: the colour change marks the channel, the smooth patch marks the shoal. "),
 "code":("def merge(a, b):\n    out = []\n    while a and b:\n        if a[0] <= b[0]:\n            out.append(a.pop(0))\n"
   "        else:\n            out.append(b.pop(0))\n    return out + a + b\n\n# usage:\n"),
 "chat":("User: I need to move about forty gigabytes of photos from an old external drive to a NAS, and I "
   "keep losing track of which folders are already copied.\nAssistant: "),
 "math":("Solve for x: 4x^2 - 9x + 2 = 0. Step 1: identify a, b and c. Step 2: compute the discriminant. "
   "Step 3: substitute into the quadratic formula and simplify both roots. "),
}
def prompt_for(tag, target_tokens):
    s=SEED[tag]; reps=max(1, int(target_tokens*3.7/len(s)))
    return f"<!--{tag}-{nonce()}-->\n" + s*reps
def one(tag, ptarget, nout):
    p=prompt_for(tag, ptarget)
    w,d=post({"model":M,"prompt":p,"max_tokens":nout,"temperature":0})
    u=d["usage"]; det=u.get("completion_tokens_details") or {}
    return u["prompt_tokens"], u["completion_tokens"], w, det.get("accepted_prediction_tokens") or 0, det.get("rejected_prediction_tokens") or 0
def med(xs): return st.median(xs)
# warmup: autotune + draft calibrator
post({"model":M,"prompt":"Warmup. " * 20,"max_tokens":256,"temperature":0})
print("=== concurrency (512 new tokens, mixed code prompt, median of 3)", flush=True)
CP="Write a Python implementation of a binary heap with push, pop, and heapify, plus a test harness."
res={}
for conc in (1,2,4,8):
    aggs=[]; pers=[]
    for _ in range(3):
        t=time.time()
        with cf.ThreadPoolExecutor(conc) as ex:
            out=list(ex.map(lambda _: post({"model":M,"prompt":CP,"max_tokens":512,"temperature":0}), range(conc)))
        wall=time.time()-t
        toks=sum(d["usage"]["completion_tokens"] for _,d in out)
        aggs.append(toks/wall); pers.append(med([d["usage"]["completion_tokens"]/w for w,d in out]))
    print(f"conc={conc}: agg median={med(aggs):6.1f} t/s (runs {[round(a,1) for a in aggs]})  per-stream median={med(pers):5.1f} t/s", flush=True)
print("=== prefill ~2048 tokens -> 1 output token (median of 3)", flush=True)
for tag in SEED:
    rows=[one(tag, 2048, 1) for _ in range(3)]
    pt=med([r[0] for r in rows]); w=med([r[2] for r in rows])
    print(f"{tag:5s}: ptok={pt} wall={w*1000:6.0f} ms prefill={pt/w:6.0f} tok/s (runs {[round(r[2]*1000) for r in rows]} ms)", flush=True)
print("=== ~128 prompt tokens -> 2048 decode (median of 3)", flush=True)
for tag in SEED:
    rows=[one(tag, 128, 2048) for _ in range(3)]
    ct=med([r[1] for r in rows]); w=med([r[2] for r in rows])
    acc=med([r[3] for r in rows]); rej=med([r[4] for r in rows])
    print(f"{tag:5s}: ctok={ct} wall={w:.2f}s decode={ct/w:6.1f} t/s acc={acc} rej={rej} (ctoks {[r[1] for r in rows]})", flush=True)
