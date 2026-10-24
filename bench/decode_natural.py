import json,time,urllib.request,statistics as st
U="http://127.0.0.1:5000/v1/completions"; M="Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8"
def post(b,t=1800):
    r=urllib.request.Request(U,json.dumps(b).encode(),{"Content-Type":"application/json"})
    s=time.time(); d=json.load(urllib.request.urlopen(r,timeout=t)); return time.time()-s,d
P={
"prose":("Write a long, detailed essay about how a coastal town's economy changed after the fishing fleet "
 "shrank: who stayed, who left, what the harbour is used for now, and how the town's self-image adapted. "
 "Use concrete detail throughout and do not summarise."),
"code":("Write a complete Python module implementing an LRU cache with a doubly linked list and a dict, "
 "including type hints, docstrings, a thread lock, and a pytest test file that covers eviction order, "
 "capacity one, and concurrent access. Output code only."),
"chat":("You are a support agent. A customer says their laptop battery drains from full to 20% in ninety minutes "
 "and they have already tried a restart. Write the full troubleshooting reply: diagnostics to run, what each "
 "result means, when to replace the battery, and what to do if the problem persists."),
"math":("Derive the closed form for the sum of the first n squares, prove it by induction, then use it to "
 "compute the sum from 1 to 2500 and verify the result with a second method. Show every step."),
}
post({"model":M,"prompt":"Warmup. "*20,"max_tokens":128,"temperature":0})
print("=== natural ~60-token prompt -> forced 2048 decode, median of 3", flush=True)
for tag,p in P.items():
    rows=[]
    for _ in range(3):
        w,d=post({"model":M,"prompt":p,"max_tokens":2048,"min_tokens":2048,"temperature":0,"loop_detect_window":0})
        u=d["usage"]; det=d["usage"].get("completion_tokens_details") or {}
        rows.append((u["prompt_tokens"],u["completion_tokens"],w,det.get("accepted_prediction_tokens") or 0,det.get("rejected_prediction_tokens") or 0))
    ct=st.median([r[1] for r in rows]); w=st.median([r[2] for r in rows])
    print(f"{tag:5s}: ptok={st.median([r[0] for r in rows]):.0f} ctok={ct:.0f} wall={w:.2f}s decode={ct/w:6.1f} t/s acc={st.median([r[3] for r in rows]):.0f} rej={st.median([r[4] for r in rows]):.0f}", flush=True)
