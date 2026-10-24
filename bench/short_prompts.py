import json,time,urllib.request,sys
U="http://127.0.0.1:5000/v1/completions"; M="Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8"
CODE=("def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs)//2]\n"
      "    lo=[x for x in xs if x<pivot]\n    hi=[x for x in xs if x>pivot]\n"
      "    return quicksort(lo)+[pivot]+quicksort(hi)\n\n# Explain in detail, then rewrite in-place:\n")
LIVE="Write a Python function that reverses a linked list."
PROSE="Explain why the sky is blue, starting from Rayleigh scattering and going through the perception side."
label=sys.argv[1] if len(sys.argv)>1 else "arm"
tot=[]
for tag,p in (("code",CODE),("live",LIVE),("prose",PROSE)):
    for i in range(2):
        body=json.dumps({"model":M,"prompt":p,"max_tokens":512,"temperature":0}).encode()
        t=time.time()
        d=json.load(urllib.request.urlopen(urllib.request.Request(U,body,{"Content-Type":"application/json"}), timeout=900))
        w=time.time()-t; u=d["usage"]; det=u.get("completion_tokens_details") or {}
        print(f"{label} {tag:5s} p{i} {u['completion_tokens']:4d}t {w*1000:6.0f}ms {u['completion_tokens']/w:6.1f} t/s acc={det.get('accepted_prediction_tokens')} rej={det.get('rejected_prediction_tokens')}", flush=True)
        if i==1: tot.append(u["completion_tokens"]/w)
print(f"{label} mean(code,live,prose) = {sum(tot)/len(tot):.1f} t/s", flush=True)
