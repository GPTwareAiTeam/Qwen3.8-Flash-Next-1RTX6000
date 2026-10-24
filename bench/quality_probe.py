import json,urllib.request,sys,hashlib
U="http://127.0.0.1:5000/v1/completions"; M="Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8"
Q=["2+2=","7*8=","17*23=","2**10=","The square root of 144 is","What is the capital of France? Answer:",
   "def add(a, b):\n    return","for i in range(3):\n    print(i)\n\n# prints","The first four prime numbers are",
   "In JSON, an array of three zeros looks like","SELECT id FROM users WHERE age > ","The chemical symbol for gold is"]
lab=sys.argv[1]; out={}
for q in Q:
    b=json.dumps({"model":M,"prompt":q,"max_tokens":24,"temperature":0}).encode()
    d=json.load(urllib.request.urlopen(urllib.request.Request(U,b,{"Content-Type":"application/json"}),timeout=300))
    out[q]=d["choices"][0]["text"]
json.dump(out, open(f'/tmp/qual_{lab}.json','w'), indent=1)
print(lab, "captured", len(out))
