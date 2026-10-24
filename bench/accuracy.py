import json,urllib.request,sys,re
U="http://127.0.0.1:5000/v1/completions"; M="Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8"
ITEMS=[("2+2=","4"),("7*8=","56"),("17*23=","391"),("13*13=","169"),("144/12=","12"),("2**10=","1024"),
("37+48=","85"),("91-57=","34"),("sqrt(225)=","15"),("5!=","120"),
("What is the capital of France? Answer:","Paris"),("What is the capital of Japan? Answer:","Tokyo"),
("What is the capital of Brazil? Answer:","Bras"),("What is the capital of Canada? Answer:","Ottawa"),
("The chemical symbol for gold is","Au"),("The chemical symbol for silver is","Ag"),
("The binary representation of 5 is","101"),("The hexadecimal representation of 255 is","FF"),
("len('abc') in Python is","3"),("sum(range(10)) in Python is","45"),
("In Python, 'a' in 'cat' evaluates to","True"),("The first four prime numbers are","2"),
("1 kilometre equals","1000"),("A byte has this many bits:","8")]
lab=sys.argv[1]; ok=0; fails=[]
for q,exp in ITEMS:
    b=json.dumps({"model":M,"prompt":q,"max_tokens":16,"temperature":0}).encode()
    t=json.load(urllib.request.urlopen(urllib.request.Request(U,b,{"Content-Type":"application/json"}),timeout=300))["choices"][0]["text"]
    good = exp.lower() in t.lower()
    ok += good
    if not good: fails.append((q,exp,t[:30]))
print(f"{lab}: {ok}/{len(ITEMS)} correct")
for f in fails: print("   miss", repr(f[0]), "want", f[1], "got", repr(f[2]))
