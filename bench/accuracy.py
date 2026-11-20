"""
Exact-answer probe: 50 unambiguous questions sent as chat messages (the model's chat template, thinking on), graded by
exact match on the final answer.

    BASE_URL=http://127.0.0.1:5000 API_KEY=... python bench/accuracy.py <label> [--out results.json] [--max-tokens 4096]

Every question asks for one answer and says how to give it ("Reply with only the number."). The server separates the
thinking (`reasoning_content`) from the answer (`content`, the text after </think>); the answer is trimmed (whitespace,
one pair of surrounding quotes or backticks, a trailing period) and compared case-sensitively with the expected
string. A reply that never closes its thinking within --max-tokens has no answer and counts as a miss. Greedy
(temperature 0), one request at a time.

`accuracy_raw.py` is the earlier raw-completion probe (24 prompts, answer searched anywhere in the continuation),
kept only to compare with results published before V3; it is not a quality gate.
"""
import argparse
import json
import os
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:5000")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8")
HDR = {"Content-Type": "application/json",
       **({"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {})}

N = "Reply with only the number."
W = "Reply with only the word."
S = "Reply with only the symbol."
B = "Reply with only True or False."

ITEMS = [
    # arithmetic
    (f"What is 2 + 2? {N}", "4"),
    (f"What is 7 × 8? {N}", "56"),
    (f"What is 17 × 23? {N}", "391"),
    (f"What is 13 squared? {N}", "169"),
    (f"What is 144 divided by 12? {N}", "12"),
    (f"What is 2 to the power of 10? {N}", "1024"),
    (f"What is 37 + 48? {N}", "85"),
    (f"What is 91 − 57? {N}", "34"),
    (f"What is the square root of 225? {N}", "15"),
    (f"What is 5 factorial (5!)? {N}", "120"),
    (f"What is 1000 − 1? {N}", "999"),
    (f"What is 25% of 80? {N}", "20"),
    # capitals
    (f"What is the capital of France? {W}", "Paris"),
    (f"What is the capital of Japan? {W}", "Tokyo"),
    (f"What is the capital of Brazil? {W}", "Brasília"),
    (f"What is the capital of Canada? {W}", "Ottawa"),
    (f"What is the capital of Australia? {W}", "Canberra"),
    (f"What is the capital of Germany? {W}", "Berlin"),
    (f"What is the capital of Egypt? {W}", "Cairo"),
    (f"What is the capital of Kenya? {W}", "Nairobi"),
    # chemistry
    (f"What is the chemical symbol for gold? {S}", "Au"),
    (f"What is the chemical symbol for silver? {S}", "Ag"),
    (f"What is the chemical symbol for sodium? {S}", "Na"),
    (f"What is the chemical symbol for iron? {S}", "Fe"),
    (f"What is the atomic number of carbon? {N}", "6"),
    (f"What is the atomic number of oxygen? {N}", "8"),
    (f"How many hydrogen atoms are in one molecule of water? {N}", "2"),
    (f"Which noble gas has the chemical symbol Ne? {W}", "Neon"),
    # number bases
    (f"What is the decimal number 5 written in binary? {N}", "101"),
    ("What is the decimal number 255 written in hexadecimal, using uppercase letters? Reply with only the hexadecimal "
     "digits, without a prefix.", "FF"),
    (f"What is the binary number 1010 in decimal? {N}", "10"),
    (f"What is the hexadecimal number 1F in decimal? {N}", "31"),
    (f"What is the decimal number 8 written in octal? {N}", "10"),
    (f"How many bits are in one byte? {N}", "8"),
    # Python
    (f"What does len('abc') return in Python? {N}", "3"),
    (f"What does sum(range(10)) return in Python? {N}", "45"),
    (f"What does 'a' in 'cat' evaluate to in Python? {B}", "True"),
    (f"What does 7 // 2 return in Python? {N}", "3"),
    (f"What does 7 % 3 return in Python? {N}", "1"),
    (f"What does len([1, 2, 3, 4]) return in Python? {N}", "4"),
    (f"What does bool(0) return in Python? {B}", "False"),
    (f"What does 2 ** 5 return in Python? {N}", "32"),
    ("What does 'ab' * 3 return in Python? Reply with only the resulting string, without quotes.", "ababab"),
    # units
    (f"How many metres are in one kilometre? {N}", "1000"),
    (f"How many centimetres are in one metre? {N}", "100"),
    (f"How many grams are in one kilogram? {N}", "1000"),
    (f"How many seconds are in one minute? {N}", "60"),
    (f"How many minutes are in one hour? {N}", "60"),
    (f"How many hours are in one day? {N}", "24"),
    (f"How many millilitres are in one litre? {N}", "1000"),
]


def ask(question: str, max_tokens: int) -> tuple[str, str, str]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": question}], "max_tokens": max_tokens,
            "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    d = json.load(urllib.request.urlopen(req, timeout = 900))
    c = d["choices"][0]
    m = c["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), (c.get("finish_reason") or "")


def normalize(answer: str) -> str:
    a = answer.strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'`":
        a = a[1:-1].strip()
    if a.endswith("."):
        a = a[:-1].rstrip()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type = int, default = 4096)
    a = ap.parse_args()
    rows, ok = [], 0
    for q, exp in ITEMS:
        content, reasoning, finish = ask(q, a.max_tokens)
        got = normalize(content)
        good = got == exp
        ok += good
        rows.append(dict(question = q, expected = exp, answer = content, normalized = got, correct = good,
                         finish_reason = finish, reasoning_chars = len(reasoning)))
    print(f"{a.label}: {ok}/{len(ITEMS)} correct")
    for r in rows:
        if not r["correct"]:
            print(f"   miss {r['question']!r}\n        want {r['expected']!r} got {r['answer']!r} (finish {r['finish_reason']})")
    if a.out:
        json.dump(dict(label = a.label, score = ok, n = len(ITEMS), rows = rows), open(a.out, "w"), indent = 1)


if __name__ == "__main__":
    main()
