"""
Per-round component breakdown of a prof_prefill.py (or any NVTX-ranged) nsys capture (sqlite export, one file per capture range).

Attribution: each kernel -> its launching runtime API call (correlationId; graph launches carry every node's kernel
with --cuda-graph-trace=node) -> innermost NVTX range open on that thread at the call. Kernels are
also bucketed by name, so graph-replayed MoE kernels launched from inside a GDN/attention graph are still visible.

    python analyze.py out/<tag>*.sqlite > out/<tag>_table.md
Reports µs per round: GPU kernel time per component, kernel count, round span (first kernel start -> last kernel
end), GPU-busy union, and idle gaps inside the span (launch / host / sync gaps).
"""
import bisect
import collections
import re
import sqlite3
import sys

NAME_BUCKETS = [
    ("moe_coop", r"moe_coop"),
    ("moe_other", r"exl3_moe|moe_|routing|topk_route|router"),
    ("gemv_exl3", r"exl3_gemv|gemv_int8|exl3_mgemv|exl3_gemm"),
    ("gdn", r"gdn|gated_delta|chunk_gated|fused_recurrent|causal_conv|delta_rule"),
    ("attn", r"flash|fmha|attn|attention|qsa|sparse_att|indexer|dsa_topk|paged"),
    ("hc", r"hc_mix|hyper|hc_"),
    ("norm_rope_act", r"norm|rope|silu|gelu|act_|softcap"),
    ("dflash", r"dflash"),
    ("sampler", r"sample|argmax|softmax|logit"),
    ("hgemm", r"gemm|cutlass|sm90|sm100|sm120|cublas|nvjet"),
    ("ngram_ple", r"ngram|ple_"),
    ("copy_misc", r"copy|fill|elementwise|reduce|index|cat|memset|vectorized|unrolled"),
]


def bucket(name):
    n = name.lower()
    for b, pat in NAME_BUCKETS:
        if re.search(pat, n):
            return b
    return "other"


def analyze(path):
    db = sqlite3.connect(path)
    strs = dict(db.execute("select id, value from StringIds"))
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    # NVTX push/pop ranges (eventType 59) with their thread
    nv = []
    for start, end, text, tid_, gtid in db.execute(
            "select start, end, text, textId, globalTid from NVTX_EVENTS where eventType in (59, 60) and end is not null"):
        label = text if text is not None else strs.get(tid_, "")
        nv.append((gtid, start, end, label))
    by_thread = collections.defaultdict(list)
    for gtid, s, e, l in nv:
        by_thread[gtid].append((s, e, l))
    for v in by_thread.values():
        v.sort()
    rounds = sorted([(s, e, l) for g, s, e, l in nv if l.startswith("round/")])
    # Steady state only: rounds longer than 2x the median host round (one-time Triton JIT compiles, ~220 ms each,
    # seen on new verify widths in the QSA sparse regime) are dropped and reported
    med = sorted(e - s for s, e, l in rounds)[len(rounds) // 2] if rounds else 0
    outliers = [round((e - s) / 1e6, 1) for s, e, l in rounds if e - s > 2 * med]
    rounds = [r for r in rounds if r[1] - r[0] <= 2 * med]

    api = {}
    for corr, s, gtid in db.execute("select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[corr] = (s, gtid)
    kern = []
    for s, e, corr, nid in db.execute("select start, end, correlationId, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"):
        kern.append((s, e, corr, strs.get(nid, "?")))
    mem = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        mem += [(s, e, c, "memcpy") for s, e, c in db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY")]
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        mem += [(s, e, c, "memset") for s, e, c in db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET")]

    starts = {g: [s for s, e, l in v] for g, v in by_thread.items()}

    def comp_of(gtid, t):
        # Push/pop ranges nest: the innermost range open at t is the latest-started one that has not ended
        rs = by_thread.get(gtid, [])
        i = bisect.bisect_right(starts.get(gtid, []), t) - 1
        while i >= 0:
            s, e, l = rs[i]
            if e >= t:
                return "round" if l.startswith("round/") else l
            i -= 1
        return "none"

    # round of a kernel = round whose API-call interval contains the launch
    rstarts = [r[0] for r in rounds]

    def round_of(t):
        i = bisect.bisect_right(rstarts, t) - 1
        if i >= 0 and rounds[i][0] <= t <= rounds[i][1]:
            return i
        return None

    comp_us = collections.defaultdict(float)
    comp_n = collections.Counter()
    bucket_us = collections.defaultdict(float)
    bucket_n = collections.Counter()
    top = collections.defaultdict(float)
    topn = collections.Counter()
    per_round = collections.defaultdict(list)
    for s, e, corr, name in kern + mem:
        a = api.get(corr)
        if a is None:
            continue
        ri = round_of(a[0])
        if ri is None:
            continue
        c = comp_of(a[1], a[0])
        d = (e - s) / 1e3
        comp_us[c] += d
        comp_n[c] += 1
        b = bucket(name) if name not in ("memcpy", "memset") else name
        bucket_us[b] += d
        bucket_n[b] += 1
        short = re.sub(r"\(.*", "", name)[:110]
        top[(c, short)] += d
        topn[(c, short)] += 1
        per_round[ri].append((s, e))

    nr = len(rounds)
    spans, busy, host = [], [], []
    for ri, iv in per_round.items():
        iv.sort()
        spans.append((iv[-1][1] - iv[0][0]) / 1e3)
        u = 0
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s > ce:
                u += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        u += ce - cs
        busy.append(u / 1e3)
        host.append((rounds[ri][1] - rounds[ri][0]) / 1e3)
    arm = rounds[0][2].split("/", 1)[1] if rounds else "?"
    return dict(arm = arm, rounds = nr, outliers = outliers, comp_us = comp_us, comp_n = comp_n, bucket_us = bucket_us, bucket_n = bucket_n,
                top = top, topn = topn, span = sum(spans) / max(1, len(spans)), busy = sum(busy) / max(1, len(busy)),
                host = sum(host) / max(1, len(host)))


def main():
    for path in sys.argv[1:]:
        r = analyze(path)
        n = max(1, r["rounds"])
        print(f"\n## {r['arm']}  ({path}, {r['rounds']} steady rounds; dropped outlier rounds ms: {r['outliers']})\n")
        print(f"host round (NVTX) {r['host']:.0f} µs · GPU span {r['span']:.0f} µs · GPU busy {r['busy']:.0f} µs · "
              f"idle in span {r['span'] - r['busy']:.0f} µs ({100 * (r['span'] - r['busy']) / max(1e-9, r['span']):.1f} %)\n")
        tot = sum(r["comp_us"].values())
        print("| component (NVTX of launch) | µs/round | % kernel time | kernels/round |\n|---|---|---|---|")
        for c, v in sorted(r["comp_us"].items(), key = lambda kv: -kv[1]):
            print(f"| {c} | {v / n:.0f} | {100 * v / tot:.1f} | {r['comp_n'][c] / n:.0f} |")
        print(f"| **total kernels** | **{tot / n:.0f}** | 100 | {sum(r['comp_n'].values()) / n:.0f} |")
        print("\n| kernel-name bucket | µs/round | % | launches/round |\n|---|---|---|---|")
        tb = sum(r["bucket_us"].values())
        for b, v in sorted(r["bucket_us"].items(), key = lambda kv: -kv[1]):
            print(f"| {b} | {v / n:.0f} | {100 * v / tb:.1f} | {r['bucket_n'][b] / n:.0f} |")
        print("\n| top kernels (component, name) | µs/round | calls/round | µs/call |\n|---|---|---|---|")
        for (c, k), v in sorted(r["top"].items(), key = lambda kv: -kv[1])[:30]:
            cnt = r["topn"][(c, k)]
            print(f"| {c} · `{k}` | {v / n:.0f} | {cnt / n:.1f} | {v / max(1, cnt):.1f} |")


if __name__ == "__main__":
    main()
