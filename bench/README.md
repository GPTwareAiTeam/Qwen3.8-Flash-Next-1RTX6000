# Bench harnesses

| script | what it measures |
|---|---|
| `matrix.py` | aggregate decode throughput at 1..16 concurrent streams |
| `decode_natural.py` | single-stream decode on prose / code / chat / math prompts |
| `accuracy.py` | fixed-answer accuracy set |
| `coldttft.py` | cold time-to-first-token at 2k / 32k / 128k |
| `identity_kl.py` | greedy identity on 8 prompts and next-token top-20 KL at short and long context |
| `prefill_under_load.py` | a cold long prompt arriving while other streams decode |
| `soak.py` | long mixed-load soak (errors, memory, latency drift) |
| `profile/` | nsys-based prefill profiler |

## Gate policy for changes (since 2026-11-19)

A change passes when, against the reference build measured in the same session:

- **top-1 agreement** of the next-token distribution matches on every probe prompt (short prompts and the
  8.7k / 35k / 157k long-context prompts), and
- **KL** (reference || candidate, over the reference's top-20) stays inside the noise floor measured by running the
  reference against itself, separately for short and long context.

Greedy text / token identity is reported for every change but is **not a hard gate**. The server is not deterministic
against itself: batched verification, the growable KV pool and kernel autotune choices move near-tied tokens between
sessions (a reference build compared with its own restart has flipped 1-2 of the 8 identity prompts, and concurrent
8-stream identity varies from 4/8 to 8/8 run to run). A greedy difference without a top-1 or KL change is a near-tie
flip, not a regression.

Speed claims use back-to-back A, A, B, B runs (one server restart per configuration) with matched host load.
